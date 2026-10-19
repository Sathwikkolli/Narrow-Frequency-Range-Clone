"""Train a detector with its authors' own recipe: `train.py --recipe official`.

    python train/train.py --recipe official --model aasist --data wb --seed 1234 --run-dir ... \
        --repo ... --data-root ... --protocols ...

The shared recipe in train.py holds everything but the architecture fixed across
models. This one does the opposite: each model is trained the way its repo's own
main.py trains it, so the result is a replica of the released model, not a
controlled variant of it. What each repo does, read from its code (README command
where the code default differs):

    model    RawBoost   dev set                   stop                      checkpoint kept
    rawnet2  none       clean                     100 epochs                best dev accuracy
    aasist   algo 5     augmented (algo 5)        100 epochs                lowest dev loss
    tcm      algo 5     augmented (algo 5)        dev loss, patience 7,     mean of the 5 lowest
                                                  cap 75                    dev losses
    sls      algo 3     none (commented out)      train loss, patience 1,   lowest train loss
                                                  cap 50
    nes2net  algo 4     augmented AND clean       100 epochs                mean of 3: epochs that set
                                                                            a new min of either dev
                                                                            loss, by lowest clean EER

lr, batch, Adam with weight decay 1e-4 and CE weighted [0.1, 0.9] are the same as
train.py's MODELS and are taken from there.

THE REPO'S OWN DATA PATH. Training clips come from each repo's own Dataset class
(data_utils.py / data_utils_SSL.py), imported from the clone: librosa.load at
16 kHz, the repo's RawBoost on the whole utterance, then its own pad/crop. Nothing
in the data path is re-implemented here. RawBoost's parameters are the argparse
defaults, which are identical in all four XLS-R repos.

DELIBERATE DEPARTURES, all of them about reproducibility, none about the recipe:
  - Shuffle order is a permutation drawn from (seed, epoch), not torch's shuffle,
    so a run resumed in a new 7-hour job visits the data exactly as an
    uninterrupted one would.
  - Every DataLoader worker reseeds numpy from (seed, epoch, stream, worker).
    The repos leave numpy unseeded in workers, so on their torch the forked
    workers start from the same numpy state and RawBoost repeats its "random"
    draws across workers and epochs.
  - Only the epochs that can still be selected are kept on disk (K files), not
    every epoch; the outcome is the same.
  - nes2net: the README says to average "the epochs with the smallest validation
    EERs" among those main.py saved, without fixing how many; its released model
    averages 3 (epochs 59, 61, 69), so 3 are averaged here.
  - After training, the final model is scored once on the clean dev set and
    written to dev_scores_best.csv. This is reporting only; it selects nothing.

LFCC-LCNN is not here yet: its recipe (whole utterances in length-sorted batches,
StepLR) does not fit a fixed-window Dataset and is added separately.
"""
import csv
import os
import random
import socket
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import train as T

# ---------------------------------------------------------------- per-model recipe

# argparse defaults of the RawBoost options; the same in SSL_Anti-spoofing,
# tcm_add, SLSforADD and Nes2Net_ASVspoof_ITW. The three *_ratio options only
# exist in Nes2Net and are read only for its algo >= 10.
RAWBOOST = dict(nBands=5, minF=20, maxF=8000, minBW=100, maxBW=1000,
                minCoeff=10, maxCoeff=100, minG=0, maxG=0,
                minBiasLinNonLin=5, maxBiasLinNonLin=20, N_f=5,
                P=10, g_sd=2, SNRmin=10, SNRmax=40,
                LnL_ratio=1.0, ISD_ratio=1.0, SSI_ratio=1.0)

# dev:     (name, RawBoost algo) per dev pass, in the order main.py runs them
# stop:    None = run every epoch; else stop after `patience` epochs in a row
#          without a strict improvement of `key`
# select:  keep the `average` best epochs by `key`, among those `eligible`
OFFICIAL = {
    "rawnet2": dict(
        where="LA/Baseline-RawNet2/main.py",
        dataset=("data_utils", "Dataset_ASVspoof2019_train"), rawboost=False, algo=0,
        dev=[("dev", 0)], dev_batch=32,                     # dev_loader: args.batch_size
        max_epochs=100, stop=None,
        # main.py saves every epoch and prints the best by valid_accuracy (strict >)
        select=dict(key="dev.acc", mode="max"), average=1,
    ),
    "aasist": dict(
        where="SSL_Anti-spoofing/main_SSL_LA.py",
        dataset=("data_utils_SSL", "Dataset_ASVspoof2019_train"), rawboost=True, algo=5,
        dev=[("dev_aug", 5)], dev_batch=14,                 # dev set built with algo=args.algo
        max_epochs=100, stop=None,
        # main.py saves every epoch and logs only val_loss, so that is the criterion
        select=dict(key="dev_aug.loss", mode="min"), average=1,
    ),
    "tcm": dict(
        where="tcm_add/main.py",
        dataset=("data_utils", "Dataset_train"), rawboost=True, algo=5,
        dev=[("dev_aug", 5)], dev_batch=8,
        # `while not_improving < args.num_epochs` (7) and `if epoch > 74: break`
        max_epochs=75, stop=dict(key="dev_aug.loss", patience=7),
        # n_mejores_loss = 5 best by val_loss, then average_model over all 5
        select=dict(key="dev_aug.loss", mode="min"), average=5,
    ),
    "sls": dict(
        where="SLSforADD/main.py",
        dataset=("data_utils_SSL", "Dataset_ASVspoof2019_train"), rawboost=True, algo=3,
        dev=[], dev_batch=5,                                # the dev set is commented out
        # README --num_epochs=50; `if patience_counter >= 1: break` on running_loss,
        # "best model is epoch-1", i.e. the epoch with the lowest training loss
        max_epochs=50, stop=dict(key="train_loss", patience=1),
        select=dict(key="train_loss", mode="min"), average=1,
    ),
    "nes2net": dict(
        where="Nes2Net_ASVspoof_ITW/main.py",
        dataset=("data_utils_SSL", "Dataset_ASVspoof2019_train"), rawboost=True,
        algo=4,                                             # README command; code default is 8
        dev=[("dev_aug", 4), ("dev", 0)], dev_batch=12,
        max_epochs=100, stop=None,
        # main.py saves an epoch when either dev loss reaches a new minimum (<=)
        select=dict(key="dev.eer", mode="min", eligible="new_min_dev_loss"), average=3,
    ),
}


# ---------------------------------------------------------------- data

class Indexed(torch.utils.data.Dataset):
    """The repo's Dataset, with each item's position appended so scores line up."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds[i]
        return x, y, i


class Seeder:
    """worker_init_fn: numpy/random/torch in each worker from (seed, epoch, stream, worker)."""

    def __init__(self, seed, epoch, stream):
        self.key = [int(seed), int(epoch), int(stream)]

    def __call__(self, worker_id):
        s = int(np.random.SeedSequence(self.key + [int(worker_id)]).generate_state(1)[0])
        np.random.seed(s)
        random.seed(s)
        torch.manual_seed(s)


def repo_dataset(off, repo):
    """The Dataset class from the repo's own data_utils module."""
    import importlib
    with T.in_repo(repo):
        mod = importlib.import_module(off["dataset"][0])
    return getattr(mod, off["dataset"][1])


def make_set(cls, off, ids, labels, base_dir, algo):
    if off["rawboost"]:
        ds = cls(Namespace(**RAWBOOST), list_IDs=ids, labels=labels, base_dir=base_dir, algo=algo)
    else:
        if algo:
            sys.exit("a model without RawBoost was given a RawBoost algo")
        ds = cls(list_IDs=ids, labels=labels, base_dir=base_dir)
    return Indexed(ds)


def loader(ds, batch, workers, seeder, order=None, drop_last=False):
    if workers == 0:
        seeder(0)                     # no workers: the main process draws RawBoost's numbers
    return DataLoader(ds, batch_size=batch, sampler=order, shuffle=False, drop_last=drop_last,
                      num_workers=workers, pin_memory=True, worker_init_fn=seeder)


# ---------------------------------------------------------------- epochs

def train_epoch(model, spec, dl, optimizer, criterion, device, accum, epoch):
    """main.py's train_epoch: running_loss = sum(batch_loss * batch_size) / num_total."""
    model.train()
    total, n, t0 = 0.0, 0, time.time()
    optimizer.zero_grad()
    steps = len(dl)
    for step, (x, y, _) in enumerate(dl, 1):
        y = y.view(-1).long().to(device)
        out = spec["logits"](model(x.to(device, non_blocking=True)))
        loss = criterion(out, y)
        (loss / accum).backward()
        if step % accum == 0 or step == steps:
            optimizer.step()
            optimizer.zero_grad()
        total += loss.item() * len(y)
        n += len(y)
        if step % 200 == 0 or step == steps:
            rate = step / (time.time() - t0)
            print(f"  epoch {epoch} step {step}/{steps}  loss {total / n:.5f}  "
                  f"{rate:.2f} it/s  eta {(steps - step) / rate / 60:.1f} min", flush=True)
    return total / max(n, 1)


def dev_pass(model, spec, dl, labels, criterion, device):
    """Weighted-CE loss as main.py's evaluate_accuracy, argmax accuracy as RawNet2's,
    and EER on the bonafide logit; plus the scores, in dataset order."""
    model.eval()
    scores = np.empty(len(labels), dtype=np.float64)
    total, n, correct = 0.0, 0, 0
    with torch.no_grad():
        for x, y, idx in dl:
            y = y.view(-1).long().to(device)
            out = spec["logits"](model(x.to(device, non_blocking=True)))
            total += criterion(out, y).item() * len(y)
            n += len(y)
            correct += (out.argmax(1) == y).sum().item()
            scores[idx.numpy()] = out[:, 1].float().cpu().numpy()
    e, _ = T.eer(scores[labels == 1], scores[labels == 0])
    return dict(loss=total / max(n, 1), acc=100.0 * correct / max(n, 1), eer=e), scores


# ---------------------------------------------------------------- selection

def better(a, b, mode):
    return a < b if mode == "min" else a > b


def eligible(rule, metrics, state):
    """nes2net: `val_loss <= loss_min` for either dev pass, against the minimum so far."""
    if rule is None:
        return True
    if rule == "new_min_dev_loss":
        mins = state["dev_loss_min"]
        hit = False
        for k in ("dev_aug.loss", "dev.loss"):
            if metrics[k] <= mins.get(k, float("inf")):
                mins[k] = metrics[k]
                hit = True
        return hit
    raise ValueError(rule)


def update_topk(state, epoch, value, k, mode, run, model):
    """Keep the k best epochs on disk. A tie never displaces a kept epoch, as the
    repos' strict comparisons. Returns True if this epoch was kept."""
    kept = state["topk"]
    if len(kept) >= k and not better(value, kept[-1]["value"], mode):
        return False
    path = run / "topk" / f"epoch_{epoch}.pth"
    path.parent.mkdir(exist_ok=True)
    T.atomic_save(model.state_dict(), path)
    kept.append(dict(epoch=epoch, value=value, path=str(path)))
    # stable sort: among equal values the earlier epoch stays ahead
    kept.sort(key=lambda d: d["value"], reverse=(mode == "max"))
    while len(kept) > k:
        gone = kept.pop()
        if os.path.exists(gone["path"]):
            os.remove(gone["path"])
    return True


def average(paths):
    """TCM's / Nes2Net's average_model: the elementwise mean of the state_dicts.
    Integer buffers (BatchNorm's num_batches_tracked) are taken from the first."""
    sds = [torch.load(p, map_location="cpu") for p in paths]
    out = {}
    for k, v in sds[0].items():
        if torch.is_floating_point(v):
            out[k] = sum(sd[k] for sd in sds) / len(sds)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------- main

def run(args):
    if args.model not in OFFICIAL:
        sys.exit(f"--recipe official: no official recipe for {args.model} yet "
                 f"(have {', '.join(sorted(OFFICIAL))})")
    if args.data != "wb":
        sys.exit("--recipe official trains on the original 16 kHz ASVspoof2019 LA only (--data wb)")
    spec, off = T.MODELS[args.model], OFFICIAL[args.model]
    lr = args.lr if args.lr is not None else spec["lr"]
    batch = args.batch_size or spec["batch_size"]
    if batch % args.accum:
        sys.exit(f"--batch-size {batch} is not divisible by --accum {args.accum}")
    micro = batch // args.accum
    max_epochs = args.max_epochs or off["max_epochs"]
    dev_batch = args.dev_batch_size or off["dev_batch"]
    sel = off["select"]

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    done, last, best = run_dir / "DONE.json", run_dir / "last.pth", run_dir / "best.pth"
    if done.exists():
        if last.exists() and not args.keep_last:
            last.unlink()
        print(f"{run_dir.name} already finished:\n{done.read_text()}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        sys.exit("no CUDA device -- refusing to train on CPU")

    recipe = dict(recipe="official", where=off["where"], model=args.model, data=args.data,
                  seed=args.seed, lr=lr, batch_size=batch, accum=args.accum, weight_decay=1e-4,
                  optimizer="Adam", scheduler=None, loss="weighted CE",
                  class_weight=T.CLASS_WEIGHT, rawboost_algo=off["algo"],
                  rawboost=RAWBOOST if off["rawboost"] else None,
                  dev=off["dev"], max_epochs=max_epochs, stop=off["stop"], select=sel,
                  average=off["average"], limit_train=args.limit_train,
                  limit_dev=args.limit_dev)

    T.seed_everything(args.seed)
    flac_dir = T.CONDITIONS["wb"][1]
    tr_ids, _, tr_y = T.trials(args.protocols, "train", flac_dir(args.data_root, "train"),
                               args.limit_train, args.seed)
    dv_ids, _, dv_y = T.trials(args.protocols, "dev", flac_dir(args.data_root, "dev"),
                               args.limit_dev, args.seed)
    base = {s: str(Path(args.data_root) / f"ASVspoof2019_LA_{s}") + "/" for s in ("train", "dev")}
    tr_lab = dict(zip(tr_ids, tr_y.tolist()))
    dv_lab = dict(zip(dv_ids, dv_y.tolist()))

    model = T.build_model(spec, args.repo, device)
    cls = repo_dataset(off, args.repo)
    train_set = make_set(cls, off, tr_ids, tr_lab, base["train"], off["algo"])
    dev_sets = [(name, make_set(cls, off, dv_ids, dv_lab, base["dev"], algo))
                for name, algo in off["dev"]]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(T.CLASS_WEIGHT).to(device))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    state = dict(epoch=0, history=[], topk=[], dev_loss_min={}, stop_best=None,
                 since_best=0, stop=None)
    if last.exists():
        ck = T.load_resume(last, "cpu")
        if ck["recipe"] != recipe:
            diff = {k: (ck["recipe"].get(k), v) for k, v in recipe.items() if ck["recipe"].get(k) != v}
            sys.exit(f"{last} was started with a different recipe {diff} -- refusing to resume")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        state = ck["state"]
        T.set_rng_state(ck["rng"])
        del ck
        print(f"resumed after epoch {state['epoch']}; kept {[d['epoch'] for d in state['topk']]}",
              flush=True)

    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        T.write_json(dict(
            recipe=recipe, source=spec["source"], trainable_params=n_params,
            n_train=len(tr_ids), n_dev=len(dv_ids),
            data_root=args.data_root, protocols=args.protocols,
            protocol_sha256={k: T.sha256(Path(args.protocols) / v) for k, v in T.PROTOCOL.items()},
            repo=args.repo, repo_commit=T.git_head(args.repo), clone_commit=T.git_head(T.HERE.parent),
            python=sys.version.split()[0], torch=torch.__version__,
            cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
            gpu=torch.cuda.get_device_name(0), host=socket.gethostname(),
            started=time.strftime("%Y-%m-%d %H:%M:%S")), cfg_path)
    with open(run_dir / "starts.log", "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  job {os.environ.get('SLURM_JOB_ID', '-')}  "
                 f"{socket.gethostname()}  {torch.cuda.get_device_name(0)}  "
                 f"from epoch {state['epoch']}\n")

    print(f"{args.model} / official ({off['where']}) / seed {args.seed}: lr {lr}, batch {batch} "
          f"({args.accum} x {micro}), RawBoost algo {off['algo']}, dev {off['dev'] or 'none'}, "
          f"<= {max_epochs} epochs, keep {off['average']} by {sel['key']}, "
          f"{n_params / 1e6:.1f}M trainable", flush=True)

    metric_names = ["train_loss"] + [f"{d}.{m}" for d, _ in off["dev"] for m in ("loss", "acc", "eer")]
    log_path = run_dir / "train_log.csv"
    if not log_path.exists():
        with open(log_path, "w", newline="") as fh:
            csv.writer(fh).writerow(["epoch"] + metric_names + ["kept", "minutes", "finished"])

    while state["stop"] is None:
        epoch = state["epoch"] + 1
        t0 = time.time()
        order = np.random.RandomState(args.seed * 1000 + epoch).permutation(len(train_set)).tolist()
        dl = loader(train_set, micro, args.workers, Seeder(args.seed, epoch, 0),
                    order=order, drop_last=True)
        metrics = dict(train_loss=train_epoch(model, spec, dl, optimizer, criterion,
                                              device, args.accum, epoch))
        for s, (name, ds) in enumerate(dev_sets, 1):
            m, _ = dev_pass(model, spec, loader(ds, dev_batch, args.workers,
                                                Seeder(args.seed, epoch, s)),
                            dv_y, criterion, device)
            for k, v in m.items():
                metrics[f"{name}.{k}"] = v

        kept = False
        if eligible(sel.get("eligible"), metrics, state):
            kept = update_topk(state, epoch, metrics[sel["key"]], off["average"],
                               sel["mode"], run_dir, model)

        stop = off["stop"]
        if stop is not None:
            v = metrics[stop["key"]]
            if state["stop_best"] is None or v < state["stop_best"]:
                state["stop_best"], state["since_best"] = v, 0
            else:
                state["since_best"] += 1
            if state["since_best"] >= stop["patience"]:
                state["stop"] = f"no {stop['key']} improvement for {stop['patience']} epoch(s)"
        if state["stop"] is None and epoch >= max_epochs:
            state["stop"] = f"reached {max_epochs} epochs"

        state["epoch"] = epoch
        minutes = (time.time() - t0) / 60
        state["history"].append(dict(epoch=epoch, minutes=minutes, **metrics))
        T.atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                           state=state, rng=T.rng_state(), recipe=recipe), last)
        with open(log_path, "a", newline="") as fh:
            csv.writer(fh).writerow([epoch] + [f"{metrics[k]:.6f}" for k in metric_names]
                                    + [int(kept), f"{minutes:.1f}",
                                       time.strftime("%Y-%m-%d %H:%M:%S")])
        shown = "  ".join(f"{k} {metrics[k]:.5f}" for k in metric_names)
        print(f"epoch {epoch}: {shown}{'  * kept' if kept else ''}  "
              f"kept {[d['epoch'] for d in state['topk']]}  {minutes:.1f} min", flush=True)
        if state["stop"] is None and epoch == args.stop_after_epoch:
            print(f"--stop-after-epoch {epoch}: exiting; launch again to resume", flush=True)
            return

    # ---- final model: the kept epoch, or the mean of the kept epochs
    epochs = [d["epoch"] for d in state["topk"]]
    if not epochs:
        sys.exit("training ended with no kept epoch")
    final = average([d["path"] for d in state["topk"]])
    T.atomic_save(final, best)
    model.load_state_dict(final)

    # reporting only: the final model on the clean dev set
    clean = make_set(cls, off, dv_ids, dv_lab, base["dev"], 0)
    m, scores = dev_pass(model, spec, loader(clean, dev_batch, args.workers,
                                             Seeder(args.seed, 0, 99)),
                         dv_y, criterion, device)
    with open(run_dir / "dev_scores_best.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["utt", "label", "score"])
        for u, lab, s in zip(dv_ids, dv_y, scores):
            w.writerow([u, "bonafide" if lab else "spoof", f"{s:.6f}"])

    summary = dict(run=run_dir.name, recipe="official", stop_reason=state["stop"],
                   epochs=state["epoch"], final_from_epochs=epochs,
                   averaged=len(epochs) > 1, selected_by=sel["key"],
                   final_clean_dev_eer=m["eer"], final_clean_dev_loss=m["loss"],
                   best_sha256=T.sha256(best), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    T.write_json(summary, done)
    if not args.keep_last:
        last.unlink()
    print(f"done: {state['stop']}. final model from epoch(s) {epochs}, clean dev EER "
          f"{m['eer'] * 100:.3f}% -> {best}", flush=True)
