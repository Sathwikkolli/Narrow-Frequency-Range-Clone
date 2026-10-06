#!/usr/bin/env python3
"""Train one XLS-R detector on ITU telephone audio (C1) or on the wideband original.

    python train/train.py --model tcm --data itu --seed 1234 --run-dir $SCR/itu_train/runs/tcm_itu_s1
    python train/train.py ... --limit-train 2000 --limit-dev 1000 --max-epochs 2   # smoke test

Normally launched through train/submit.sh, which sets every path.

ONE RECIPE FOR EVERY MODEL. Only the architecture and its published learning rate
and batch size are model-specific (MODELS below). Everything a reviewer could
attribute to the recipe rather than the data is shared and fixed:

    data        itu  C1 train (8 kHz IRS8 + G.711), resampled to 16 kHz on load
                wb   original ASVspoof2019 LA train, 16 kHz, untouched
                Same 25,380 utterances and labels either way, so an epoch is the
                same number of steps for both and the only difference is the audio.
    selection   dev EER every epoch on the matching dev set (C1 dev or original dev);
                the single best epoch is kept. No checkpoint averaging.
    stopping    patience 7 epochs on dev EER, hard cap 50 epochs
    loss        cross-entropy weighted [0.1 spoof, 0.9 bonafide], as all four repos
    optimiser   Adam, weight decay 1e-4, no scheduler, as all four repos
    augment     none. RawBoost's noise spans 0-8 kHz and would put energy back into
                the 4-8 kHz band that the telephone channel removed.

SAME AUDIO PATH AS EVAL. Clips are read, resampled and windowed by eval/harness.py's
own Clips dataset, so a trained model sees exactly the samples it is later scored on.
In particular C1 goes through the harness resampler, never librosa: a resampler that
images energy into the empty 4-8 kHz band would be indistinguishable from the effect
under study. The per-file rate is checked against what the condition must be (8 kHz
for C1, 16 kHz for wideband), so pointing at the wrong tree fails immediately.

STORAGE. Two checkpoints, both in --run-dir (scratch, never home):
    best.pth   plain state_dict of the best epoch, loadable by eval/harness.py as is
    last.pth   model + Adam state + RNG, for resuming; deleted when the run ends
A finished run writes DONE.json; launching it again is then a no-op, which is what
lets submit.sh queue several chained jobs without knowing how many epochs it needs.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent / "eval"
sys.path.insert(0, str(EVAL))

from harness import RESAMPLER, TARGET_SR, Clips, read_labels   # noqa: E402
from compute_metrics import eer                                 # noqa: E402
from adapters._common import in_repo, ns, require_xlsr          # noqa: E402


# ---------------------------------------------------------------- per-model spec
#
# lr / batch are each repo's published values. arch is what the repo's Model()
# reads from its argparse Namespace. window and the score column are the same ones
# the eval adapters use (eval/adapters/<model>.py), so train and eval agree.

def _tcm_logits(out):
    return out[0]                     # Model.forward returns (logits, attention)


MODELS = {
    "tcm": dict(
        source="github.com/ductuantruong/tcm_add",
        module="model", cls="Model",
        arch=dict(emb_size=144, heads=4, kernel_size=31, num_encoders=4),
        window=66800,                 # data_utils.py: self.cut = 66800
        lr=1e-6, batch_size=20,       # main.py defaults, README: python main.py --algo 5
        base_seed=1234,               # main.py default
        logits=_tcm_logits,
    ),
}

CONDITIONS = {
    # cond: (expected source rate, path of a subset's flac folder under its root)
    "itu": (8000, lambda root, subset: Path(root) / f"ASVspoof2019_LA_{subset}" / "C1" / "flac"),
    "wb": (16000, lambda root, subset: Path(root) / f"ASVspoof2019_LA_{subset}" / "flac"),
}

PROTOCOL = {"train": "ASVspoof2019.LA.cm.train.trn.txt", "dev": "ASVspoof2019.LA.cm.dev.trl.txt"}
CLASS_WEIGHT = [0.1, 0.9]             # [spoof, bonafide]; label 1 = bonafide, as in the repos


# ---------------------------------------------------------------- data

def trials(protocols, subset, flac_dir, limit, seed):
    """Protocol -> (paths, labels). Every listed file must exist; a gap is fatal."""
    labels = read_labels(Path(protocols) / PROTOCOL[subset])
    ids = sorted(labels)
    if limit:
        # smoke runs only: a fixed random subset, both classes guaranteed by size
        ids = sorted(np.random.RandomState(seed).choice(ids, min(limit, len(ids)), replace=False))
    paths = [flac_dir / f"{u}.flac" for u in ids]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        sys.exit(f"{subset}: {len(missing)} of {len(paths)} protocol files missing, e.g. {missing[:3]}")
    y = np.array([1 if labels[u][0] == "bonafide" else 0 for u in ids], dtype=np.int64)
    print(f"{subset}: {len(ids)} trials ({int(y.sum())} bonafide / {int(len(y) - y.sum())} spoof) "
          f"from {flac_dir}", flush=True)
    return ids, paths, y


def check_rate(sr, expected, where):
    bad = (sr != expected).nonzero()
    if len(bad):
        sys.exit(f"{where}: got a {int(sr[bad[0]])} Hz file, this condition must be {expected} Hz "
                 f"-- wrong data tree?")


# ---------------------------------------------------------------- model

def build_model(spec, repo, device):
    import importlib
    with in_repo(repo) as r:
        require_xlsr(r)
        Model = getattr(importlib.import_module(spec["module"]), spec["cls"])
        model = Model(ns(**spec["arch"]), device)
    # Moved to the device before the first forward: the repos' SSLModel.extract_feat
    # calls self.model.train() when it finds XLS-R on the wrong device or dtype,
    # which would silently re-enable dropout during dev scoring.
    return model.to(device)


# ---------------------------------------------------------------- bookkeeping

def git_head(path):
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_save(obj, path):
    """Write beside, then rename: a job killed mid-save never leaves a torn checkpoint."""
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_resume(path, device):
    """last.pth holds numpy/python RNG state, which torch >= 2.6 refuses by default
    (weights_only=True). torch 1.8, which the XLS-R repos pin, has no such argument
    and would pass it on to pickle, so it is only given where it exists."""
    import inspect
    kw = {"weights_only": False} if "weights_only" in inspect.signature(torch.load).parameters else {}
    return torch.load(path, map_location=device, **kw)


def write_json(obj, path):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def set_rng_state(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True     # as the repos' reproducibility()
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------- epochs

def train_epoch(model, spec, loader, labels, optimizer, criterion, device, accum, expected_sr, epoch):
    model.train()
    total, n, t0 = 0.0, 0, time.time()
    optimizer.zero_grad()
    steps = len(loader)
    for step, (x, idx, sr, _) in enumerate(loader, 1):
        check_rate(sr, expected_sr, "train")
        y = torch.from_numpy(labels[idx.numpy()]).to(device)
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


def dev_epoch(model, spec, loader, labels, criterion, device, expected_sr):
    model.eval()
    scores = np.empty(len(labels), dtype=np.float64)
    total, n = 0.0, 0
    with torch.no_grad():
        for x, idx, sr, _ in loader:
            check_rate(sr, expected_sr, "dev")
            y = torch.from_numpy(labels[idx.numpy()]).to(device)
            out = spec["logits"](model(x.to(device, non_blocking=True)))
            total += criterion(out, y).item() * len(y)
            n += len(y)
            scores[idx.numpy()] = out[:, 1].float().cpu().numpy()   # bonafide logit, as eval
    e, _ = eer(scores[labels == 1], scores[labels == 0])
    return e, total / max(n, 1), scores


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--data", required=True, choices=sorted(CONDITIONS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--run-dir", required=True, help="checkpoints and logs (scratch)")
    ap.add_argument("--repo", required=True, help="clone of the model's official repo")
    ap.add_argument("--data-root", required=True,
                    help="itu: AsvSpoofData_2019_NB root; wb: the folder holding ASVspoof2019_LA_<subset>")
    ap.add_argument("--protocols", required=True, help="folder holding the 2019 LA train/dev protocols")
    ap.add_argument("--max-epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--lr", type=float, help="default: the model's published value")
    ap.add_argument("--batch-size", type=int, help="default: the model's published value")
    ap.add_argument("--accum", type=int, default=1,
                    help="gradient accumulation; only if the published batch does not fit the GPU. "
                         "Effective batch stays --batch-size, which is split into --accum steps")
    ap.add_argument("--dev-batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit-train", type=int, default=0, help="smoke test only")
    ap.add_argument("--limit-dev", type=int, default=0, help="smoke test only")
    ap.add_argument("--keep-last", action="store_true", help="keep last.pth after the run ends")
    ap.add_argument("--stop-after-epoch", type=int, default=0,
                    help="exit cleanly after this epoch as if the job hit its time limit "
                         "(preflight's resume test); not part of the recipe")
    args = ap.parse_args()

    spec = MODELS[args.model]
    lr = args.lr if args.lr is not None else spec["lr"]
    batch = args.batch_size or spec["batch_size"]
    if batch % args.accum:
        sys.exit(f"--batch-size {batch} is not divisible by --accum {args.accum}")
    micro = batch // args.accum
    expected_sr, flac_dir = CONDITIONS[args.data]

    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    done, last, best = run / "DONE.json", run / "last.pth", run / "best.pth"
    if done.exists():
        if last.exists() and not args.keep_last:      # killed between DONE and cleanup
            last.unlink()
        print(f"{run.name} already finished:\n{done.read_text()}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        sys.exit("no CUDA device -- refusing to train an XLS-R model on CPU")

    # The recipe. On resume it must match what the run was started with exactly,
    # or the checkpoint would continue a different experiment.
    recipe = dict(model=args.model, data=args.data, seed=args.seed, lr=lr, batch_size=batch,
                  accum=args.accum, weight_decay=1e-4, optimizer="Adam", scheduler=None,
                  loss="weighted CE", class_weight=CLASS_WEIGHT, augmentation=None,
                  window=spec["window"], arch=spec["arch"], max_epochs=args.max_epochs,
                  patience=args.patience, selection="min dev EER, single best epoch",
                  limit_train=args.limit_train, limit_dev=args.limit_dev,
                  condition="C1" if args.data == "itu" else "RAW", source_rate=expected_sr,
                  resampler=RESAMPLER, target_rate=TARGET_SR)

    seed_everything(args.seed)
    _, tr_paths, tr_y = trials(args.protocols, "train", flac_dir(args.data_root, "train"),
                               args.limit_train, args.seed)
    _, dv_paths, dv_y = trials(args.protocols, "dev", flac_dir(args.data_root, "dev"),
                               args.limit_dev, args.seed)
    dv_ids = [p.stem for p in dv_paths]

    model = build_model(spec, args.repo, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(CLASS_WEIGHT).to(device))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    state = dict(epoch=0, best_eer=float("inf"), best_epoch=None, since_best=0, history=[])
    if last.exists():
        ck = load_resume(last, device)
        if ck["recipe"] != recipe:
            diff = {k: (ck["recipe"].get(k), v) for k, v in recipe.items() if ck["recipe"].get(k) != v}
            sys.exit(f"{last} was started with a different recipe {diff} -- refusing to resume")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        state = ck["state"]
        set_rng_state(ck["rng"])
        del ck
        print(f"resumed after epoch {state['epoch']} (best EER {state['best_eer'] * 100:.3f}% "
              f"at epoch {state['best_epoch']})", flush=True)

    cfg_path = run / "config.json"
    if not cfg_path.exists():
        write_json(dict(
            recipe=recipe, source=spec["source"], trainable_params=n_params,
            n_train=len(tr_paths), n_dev=len(dv_paths),
            data_root=args.data_root, protocols=args.protocols,
            protocol_sha256={k: sha256(Path(args.protocols) / v) for k, v in PROTOCOL.items()},
            repo=args.repo, repo_commit=git_head(args.repo), clone_commit=git_head(HERE.parent),
            python=platform.python_version(), torch=torch.__version__,
            cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
            tf32_matmul=getattr(torch.backends.cuda.matmul, "allow_tf32", None),
            gpu=torch.cuda.get_device_name(0), host=socket.gethostname(),
            started=time.strftime("%Y-%m-%d %H:%M:%S")), cfg_path)
    with open(run / "starts.log", "a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  job {os.environ.get('SLURM_JOB_ID', '-')}  "
                 f"{socket.gethostname()}  {torch.cuda.get_device_name(0)}  "
                 f"from epoch {state['epoch']}\n")

    print(f"{args.model} / {args.data} / seed {args.seed}: lr {lr}, batch {batch} "
          f"({args.accum} x {micro}), {n_params / 1e6:.1f}M trainable, {RESAMPLER}", flush=True)

    dev_loader = DataLoader(Clips(dv_paths, spec["window"]), batch_size=args.dev_batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=True)
    log_path = run / "train_log.csv"
    fields = ["epoch", "train_loss", "dev_loss", "dev_eer", "best", "minutes", "finished"]
    if not log_path.exists():
        with open(log_path, "w", newline="") as fh:
            csv.writer(fh).writerow(fields)

    stop = None
    while stop is None:
        epoch = state["epoch"] + 1
        t0 = time.time()
        # Shuffle order is a function of (seed, epoch) alone, so a resumed run
        # visits the data in exactly the order an uninterrupted one would.
        order = np.random.RandomState(args.seed * 1000 + epoch).permutation(len(tr_paths)).tolist()
        train_loader = DataLoader(Clips(tr_paths, spec["window"]), batch_size=micro,
                                  sampler=order, drop_last=True,
                                  num_workers=args.workers, pin_memory=True)
        tr_loss = train_epoch(model, spec, train_loader, tr_y, optimizer, criterion,
                              device, args.accum, expected_sr, epoch)
        dv_eer, dv_loss, dv_scores = dev_epoch(model, spec, dev_loader, dv_y, criterion,
                                               device, expected_sr)

        improved = dv_eer < state["best_eer"]
        if improved:
            atomic_save(model.state_dict(), best)
            with open(run / "dev_scores_best.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["utt", "label", "score"])
                for u, lab, s in zip(dv_ids, dv_y, dv_scores):
                    w.writerow([u, "bonafide" if lab else "spoof", f"{s:.6f}"])
            state.update(best_eer=dv_eer, best_epoch=epoch, since_best=0)
        else:
            state["since_best"] += 1
        state["epoch"] = epoch
        minutes = (time.time() - t0) / 60
        state["history"].append(dict(epoch=epoch, train_loss=tr_loss, dev_loss=dv_loss,
                                     dev_eer=dv_eer, minutes=minutes))

        if state["since_best"] >= args.patience:
            stop = f"no dev EER improvement for {args.patience} epochs"
        elif epoch >= args.max_epochs:
            stop = f"reached max {args.max_epochs} epochs"

        atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                         state=state, rng=rng_state(), recipe=recipe), last)
        with open(log_path, "a", newline="") as fh:
            csv.writer(fh).writerow([epoch, f"{tr_loss:.6f}", f"{dv_loss:.6f}", f"{dv_eer:.6f}",
                                     int(improved), f"{minutes:.1f}",
                                     time.strftime("%Y-%m-%d %H:%M:%S")])
        print(f"epoch {epoch}: train loss {tr_loss:.5f}  dev loss {dv_loss:.5f}  "
              f"dev EER {dv_eer * 100:.3f}%{'  * best' if improved else ''}  "
              f"(best {state['best_eer'] * 100:.3f}% @ {state['best_epoch']}, "
              f"{state['since_best']}/{args.patience})  {minutes:.1f} min", flush=True)
        if stop is None and epoch == args.stop_after_epoch:
            print(f"--stop-after-epoch {epoch}: exiting; launch again to resume", flush=True)
            return

    summary = dict(run=run.name, stop_reason=stop, epochs=state["epoch"],
                   best_epoch=state["best_epoch"], best_dev_eer=state["best_eer"],
                   best_sha256=sha256(best), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    write_json(summary, done)
    if not args.keep_last:
        last.unlink()
    print(f"done: {stop}. best dev EER {state['best_eer'] * 100:.3f}% at epoch "
          f"{state['best_epoch']} -> {best}", flush=True)


if __name__ == "__main__":
    main()
