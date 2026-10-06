#!/usr/bin/env python3
"""Find everything that could kill a multi-day training run, before it starts.

    python train/preflight.py --model tcm --data itu --repo ... --data-root ... --protocols ... \
        --scratch $SCR/itu_train --final-root $T/itu_ckpt

Run by train/preflight.sbatch on the same partition and env as the real job. Each
check prints PASS / WARN / FAIL and the script exits non-zero if anything FAILed.
A run that passes can still be unlucky (a node dies), but it cannot fail for a
reason that was knowable in advance:

  env        torch, CUDA, GPU arch supported by this torch build, fairseq imports
  resampler  8 kHz -> 16 kHz leaves 4-8 kHz empty (no imaging into the studied band)
  data       EVERY train and dev file in the protocol: exists, decodes, right rate,
             finite, not empty. One corrupt FLAC would otherwise crash epoch 1 at
             hour N, or worse, every epoch.
  memory     real XLS-R model, published batch, real clips: forward + backward +
             Adam step (Adam's state is allocated on the first step, so peak memory
             is only known after one), then a dev batch. Peak must leave headroom.
  speed      GPU step time and data-loader throughput -> projected epoch time and
             whole-run time against the chained job budget
  storage    a full resume checkpoint (model + Adam) is saved, timed, reloaded and
             compared on scratch; turbo is written to and checksummed; free space
"""
import argparse
import hashlib
import os
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T                                                # noqa: E402

RESULTS = []


def report(status, name, msg):
    RESULTS.append((status, name))
    print(f"[{status}] {name}: {msg}", flush=True)


# ---------------------------------------------------------------- checks

def check_env():
    import torch
    print(f"python {sys.version.split()[0]}  torch {torch.__version__}  cuda {torch.version.cuda}  "
          f"cudnn {torch.backends.cudnn.version()}", flush=True)
    if not torch.cuda.is_available():
        report("FAIL", "env", "no CUDA device")
        return False
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    archs = getattr(torch.cuda, "get_arch_list", lambda: [])()
    if archs and f"sm_{major}{minor}" not in archs and \
            not any(a.startswith("compute_") for a in archs):
        report("FAIL", "env", f"{name} is sm_{major}{minor}; this torch build has {archs}")
        return False
    try:                                   # catches "no kernel image is available"
        a = torch.randn(256, 256, device="cuda")
        (a @ a).sum().item()
    except Exception as e:
        report("FAIL", "env", f"CUDA kernel failed on {name}: {e}")
        return False
    try:
        import fairseq
        fv = fairseq.__version__
    except Exception as e:
        report("FAIL", "env", f"fairseq does not import: {e}")
        return False
    report("PASS", "env", f"{name} sm_{major}{minor}, fairseq {fv}")
    return True


def check_resampler():
    from harness import RESAMPLER, resample
    sr = 8000
    t = np.arange(sr * 2) / sr
    x = (0.5 * np.sin(2 * np.pi * 3000 * t)).astype("float32")   # near the 4 kHz edge
    y = np.asarray(resample(x, sr), dtype="float64")[1000:-1000]
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    f = np.fft.rfftfreq(len(y), 1 / 16000)
    leak = 10 * np.log10(spec[f > 4200].sum() / spec.sum() + 1e-30)
    status = "PASS" if leak < -60 else "FAIL"
    report(status, "resampler", f"{RESAMPLER}: energy above 4.2 kHz {leak:.1f} dB (need < -60)")


def _probe(args):
    path, expected = args
    try:
        x, sr = sf.read(path, dtype="float32", always_2d=True)
    except Exception as e:
        return path, f"decode error: {e}"
    if sr != expected:
        return path, f"{sr} Hz, expected {expected}"
    if len(x) == 0:
        return path, "empty"
    if not np.isfinite(x).all():
        return path, "NaN/Inf samples"
    if np.abs(x).max() == 0:
        return path, "all-zero"
    return None


def check_data(data, root, protocols, workers):
    expected, flac_dir = T.CONDITIONS[data]
    ok = True
    for subset in ("train", "dev"):
        t0 = time.time()
        try:
            _, paths, y = T.trials(protocols, subset, flac_dir(root, subset), 0, 0)
        except SystemExit as e:
            report("FAIL", f"data/{data}/{subset}", str(e))
            ok = False
            continue
        with Pool(workers) as pool:
            bad = [r for r in pool.imap_unordered(_probe, [(str(p), expected) for p in paths],
                                                  chunksize=64) if r]
        mins = (time.time() - t0) / 60
        if bad:
            report("FAIL", f"data/{data}/{subset}",
                   f"{len(bad)} of {len(paths)} bad, e.g. {bad[:3]}")
            ok = False
        else:
            report("PASS", f"data/{data}/{subset}",
                   f"all {len(paths)} files decode at {expected} Hz "
                   f"({int(y.sum())} bonafide / {int(len(y) - y.sum())} spoof), {mins:.1f} min")
    return ok


def check_gpu(model_name, data, repo, root, protocols, workers, chain_hours, max_epochs, accum,
              dev_batch):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from harness import Clips

    spec = T.MODELS[model_name]
    batch = spec["batch_size"]
    micro = batch // accum
    expected, flac_dir = T.CONDITIONS[data]
    _, tr_paths, tr_y = T.trials(protocols, "train", flac_dir(root, "train"), 0, 0)
    _, dv_paths, _ = T.trials(protocols, "dev", flac_dir(root, "dev"), 0, 0)

    # loader throughput on its own: is decoding + resampling off turbo the bottleneck?
    order = np.random.RandomState(0).permutation(len(tr_paths))[: micro * 40].tolist()
    loader = DataLoader(Clips(tr_paths, spec["window"]), batch_size=micro, sampler=order,
                        num_workers=workers, drop_last=True)
    it = iter(loader)
    next(it)                                # worker start-up is not throughput
    t0 = time.time()
    n = sum(1 for _ in it)
    load_s = (time.time() - t0) / max(n, 1)

    device = "cuda"
    torch.cuda.reset_peak_memory_stats()
    model = T.build_model(spec, repo, device)
    opt = torch.optim.Adam(model.parameters(), lr=spec["lr"], weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=torch.FloatTensor(T.CLASS_WEIGHT).to(device))
    total_mem = torch.cuda.get_device_properties(0).total_memory

    model.train()
    batches = DataLoader(Clips(tr_paths, spec["window"]), batch_size=micro, sampler=order,
                         num_workers=workers, drop_last=True)
    times = []
    try:
        for step, (x, idx, sr, _) in enumerate(batches, 1):
            torch.cuda.synchronize()
            t0 = time.time()
            y = torch.from_numpy(tr_y[idx.numpy()]).to(device)
            loss = crit(spec["logits"](model(x.to(device))), y)
            (loss / accum).backward()
            if step % accum == 0:
                opt.step()
                opt.zero_grad()
            torch.cuda.synchronize()
            times.append(time.time() - t0)
            if not torch.isfinite(loss):
                report("FAIL", "memory", f"non-finite loss {loss.item()} at step {step}")
                return None
            if step >= 3 * accum + 2:
                break
    except RuntimeError as e:
        if "out of memory" in str(e):
            report("FAIL", "memory", f"OOM at batch {batch} (micro {micro}) -- retry with ACCUM=2")
            return None
        raise
    train_peak = torch.cuda.max_memory_allocated()

    model.eval()
    dv = DataLoader(Clips(dv_paths, spec["window"]), batch_size=dev_batch, num_workers=workers)
    with torch.no_grad():
        x, _, _, _ = next(iter(dv))
        model(x.to(device))
    peak = torch.cuda.max_memory_allocated()
    frac = peak / total_mem
    status = "PASS" if frac < 0.85 else ("WARN" if frac < 0.95 else "FAIL")
    report(status, "memory", f"peak {peak / 2**30:.1f} of {total_mem / 2**30:.1f} GiB ({frac:.0%}); "
                             f"train {train_peak / 2**30:.1f} GiB at batch {batch} = {accum} x {micro}")

    # GPU time per micro-step, steady state (first steps include cudnn warm-up)
    gpu_s = float(np.median(times[2:])) if len(times) > 3 else float(np.median(times))
    step_s = max(gpu_s, load_s)
    steps = len(tr_paths) // micro
    train_min = steps * step_s / 60
    # dev is forward-only (~1/3 of a training step) per clip, or loader-bound
    dev_min = len(dv_paths) * max(gpu_s / 3, load_s) / micro / 60
    epoch_h = (train_min + dev_min) / 60
    budget_h = chain_hours
    worst_h = epoch_h * max_epochs
    bottleneck = "data loader" if load_s > gpu_s else "GPU"
    msg = (f"{gpu_s:.2f} s/step GPU, {load_s:.2f} s/batch loader ({bottleneck}-bound); "
           f"~{epoch_h:.2f} h/epoch, worst case {max_epochs} epochs = {worst_h:.0f} h; "
           f"chain budget {budget_h:.0f} h")
    report("PASS" if worst_h < 0.9 * budget_h else "WARN", "speed", msg)
    if load_s > gpu_s:
        print("       the loader is slower than the GPU: request more CPUs (--cpus-per-task)", flush=True)
    return model, opt


def check_storage(model, opt, scratch, final_root):
    import torch
    ok = True
    scratch, final_root = Path(scratch), Path(final_root)
    scratch.mkdir(parents=True, exist_ok=True)
    for where, path, need in (("scratch", scratch, 30), ("turbo", final_root.parent, 40)):
        free = shutil.disk_usage(str(path)).free / 2**30
        report("PASS" if free > need else "FAIL", f"space/{where}", f"{free:,.0f} GiB free at {path}")
        ok &= free > need

    if model is not None:
        ck = scratch / "_preflight_last.pth"
        t0 = time.time()
        T.atomic_save(dict(model=model.state_dict(), optimizer=opt.state_dict(),
                           state={}, rng=T.rng_state(), recipe={}), ck)
        save_s, size = time.time() - t0, ck.stat().st_size / 2**30
        t0 = time.time()
        back = T.load_resume(ck, "cpu")
        load_s = time.time() - t0
        k = next(iter(back["model"]))
        same = torch.equal(back["model"][k].cpu(), model.state_dict()[k].cpu()) \
            and len(back["optimizer"]["state"]) > 0
        ck.unlink()
        report("PASS" if same else "FAIL", "checkpoint",
               f"resume checkpoint {size:.2f} GiB: save {save_s:.0f} s, load {load_s:.0f} s, "
               f"round trip {'identical' if same else 'DIFFERS'}")
        ok &= same

    # turbo: the finished run's best.pth is copied here, so it must be writable
    probe = final_root / "_preflight_probe"
    try:
        final_root.mkdir(parents=True, exist_ok=True)
        blob = os.urandom(64 << 20)
        probe.write_bytes(blob)
        same = hashlib.sha256(probe.read_bytes()).digest() == hashlib.sha256(blob).digest()
        probe.unlink()
        report("PASS" if same else "FAIL", "turbo", f"{final_root} writable, checksum "
                                                    f"{'ok' if same else 'MISMATCH'}")
        ok &= same
    except OSError as e:
        report("FAIL", "turbo", f"cannot write {final_root}: {e}")
        ok = False
    return ok


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, choices=sorted(T.MODELS))
    ap.add_argument("--data", nargs="+", required=True, choices=sorted(T.CONDITIONS),
                    help="data trees to scan; the GPU checks use the first")
    ap.add_argument("--data-root", nargs="+", required=True, help="one per --data, same order")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--protocols", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--final-root", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--dev-batch-size", type=int, default=32)
    ap.add_argument("--max-epochs", type=int, default=50)
    ap.add_argument("--chain-hours", type=float, default=4 * 48)
    ap.add_argument("--skip-data", action="store_true", help="skip the full file scan")
    args = ap.parse_args()
    if len(args.data) != len(args.data_root):
        sys.exit("--data and --data-root must pair up")

    print("=== env", flush=True)
    if not check_env():
        sys.exit(1)
    print("=== resampler", flush=True)
    check_resampler()
    if not args.skip_data:
        for d, r in zip(args.data, args.data_root):
            print(f"=== data: {d} ({r})", flush=True)
            check_data(d, r, args.protocols, args.workers)
    print(f"=== GPU memory and speed: {args.model} on {args.data[0]}", flush=True)
    got = check_gpu(args.model, args.data[0], args.repo, args.data_root[0], args.protocols,
                    args.workers, args.chain_hours, args.max_epochs, args.accum,
                    args.dev_batch_size)
    print("=== storage", flush=True)
    check_storage(*(got or (None, None)), args.scratch, args.final_root)

    fails = [n for s, n in RESULTS if s == "FAIL"]
    warns = [n for s, n in RESULTS if s == "WARN"]
    print(f"\npreflight: {len(RESULTS) - len(fails) - len(warns)} pass, {len(warns)} warn, "
          f"{len(fails)} fail" + (f"  -> FAILED: {', '.join(fails)}" if fails else ""), flush=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
