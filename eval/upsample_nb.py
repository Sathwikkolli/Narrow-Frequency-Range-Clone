#!/usr/bin/env python3
"""Write 16 kHz copies of the 8 kHz ITU conditions, for LCNN only.

    python eval/upsample_nb.py --nb-root $NB_ROOT --out $OUT --cond B1
    python eval/upsample_nb.py ... --shard 3 --nshards 20     # SLURM array

WHY THIS EXISTS. Five of the six detectors go through eval/harness.py, which
resamples per file from the header on load, so they never need this. LCNN does
not: it is a project-NN-Pytorch-scripts project driven by its own scripts, and
its config hardcodes

    wav_samp_rate = 16000

as a constant, not a conversion. Feeding it the 8 kHz B1/C1 files would compute
LFCCs against the wrong frame rate and filter spacing -- scores that look like a
narrowband effect but are a configuration bug.

Only B1 and C1 need this. RAW, P0 and F0-F5 are already 16 kHz.

THE RESAMPLER IS IMPORTED FROM harness.py ON PURPOSE. LCNN must see the same
samples the other five models see, or a difference between LCNN and the rest
would be partly the resampler. Sharing the function makes that identity provable
rather than asserted: soxr HQ where available, scipy resample_poly otherwise,
both sinc/polyphase. A naive resampler would image energy into the empty 4-8 kHz
band, which is precisely the band under study.

CLIPPING is reported, never corrected. Correcting it would be a per-file gain
change -- the level normalisation this corpus deliberately does not do.
"""
import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import RESAMPLER, TARGET_SR, resample

FIELDS = ["cond", "subset", "file_id", "in_rate", "out_rate", "in_samples",
          "out_samples", "peak", "clipped", "resampler", "status"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nb-root", required=True, help="the 8 kHz corpus root")
    ap.add_argument("--out", required=True, help="where the 16 kHz copies go")
    ap.add_argument("--cond", required=True, help="B1 or C1")
    ap.add_argument("--subset", default="eval")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true",
                    help="rebuild files that already exist (default: skip and resume)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    src_dir = Path(args.nb_root) / f"ASVspoof2019_LA_{args.subset}" / args.cond / "flac"
    if not src_dir.is_dir():
        sys.exit(f"not a directory: {src_dir}")
    out_dir = Path(args.out) / f"ASVspoof2019_LA_{args.subset}" / args.cond / "flac"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob("*.flac"))
    if args.nshards > 1:
        files = files[args.shard::args.nshards]
    if args.limit:
        files = files[:args.limit]
    if not files:
        sys.exit("no files in this shard")

    man_dir = Path(args.out) / "manifest"
    man_dir.mkdir(parents=True, exist_ok=True)
    man = man_dir / f"{args.cond}_{args.subset}_{args.shard:04d}.csv"

    print(f"src        {src_dir}  ({len(files)} files in this shard)")
    print(f"out        {out_dir}")
    print(f"resampler  {RESAMPLER}  ->  {TARGET_SR} Hz", flush=True)

    rows, t0, n_clip, n_skip, failed, rates = [], time.time(), 0, 0, 0, {}
    for i, srcp in enumerate(files, 1):
        fid = srcp.stem
        dst = out_dir / f"{fid}.flac"
        row = dict.fromkeys(FIELDS, "")
        row.update(cond=args.cond, subset=args.subset, file_id=fid,
                   out_rate=TARGET_SR, resampler=RESAMPLER, status="ok")
        try:
            if dst.exists() and not args.overwrite:
                n_skip += 1
                row["status"] = "exists"
                rows.append(row)
                continue
            x, sr = sf.read(str(srcp), dtype="float32", always_2d=True)
            x = x.mean(axis=1)
            rates[sr] = rates.get(sr, 0) + 1
            row["in_rate"], row["in_samples"] = sr, len(x)
            y = np.asarray(resample(x, sr), dtype="float64")
            peak = float(np.max(np.abs(y))) if len(y) else 0.0
            i16 = np.round(y * 32767.0)
            clipped = int(np.count_nonzero((i16 > 32767) | (i16 < -32768)))
            i16 = np.clip(i16, -32768, 32767).astype("<i2")
            sf.write(str(dst), i16, TARGET_SR, subtype="PCM_16")
            row.update(out_samples=len(i16), peak=round(peak, 6), clipped=clipped)
            n_clip += clipped
        except Exception as e:
            failed += 1
            row["status"] = f"FAILED: {e}"
        rows.append(row)
        if i % 2000 == 0 or i == len(files):
            rate = i / max(time.time() - t0, 1e-9)
            print(f"  {i}/{len(files)}  {rate:.0f} files/s  "
                  f"eta {(len(files) - i) / max(rate, 1e-9) / 60:.1f} min", flush=True)

    with open(man, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"  wrote {man}")
    print(f"  {len(rows) - failed}/{len(rows)} ok"
          + (f", {n_skip} already existed" if n_skip else "")
          + (f", {failed} FAILED" if failed else ""))
    print(f"  input rates seen: {rates}")
    if n_clip:
        # Resampling a band-limited signal can ring slightly past full scale. It is
        # reported so it can be audited, not corrected.
        print(f"  NOTE: {n_clip} samples clipped at full scale (not corrected)")
    if rates and set(rates) == {TARGET_SR}:
        print(f"  WARNING: every input was already {TARGET_SR} Hz -- this condition "
              f"did not need upsampling; check --cond")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
