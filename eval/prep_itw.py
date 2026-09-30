#!/usr/bin/env python3
"""Normalise In-the-Wild into a corpus the NB/NFR pipelines accept.

    python eval/prep_itw.py \\
        --meta /nfs/turbo/umd-hafiz/issf_server_data/ds_wild/protocols/meta.csv \\
        --audio /nfs/turbo/umd-hafiz/issf_server_data/ds_wild/release_in_the_wild \\
        --out /nfs/turbo/umd-hafiz/issf_server_data/ITW_16k

In-the-Wild is far cleaner than Famous Figures: uniform 16 kHz, no filename
collisions, nothing shorter than 0.5 s. So this does no resampling, no renaming
for uniqueness and no length filtering. Three things still need doing.

1. .wav -> .flac
   Both pipelines glob for "*.flac" only.

2. IDS NEED TWO UNDERSCORES
   harness.read_labels matches the utterance id on underscore count, so bare
   "1" or "ITW_1" would never be found. Ids are emitted as ITW_wild_00001.

   The speaker goes in the attack field so --per-attack gives per-speaker EER,
   which is meaningful for a celebrity corpus. Speaker names are stripped to
   bare alphanumerics -- "Alec Guinness" -> "AlecGuinness" -- deliberately
   leaving them with NO underscores: the CM line puts the speaker before the
   utterance id, so a slug like "Mary_Jo_White" would be picked up as the id
   instead.

3. THE LABEL COLUMN HAS TYPOS
   meta.csv contains 19958 "bona-fide", 4 "bonafide", 1 "bonafilde" and 11815
   "spoof". "bonafilde" is in no label vocabulary, so a strict reader drops that
   row without saying so. Labels are normalised here, at the boundary where the
   messy data enters, and every normalisation is counted and printed -- rather
   than teaching the shared LABEL_WORDS table to recognise a typo.

   Anything that is not recognisably bonafide or spoof is a hard error, not a
   silent skip.

Also note meta.csv has 31778 rows against 31779 audio files: 0.wav carries no
label. Unlabelled files are skipped and counted.

Output:
    <out>/ASVspoof2019_LA_eval/flac/<uid>.flac        16 kHz, PCM_16, mono
    <out>/protocol/ITW.cm.eval.trl.txt                ASVspoof CM format
    <out>/manifest/ITW_eval_<shard>.csv               per-file provenance

The directory name ASVspoof2019_LA_eval is a pipeline requirement -- both
builders hardcode that path -- not a claim about the data.
"""
import argparse
import csv
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import RESAMPLER, TARGET_SR, resample

BONA = {"bonafide", "bona-fide", "bona_fide", "bonafilde", "real", "genuine", "human"}
SPOOF = {"spoof", "fake", "spoofed", "synthetic"}
FIELDS = ["cond", "subset", "file_id", "speaker", "label", "raw_label", "src_rate",
          "src_path", "in_samples", "out_samples", "peak", "clipped", "resampler",
          "status"]


def norm_label(s):
    t = s.strip().lower()
    if t in BONA:
        return "bonafide"
    if t in SPOOF:
        return "spoof"
    return None


def slug(name):
    """Speaker -> bare alphanumerics, no underscores (see module docstring)."""
    return re.sub(r"[^A-Za-z0-9]", "", name) or "unknown"


def read_meta(path):
    """meta.csv -> [(stem, speaker, label, raw_label)]"""
    rows, bad, counts = [], [], Counter()
    with open(path, newline="") as fh:
        for line in fh:
            f = [t.strip() for t in line.rstrip("\n").split(",")]
            if len(f) < 3 or not f[0]:
                continue
            if f[0].lower() in ("filename", "file", "audiofilename"):
                continue                                  # header, if present
            stem = f[0][:-4] if f[0].lower().endswith(".wav") else f[0]
            lab = norm_label(f[2])
            counts[f[2].strip()] += 1
            if lab is None:
                bad.append((f[0], f[2]))
                continue
            rows.append((stem, f[1], lab, f[2].strip()))
    if bad:
        sys.exit(f"{len(bad)} row(s) with an unrecognised label, e.g. {bad[:3]}.\n"
                 f"Add the spelling to BONA/SPOOF in this file if it is legitimate.")
    if not rows:
        sys.exit(f"no usable rows in {path}")
    print("  label column as found:", dict(counts))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta", required=True, help="ds_wild/protocols/meta.csv")
    ap.add_argument("--audio", required=True, help="ds_wild/release_in_the_wild")
    ap.add_argument("--out", required=True)
    ap.add_argument("--subset", default="eval")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-samples", type=int, default=4000,
                    help="drop clips shorter than this many 16 kHz samples. The DSP "
                         "methods use ~875-tap FIR designs and filtfilt needs input "
                         "longer than 2625 samples. In-the-Wild's shortest clip is "
                         "0.506 s (8096 samples) so nothing should be dropped -- this "
                         "is a guard, and any drop is reported.")
    args = ap.parse_args()

    rows = read_meta(args.meta)
    if args.nshards > 1:
        rows = rows[args.shard::args.nshards]
    if args.limit:
        rows = rows[:args.limit]

    out_dir = Path(args.out) / f"ASVspoof2019_LA_{args.subset}" / "flac"
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("protocol", "manifest"):
        (Path(args.out) / sub).mkdir(parents=True, exist_ok=True)

    print(f"meta       {args.meta}  ({len(rows)} rows in this shard)")
    print(f"audio      {args.audio}")
    print(f"out        {out_dir}")
    print(f"resampler  {RESAMPLER}  ->  {TARGET_SR} Hz", flush=True)

    man, prot, rates, n_clip, short, missing, failed, t0 = [], [], Counter(), 0, 0, 0, 0, time.time()
    for i, (stem, speaker, label, raw) in enumerate(rows, 1):
        uid = f"ITW_wild_{stem}"
        r = dict.fromkeys(FIELDS, "")
        r.update(cond="ITW", subset=args.subset, file_id=uid, speaker=speaker,
                 label=label, raw_label=raw, resampler=RESAMPLER, status="ok")
        src = Path(args.audio) / f"{stem}.wav"
        r["src_path"] = str(src)
        dst = out_dir / f"{uid}.flac"
        try:
            if not src.is_file():
                missing += 1
                r["status"] = "missing source"
                man.append(r)
                continue
            if dst.exists() and not args.overwrite:
                r["status"] = "exists"
                r["out_samples"] = sf.info(str(dst)).frames
            else:
                x, sr = sf.read(str(src), dtype="float32", always_2d=True)
                x = x.mean(axis=1)
                rates[sr] += 1
                r["src_rate"], r["in_samples"] = sr, len(x)
                y = np.asarray(resample(x, sr), dtype="float64")
                if args.min_samples and len(y) < args.min_samples:
                    short += 1
                    r.update(out_samples=len(y),
                             status=f"dropped: {len(y)} < {args.min_samples} samples")
                    man.append(r)
                    continue
                peak = float(np.max(np.abs(y))) if len(y) else 0.0
                i16 = np.round(y * 32767.0)
                clipped = int(np.count_nonzero((i16 > 32767) | (i16 < -32768)))
                i16 = np.clip(i16, -32768, 32767).astype("<i2")
                sf.write(str(dst), i16, TARGET_SR, subtype="PCM_16")
                r.update(out_samples=len(i16), peak=round(peak, 6), clipped=clipped)
                n_clip += clipped
            # CM format: <speaker> <utt> - <attack> <label>. Speaker slug carries no
            # underscores so it cannot be mistaken for the utterance id.
            sp = slug(speaker)
            prot.append(f"{sp} {uid} - {'-' if label == 'bonafide' else sp} {label}")
        except Exception as e:
            failed += 1
            r["status"] = f"FAILED: {e}"
        man.append(r)
        if i % 2000 == 0 or i == len(rows):
            rate = i / max(time.time() - t0, 1e-9)
            print(f"  {i}/{len(rows)}  {rate:.0f} files/s  "
                  f"eta {(len(rows) - i) / max(rate, 1e-9) / 60:.1f} min", flush=True)

    tag = f"{args.subset}_{args.shard:04d}"
    mpath = Path(args.out) / "manifest" / f"ITW_{tag}.csv"
    with open(mpath, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(man)
    ppath = Path(args.out) / "protocol" / f"ITW.cm.{args.subset}.trl.txt"
    mode = "a" if (args.shard and ppath.exists()) else "w"
    with open(ppath, mode) as fh:
        fh.write("\n".join(prot) + "\n")

    kept = [r for r in man if not str(r["status"]).startswith(("dropped", "FAILED", "missing"))]
    n_bona = sum(1 for r in kept if r["label"] == "bonafide")
    print(f"\n  wrote {mpath}")
    print(f"  wrote {ppath}  ({len(prot)} trials)")
    print(f"  {len(kept)}/{len(man)} kept"
          + (f", {short} too short" if short else "")
          + (f", {missing} missing source" if missing else "")
          + (f", {failed} FAILED" if failed else ""))
    print(f"  {n_bona} bonafide / {len(kept) - n_bona} spoof")
    print(f"  source rates: {dict(rates)}")
    if n_clip:
        print(f"  NOTE: {n_clip} samples clipped at full scale (not corrected)")
    if len(rates) > 1:
        print("\n  WARNING: mixed source rates. Check they do not correlate with the "
              "label before attributing any result to a condition.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
