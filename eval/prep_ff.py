#!/usr/bin/env python3
"""Normalise a Famous Figures speaker into a corpus the NB/NFR pipelines accept.

    python eval/prep_ff.py --protocol /nfs/.../famousfigures/protocol.txt \\
                           --speaker Donald_Trump \\
                           --out /scratch/.../FF_Donald_Trump_16k

Famous Figures cannot be fed to dataset/ or nfr_dataset/ as it stands. Three
things are in the way, and this fixes all three in one pass.

1. MIXED SAMPLE RATES, CORRELATED WITH THE LABEL
   bonafide and two attacks are 16 kHz; six attacks are 24 kHz; two are 44.1 kHz.
   nfr_dataset/build_shard.py rejects anything that is not 16 kHz outright, and
   the ITU chain assumes 16 kHz input.

   More importantly this is a confound in the corpus itself: 83% of spoofs are
   identifiable from the file header alone. Resampling removes the header cue but
   leaves a rolloff signature near 8 kHz that natively-16 kHz files do not have.
   Since that signature lives in the band narrowband filtering discards, a
   detector leaning on it would show a large apparent narrowband effect that is
   really the confound being removed. The `src_rate` column is written to the
   manifest so results can be split by native rate and this can be checked
   rather than assumed.

2. FILENAMES AND AudioNames BOTH COLLIDE
   11,320 files share 7,912 basenames, and the protocol's own AudioName field has
   655 duplicates (STYLETTS2 uses plain names, COZYVOICE2 composite ones). The
   unique key is Source + AudioName, so ids are built as

       FF_<SOURCE>_<AudioName>          e.g. FF_STYLETTS2_Donald_Trump_00001
       FF_BONAFIDE_<AudioName>          for the "-" directory

3. LAYOUT
   Both pipelines hardcode `ASVspoof2019_LA_<subset>/flac`, so the output uses
   that path inside its own root. The directory name is a pipeline requirement,
   not a claim that this is ASVspoof data.

Output:
    <out>/ASVspoof2019_LA_eval/flac/<uid>.flac      16 kHz, PCM_16, mono
    <out>/protocol/FF_<speaker>.cm.eval.trl.txt     ASVspoof CM format
    <out>/manifest/FF_<speaker>_eval_<shard>.csv    per-file provenance

The resampler is imported from harness.py so these files are built with the same
sinc/polyphase conversion every model later uses on load.
"""
import argparse
import csv
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import RESAMPLER, TARGET_SR, resample

BONAFIDE_DIR = "-"
FIELDS = ["cond", "subset", "file_id", "speaker", "source", "label", "src_rate",
          "src_path", "in_samples", "out_samples", "peak", "clipped", "resampler",
          "status"]


def read_protocol(path, speaker):
    """FF protocol -> [(uid, source, label, src_path)], deduped on uid."""
    rows, seen, dupes = [], set(), 0
    with open(path) as fh:
        header = fh.readline()
        if "AudioName" not in header:       # not a header after all; rewind
            fh.seek(0)
        for line in fh:
            f = [t.strip() for t in line.rstrip("\n").split("\t")]
            if len(f) < 5 or f[1] != speaker:
                continue
            name, src, label, src_path = f[0], f[2], f[3].lower(), f[4]
            tag = "BONAFIDE" if src == BONAFIDE_DIR else src
            uid = f"FF_{tag}_{name}"
            if uid in seen:
                dupes += 1
                continue
            seen.add(uid)
            rows.append((uid, tag, label, src_path))
    if not rows:
        sys.exit(f"no rows for speaker {speaker!r} in {path}")
    if dupes:
        print(f"  {dupes} duplicate ids dropped (Source+AudioName still collided)",
              flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", required=True, help="famousfigures/protocol.txt")
    ap.add_argument("--speaker", required=True, help="e.g. Donald_Trump")
    ap.add_argument("--out", required=True, help="output corpus root")
    ap.add_argument("--subset", default="eval")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = read_protocol(args.protocol, args.speaker)
    if args.nshards > 1:
        rows = rows[args.shard::args.nshards]
    if args.limit:
        rows = rows[:args.limit]

    out_dir = Path(args.out) / f"ASVspoof2019_LA_{args.subset}" / "flac"
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("protocol", "manifest"):
        (Path(args.out) / sub).mkdir(parents=True, exist_ok=True)

    print(f"speaker    {args.speaker}  ({len(rows)} files in this shard)")
    print(f"out        {out_dir}")
    print(f"resampler  {RESAMPLER}  ->  {TARGET_SR} Hz", flush=True)

    man, prot, rates, n_clip, failed, t0 = [], [], Counter(), 0, 0, time.time()
    for i, (uid, source, label, src_path) in enumerate(rows, 1):
        r = dict.fromkeys(FIELDS, "")
        r.update(cond="FF", subset=args.subset, file_id=uid, speaker=args.speaker,
                 source=source, label=label, src_path=src_path,
                 resampler=RESAMPLER, status="ok")
        dst = out_dir / f"{uid}.flac"
        try:
            if dst.exists() and not args.overwrite:
                r["status"] = "exists"
            else:
                x, sr = sf.read(src_path, dtype="float32", always_2d=True)
                x = x.mean(axis=1)
                rates[sr] += 1
                r["src_rate"], r["in_samples"] = sr, len(x)
                y = np.asarray(resample(x, sr), dtype="float64")
                peak = float(np.max(np.abs(y))) if len(y) else 0.0
                i16 = np.round(y * 32767.0)
                clipped = int(np.count_nonzero((i16 > 32767) | (i16 < -32768)))
                i16 = np.clip(i16, -32768, 32767).astype("<i2")
                sf.write(str(dst), i16, TARGET_SR, subtype="PCM_16")
                r.update(out_samples=len(i16), peak=round(peak, 6), clipped=clipped)
                n_clip += clipped
            # ASVspoof CM format: <speaker> <utt> - <attack> <label>
            attack = "-" if label == "bonafide" else source
            prot.append(f"{args.speaker} {uid} - {attack} {label}")
        except Exception as e:
            failed += 1
            r["status"] = f"FAILED: {e}"
        man.append(r)
        if i % 1000 == 0 or i == len(rows):
            rate = i / max(time.time() - t0, 1e-9)
            print(f"  {i}/{len(rows)}  {rate:.0f} files/s  "
                  f"eta {(len(rows) - i) / max(rate, 1e-9) / 60:.1f} min", flush=True)

    tag = f"{args.speaker}_{args.subset}_{args.shard:04d}"
    mpath = Path(args.out) / "manifest" / f"FF_{tag}.csv"
    with open(mpath, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(man)
    ppath = Path(args.out) / "protocol" / f"FF_{args.speaker}.cm.{args.subset}.trl.txt"
    mode = "a" if (args.shard and ppath.exists()) else "w"
    with open(ppath, mode) as fh:
        fh.write("\n".join(prot) + "\n")

    n_bona = sum(1 for r in man if r["label"] == "bonafide")
    print(f"\n  wrote {mpath}")
    print(f"  wrote {ppath}")
    print(f"  {len(man) - failed}/{len(man)} ok"
          + (f", {failed} FAILED" if failed else ""))
    print(f"  {n_bona} bonafide / {len(man) - n_bona} spoof")
    print(f"  source rates: {dict(rates)}")
    if n_clip:
        print(f"  NOTE: {n_clip} samples clipped at full scale (not corrected)")
    if len(rates) > 1:
        print("\n  WARNING: this speaker has MIXED source rates. They correlate with\n"
              "  the label in Famous Figures, so any result should be checked against\n"
              "  the src_rate column before it is attributed to the condition.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
