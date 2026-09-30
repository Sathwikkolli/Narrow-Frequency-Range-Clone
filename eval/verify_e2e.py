#!/usr/bin/env python3
"""End-to-end verification: original audio and labels -> scores -> EER.

    python eval/verify_e2e.py \\
        --orig-meta  $ITW_SRC/protocols/meta.csv \\
        --orig-audio $ITW_SRC/release_in_the_wild \\
        --root       $ITW_ROOT --nb $ITW_NB --nfr $ITW_NFR \\
        --labels     $ITW_LABELS \\
        --scores     runs/itw \\
        --json-dir   results --json-prefix itw_ \\
        --n 60

Re-reading the metric JSONs proves nothing -- they are the thing under test. This
walks the chain from the corpus as shipped to the numbers being reported, and
every check is computed here with code that does not share an implementation with
compute_metrics.py or harness.py.

  1 LABELS     Sample N ids. Trace each back through the generated protocol to
               the ORIGINAL meta file and confirm bonafide/spoof agree. Catches a
               protocol built from the wrong column, a mis-parsed header, or an
               id mapping that drifted.

  2 AUDIO      For the same ids, open every condition and measure:
                 - sample rate (8 kHz only for the ITU telephone conditions)
                 - share of energy inside 300-3400 Hz, by FFT
               A band-limited condition that is not actually band-limited would
               otherwise be invisible -- the scores would look plausible and the
               conclusion would be wrong.

  3 TRIALS     Every condition of every model must score the SAME id set. If one
               condition silently dropped clips, cross-condition deltas compare
               different populations.

  4 EER        Recompute EER from the scores CSVs with an independent threshold
               sweep and compare against the published JSON. Also recompute the
               class counts and the direction of separation.

Exit code is 0 only if every check passes.
"""
import argparse
import csv
import glob
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import soundfile as sf

CONDS = ["RAW", "P0", "B1", "C1", "F0", "F1", "F2", "F3", "F4", "F5"]
EIGHT_K = {"B1", "C1"}                 # ITU telephone conditions keep 8 kHz
BANDLIMITED = {"B1", "C1", "F1", "F2", "F3", "F4", "F5"}
LO, HI = 300.0, 3400.0
fails, warns = [], []


def fail(msg):
    fails.append(msg)
    print(f"    FAIL  {msg}")


def warn(msg):
    warns.append(msg)
    print(f"    warn  {msg}")


def band_share(x, sr):
    """Share of total energy inside LO..HI Hz. Plain FFT, no shared helper."""
    n = 1 << (len(x) - 1).bit_length() if len(x) else 2
    n = min(max(n, 2048), 1 << 16)
    w = np.hanning(min(len(x), n))
    seg = x[:len(w)] * w
    P = np.abs(np.fft.rfft(seg, n)) ** 2
    f = np.fft.rfftfreq(n, 1 / sr)
    tot = P.sum() + 1e-30
    return float(P[(f >= LO) & (f <= HI)].sum() / tot)


def cond_path(cond, uid, root, nb, nfr):
    if cond == "RAW":
        return os.path.join(root, "ASVspoof2019_LA_eval", "flac", uid + ".flac")
    base = nb if cond in ("P0", "B1", "C1") else nfr
    return os.path.join(base, "ASVspoof2019_LA_eval", cond, "flac", uid + ".flac")


def eer_independent(scores, labels):
    """EER by sweeping every distinct threshold. Deliberately not the same
    algorithm compute_metrics.py uses -- an agreeing bug is then unlikely."""
    s = np.asarray(scores, dtype=float)
    y = np.asarray([1 if l == "bonafide" else 0 for l in labels])
    order = np.argsort(-s)                       # descending: accept top-down
    y = y[order]
    P, N = y.sum(), len(y) - y.sum()
    if P == 0 or N == 0:
        return float("nan")
    tp = np.cumsum(y)                            # bonafide accepted
    fp = np.cumsum(1 - y)                        # spoof accepted  -> false accept
    frr = 1.0 - tp / P
    far = fp / N
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2 * 100)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orig-meta", required=True)
    ap.add_argument("--orig-audio", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--nb", required=True)
    ap.add_argument("--nfr", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--scores", required=True, help="merged CSV dir, e.g. runs/itw")
    ap.add_argument("--json-dir", default="results")
    ap.add_argument("--json-prefix", default="itw_")
    ap.add_argument("--models", nargs="+",
                    default=["rawnet2", "aasist", "nes2net", "tcm", "sls", "lcnn"])
    ap.add_argument("--n", type=int, default=60, help="clips to open and measure")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eer-tol", type=float, default=0.05,
                    help="pp difference allowed between our EER and the JSON")
    args = ap.parse_args()
    rnd = random.Random(args.seed)

    # ---------------------------------------------------------------- 1 labels
    print("\n[1] LABELS  generated protocol vs original meta")
    prot = {}
    with open(args.labels) as fh:
        for line in fh:
            f = line.split()
            if len(f) >= 5:
                prot[f[1]] = f[4].lower()
    print(f"    protocol: {len(prot)} trials, "
          f"{sum(1 for v in prot.values() if v=='bonafide')} bonafide")

    orig = {}
    with open(args.orig_meta, newline="") as fh:
        for line in fh:
            f = [t.strip() for t in line.rstrip("\n").split(",")]
            if len(f) < 3 or f[0].lower() in ("filename", "file"):
                continue
            stem = f[0][:-4] if f[0].lower().endswith(".wav") else f[0]
            t = f[2].strip().lower()
            orig[stem] = "bonafide" if t.startswith("bona") else (
                "spoof" if t in ("spoof", "fake", "spoofed", "synthetic") else t)
    print(f"    original: {len(orig)} rows")

    ids = sorted(prot)
    sample = rnd.sample(ids, min(args.n, len(ids)))
    mism = 0
    for uid in ids:                                   # check ALL, not just sample
        stem = uid.split("_", 2)[-1]
        if stem not in orig:
            fail(f"{uid}: stem {stem!r} absent from original meta"); break
        if orig[stem] != prot[uid]:
            mism += 1
            if mism <= 3:
                fail(f"{uid}: protocol {prot[uid]} != original {orig[stem]}")
    if mism == 0 and not fails:
        print(f"    ok    all {len(prot)} labels match the original meta file")
    elif mism:
        fail(f"{mism} label mismatches in total")

    # ---------------------------------------------------------------- 2 audio
    print(f"\n[2] AUDIO  {len(sample)} clips x {len(CONDS)} conditions")
    stats = defaultdict(list)
    for uid in sample:
        for c in CONDS:
            p = cond_path(c, uid, args.root, args.nb, args.nfr)
            if not os.path.isfile(p):
                fail(f"{c}/{uid}: missing on disk"); continue
            x, sr = sf.read(p, dtype="float64", always_2d=True)
            x = x.mean(axis=1)
            stats[(c, "sr")].append(sr)
            stats[(c, "band")].append(band_share(x, sr))
            stats[(c, "rms")].append(float(np.sqrt((x ** 2).mean())) if len(x) else 0.0)
    print(f"    {'cond':<5}{'rate':>7}{'in 300-3400Hz':>16}{'rms':>10}")
    for c in CONDS:
        if (c, "sr") not in stats:
            continue
        rates = set(stats[(c, "sr")])
        band = float(np.mean(stats[(c, "band")])) * 100
        rms = float(np.mean(stats[(c, "rms")]))
        print(f"    {c:<5}{str(sorted(rates)):>7}{band:>15.2f}%{rms:>10.5f}")
        want = {8000} if c in EIGHT_K else {16000}
        if rates != want:
            fail(f"{c}: rate {sorted(rates)}, expected {sorted(want)}")
        if c in BANDLIMITED and band < 90:
            fail(f"{c}: only {band:.1f}% of energy inside {LO:.0f}-{HI:.0f} Hz "
                 f"-- this condition is not band-limited")
        if c in ("RAW", "P0", "F0") and band > 97:
            warn(f"{c}: {band:.1f}% of energy already inside the band -- expected a "
                 f"full-band control to have more energy outside it")

    # ------------------------------------------------------- 3 trials / 4 EER
    print("\n[3] TRIALS and [4] EER  (independent recomputation)")
    print(f"    {'model':<9}{'cond':<5}{'n':>7}{'bona':>7}{'ours':>9}{'json':>9}{'diff':>8}")
    for m in args.models:
        jp = os.path.join(args.json_dir, f"{args.json_prefix}{m}.json")
        pub = {}
        if os.path.exists(jp):
            rows = json.load(open(jp)); rows = rows if isinstance(rows, list) else [rows]
            pub = {r["cond"]: r for r in rows}
        else:
            warn(f"{m}: no {jp}")
        idsets = {}
        for c in CONDS:
            p = os.path.join(args.scores, f"{m}_{c}.csv")
            if not os.path.isfile(p):
                fail(f"{m}/{c}: {p} missing"); continue
            u, sc, lab = [], [], []
            with open(p, newline="") as fh:
                rd = csv.DictReader(fh)
                for r in rd:
                    u.append(r["utt_id"]); sc.append(float(r["score"])); lab.append(r["label"])
            idsets[c] = set(u)
            if len(u) != len(idsets[c]):
                fail(f"{m}/{c}: duplicate utt_ids")
            bad = idsets[c] - set(prot)
            if bad:
                fail(f"{m}/{c}: {len(bad)} scored ids are not in the protocol")
            wrong = sum(1 for i, l in zip(u, lab) if prot.get(i) != l)
            if wrong:
                fail(f"{m}/{c}: {wrong} rows whose label disagrees with the protocol")
            ours = eer_independent(sc, lab)
            ref = pub.get(c, {}).get("eer")
            ref = ref * 100 if ref is not None else float("nan")
            diff = abs(ours - ref) if ref == ref else float("nan")
            print(f"    {m:<9}{c:<5}{len(u):>7}{sum(1 for l in lab if l=='bonafide'):>7}"
                  f"{ours:>9.3f}{ref:>9.3f}{diff:>8.3f}")
            if diff == diff and diff > args.eer_tol:
                fail(f"{m}/{c}: our EER {ours:.3f}% vs reported {ref:.3f}% "
                     f"(differ by {diff:.3f} pp)")
        if len(idsets) == len(CONDS):
            base = idsets["RAW"]
            for c, s in idsets.items():
                if s != base:
                    fail(f"{m}/{c}: scores a different id set than RAW "
                         f"({len(s ^ base)} differing ids) -- cross-condition deltas "
                         f"would compare different populations")

    # ---------------------------------------------------------------- verdict
    print(f"\n{'='*70}")
    if fails:
        print(f"FAILED: {len(fails)} problem(s)")
        for f in fails[:20]:
            print(f"  - {f}")
    else:
        print("PASSED: labels, audio, trial sets and EER all verified independently")
    if warns:
        print(f"\n{len(warns)} warning(s):")
        for w in warns[:10]:
            print(f"  - {w}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
