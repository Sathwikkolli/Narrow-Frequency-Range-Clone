#!/usr/bin/env python3
"""FAR / FRR at one fixed operating point per detector, across every condition
and dataset. CPU only, reads the merged scores CSVs.

THE THRESHOLD. For each model, tau = its EER threshold on ASVspoof2019 LA RAW
(the untouched, in-domain condition it was trained for), computed with the same
det_curve/eer as compute_metrics.py. tau is then frozen and applied to every
condition on every dataset. That is what a deployed detector experiences: one
threshold, chosen on clean in-domain data, meeting audio it was not tuned for.
It needs no assumption about score scale, so it treats all six models the same
(a fixed 0 does not: RawNet2 and SLS emit log-probabilities, all <= 0).

THE RATES. Bonafide is accepted when score > tau (compute_metrics' convention).
    FAR = spoof trials with score >  tau / spoof trials
    FRR = bonafide trials with score <= tau / bonafide trials
These are exact counts; the raw counts are written alongside the percentages.

SELF-CHECKS (the script exits non-zero if any fails):
  1. every CSV has the expected trial and class counts and no duplicate utt_ids
  2. the counts agree with a second, independent count (sort + searchsorted)
  3. on ASVspoof2019 RAW, (FAR + FRR) / 2 at tau equals that model's EER

    python eval/far_frr.py --out results/far_frr
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_metrics import eer  # noqa: E402  same threshold rule as the EER tables

MODELS = [("rawnet2", "RawNet2"), ("lcnn", "LCNN"), ("aasist", "AASIST"),
          ("nes2net", "Nes2Net"), ("tcm", "TCM"), ("sls", "SLS")]
CONDS = [("RAW", "original"), ("P0", "level only"), ("B1", "phone filter"),
         ("C1", "+ codec"), ("F0", "control"), ("F1", "Parks–McClellan"),
         ("F2", "Kaiser FIR"), ("F3", "Elliptic IIR"), ("F4", "Polyphase"),
         ("F5", "FFT overlap")]
# dataset key -> (title, default scores dir, expected (bonafide, spoof))
DATASETS = {
    "asv": ("ASVspoof2019 LA", "runs/suite", (7355, 63882)),
    "ff":  ("FakeOrReal (FF)", "runs/ff_trump", (4754, 6536)),
    "itw": ("In-the-Wild (ITW)", "runs/itw", (19963, 11815)),
}


def load(path):
    ids, bona, spoof = set(), [], []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            uid = r.get("utt_id") or next(iter(r.values()))
            if uid in ids:
                sys.exit(f"FAIL {path}: duplicate utt_id {uid}")
            ids.add(uid)
            s = float(r["score"])
            if not np.isfinite(s):
                sys.exit(f"FAIL {path}: non-finite score for {uid}")
            if r["label"] == "bonafide":
                bona.append(s)
            elif r["label"] == "spoof":
                spoof.append(s)
            else:
                sys.exit(f"FAIL {path}: unknown label {r['label']!r}")
    return np.array(bona), np.array(spoof)


def rates(bona, spoof, tau):
    fa = int((spoof > tau).sum())
    fr = int((bona <= tau).sum())
    # independent second count
    fa2 = len(spoof) - int(np.searchsorted(np.sort(spoof), tau, side="right"))
    fr2 = int(np.searchsorted(np.sort(bona), tau, side="right"))
    if (fa, fr) != (fa2, fr2):
        sys.exit(f"FAIL count mismatch: {fa}/{fr} vs {fa2}/{fr2}")
    return fa, fr


def main():
    ap = argparse.ArgumentParser()
    for k, (_, d, _) in DATASETS.items():
        ap.add_argument(f"--{k}-dir", default=d)
    ap.add_argument("--out", default="results/far_frr", help="writes <out>.md and <out>.json")
    ap.add_argument("--no-count-check", action="store_true",
                    help="skip the expected-trial-count check (e.g. a partial run)")
    args = ap.parse_args()
    dirs = {k: getattr(args, f"{k}_dir") for k in DATASETS}

    # 1. thresholds from ASVspoof2019 RAW
    tau = {}
    for m, _ in MODELS:
        b, s = load(os.path.join(dirs["asv"], f"{m}_RAW.csv"))
        e, t = eer(b, s)
        fa, fr = rates(b, s, t)
        far, frr = fa / len(s), fr / len(b)
        # det_curve reads a tied block of scores as partly accepted; exact counting
        # cannot, so allow at most the tied trials' share of disagreement
        ties = int((b == t).sum() + (s == t).sum())
        if abs((far + frr) / 2 - e) > ties / min(len(b), len(s)) + 1e-9:
            sys.exit(f"FAIL {m}: (FAR+FRR)/2 {(far + frr) / 2:.6f} != EER {e:.6f}")
        tau[m] = t
        print(f"{m:<8} tau {t:+.6f}   ASV RAW EER {e * 100:.3f}%  "
              f"(FAR {far * 100:.3f}%, FRR {frr * 100:.3f}%)  check ok")

    # 2. apply everywhere
    out, md = [], []
    for k, (title, _, (nb, ns)) in DATASETS.items():
        md += [f"### {title} — FAR / FRR (%)", "",
               "| Condition | What | " + " | ".join(n for _, n in MODELS) + " |",
               "|---|---|" + "---|" * len(MODELS)]
        for c, what in CONDS:
            cells = []
            for m, _ in MODELS:
                path = os.path.join(dirs[k], f"{m}_{c}.csv")
                if not os.path.exists(path):
                    cells.append("missing")
                    print(f"WARN missing {path}")
                    continue
                b, s = load(path)
                if not args.no_count_check and (len(b), len(s)) != (nb, ns):
                    sys.exit(f"FAIL {path}: {len(b)}/{len(s)} trials, expected {nb}/{ns}")
                fa, fr = rates(b, s, tau[m])
                far, frr = fa / len(s), fr / len(b)
                cells.append(f"{far * 100:.2f} / {frr * 100:.2f}")
                out.append(dict(dataset=k, cond=c, model=m, threshold=tau[m],
                                n_bonafide=len(b), n_spoof=len(s),
                                false_accepts=fa, false_rejects=fr, far=far, frr=frr))
            md.append(f"| {c} | {what} | " + " | ".join(cells) + " |")
        md.append("")
    md += ["Threshold per model = its EER threshold on ASVspoof2019 LA `RAW`, frozen "
           "and applied to every condition and dataset. Bonafide accepted when score > threshold.",
           "", "| Model | Threshold |", "|---|---|"]
    md += [f"| {n} | {tau[m]:+.6f} |" for m, n in MODELS]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out + ".md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    with open(args.out + ".json", "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n" + "\n".join(md))
    print(f"\nall checks passed; wrote {args.out}.md and {args.out}.json")


if __name__ == "__main__":
    main()
