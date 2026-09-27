#!/usr/bin/env python3
"""Metrics for each condition's scores CSV from score_df_arena.py.

Reports, per condition:

  EER + bootstrap CI   the headline. Threshold-free, so it needs no tuning set.
  AUC                  a sanity companion to EER
  d-prime              separation of the two score distributions. This is what
                       distinguishes "the classes collapsed into each other" (the
                       model genuinely cannot tell them apart any more) from "both
                       distributions slid sideways but stayed apart" (the model
                       still discriminates, it is just mis-calibrated). Those two
                       have opposite engineering conclusions and identical EERs
                       are possible for either, so EER alone cannot settle it.
  acc / F1 / P / R     at the EER threshold, and at any threshold passed in
  FAR / FRR            at a fixed operating point, which is what a deployed
                       detector with one threshold actually experiences
  per-attack EER       each attack vs. all bonafide; shows which attacks stop
                       being detectable
  data characteristics n, class balance, durations, and the fraction of clips
                       shorter than the model's 4.04 s window (those get
                       tile-repeated, which is worth disclosing)

Writes a DET curve (and its points) per condition when --plot-dir is given.
Everything here runs on CPU from the CSVs alone, so it is re-runnable without
touching a GPU.
"""
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

MODEL_WINDOW_S = 64600 / 16000      # 4.0375 s


# ---------------------------------------------------------------- metrics

def det_curve(bona, spoof):
    """False-reject and false-accept rates over every threshold, plus thresholds.

    Scores are bonafide-ness, so a trial is accepted as bonafide when its score is
    above the threshold. Sweeping the threshold upward, FRR rises and FAR falls.
    """
    scores = np.concatenate([bona, spoof])
    is_bona = np.concatenate([np.ones(len(bona), bool), np.zeros(len(spoof), bool)])
    order = np.argsort(scores, kind="mergesort")
    scores, is_bona = scores[order], is_bona[order]
    frr = np.concatenate([[0.0], np.cumsum(is_bona) / len(bona)])
    far = np.concatenate([[1.0], 1.0 - np.cumsum(~is_bona) / len(spoof)])
    thr = np.concatenate([[scores[0] - 1e-9], scores])
    return frr, far, thr


def eer(bona, spoof):
    if len(bona) == 0 or len(spoof) == 0:
        return float("nan"), float("nan")
    frr, far, thr = det_curve(bona, spoof)
    i = int(np.nanargmin(np.abs(frr - far)))
    return float((frr[i] + far[i]) / 2.0), float(thr[i])


def auc(bona, spoof):
    """ROC AUC via rank sums (Mann-Whitney U), ties averaged."""
    if len(bona) == 0 or len(spoof) == 0:
        return float("nan")
    s = np.concatenate([bona, spoof])
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    r = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n1 = len(bona)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * len(spoof)))


def dprime(bona, spoof):
    """Standardised separation of the two score distributions (Cohen's d)."""
    if len(bona) < 2 or len(spoof) < 2:
        return float("nan")
    vb, vs = bona.var(ddof=1), spoof.var(ddof=1)
    pooled = math.sqrt((vb + vs) / 2.0)
    return float((bona.mean() - spoof.mean()) / pooled) if pooled > 0 else float("nan")


def bootstrap_eer(bona, spoof, n=1000, seed=0):
    """Percentile CI, resampling each class independently with replacement."""
    if len(bona) < 2 or len(spoof) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for k in range(n):
        b = bona[rng.integers(0, len(bona), len(bona))]
        s = spoof[rng.integers(0, len(spoof), len(spoof))]
        out[k] = eer(b, s)[0]
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def at_threshold(bona, spoof, t):
    """Bonafide is the positive class. FRR = bonafide rejected, FAR = spoof accepted."""
    tp = int((bona > t).sum()); fn = len(bona) - tp
    fp = int((spoof > t).sum()); tn = len(spoof) - fp
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return dict(acc=acc, f1=f1, precision=prec, recall=rec,
                far=fp / max(len(spoof), 1), frr=fn / max(len(bona), 1))


# ---------------------------------------------------------------- io

def load(path):
    rows = []
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        for r in rd:
            try:
                r["_score"] = float(r["score"])
            except (KeyError, ValueError):
                continue
            r["_dur"] = float(r.get("dur_s") or "nan")
            rows.append(r)
    if not rows:
        sys.exit(f"{path}: no usable rows")
    return rows


def split(rows):
    return (np.array([r["_score"] for r in rows if r["label"] == "bonafide"]),
            np.array([r["_score"] for r in rows if r["label"] == "spoof"]))


# ---------------------------------------------------------------- report

def report(name, rows, args):
    b, s = split(rows)
    e, thr = eer(b, s)
    lo, hi = bootstrap_eer(b, s, args.bootstrap, args.seed) if args.bootstrap else (float("nan"),) * 2
    durs = np.array([r["_dur"] for r in rows if not math.isnan(r["_dur"])])
    short = float((durs < MODEL_WINDOW_S).mean() * 100) if len(durs) else float("nan")
    conds = sorted({r.get("cond", "") for r in rows}) or [""]
    srs = sorted({r.get("orig_sr", "") for r in rows})

    print(f"\n{'=' * 64}\n{name}   [cond {','.join(conds)}]\n{'=' * 64}")
    print(f"  trials            {len(rows)}  ({len(b)} bonafide / {len(s)} spoof)")
    print(f"  source rate(s)    {','.join(srs)} Hz")
    if len(durs):
        print(f"  duration          median {np.median(durs):.2f} s, "
              f"mean {durs.mean():.2f} s, {short:.1f}% under the {MODEL_WINDOW_S:.2f} s window")
    print(f"  EER               {e * 100:.3f}%"
          + (f"   95% CI [{lo * 100:.3f}, {hi * 100:.3f}]" if args.bootstrap else ""))
    print(f"  EER threshold     {thr:+.4f}")
    print(f"  AUC               {auc(b, s):.5f}")
    print(f"  d-prime           {dprime(b, s):.3f}")
    print(f"  score mean/sd     bonafide {b.mean():+.3f}/{b.std():.3f}   "
          f"spoof {s.mean():+.3f}/{s.std():.3f}")

    m = at_threshold(b, s, thr)
    print(f"  at EER threshold  acc {m['acc'] * 100:.2f}%  F1 {m['f1']:.4f}  "
          f"P {m['precision']:.3f}  R {m['recall']:.3f}")
    for t in ([0.0] + ([args.threshold] if args.threshold is not None else [])):
        m = at_threshold(b, s, t)
        print(f"  at thr {t:+.4f}    acc {m['acc'] * 100:.2f}%  F1 {m['f1']:.4f}  "
              f"FAR {m['far'] * 100:.2f}%  FRR {m['frr'] * 100:.2f}%")

    if args.per_attack:
        atks = sorted({r["attack_id"] for r in rows
                       if r["label"] == "spoof" and r.get("attack_id") not in ("-", "", None)})
        if atks:
            print("  per-attack EER (attack vs. all bonafide):")
            for a in atks:
                sa = np.array([r["_score"] for r in rows
                               if r["label"] == "spoof" and r["attack_id"] == a])
                print(f"    {a:<5} n={len(sa):<6} {eer(b, sa)[0] * 100:7.3f}%")

    if args.plot_dir:
        write_det(name, b, s, args.plot_dir)

    return dict(name=name, cond=",".join(conds), n=len(rows), n_bonafide=len(b),
                n_spoof=len(s), eer=e, eer_ci=[lo, hi], eer_threshold=thr,
                auc=auc(b, s), dprime=dprime(b, s),
                bonafide_mean=float(b.mean()), bonafide_sd=float(b.std()),
                spoof_mean=float(s.mean()), spoof_sd=float(s.std()),
                pct_under_window=short)


def write_det(name, b, s, out_dir):
    """DET points as CSV, plus a plot if matplotlib is importable."""
    os.makedirs(out_dir, exist_ok=True)
    frr, far, thr = det_curve(b, s)
    stem = os.path.join(out_dir, os.path.basename(name).replace(".csv", ""))
    step = max(1, len(frr) // 5000)                 # thin for a usable file size
    with open(f"{stem}_det.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["threshold", "frr", "far"])
        for i in range(0, len(frr), step):
            w.writerow([f"{thr[i]:.6f}", f"{frr[i]:.6f}", f"{far[i]:.6f}"])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.stats import norm
    except ImportError:
        return
    ok = (frr > 0) & (far > 0) & (frr < 1) & (far < 1)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot(norm.ppf(far[ok]), norm.ppf(frr[ok]), lw=1.4)
    ticks = [0.001, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.9]
    ax.set_xticks(norm.ppf(ticks)); ax.set_xticklabels([f"{t * 100:g}" for t in ticks])
    ax.set_yticks(norm.ppf(ticks)); ax.set_yticklabels([f"{t * 100:g}" for t in ticks])
    ax.set_xlabel("false accept rate (%)"); ax.set_ylabel("false reject rate (%)")
    ax.set_title(os.path.basename(stem)); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{stem}_det.png", dpi=180); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores", nargs="+", help="one or more scores CSVs")
    ap.add_argument("--threshold", type=float, default=None,
                    help="an extra operating point to report at, e.g. the EER "
                         "threshold measured on the full-band condition")
    ap.add_argument("--per-attack", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=1000,
                    help="bootstrap resamples for the EER CI; 0 to skip")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot-dir", default=None, help="write DET curves here")
    ap.add_argument("--json-out", default=None, help="write the summary as JSON")
    args = ap.parse_args()

    summaries = [report(p, load(p), args) for p in args.scores]

    if len(summaries) > 1:
        print(f"\n{'=' * 64}\nsummary\n{'=' * 64}")
        print(f"  {'condition':<12} {'n':>7} {'EER %':>9} {'95% CI':>20} "
              f"{'AUC':>8} {'d-prime':>9}")
        for s in summaries:
            ci = (f"[{s['eer_ci'][0] * 100:.2f}, {s['eer_ci'][1] * 100:.2f}]"
                  if not math.isnan(s["eer_ci"][0]) else "-")
            print(f"  {s['cond'] or s['name'][:12]:<12} {s['n']:>7} "
                  f"{s['eer'] * 100:>9.3f} {ci:>20} {s['auc']:>8.4f} {s['dprime']:>9.3f}")

    if args.json_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_out)) or ".", exist_ok=True)
        with open(args.json_out, "w") as fh:
            json.dump(summaries, fh, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
