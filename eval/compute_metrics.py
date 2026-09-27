#!/usr/bin/env python3
"""Metrics from one or more scores CSVs written by score_df_arena.py.

Per file: EER (and its threshold), AUC, accuracy/F1 at threshold 0, accuracy/F1 at
a threshold you pass in with --threshold (tune it on dev, apply it to eval), and
per-attack EER. Across files: pooled EER over all rows concatenated, and the mean
of the per-file EERs -- the Arena leaderboard reports both, and they differ when
the conditions have different score offsets.

EER is threshold-free, so it is the number to lead with; accuracy and F1 depend on
a threshold and will look flattering or terrible depending on which you pick.
"""
import argparse, csv, sys
from collections import defaultdict

import numpy as np


def eer(bona, spoof):
    """Equal error rate from bonafide (target) and spoof (nontarget) score arrays."""
    if len(bona) == 0 or len(spoof) == 0:
        return float("nan"), float("nan")
    scores = np.concatenate([bona, spoof])
    labels = np.concatenate([np.ones(len(bona)), np.zeros(len(spoof))])
    order = np.argsort(scores, kind="mergesort")
    scores, labels = scores[order], labels[order]
    # sweep the threshold upward: everything at or below it is called spoof
    fn = np.cumsum(labels) / len(bona)                      # bonafide rejected
    fp = 1.0 - (np.cumsum(1 - labels) / len(spoof))          # spoof accepted
    frr = np.concatenate([[0.0], fn])
    far = np.concatenate([[1.0], fp])
    thr = np.concatenate([[scores[0] - 1e-6], scores])
    i = np.nanargmin(np.abs(frr - far))
    return (frr[i] + far[i]) / 2.0, float(thr[i])


def auc(bona, spoof):
    """ROC AUC via the rank identity (equivalent to Mann-Whitney U)."""
    if len(bona) == 0 or len(spoof) == 0:
        return float("nan")
    s = np.concatenate([bona, spoof])
    r = np.empty(len(s))
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    i = 0
    while i < len(s):                                        # average ties
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n1 = len(bona)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * len(spoof))


def at_threshold(bona, spoof, t):
    """Accuracy and F1 with bonafide as the positive class."""
    tp = int((bona > t).sum()); fn = len(bona) - tp
    fp = int((spoof > t).sum()); tn = len(spoof) - fp
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return acc, f1, prec, rec


def load(path):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append((r["label"], r.get("attack_id", "-"), float(r["score"])))
    if not rows:
        sys.exit(f"{path} is empty")
    return rows


def split(rows):
    b = np.array([s for lab, _, s in rows if lab == "bonafide"])
    s = np.array([s for lab, _, s in rows if lab == "spoof"])
    return b, s


def report(name, rows, thr_in, per_attack):
    b, s = split(rows)
    e, t = eer(b, s)
    print(f"\n=== {name} ===")
    print(f"  clips            {len(rows)}  ({len(b)} bonafide / {len(s)} spoof)")
    print(f"  EER              {e * 100:.2f}%   (at threshold {t:+.3f})")
    print(f"  AUC              {auc(b, s):.4f}")
    for label, tt in [("thr=0", 0.0)] + ([("thr=%+.3f" % thr_in, thr_in)] if thr_in is not None else []):
        acc, f1, p, r = at_threshold(b, s, tt)
        print(f"  acc/F1 @ {label:<12} {acc * 100:.2f}% / {f1:.4f}   (P {p:.3f} R {r:.3f})")
    if per_attack:
        attacks = sorted({a for lab, a, _ in rows if lab == "spoof" and a not in ("-", "")})
        if attacks:
            print("  per-attack EER (each attack vs. all bonafide):")
            for a in attacks:
                sa = np.array([sc for lab, aa, sc in rows if lab == "spoof" and aa == a])
                ea, _ = eer(b, sa)
                print(f"    {a:<5} n={len(sa):<6} {ea * 100:6.2f}%")
    return e, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores", nargs="+", help="one or more scores CSVs")
    ap.add_argument("--threshold", type=float, default=None,
                    help="operating threshold to also report at (e.g. the EER threshold from dev)")
    ap.add_argument("--per-attack", action="store_true")
    args = ap.parse_args()

    eers, allrows = [], []
    for p in args.scores:
        rows = load(p)
        allrows += rows
        e, _ = report(p, rows, args.threshold, args.per_attack)
        eers.append(e)

    if len(args.scores) > 1:
        print("\n=== across conditions ===")
        pb, ps = split(allrows)
        pe, pt = eer(pb, ps)
        print(f"  pooled EER       {pe * 100:.2f}%   (at threshold {pt:+.3f}, {len(allrows)} clips)")
        print(f"  average EER      {np.nanmean(eers) * 100:.2f}%   (mean of the {len(eers)} per-condition EERs)")


if __name__ == "__main__":
    main()
