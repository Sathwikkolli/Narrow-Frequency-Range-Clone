#!/usr/bin/env python3
"""Cut a CM protocol to a fixed fraction of its trials, stratified by (label, attack).

    python eval/make_subset.py --fraction 0.4 --in <protocol> --out <subset protocol>

Normally run through eval/make_subset_protocols.sh, which cuts all four protocols
(2019 LA train, dev, eval and In-the-Wild eval) into one folder.

WHY A PROTOCOL AND NOT A FLAG. Both train/train.py and eval/harness.py take the
trial list from a protocol (harness scores only clips that are in --labels), so a
cut protocol restricts training, dev selection and scoring with no code change,
and every model, every condition (RAW, P0, B1, C1) and every training data choice
(wb, itu, mix) sees exactly the same utterances.

FIXED. The draw depends only on --seed (default 0) and the protocol, never on a
model's training seed, so it is the same subset for every run. Lines are copied
verbatim, in the original order.

STRATIFIED. Each (label, attack) group keeps round(fraction * n) trials, at least
one, so the bonafide/spoof ratio and every attack's share are preserved. Attack is
what harness.read_labels reads: A01-A19 for ASVspoof, the speaker field for
In-the-Wild spoofs.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import read_labels   # noqa: E402


def utt_of(line, labels):
    toks = line.replace(",", " ").split()
    return next((t for t in toks if t in labels), None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fraction", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not 0 < args.fraction <= 1:
        sys.exit("--fraction must be in (0, 1]")

    labels = read_labels(args.src)
    groups = defaultdict(list)
    for u in sorted(labels):
        groups[labels[u]].append(u)          # key: (label, attack)

    rng = np.random.RandomState(args.seed)
    keep = set()
    for key in sorted(groups):
        ids = groups[key]
        k = max(1, int(round(args.fraction * len(ids))))
        keep.update(rng.choice(ids, k, replace=False).tolist())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_out = 0
    with open(args.src) as fh, open(out, "w") as fo:
        for line in fh:
            if utt_of(line, labels) in keep:
                fo.write(line if line.endswith("\n") else line + "\n")
                n_out += 1
    if n_out != len(keep):
        sys.exit(f"wrote {n_out} lines for {len(keep)} kept trials -- duplicate ids in {args.src}?")

    bona = sum(1 for u in keep if labels[u][0] == "bonafide")
    bona_all = sum(1 for v in labels.values() if v[0] == "bonafide")
    print(f"{out.name}: {n_out} of {len(labels)} trials ({n_out / len(labels):.1%}); "
          f"bonafide {bona} of {bona_all}, spoof {n_out - bona} of {len(labels) - bona_all}; "
          f"{len(groups)} (label, attack) groups; seed {args.seed}")


if __name__ == "__main__":
    main()
