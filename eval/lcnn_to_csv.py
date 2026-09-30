#!/usr/bin/env python3
"""Convert the LFCC-LCNN baseline's score file into the harness CSV schema.

LCNN is scored through its own tool rather than through harness.py -- see
adapters/lcnn_NOTES.md for why. This puts its output on the same footing as every
other model so compute_metrics.py needs no special case.

    cd project/baseline_LA
    bash 02_eval_alternative.sh <WAV_DIR> <SET_NAME> __pretrained/trained_network.pt

    python eval/lcnn_to_csv.py \\
        --scores  log_output_<SET_NAME> \\
        --labels  $HOME/asvspoof_protocols/ASVspoof2019.LA.cm.eval.trl.txt \\
        --cond    B1 \\
        --out     runs/suite/lcnn_B1.csv

SCORE DIRECTION. project-NN's LCNN is trained with a sigmoid/CE output where a
HIGHER score means more bonafide, matching the convention used everywhere else
here (score = logit_bonafide). Confirm it on the first condition: RAW or P0
should give a low EER, and an EER near 100-x% means the sign is inverted -- pass
--negate rather than editing metrics downstream.

The two logit columns are left empty: LCNN emits a single score, so there is no
second logit to record. compute_metrics.py reads `score`, so nothing downstream
depends on them.
"""
import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from harness import CSV_FIELDS, read_labels


def parse_scores(path):
    """LCNN score file -> [(utt_id, score)].

    project-NN writes one line per trial. Two shapes are accepted, because the
    exact format varies between releases of the framework:

        Output, LA_E_1234567, 0, -1.234567      comma separated, score last
        LA_E_1234567 -1.234567                  whitespace separated

    Anything else is a hard error rather than a silent skip -- a converter that
    quietly drops rows would show up as a suspiciously small trial count much
    later, if at all.
    """
    out, bad = [], []
    num = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")
    with open(path) as fh:
        for ln, line in enumerate(fh, 1):
            s = line.strip()
            if not s:
                continue
            fields = [t.strip() for t in s.split(",")] if "," in s else s.split()
            fields = [t for t in fields if t]
            if fields and fields[0].lower() == "output":
                fields = fields[1:]
            # Two corpora: ASVspoof ids look like LA_E_9332881, Famous Figures ids
            # like FF_STYLETTS2_Donald_Trump_00001. Match on underscore count rather
            # than an "LA_" prefix so both parse, excluding the numeric score field.
            utt = next((t for t in fields
                        if t.count("_") >= 2 and not num.match(t)), None)
            score = next((t for t in reversed(fields) if num.match(t)), None)
            if utt is None or score is None:
                bad.append((ln, s[:120]))
                continue
            out.append((utt, float(score)))
    if bad:
        sys.exit(f"{path}: {len(bad)} line(s) not parsed, first at line {bad[0][0]}:\n"
                 f"  {bad[0][1]}\n"
                 f"Check the score-file format and extend parse_scores() rather than "
                 f"letting rows be dropped.")
    if not out:
        sys.exit(f"{path}: no scores parsed")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True, help="LCNN score file")
    ap.add_argument("--labels", required=True, help="ASVspoof2019 LA CM protocol")
    ap.add_argument("--cond", required=True, help="condition name, e.g. B1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="lcnn")
    ap.add_argument("--negate", action="store_true",
                    help="flip the sign if higher turns out to mean more spoof")
    args = ap.parse_args()

    scores = parse_scores(args.scores)
    labels = read_labels(args.labels)

    rows, no_label, seen = [], 0, set()
    for utt, sc in scores:
        if utt in seen:
            continue
        seen.add(utt)
        if utt not in labels:
            no_label += 1
            continue
        label, attack = labels[utt]
        if args.negate:
            sc = -sc
        rows.append([utt, args.cond, label, attack, f"{sc:.6f}", "", "",
                     "", "", "", "", "", args.model])

    if not rows:
        sys.exit("no scored clips intersect the protocol")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_FIELDS)
        w.writerows(rows)

    nb = sum(1 for r in rows if r[2] == "bonafide")
    print(f"{args.scores} -> {args.out}")
    print(f"  {len(rows)} trials ({nb} bonafide / {len(rows) - nb} spoof), cond {args.cond}")
    if no_label:
        print(f"  {no_label} scored clips are not trials in the protocol "
              f"(expected: 71933 files against 71237 trials)")
    if len(rows) != 71237:
        print(f"  NOTE: expected 71237 trials for the LA eval protocol, got {len(rows)}")


if __name__ == "__main__":
    main()
