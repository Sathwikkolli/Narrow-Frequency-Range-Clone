#!/usr/bin/env python3
"""Cross-model grid: turn per-model metrics into the comparisons the study asks for.

    # one JSON per model first
    for m in rawnet2 aasist tcm nes2net sls lcnn; do
        python eval/compute_metrics.py runs/suite/${m}_*.csv --json-out runs/suite/${m}.json
    done

    python eval/grid.py runs/suite/*.json

compute_metrics.py reports one condition at a time. This reads those JSONs and
lays them out as models x conditions, then computes the contrasts that answer the
question rather than leaving them to be eyeballed.

WHICH BASELINE. The two corpora are built differently -- the ITU chain applies
P.56 level normalisation and drops to 8 kHz, the NFR chain applies neither -- and
the pipeline's own README says they "should not be mixed in one table without
saying so". So each arm is measured against its own control:

    ITU  (B1, C1)      vs  P0      level held constant
    NFR  (F1..F5)      vs  F0      headroom held constant
    both               vs  RAW     the shared absolute reference

Reporting a B1 drop against RAW would fold the level normalisation into what
looks like a bandwidth effect. RawNet2 shows that is worth 1.5 pp on its own.

LEAKAGE. F1-F5 all target 300-3400 Hz, so any spread between them should be
checked against how much energy each filter actually lets past the band edges
before it is attributed to the algorithm. Those figures come from `python nfr.py`
in the pipeline repo and are hardcoded below; the Spearman correlation against
EER is printed so the check is not optional.
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

# share of surviving energy outside 300-3400 Hz, from nfr.py's own measurement
LEAKAGE = {"F1": 0.82, "F2": 0.80, "F3": 0.10, "F4": 0.16, "F5": 1.51}

ORDER = ["RAW", "P0", "B1", "C1", "F0", "F1", "F2", "F3", "F4", "F5"]
ITU = ["B1", "C1"]
NFR = ["F1", "F2", "F3", "F4", "F5"]


def model_of(path, cond):
    """runs/suite/rawnet2_B1.csv -> rawnet2"""
    base = os.path.basename(path)
    for ext in (".csv", ".json"):
        if base.endswith(ext):
            base = base[: -len(ext)]
    if cond and base.endswith("_" + cond):
        base = base[: -(len(cond) + 1)]
    return base


def load(paths):
    """{model: {cond: row}}"""
    grid = defaultdict(dict)
    for p in paths:
        with open(p) as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            data = [data]
        for row in data:
            cond = (row.get("cond") or "").strip()
            name = row.get("name") or p
            if not cond:
                continue
            grid[model_of(name, cond)][cond] = row
    if not grid:
        sys.exit(f"no usable rows in {paths}")
    return grid


def spearman(xs, ys):
    """Rank correlation, no scipy dependency."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def pct(row, key="eer"):
    v = row.get(key)
    return None if v is None else v * 100.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", nargs="+", help="per-model JSON from compute_metrics.py")
    ap.add_argument("--metric", default="eer", choices=["eer", "auc", "dprime"])
    args = ap.parse_args()

    paths = sorted({f for pat in args.json for f in glob.glob(pat)}) or args.json
    grid = load(paths)
    models = sorted(grid)
    conds = [c for c in ORDER if any(c in grid[m] for m in models)]

    scale = 100.0 if args.metric == "eer" else 1.0
    unit = " %" if args.metric == "eer" else ""

    # ---------------------------------------------------------- the grid
    print(f"\n{'=' * 78}\n{args.metric.upper()}{unit} by model and condition\n{'=' * 78}")
    print(f"{'model':<14}" + "".join(f"{c:>7}" for c in conds))
    for m in models:
        cells = []
        for c in conds:
            r = grid[m].get(c)
            cells.append("     ." if r is None else f"{r[args.metric] * scale:7.2f}")
        print(f"{m:<14}" + "".join(cells))

    if args.metric != "eer":
        return

    # ---------------------------------------------------------- contrasts
    print(f"\n{'=' * 78}\ncontrasts (EER percentage points; each arm vs its own control)\n{'=' * 78}")
    print(f"{'model':<14}{'RAW':>8}{'lvl':>8}{'B1-P0':>8}{'C1-B1':>8}"
          f"{'NFR-F0':>9}{'spread':>8}")
    print(f"{'':14}{'abs':>8}{'P0-RAW':>8}{'band':>8}{'codec':>8}{'mean':>9}{'F1-F5':>8}")
    for m in models:
        g = grid[m]
        raw, p0, f0 = pct(g.get("RAW", {})), pct(g.get("P0", {})), pct(g.get("F0", {}))
        b1, c1 = pct(g.get("B1", {})), pct(g.get("C1", {}))
        nfr = [pct(g[c]) for c in NFR if c in g]
        def f(v, w=8):
            return f"{v:>{w}.2f}" if v is not None else f"{'.':>{w}}"
        lvl = p0 - raw if (p0 is not None and raw is not None) else None
        band = b1 - p0 if (b1 is not None and p0 is not None) else None
        codec = c1 - b1 if (c1 is not None and b1 is not None) else None
        nfrd = (sum(nfr) / len(nfr) - f0) if (nfr and f0 is not None) else None
        spread = (max(nfr) - min(nfr)) if len(nfr) > 1 else None
        print(f"{m:<14}{f(raw)}{f(lvl)}{f(band)}{f(codec)}{f(nfrd, 9)}{f(spread)}")

    print("""
  lvl     P.56 level normalisation alone. The ITU README warns that if this moves,
          the pipeline is a confound -- so B1/C1 must be read against P0, not RAW.
  band    the handset filter, with level held constant. This is the narrowband cost.
  codec   G.711 mu-law on top of the handset filter.
  NFR-F0  mean cost of flat band-limiting, no level treatment anywhere.
  spread  disagreement among five filters targeting the SAME band -- see below.""")

    # ---------------------------------------------------------- leakage check
    print(f"\n{'=' * 78}\nis the F1-F5 spread the algorithm, or the leakage?\n{'=' * 78}")
    print("  out-of-band energy surviving each filter, from nfr.py:")
    print("    " + "  ".join(f"{c} {LEAKAGE[c]:.2f}%" for c in NFR))
    print()
    for m in models:
        have = [c for c in NFR if c in grid[m]]
        if len(have) < 3:
            continue
        xs = [LEAKAGE[c] for c in have]
        ys = [pct(grid[m][c]) for c in have]
        rho = spearman(xs, ys)
        verdict = ("tracks leakage -- NOT evidence about the algorithm"
                   if rho <= -0.7 else
                   "no clear leakage relation -- the spread may be real"
                   if abs(rho) < 0.7 else
                   "EER RISES with leakage, which is backwards -- investigate")
        print(f"  {m:<14} spearman(leakage, EER) = {rho:+.2f}   {verdict}")
    print("""
  A strongly negative correlation means more surviving out-of-band energy gives
  the detector more to work with, so the five filters differ because of leakage
  rather than because the algorithm matters. That is the check nfr_dataset's
  README asks for before any method-to-method claim.""")


if __name__ == "__main__":
    main()
