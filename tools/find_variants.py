"""
Find base Trump clips that several attacks all cloned, i.e. the same sentence
spoken by several different synthesizers. Run this ON GREAT LAKES (or the lab
server) where DATA_ROOT is readable.

    python tools/find_variants.py --data-root "$DATA_ROOT" --min-attacks 3

Layout is {Source}/{AudioName}.wav under DATA_ROOT. A fake is named after the
real clip it clones, so the leading Donald_Trump_<id> is the base clip id:
    F5TTS/Donald_Trump_00150.wav                              -> 00150
    COZYVOICE2/Donald_Trump_00353_COZYVOICE2__Donald_Trump_00001.wav -> 00353
Prints a table of base ids with the most attack coverage, plus a ready to paste
list of paths for tools/fetch_clips.sh.
"""
import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

REAL_DIRS = {"-", "original", "bonafide"}
BASE_ID = re.compile(r"^(?P<speaker>[A-Za-z_]+?)_(?P<id>\d{4,6})")


def base_id(filename):
    m = BASE_ID.match(filename)
    return m.group("id") if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--min-attacks", type=int, default=3)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--require-real", action="store_true",
                    help="only list base ids whose real clip is also present")
    args = ap.parse_args()

    root = Path(args.data_root)
    # base id -> attack -> path      and      base id -> real path
    fakes = defaultdict(dict)
    reals = {}
    n = 0
    for dirpath, _, files in os.walk(root):
        d = Path(dirpath)
        source = d.name
        for f in files:
            if not f.lower().endswith(".wav"):
                continue
            n += 1
            bid = base_id(f)
            if bid is None:
                continue
            if source.lower() in REAL_DIRS:
                reals[bid] = d / f
            else:
                fakes[bid].setdefault(source, d / f)

    print(f"scanned {n} wav files under {root}")
    print(f"{len(reals)} real clips, {len(fakes)} base ids with at least one fake\n")

    rows = [(bid, atk) for bid, atk in fakes.items() if len(atk) >= args.min_attacks]
    if args.require_real:
        rows = [r for r in rows if r[0] in reals]
    rows.sort(key=lambda r: (-len(r[1]), r[0]))

    if not rows:
        print(f"no base id is covered by {args.min_attacks}+ attacks; try --min-attacks 2")
        return

    print(f"base ids covered by {args.min_attacks}+ attacks (top {args.top}):")
    for bid, atk in rows[:args.top]:
        real = "real:yes" if bid in reals else "real:no "
        print(f"  {bid}  {len(atk)} attacks  {real}  {', '.join(sorted(atk))}")

    bid, atk = rows[0]
    print(f"\n--- paths for base id {bid} (paste into tools/fetch_clips.sh CLIPS) ---")
    if bid in reals:
        print(reals[bid].relative_to(root).as_posix())
    for source in sorted(atk):
        print(atk[source].relative_to(root).as_posix())


if __name__ == "__main__":
    main()
