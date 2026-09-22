"""
Find the clips listed in a samples CSV inside the Famous Figures folder and
link them into one input folder for the narrowband pipeline.

    python prepare_samples.py --data-root DATA_ROOT --samples samples.csv --out RUN_DIR/input

Real and fake clips can share a filename (FISHSPEECH/Donald_Trump_00150.wav
is a clone of the real Donald_Trump_00150.wav), so they are linked as
    input/real/<filename>
    input/fake/<ATTACK>/<filename>
Real files are the ones in a folder named "-" or "Original"; a fake file must
sit under a folder named after its attack. Same rule as Deep_SVDD's
extract_embeddings.py.
"""
import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

REAL_DIRS = {"-", "original", "bonafide"}


def build_index(root):
    index = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".wav"):
                index.setdefault(f, []).append(Path(dirpath) / f)
    return index


def pick(candidates, label, attack, root):
    def dirs(p):
        return {d.lower() for d in p.relative_to(root).parts[:-1]}
    if label == "real":
        hits = [c for c in candidates if dirs(c) & REAL_DIRS]
    else:
        hits = [c for c in candidates if attack.lower() in dirs(c)]
    return hits[0] if len(hits) == 1 else None, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.data_root)
    with open(args.samples, newline="") as f:
        rows = list(csv.DictReader(f))
    print(f"indexing {root} ...")
    index = build_index(root)
    print(f"  {sum(len(v) for v in index.values())} wav files")

    out = Path(args.out)
    manifest, missing = [], 0
    for r in rows:
        label, attack, name = r["label"].strip(), r["attack"].strip(), r["filename"].strip()
        src, hits = pick(index.get(name, []), label, attack, root)
        if src is None:
            missing += 1
            why = "not found" if not hits else f"{len(hits)} matches: {[str(h) for h in hits]}"
            print(f"[MISSING] {label}/{attack}/{name}: {why}")
            continue
        dst = out / "real" / name if label == "real" else out / "fake" / attack / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        try:
            dst.symlink_to(src.resolve())
        except OSError:            # no symlink rights (e.g. Windows): copy instead
            shutil.copy2(src, dst)
        manifest.append({"label": label, "attack": attack, "input": dst.relative_to(out).as_posix(), "source": str(src)})
        print(f"[OK] {label:4s} {attack:10s} {src}")

    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["label", "attack", "input", "source"])
        w.writeheader()
        w.writerows(manifest)
    print(f"\n{len(manifest)}/{len(rows)} linked into {out}")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
