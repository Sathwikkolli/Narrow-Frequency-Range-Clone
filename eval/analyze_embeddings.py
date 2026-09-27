#!/usr/bin/env python3
"""Embedding analysis for the .npz files score_df_arena.py writes.

The question EER cannot answer: when performance drops on narrowband, is the
information gone, or is the classifier head simply pointed the wrong way?

  linear probe   train a fresh linear classifier on the frozen embeddings of this
                 condition. If the probe recovers a low EER where the model's own
                 head did not, the front-end still encodes what separates real from
                 fake and the fix is re-heading or fine-tuning. If the probe fails
                 too, the representation itself has lost it and a different or
                 retrained backbone is needed. This is the finding; the rest is
                 supporting evidence.
  silhouette     how cleanly the two classes occupy separate regions
  kNN purity     the same thing, locally and without assuming convex clusters
  UMAP / PCA     the picture. UMAP is illustration only -- its cluster sizes, gaps
                 and distances carry no quantitative meaning, so no claim rests on
                 it. PCA is plotted alongside as a linear, deterministic check that
                 the structure is not a UMAP artefact.
  layer gates    the mean sigmoid gate per XLS-R layer. Shows which layers the
                 model leans on, and whether narrowband shifts that reliance.

CPU only. Reads what the scoring run already saved, so it is re-runnable.
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np


# ---------------------------------------------------------------- loading

def load_npz(patterns):
    """Concatenate .npz shards, keeping first occurrence of each utt_id."""
    files = sorted({f for pat in patterns for f in glob.glob(pat)})
    if not files:
        sys.exit(f"no .npz matched: {patterns}")
    keys = ("utt_id", "cond", "label", "attack", "embedding")
    parts, gates, seen_meta = {k: [] for k in keys}, [], None
    for f in files:
        z = np.load(f, allow_pickle=True)
        missing = [k for k in keys if k not in z]
        if missing:
            print(f"  skipping {os.path.basename(f)}: missing {missing}")
            continue
        for k in keys:
            parts[k].append(z[k])
        gates.append(z["gates"] if "gates" in z else None)
        if seen_meta is None and "meta" in z:
            seen_meta = str(z["meta"][0])
    if not parts["utt_id"]:
        sys.exit("no usable .npz shards")
    out = {k: np.concatenate(parts[k]) for k in keys}
    _, first = np.unique(out["utt_id"], return_index=True)
    first.sort()
    out = {k: v[first] for k, v in out.items()}
    if all(g is not None for g in gates) and gates:
        g = np.concatenate(gates)[first]
        out["gates"] = g
    out["meta"] = seen_meta
    print(f"  {len(files)} shard(s) -> {len(out['utt_id'])} clips, "
          f"embedding {out['embedding'].shape[1]}d"
          + (f", gates {out['gates'].shape[1]} layers" if "gates" in out else ""))
    return out


def eer_from_scores(score, is_bona):
    """Same convention as compute_metrics: higher score = more bonafide."""
    b, s = score[is_bona], score[~is_bona]
    if len(b) == 0 or len(s) == 0:
        return float("nan")
    allsc = np.concatenate([b, s])
    lab = np.concatenate([np.ones(len(b), bool), np.zeros(len(s), bool)])
    o = np.argsort(allsc, kind="mergesort")
    lab = lab[o]
    frr = np.concatenate([[0.0], np.cumsum(lab) / len(b)])
    far = np.concatenate([[1.0], 1.0 - np.cumsum(~lab) / len(s)])
    i = int(np.nanargmin(np.abs(frr - far)))
    return float((frr[i] + far[i]) / 2.0)


# ---------------------------------------------------------------- analyses

def linear_probe(X, y, seed=0, folds=2):
    """Train/test a linear classifier on frozen embeddings; report EER and accuracy.

    Uses scikit-learn's logistic regression when available, otherwise a closed-form
    ridge regression onto +-1 targets, which is deterministic and needs no solver.
    Class weights are balanced because the trial list is ~9:1 spoof.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    X, y = X[idx].astype("float32"), y[idx]
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
    X = (X - mu) / sd

    cut = len(y) // folds
    eers, accs = [], []
    for k in range(folds):
        te = np.zeros(len(y), bool)
        te[k * cut:(k + 1) * cut] = True
        tr = ~te
        try:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
            clf.fit(X[tr], y[tr])
            score = clf.decision_function(X[te])
        except ImportError:
            # ridge onto +-1 with class balancing, solved in closed form
            t = np.where(y[tr], 1.0, -1.0)
            w_pos = 0.5 / max(y[tr].mean(), 1e-6)
            w_neg = 0.5 / max(1 - y[tr].mean(), 1e-6)
            sw = np.where(y[tr], w_pos, w_neg)
            A = np.hstack([X[tr], np.ones((tr.sum(), 1), "float32")])
            Aw = A * sw[:, None]
            W = np.linalg.solve(A.T @ Aw + 1.0 * np.eye(A.shape[1]), Aw.T @ t)
            score = np.hstack([X[te], np.ones((te.sum(), 1), "float32")]) @ W
        eers.append(eer_from_scores(score, y[te]))
        accs.append(float(((score > 0) == y[te]).mean()))
    return float(np.mean(eers)), float(np.mean(accs))


def silhouette(X, y, sample=4000, seed=0, n_components=32):
    """Mean silhouette over a sample, Euclidean on standardised features.

    Euclidean rather than cosine: when the class signal lives in a few dimensions
    of a high-dimensional embedding, the remaining dimensions dominate the angle
    and cosine silhouette reads near zero even for cleanly separable classes.
    Standardising first stops any single large-variance dimension dominating.

    Computed after a PCA reduction for the same reason. Silhouette stays
    conservative in high dimensions regardless, so treat it as a descriptor whose
    ordering across conditions is meaningful while its absolute magnitude is not;
    the linear probe and kNN purity carry the argument.
    """
    rng = np.random.default_rng(seed)
    n = min(sample, len(y))
    i = rng.choice(len(y), n, replace=False)
    Z = X[i].astype("float32")
    Z = (Z - Z.mean(0, keepdims=True)) / (Z.std(0, keepdims=True) + 1e-6)
    if Z.shape[1] > n_components:          # reduce first: silhouette degrades as
        _, _, Vt = np.linalg.svd(Z, full_matrices=False)   # dimensions grow and
        Z = Z @ Vt[:n_components].T                        # distances concentrate
    sq = (Z * Z).sum(1)
    D = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T), 0.0))
    np.fill_diagonal(D, np.nan)
    yy = y[i]
    out = np.empty(n)
    for c in (True, False):
        m = yy == c
        if m.sum() < 2 or (~m).sum() < 1:
            return float("nan")
        a = np.nanmean(D[np.ix_(m, m)], axis=1)
        b = np.nanmean(D[np.ix_(m, ~m)], axis=1)
        out[m] = (b - a) / np.maximum(a, b)
    return float(np.nanmean(out))


def knn_purity(X, y, k=10, sample=4000, seed=0):
    """Fraction of each point's k nearest neighbours sharing its label."""
    rng = np.random.default_rng(seed)
    n = min(sample, len(y))
    i = rng.choice(len(y), n, replace=False)
    Z = X[i].astype("float32")
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9
    S = Z @ Z.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argpartition(-S, k, axis=1)[:, :k]
    return float((y[i][nn] == y[i][:, None]).mean())


def project(X, method, seed=0, **kw):
    if method == "umap":
        try:
            import umap
        except ImportError:
            return None
        return umap.UMAP(random_state=seed, **kw).fit_transform(X.astype("float32"))
    Z = X.astype("float32") - X.astype("float32").mean(0, keepdims=True)
    # PCA by SVD; deterministic, no dependencies
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    return Z @ Vt[:2].T


def scatter(ax, P, groups, title):
    for g in sorted(set(groups)):
        m = np.array([x == g for x in groups])
        ax.scatter(P[m, 0], P[m, 1], s=2, alpha=.35, label=str(g), linewidths=0)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(markerscale=5, fontsize=6, loc="best", framealpha=.6)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+", help="embedding .npz files or globs, one condition per group")
    ap.add_argument("--out-dir", default="analysis")
    ap.add_argument("--sample", type=int, default=12000,
                    help="clips to project; UMAP on 70k is slow and no clearer")
    ap.add_argument("--probe-folds", type=int, default=2)
    ap.add_argument("--umap-neighbors", type=int, default=30)
    ap.add_argument("--umap-min-dist", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    summaries = []
    for pat in args.npz:
        print(f"\n=== {pat} ===")
        d = load_npz([pat])
        y = d["label"] == "bonafide"
        X = d["embedding"].astype("float32")
        cond = ",".join(sorted(set(d["cond"].tolist())))

        probe_eer, probe_acc = linear_probe(X, y, args.seed, args.probe_folds)
        sil = silhouette(X, y, seed=args.seed)
        pur = knn_purity(X, y, seed=args.seed)
        print(f"  condition        {cond}")
        print(f"  clips            {len(y)}  ({int(y.sum())} bonafide / {int((~y).sum())} spoof)")
        print(f"  linear probe     EER {probe_eer * 100:.3f}%   acc {probe_acc * 100:.2f}%")
        print(f"  silhouette       {sil:+.4f}   (1 = clean split, 0 = overlapping)")
        print(f"  kNN purity k=10  {pur * 100:.2f}%   (50% = chance for balanced, "
              f"{max(y.mean(), 1 - y.mean()) * 100:.1f}% = majority here)")

        rec = dict(condition=cond, n=int(len(y)), n_bonafide=int(y.sum()),
                   probe_eer=probe_eer, probe_acc=probe_acc,
                   silhouette=sil, knn_purity=pur, meta=d.get("meta"))

        if "gates" in d:
            g = 1.0 / (1.0 + np.exp(-d["gates"]))          # stored pre-sigmoid
            rec["gate_mean"] = g.mean(0).tolist()
            rec["gate_mean_bonafide"] = g[y].mean(0).tolist()
            rec["gate_mean_spoof"] = g[~y].mean(0).tolist()
            top = np.argsort(-g.mean(0))[:5]
            print(f"  layer gates      {g.shape[1]} layers; most-weighted: "
                  + ", ".join(f"L{int(i)} {g.mean(0)[i]:.3f}" for i in top))
        summaries.append(rec)

        if args.no_plots:
            continue
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("  (matplotlib missing -- skipping plots)")
            continue

        rng = np.random.default_rng(args.seed)
        i = rng.choice(len(y), min(args.sample, len(y)), replace=False)
        for method in ("umap", "pca"):
            P = project(X[i], method, seed=args.seed,
                        **(dict(n_neighbors=args.umap_neighbors,
                                min_dist=args.umap_min_dist) if method == "umap" else {}))
            if P is None:
                print("  (umap-learn missing -- PCA only)")
                continue
            fig, axes = plt.subplots(1, 2, figsize=(9, 4.4))
            scatter(axes[0], P, ["bonafide" if v else "spoof" for v in y[i]],
                    f"{cond} {method.upper()} by class")
            scatter(axes[1], P, d["attack"][i].tolist(),
                    f"{cond} {method.upper()} by attack")
            fig.suptitle(
                f"{cond}  n={len(i)}"
                + (f"  UMAP(n_neighbors={args.umap_neighbors}, "
                   f"min_dist={args.umap_min_dist}, seed={args.seed}) "
                   "- layout is illustrative; distances are not meaningful"
                   if method == "umap" else "  PCA - linear, deterministic"),
                fontsize=8)
            fig.tight_layout()
            out = os.path.join(args.out_dir, f"{cond.replace(',', '_')}_{method}.png")
            fig.savefig(out, dpi=180); plt.close(fig)
            print(f"  wrote {out}")

    with open(os.path.join(args.out_dir, "embedding_summary.json"), "w") as fh:
        json.dump(summaries, fh, indent=2)
    print(f"\nwrote {os.path.join(args.out_dir, 'embedding_summary.json')}")

    if len(summaries) > 1:
        print(f"\n{'condition':<12} {'probe EER %':>12} {'probe acc %':>12} "
              f"{'silhouette':>11} {'kNN purity %':>13}")
        for s in summaries:
            print(f"{s['condition']:<12} {s['probe_eer'] * 100:>12.3f} "
                  f"{s['probe_acc'] * 100:>12.2f} {s['silhouette']:>11.4f} "
                  f"{s['knn_purity'] * 100:>13.2f}")


if __name__ == "__main__":
    main()
