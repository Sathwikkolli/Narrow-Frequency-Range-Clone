#!/usr/bin/env python3
"""Spectrograms of narrowband (ITU C1) clips after RawBoost, for RawBoost algos 1-5.

    python eval/rawboost_spectrograms.py --out runs/rawboost_spec            # Great Lakes defaults
    sbatch eval/rawboost_spectrograms.sbatch                                  # same, as a CPU job

WHAT IT SHOWS. The narrowband models are trained on C1 audio with RawBoost on, and
must never see energy above 4 kHz, because C1 test audio has none. Each clip gets
one figure with two columns:

    left   "ours"  train/official.py's NarrowbandSet path:
                   8 kHz read -> RawBoost at fs=8000, maxF=4000 -> soxr to 16 kHz
    right  "repo"  the authors' own data path, had it been run on C1:
                   librosa.load(sr=16000) -> RawBoost at fs=16000, maxF=8000

and one row per condition: no RawBoost, then algo 1..5. The dashed line is 4 kHz.
Each panel is labelled with its energy above 4.2 kHz relative to the whole clip
(dB). "ours" should stay near the noise floor; "repo" shows the leak.

    algo 1  LnL   linear + non-linear convolutive noise (notches + harmonics)
    algo 2  ISD   impulsive signal-dependent noise (clicks)
    algo 3  SSI   stationary coloured additive noise (hiss)      <- SLS
    algo 4  1+2+3 in series                                      <- Nes2Net
    algo 5  1+2 in series                                        <- AASIST, TCM

Every number that defines the processing is imported, not copied: RawBoost's
parameters (RAWBOOST / RAWBOOST_NB), the 8 kHz rate and the resampler come from
train/official.py and eval/harness.py, and process_Rawboost_feature from the
model repo's own code. Draws are seeded per (seed, clip, algo), and both columns
of a row use the same seed.

Outputs in --out:
    spec_<n>_<utt>.png    one figure per clip
    psd_summary.png       mean spectrum over the clips, per algo, ours vs repo
    band_energy.csv       energy above 4.2 kHz per clip / algo / path
    audio/*.wav           every processed clip at 16 kHz, to listen to
    clips.csv             which clips were picked
"""
import argparse
import csv
import importlib
import os
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "train"))
sys.path.insert(0, str(HERE))

import official                       # noqa: E402  RAWBOOST, RAWBOOST_NB, NB_RATE
from harness import RESAMPLER, TARGET_SR, read_labels, resample   # noqa: E402
from adapters._common import in_repo  # noqa: E402

TURBO = "/nfs/turbo/umd-hafiz/issf_server_data"
ALGOS = {1: "LnL (convolutive)", 2: "ISD (impulsive)", 3: "SSI (coloured noise)",
         4: "1+2+3 series", 5: "1+2 series"}
USED_BY = {3: "SLS", 4: "Nes2Net", 5: "AASIST, TCM"}
HIGH_HZ = 4200            # same edge as the preflight leak check


# ---------------------------------------------------------------- processing

def load_rawboost(repo):
    """process_Rawboost_feature from the repo's own code (RawBoost.py in tcm_add,
    data_utils_SSL.py in the other three; the function is identical)."""
    with in_repo(repo):
        for mod in ("RawBoost", "data_utils_SSL", "data_utils"):
            try:
                f = getattr(importlib.import_module(mod), "process_Rawboost_feature", None)
            except ImportError:
                continue
            if f is not None:
                return f, mod
    sys.exit(f"no process_Rawboost_feature found in {repo}")


def fit(x, window):
    """The repos' pad(): crop, or tile-repeat a short clip. window 0 = whole clip."""
    if not window:
        return x
    if len(x) >= window:
        return x[:window]
    return np.tile(x, int(np.ceil(window / max(len(x), 1))))[:window]


def seed_for(seed, clip, algo):
    np.random.seed(int(np.random.SeedSequence([seed, clip, algo]).generate_state(1)[0]))


def ours(x8, rb, algo, window):
    """train/official.py NarrowbandSet: RawBoost at 8 kHz, then the harness resampler."""
    if algo:
        x8 = rb(x8, official.NB_RATE, Namespace(**official.RAWBOOST_NB), algo)
    x = np.asarray(resample(np.asarray(x8, dtype="float32"), official.NB_RATE), dtype="float32")
    return fit(x, window)


def repo_path(x16_librosa, rb, algo, window):
    """The repos' own Dataset: librosa.load(sr=16000), then RawBoost at 16 kHz."""
    x = x16_librosa
    if algo:
        x = rb(x, TARGET_SR, Namespace(**official.RAWBOOST), algo)
    return fit(np.asarray(x, dtype="float32"), window)


def high_band_db(x, sr=TARGET_SR):
    """Energy above HIGH_HZ relative to the whole clip, from a Welch PSD."""
    f, p = signal.welch(x, fs=sr, window="hann", nperseg=1024)
    return 10 * np.log10(p[f >= HIGH_HZ].sum() / max(p.sum(), 1e-30) + 1e-30)


# ---------------------------------------------------------------- picking clips

def pick(protocols, subset, flac_dir, n, seed, min_sec):
    """About 2 bonafide for every 3 spoof, spoofs from different attacks, each at
    least min_sec long. Deterministic for a given seed."""
    proto = official.T.PROTOCOL[subset]
    labels = read_labels(Path(protocols) / proto)
    rng = np.random.RandomState(seed)
    ids = sorted(labels)
    rng.shuffle(ids)
    n_bona = max(1, round(n * 2 / 5))
    out, attacks = [], set()
    for u in ids:
        if len(out) == n:
            break
        lab, att = labels[u]
        n_have = sum(1 for _, l, _ in out if l == "bonafide")
        if lab == "bonafide" and n_have >= n_bona:
            continue
        if lab != "bonafide" and (len(out) - n_have >= n - n_bona or att in attacks):
            continue
        p = Path(flac_dir) / f"{u}.flac"
        if not p.is_file():
            continue
        info = sf.info(str(p))
        if info.samplerate != official.NB_RATE:
            sys.exit(f"{p}: {info.samplerate} Hz -- expected the {official.NB_RATE} Hz C1 tree")
        if info.duration < min_sec:
            continue
        out.append((u, lab, att))
        if lab != "bonafide":
            attacks.add(att)
    if len(out) < n:
        sys.exit(f"only found {len(out)} clips of >= {min_sec} s in {flac_dir}")
    return out


# ---------------------------------------------------------------- plotting

def stft_db(x, sr=TARGET_SR):
    f, t, z = signal.stft(x, fs=sr, window="hann", nperseg=512, noverlap=384)
    return f, t, 20 * np.log10(np.abs(z) + 1e-10)


def plot_clip(path, title, rows, col_names):
    """rows: [(row label, [x for each column])]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    specs = [[stft_db(x) for x in xs] for _, xs in rows]
    top = max(s[2].max() for r in specs for s in r)
    vmin, vmax = top - 90, top
    fig, axes = plt.subplots(len(rows), len(col_names), figsize=(6.2 * len(col_names), 2.1 * len(rows)),
                             sharex=True, sharey=True, squeeze=False)
    for i, ((label, xs), row) in enumerate(zip(rows, specs)):
        for j, ((f, t, s), x) in enumerate(zip(row, xs)):
            ax = axes[i, j]
            im = ax.pcolormesh(t, f / 1000, s, shading="auto", cmap="magma", vmin=vmin, vmax=vmax)
            ax.axhline(4.0, color="white", ls="--", lw=0.8)
            ax.text(0.99, 0.95, f">4.2 kHz: {high_band_db(x):.0f} dB", transform=ax.transAxes,
                    ha="right", va="top", color="white", fontsize=8,
                    bbox=dict(facecolor="black", alpha=0.5, lw=0))
            if i == 0:
                ax.set_title(col_names[j], fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{label}\nkHz", fontsize=9)
            if i == len(rows) - 1:
                ax.set_xlabel("s")
    fig.suptitle(title, fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.6, label="dB")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


PANEL_NOTE = {
    "clean": "The phone clip as the model gets it without RawBoost. Speech stops near 3.5 kHz; "
             "above the dashed line there is nothing.",
    "ours": "RawBoost run on the 8 kHz audio, then upsampled (train/official.py). The damage "
            "stays below the dashed line; above it stays empty.",
    "repo": "RawBoost run after upsampling to 16 kHz (the authors' data path). Its noise and "
            "clicks spill above the dashed line, into a band real phone audio never has.",
}


def plot_single(path, x, title, note, vmin, vmax):
    """One condition of one clip as its own large, labelled image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f, t, s = stft_db(x)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    im = ax.pcolormesh(t, f / 1000, s, shading="auto", cmap="magma", vmin=vmin, vmax=vmax)
    ax.axhline(4.0, color="white", ls="--", lw=1.2)
    ax.text(0.005, 4.12, "phone-band limit, 4 kHz", color="white", fontsize=9,
            transform=ax.get_yaxis_transform())
    ax.text(0.995, 0.96, f"energy above 4.2 kHz: {high_band_db(x):.0f} dB of the clip",
            transform=ax.transAxes, ha="right", va="top", color="white", fontsize=10,
            bbox=dict(facecolor="black", alpha=0.6, lw=0))
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (kHz)")
    ax.set_title(title, fontsize=12)
    fig.colorbar(im, ax=ax, label="loudness (dB)", pad=0.01)
    fig.text(0.01, -0.02, note, fontsize=9, ha="left", va="top", wrap=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_psd_single(path, curves, algo):
    """Average spectrum for one algo: no RawBoost vs ours vs repo."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.6))
    for label, (f, p), style in curves:
        ax.plot(f / 1000, 10 * np.log10(p + 1e-30), lw=1.5, label=label, **style)
    ax.axvline(4.0, color="grey", ls="--", lw=1)
    ax.text(4.05, 0.97, "phone-band limit", transform=ax.get_xaxis_transform(), va="top",
            fontsize=9, color="grey")
    ax.set_xlabel("frequency (kHz)")
    ax.set_ylabel("average loudness (dB)")
    ax.set_title(f"Average spectrum over the clips: algo {algo} ({ALGOS[algo]})", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_psd(path, psd, algos, col_names):
    """psd[(algo, col)] = (f, mean PSD over clips)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(col_names), figsize=(6.2 * len(col_names), 4), sharey=True,
                             squeeze=False)
    for j, name in enumerate(col_names):
        ax = axes[0, j]
        for a in [0] + algos:
            f, p = psd[(a, j)]
            ax.plot(f / 1000, 10 * np.log10(p + 1e-30), lw=1.2,
                    label="no RawBoost" if a == 0 else f"algo {a}: {ALGOS[a]}")
        ax.axvline(4.0, color="grey", ls="--", lw=0.8)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("kHz")
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("mean PSD (dB)")
    axes[0, -1].legend(fontsize=8)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", default=f"{TURBO}/AsvSpoofData_2019_NB",
                    help="narrowband root holding ASVspoof2019_LA_<subset>/C1/flac")
    ap.add_argument("--protocols", default=f"{TURBO}/AsvSpoofData_2019/train/LA/ASVspoof2019_LA_cm_protocols")
    ap.add_argument("--subset", default="train", choices=["train", "dev"])
    ap.add_argument("--repo", default=os.path.expanduser("~/models/tcm_repo"),
                    help="a model repo with process_Rawboost_feature (any of the four XLS-R repos)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5, help="number of clips")
    ap.add_argument("--clips", nargs="+", help="explicit utterance ids instead of a random pick")
    ap.add_argument("--algos", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--window", type=int, default=64600,
                    help="crop/tile to this many 16 kHz samples, as training does; 0 = whole clip")
    ap.add_argument("--min-sec", type=float, default=3.0, help="shortest clip to pick")
    ap.add_argument("--no-repo-path", action="store_true", help="only the 'ours' column")
    ap.add_argument("--panels", action="store_true",
                    help="also write every panel as its own image in <out>/panels/, plus one "
                         "average-spectrum image per algo")
    ap.add_argument("--model", default="",
                    help="name shown in the --panels titles, e.g. 'W2V2-AASIST'")
    args = ap.parse_args()

    bad = [a for a in args.algos if a not in ALGOS]
    if bad:
        sys.exit(f"algos must be in {sorted(ALGOS)}, got {bad}")
    if not RESAMPLER.startswith("soxr"):
        sys.exit(f"resampler is {RESAMPLER}, not soxr -- this env differs from the training env")

    flac_dir = Path(args.data_root) / f"ASVspoof2019_LA_{args.subset}" / "C1" / "flac"
    out = Path(args.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    rb, rb_mod = load_rawboost(args.repo)
    print(f"RawBoost from {args.repo}/{rb_mod}.py, resampler {RESAMPLER}, clips from {flac_dir}",
          flush=True)

    if args.clips:
        labels = read_labels(Path(args.protocols) / official.T.PROTOCOL[args.subset])
        clips = [(u, *labels[u]) for u in args.clips]
    else:
        clips = pick(args.protocols, args.subset, flac_dir, args.n, args.seed, args.min_sec)
    with open(out / "clips.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["utt", "label", "attack", "path"])
        for u, lab, att in clips:
            w.writerow([u, lab, att, flac_dir / f"{u}.flac"])

    want_repo = not args.no_repo_path
    if want_repo:
        import librosa
    cols = ["ours: RawBoost at 8 kHz -> soxr 16 kHz"]
    if want_repo:
        cols.append("repo: librosa 16 kHz -> RawBoost at 16 kHz")
    conds = [0] + args.algos
    psd_sum, rows_csv = {}, []

    for ci, (u, lab, att) in enumerate(clips, 1):
        path = flac_dir / f"{u}.flac"
        x8, sr = sf.read(str(path), dtype="float32")
        if sr != official.NB_RATE:
            sys.exit(f"{path}: {sr} Hz, expected {official.NB_RATE}")
        if x8.ndim > 1:
            x8 = x8.mean(axis=1)
        x16_lr = librosa.load(str(path), sr=TARGET_SR)[0] if want_repo else None

        rows = []
        for a in conds:
            xs = []
            seed_for(args.seed, ci, a)
            xs.append(ours(x8.copy(), rb, a, args.window))
            if want_repo:
                seed_for(args.seed, ci, a)             # same draws as the left column
                xs.append(repo_path(x16_lr.copy(), rb, a, args.window))
            name = "no RawBoost" if a == 0 else f"algo {a}: {ALGOS[a]}" + (
                f"\n({USED_BY[a]})" if a in USED_BY else "")
            rows.append((name, xs))
            for j, x in enumerate(xs):
                tag = ["ours", "repo"][j]
                sf.write(str(out / "audio" / f"{ci}_{u}_algo{a}_{tag}.wav"), x, TARGET_SR)
                f, p = signal.welch(x, fs=TARGET_SR, window="hann", nperseg=1024)
                p = p / max(p.sum(), 1e-30)            # each clip weighs the same
                psd_sum[(a, j)] = (f, psd_sum.get((a, j), (f, 0))[1] + p)
                rows_csv.append([u, lab, att, a, tag, f"{high_band_db(x):.2f}"])

        title = f"{u}  ({lab}{'' if lab == 'bonafide' else ', ' + att})  C1 {args.subset}"
        png = out / f"spec_{ci}_{u}.png"
        plot_clip(png, title, rows, cols)
        print(f"  {png.name}", flush=True)

        if args.panels:
            pdir = out / "panels"
            pdir.mkdir(exist_ok=True)
            top = max(stft_db(x)[2].max() for _, xs in rows for x in xs)
            who = f"{args.model}, " if args.model else ""
            clip = f"{u} ({lab}{'' if lab == 'bonafide' else ', attack ' + att})"
            for a, (_, xs) in zip(conds, rows):
                if a == 0:
                    shots = [("clean", xs[0], "no RawBoost")]
                else:
                    shots = [("ours", xs[0], f"{who}RawBoost algo {a}, our 8 kHz path")]
                    if want_repo:
                        shots.append(("repo", xs[1], f"{who}RawBoost algo {a}, authors' 16 kHz path"))
                for tag, x, what in shots:
                    name = f"{ci}_{u}_{'clean' if a == 0 else f'algo{a}_{tag}'}.png"
                    plot_single(pdir / name, x, f"{clip}: {what}", PANEL_NOTE[tag],
                                top - 90, top)
            print(f"  panels/{ci}_{u}_*.png", flush=True)

    with open(out / "band_energy.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["utt", "label", "attack", "algo", "path", "energy_above_4.2kHz_db"])
        w.writerows(rows_csv)
    psd = {k: (f, s / len(clips)) for k, (f, s) in psd_sum.items()}
    plot_psd(out / "psd_summary.png", psd, args.algos, cols)
    if args.panels:
        for a in args.algos:
            curves = [("no RawBoost", psd[(0, 0)], dict(color="tab:blue")),
                      (f"algo {a}, our 8 kHz path", psd[(a, 0)], dict(color="tab:green"))]
            if want_repo:
                curves.append((f"algo {a}, authors' 16 kHz path", psd[(a, 1)],
                               dict(color="tab:red")))
            plot_psd_single(out / "panels" / f"psd_algo{a}.png", curves, a)

    print("\nenergy above 4.2 kHz, mean over clips (dB re whole clip):")
    print(f"  {'':<26}" + "".join(f"{c.split(':')[0]:>10}" for c in cols))
    for a in conds:
        vals = [np.mean([float(r[5]) for r in rows_csv if r[3] == a and r[4] == t])
                for t in ["ours", "repo"][:len(cols)]]
        name = "no RawBoost" if a == 0 else f"algo {a} {ALGOS[a]}"
        print(f"  {name:<26}" + "".join(f"{v:>10.1f}" for v in vals))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
