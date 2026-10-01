#!/usr/bin/env python3
"""Score one condition with one detector, writing the schema compute_metrics.py reads.

This is score_df_arena.py generalised. Everything that is not the model itself --
finding the clips, joining them to the protocol, decoding, downmixing, resampling,
windowing, writing scores, tapping embeddings -- lives here and is written once.
Each detector contributes a small adapter under eval/adapters/ that answers four
questions and nothing else:

    window      how many 16 kHz samples the model wants
    batch_ok    whether it accepts a real batch or one clip per forward
    build()     how to construct it and load the checkpoint
    score()     how to turn its output into (score, logit_spoof, logit_bonafide)

WHY ONE LOADER. Six detectors across four conda environments would otherwise each
decode and resample through their own library -- librosa, soundfile, torchaudio --
with different default resamplers. B1 and C1 are 8 kHz, so they are the only
conditions that get resampled at all, and a resampler that images energy into the
empty 4-8 kHz band would be indistinguishable from the narrowband effect under
study. One loader means every detector sees the same samples, so any difference
between them is the detector.

SAMPLE RATE. Decided per file from its own header, never per condition: RAW, P0
and F0-F5 are 16 kHz and pass through untouched, B1 and C1 are 8 kHz and are
resampled up. A sinc/polyphase resampler is used, for the reason above.

SCORE. The `score` column is whatever the adapter declares as that model's own
convention, so a published number can be reproduced exactly (RawNet2 and TCM both
use logits[bonafide]; DF Arena uses the difference). Both raw logits are written
alongside, so the other convention is recoverable without rescoring.

WINDOW. Per model, not global -- RawNet2 and DF Arena want 64600 samples, TCM
wants 66800. Short clips are tile-repeated, matching what each of these repos
does in its own dataset class.
"""
import argparse
import csv
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TARGET_SR = 16000

# torch.inference_mode() landed in torch 1.9. The SSL models (AASIST, TCM,
# Nes2Net) pin torch 1.8.1 through their fairseq commit, so fall back to
# no_grad there -- equivalent for scoring, marginally slower.
NO_GRAD = getattr(torch, "inference_mode", torch.no_grad)

LABEL_WORDS = {"bonafide": "bonafide", "bona-fide": "bonafide", "bona_fide": "bonafide",
               "real": "bonafide", "genuine": "bonafide", "human": "bonafide",
               "spoof": "spoof", "fake": "spoof", "spoofed": "spoof", "synthetic": "spoof"}

try:
    import soxr

    def resample(x, sr):
        return x if sr == TARGET_SR else soxr.resample(x, sr, TARGET_SR, quality="HQ")

    RESAMPLER = "soxr:HQ"
except ImportError:                                  # scipy is present in every env
    from math import gcd
    from scipy.signal import resample_poly

    def resample(x, sr):
        if sr == TARGET_SR:
            return x
        g = gcd(int(sr), TARGET_SR)
        return resample_poly(x, TARGET_SR // g, int(sr) // g).astype("float32")

    RESAMPLER = "scipy:resample_poly"


# ---------------------------------------------------------------- inputs

def read_labels(path):
    """CM protocol -> {file_id: (label, attack_id)}.

    Fields are identified by content, not position, so the .trn/.trl variants and
    the whitespace or comma separated forms all parse.

    Two corpora use this. ASVspoof2019 LA looks like

        LA_0069  LA_E_9332881  -  A13  spoof

    and Famous Figures, prepared by eval/prep_ff.py, looks like

        Donald_Trump  FF_STYLETTS2_Donald_Trump_00001  -  STYLETTS2  spoof

    so the utterance id is matched on having two or more underscores rather than
    an "LA_" prefix -- that still picks LA_E_9332881 over LA_0069 (one
    underscore) and the speaker field over neither. The attack id falls back to
    position when the A-nn pattern does not apply, because FF names its attacks
    in full and per-attack EER is worth keeping.
    """
    out = {}
    with open(path) as fh:
        for line in fh:
            f = next(csv.reader([line])) if line.count(",") >= 2 else line.split()
            f = [t.strip() for t in f if t.strip()]
            if not f:
                continue
            label = next((LABEL_WORDS[t.lower()] for t in f if t.lower() in LABEL_WORDS), None)
            utt = next((t for t in f
                        if t.count("_") >= 2 and t.lower() not in LABEL_WORDS), None)
            if label is None or utt is None:
                continue
            attack = next((t for t in f
                           if len(t) == 3 and t[0] == "A" and t[1:].isdigit()), None)
            if attack is None:
                attack = f[3] if len(f) >= 5 else "-"
            out[utt] = (label, attack)
    if not out:
        sys.exit(f"no bonafide/spoof lines in {path} -- is it really a CM protocol?")
    return out


def read_manifest(patterns, keep_failed=False):
    """Manifest shards -> list of row dicts, deduped on file_id, FAILED builds dropped."""
    files = sorted({Path(f) for pat in patterns for f in glob.glob(pat)})
    if not files:
        sys.exit(f"no manifest files matched: {patterns}")
    rows, seen, dropped = [], set(), 0
    for f in files:
        with open(f, newline="") as fh:
            rd = csv.DictReader(fh)
            missing = {"cond", "subset", "file_id"} - set(rd.fieldnames or [])
            if missing:
                sys.exit(f"{f}: missing column(s) {sorted(missing)}; got {rd.fieldnames}")
            for r in rd:
                fid = (r.get("file_id") or "").strip()
                if not fid or fid in seen:
                    continue
                # "exists" is a resumed build and "ok_no_active_speech" is the ITU
                # chain's silent-file path; both are successful builds. Only FAILED
                # goes. Requiring an exact "ok" silently discards resumed shards --
                # 93509 of the NFR eval rows are "exists".
                status = (r.get("status") or "ok").strip().lower()
                if not keep_failed and status.startswith("failed"):
                    dropped += 1
                    continue
                seen.add(fid)
                rows.append(r)
    if not rows:
        sys.exit(f"no usable rows across {len(files)} manifest file(s)")
    print(f"manifest: {len(files)} shard(s), {len(rows)} clips"
          + (f", {dropped} dropped as FAILED in the build" if dropped else ""), flush=True)
    return rows


def rows_from_dir(audio_dir, cond):
    """A condition with no manifest -- RAW, which neither pipeline built."""
    files = sorted(Path(audio_dir).glob("*.flac")) or sorted(Path(audio_dir).glob("*.wav"))
    if not files:
        sys.exit(f"no .flac or .wav under {audio_dir}")
    print(f"audio-dir: {len(files)} clips in {audio_dir} (cond={cond})", flush=True)
    return [dict(cond=cond, subset="eval", file_id=p.stem, _path=p) for p in files]


def build(manifest_rows, labels, nb_root):
    """Intersect with the protocol and resolve every clip to a real file."""
    out, no_label, no_file = [], [], []
    for r in manifest_rows:
        fid = r["file_id"].strip()
        if fid not in labels:
            no_label.append(fid)
            continue
        if r.get("_path") is not None:
            p = r["_path"]
        else:
            cond, subset = r["cond"].strip(), r["subset"].strip()
            p = Path(nb_root) / f"ASVspoof2019_LA_{subset}" / cond / "flac" / f"{fid}.flac"
        if not Path(p).exists():
            no_file.append(str(p))
            continue
        label, attack = labels[fid]
        out.append(dict(utt=fid, path=p, label=label, attack=attack,
                        cond=r["cond"].strip(),
                        clipped=(r.get("clipped") or "").strip(),
                        padded=(r.get("padded") or "").strip()))
    if no_label:
        print(f"  {len(no_label)} clips are not trials in the protocol -- expected, the "
              f"corpora hold 71933 files against 71237 trials", flush=True)
    if no_file:
        print(f"  WARNING: {len(no_file)} clips missing on disk, e.g. {no_file[:2]}", flush=True)
    if not out:
        sys.exit("nothing to score: manifest, protocol and audio tree do not intersect")
    return out


class Clips(Dataset):
    """Reads, downmixes, resamples to 16 kHz, then fits this model's fixed window."""

    def __init__(self, paths, window):
        self.paths = paths
        self.window = int(window)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x, sr = sf.read(str(self.paths[i]), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        dur = len(x) / sr
        x = np.asarray(resample(x, sr), dtype="float32")
        n, w = len(x), self.window
        if n >= w:
            x = x[:w]
        else:                          # tile-repeat, as each of these repos does itself
            x = np.tile(x, int(np.ceil(w / max(n, 1))))[:w]
        return torch.from_numpy(np.ascontiguousarray(x)), i, sr, dur


# ---------------------------------------------------------------- embedding taps

class Taps:
    """Forward hooks that stash the classifier's input as the pooled embedding.

    The classifier is found as the last Linear whose out_features equals the number
    of classes; its *input* is the pre-head embedding. That resolves correctly for
    every back-end in the suite -- TCM's fc5 (144->2), AASIST, Nes2Net and SLS all
    end in a 2-way Linear. A gate module is tapped by name when a model has one.
    """

    def __init__(self, model, num_labels=2, gate_name=None, classifier_name=None):
        self.emb = self.gates = None
        self.handles = []
        self.emb_module = self.gate_module = None

        linears = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
        if classifier_name:
            cand = [(n, m) for n, m in linears if n == classifier_name]
            if not cand:
                sys.exit(f"--classifier-name {classifier_name!r} not found")
        else:
            cand = [(n, m) for n, m in linears if m.out_features == num_labels]
        if cand:
            self.emb_module = cand[-1][0]
            self.handles.append(cand[-1][1].register_forward_hook(self._take_emb))

        if gate_name:
            gate = [(n, m) for n, m in linears if n.split(".")[-1] == gate_name]
            if gate:
                self.gate_module = gate[0][0]
                self.handles.append(gate[0][1].register_forward_hook(self._take_gates))

    def reset(self):
        """Clear before every forward so a hook that does not fire is detected.

        Without this a missed hook leaves the previous clip's tensor in place and it
        is recorded against the next utterance. The counts still line up, so the
        misalignment is silent: it corrupts the embeddings while leaving the scores
        untouched.
        """
        self.emb = self.gates = None

    def _take_emb(self, mod, inp, out):
        t = inp[0].detach()
        self.emb = t.reshape(t.shape[0], -1).float().cpu().numpy()

    def _take_gates(self, mod, inp, out):
        t = out.detach()
        self.gates = t.reshape(t.shape[0], -1).float().cpu().numpy()

    def close(self):
        for h in self.handles:
            h.remove()


# ---------------------------------------------------------------- main

CSV_FIELDS = ["utt_id", "cond", "label", "attack_id", "score", "logit_spoof",
              "logit_bonafide", "orig_sr", "dur_s", "tiled", "clipped", "padded", "model"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True, help="model adapter name, e.g. rawnet2")
    ap.add_argument("--ckpt", required=True, help="checkpoint for that adapter")
    ap.add_argument("--repo", default=None, help="the model's own repo, if it imports from one")
    ap.add_argument("--labels", required=True, help="ASVspoof2019 LA CM protocol")
    ap.add_argument("--out", required=True, help="output scores CSV")
    ap.add_argument("--manifest", nargs="+", help="manifest shard(s); globs allowed")
    ap.add_argument("--nb-root", default=None, help="corpus root for manifest-driven conditions")
    ap.add_argument("--audio-dir", default=None, help="one folder of clips (for RAW)")
    ap.add_argument("--cond", default="RAW", help="condition name when using --audio-dir")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--save-embeddings", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--precision", choices=["fp32", "tf32", "fp16", "bf16"], default="fp32")
    ap.add_argument("--keep-failed", action="store_true")
    args = ap.parse_args()

    from adapters import get_adapter
    ad = get_adapter(args.adapter)

    if args.audio_dir:
        rows = rows_from_dir(args.audio_dir, args.cond)
    else:
        if not args.manifest or not args.nb_root:
            sys.exit("pass --manifest and --nb-root, or --audio-dir")
        rows = read_manifest(args.manifest, args.keep_failed)

    # A queued sbatch task cannot have its CONDITIONS changed, so a condition is
    # dropped from a run in flight by listing it in <scores dir>/SKIP_CONDITIONS.
    # Checked before the protocol join and the model load, so a skip costs seconds.
    # Scoped to that one scores folder, and nothing is written for the condition.
    skip_file = Path(os.path.dirname(os.path.abspath(args.out))) / "SKIP_CONDITIONS"
    if skip_file.is_file():
        cond = rows[0]["cond"].strip()
        if cond in skip_file.read_text().split():
            print(f"SKIPPED: {cond} is listed in {skip_file}; nothing scored", flush=True)
            return
    rows = build(rows, read_labels(args.labels), args.nb_root)
    if args.nshards > 1:
        rows = rows[args.shard::args.nshards]
    if args.limit:
        rows = rows[:args.limit]

    nb = sum(1 for r in rows if r["label"] == "bonafide")
    print(f"scoring {len(rows)} clips ({nb} bonafide / {len(rows) - nb} spoof), "
          f"cond {','.join(sorted({r['cond'] for r in rows}))}, "
          f"adapter {ad.name}, window {ad.window}, resampler {RESAMPLER}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("WARNING: no CUDA device -- this will be very slow", flush=True)
    if args.precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if args.precision in ("fp16", "bf16") and not ad.half_ok:
        sys.exit(f"{ad.name} must run in fp32 -- see the note in its adapter. "
                 f"Re-run with --precision fp32.")

    model = ad.build(device=dev, ckpt=args.ckpt, repo=args.repo)
    model.eval()
    cast = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.precision)
    if cast and dev == "cuda":
        model = model.to(cast)
    taps = Taps(model, gate_name=ad.gate_name)
    print(f"model on {dev} ({args.precision}); embedding tap {taps.emb_module}", flush=True)

    loader = DataLoader(Clips([r["path"] for r in rows], ad.window),
                        batch_size=args.batch_size, num_workers=args.workers,
                        shuffle=False, pin_memory=(dev == "cuda"))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    embs, gates, rates = [], [], {}
    want_emb = bool(args.save_embeddings)
    done, t0 = 0, time.time()
    # Written under a temporary name and renamed only once every clip is scored. A
    # crash mid-run would otherwise leave a header-only CSV at args.out, which the
    # sbatch resume check ("exists, skipping") then mistakes for a finished shard.
    part = args.out + ".part"
    with open(part, "w", newline="") as fh, NO_GRAD():
        w = csv.writer(fh)
        w.writerow(CSV_FIELDS)
        for batch, idx, srs, durs in loader:
            batch = batch.to(dev, non_blocking=True)
            if cast and dev == "cuda":
                batch = batch.to(cast)
            # A model that cannot take a real batch is fed one clip per forward. The
            # DataLoader still batches either way, which is what keeps the NFS reads
            # and FLAC decoding overlapped with compute.
            chunks = [batch] if ad.batch_ok else [batch[j:j + 1] for j in range(batch.shape[0])]
            pos = 0
            for ch in chunks:
                taps.reset()
                sc, lo_s, lo_b = ad.score(model, ch)
                if want_emb:
                    if taps.emb is None:
                        sys.exit(f"embedding hook did not fire on "
                                 f"{rows[int(idx[pos])]['utt']}; refusing to continue "
                                 f"with misaligned embeddings")
                    embs.append(taps.emb.astype("float16"))
                    if taps.gate_module is not None and taps.gates is not None:
                        gates.append(taps.gates.astype("float32"))
                for k in range(len(sc)):
                    i = int(idx[pos + k])
                    sr, dur, r = int(srs[pos + k]), float(durs[pos + k]), rows[i]
                    rates[sr] = rates.get(sr, 0) + 1
                    w.writerow([r["utt"], r["cond"], r["label"], r["attack"],
                                f"{sc[k]:.6f}", f"{lo_s[k]:.6f}", f"{lo_b[k]:.6f}",
                                sr, f"{dur:.3f}", int(dur * TARGET_SR < ad.window),
                                r["clipped"], r["padded"], ad.name])
                pos += len(sc)
            done += batch.shape[0]
            if done % (args.batch_size * 50) < args.batch_size:
                rate = done / max(time.time() - t0, 1e-9)
                print(f"  {done}/{len(rows)}  {rate:.1f} clips/s  "
                      f"eta {(len(rows) - done) / max(rate, 1e-9) / 60:.1f} min", flush=True)

    os.replace(part, args.out)
    secs = time.time() - t0
    print(f"wrote {done} scores to {args.out}")
    print(f"source sample rates seen: {rates}")
    print(f"{secs / 60:.1f} min, {done / max(secs, 1e-9):.1f} clips/s")

    if args.save_embeddings and embs:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_embeddings)) or ".",
                    exist_ok=True)
        payload = dict(
            utt_id=np.array([r["utt"] for r in rows]),
            cond=np.array([r["cond"] for r in rows]),
            label=np.array([r["label"] for r in rows]),
            attack=np.array([r["attack"] for r in rows]),
            embedding=np.concatenate(embs, axis=0),
            meta=np.array([json.dumps(dict(model=ad.name, ckpt=args.ckpt,
                                           precision=args.precision, resampler=RESAMPLER,
                                           window=ad.window, emb_module=taps.emb_module))]),
        )
        if gates:
            payload["gates"] = np.concatenate(gates, axis=0)
        np.savez_compressed(args.save_embeddings, **payload)
        print(f"wrote embeddings {payload['embedding'].shape} to {args.save_embeddings}")
    taps.close()


if __name__ == "__main__":
    main()
