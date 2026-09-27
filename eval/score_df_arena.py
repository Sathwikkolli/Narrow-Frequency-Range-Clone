#!/usr/bin/env python3
"""Score narrowband ASVspoof2019 LA clips with Speech-Arena-2025/DF_Arena_*.

Captures three things in a single forward pass, because the model's own forward()
returns only {"logits": ...} and re-running 71k clips per condition to add an
analysis later is not free:

    logits      -> EER and every metric derived from it
    embedding   -> the input to the final classifier layer (UMAP, linear probe)
    layer gates -> the per-XLS-R-layer gate logits, if the module can be found

Inputs are joined on file_id:

  --manifest  the NB dataset's processing-log shards, e.g. manifest/P0_eval_0000.csv
              (cond,subset,file_id,...,clipped,padded,...,status,sha256). File list
              and QC record; carries no labels. It is a SUPERSET of the trial list.
  --labels    the ASVspoof2019 CM protocol, which is where bonafide/spoof and the
              attack id come from. The join keeps the intersection, which is what
              makes the score comparable to a published number on the same protocol.

Clip paths are built from the manifest's own cond/subset columns:
    <nb-root>/ASVspoof2019_LA_<subset>/<cond>/flac/<file_id>.flac

SAMPLE RATE: the model's feature extractor is fixed at 16 kHz. P0 is already 16 kHz;
B1/C1 are 8 kHz. The decision is made per file from its own header, never per
condition, because skipping the conversion on an 8 kHz file yields plausible-looking
but meaningless scores. Resampling also keeps the model's fixed 64600-sample window
equal to 4.04 s of speech in every condition, so the arms see the same amount of
content. A sinc/polyphase resampler is used: a naive one would image energy into the
empty 4-8 kHz band, which is precisely the band the experiment is about.

SCORE: the score column is logits[bonafide] - logits[spoof], so higher means more
bonafide. Do not use the HF pipeline's "score" field for EER -- that is the winning
class's confidence, which is high at both ends of the scale and not monotone.
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

MAX_LEN = 64600          # the feature extractor's fixed window
TARGET_SR = 16000        # the only rate the model accepts

LABEL_WORDS = {"bonafide": "bonafide", "bona-fide": "bonafide", "bona_fide": "bonafide",
               "real": "bonafide", "genuine": "bonafide", "human": "bonafide",
               "spoof": "spoof", "fake": "spoof", "spoofed": "spoof", "synthetic": "spoof"}

try:
    import soxr

    def resample(x, sr):
        return x if sr == TARGET_SR else soxr.resample(x, sr, TARGET_SR, quality="HQ")

    RESAMPLER = "soxr:HQ"
except ImportError:                                  # scipy is a safe fallback
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
    """
    out = {}
    with open(path) as fh:
        for line in fh:
            f = next(csv.reader([line])) if line.count(",") >= 2 else line.split()
            f = [t.strip() for t in f if t.strip()]
            if not f:
                continue
            label = next((LABEL_WORDS[t.lower()] for t in f if t.lower() in LABEL_WORDS), None)
            utt = next((t for t in f if t.startswith("LA_") and t.count("_") >= 2), None)
            if label is None or utt is None:
                continue
            attack = next((t for t in f
                           if len(t) == 3 and t[0] == "A" and t[1:].isdigit()), "-")
            out[utt] = (label, attack)
    if not out:
        sys.exit(f"no bonafide/spoof lines in {path} -- is it really a CM protocol?")
    return out


def read_manifest(patterns, keep_failed=False):
    """Manifest shards -> list of row dicts, deduped on file_id, status != ok dropped."""
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
                if not keep_failed and (r.get("status") or "ok").strip().lower() != "ok":
                    dropped += 1
                    continue
                seen.add(fid)
                rows.append(r)
    if not rows:
        sys.exit(f"no usable rows across {len(files)} manifest file(s)")
    print(f"manifest: {len(files)} shard(s), {len(rows)} clips"
          + (f", {dropped} dropped for status != ok" if dropped else ""), flush=True)
    return rows


def build(manifest_rows, labels, nb_root, audio_dir):
    """Intersect manifest with protocol and resolve each clip to a real file."""
    out, no_label, no_file = [], [], []
    for r in manifest_rows:
        fid = r["file_id"].strip()
        if fid not in labels:
            no_label.append(fid)
            continue
        cond, subset = r["cond"].strip(), r["subset"].strip()
        if audio_dir:
            p = Path(audio_dir) / f"{fid}.flac"
            if not p.exists():
                p = Path(audio_dir) / "flac" / f"{fid}.flac"
        else:
            p = Path(nb_root) / f"ASVspoof2019_LA_{subset}" / cond / "flac" / f"{fid}.flac"
        if not p.exists():
            no_file.append(str(p))
            continue
        label, attack = labels[fid]
        out.append(dict(utt=fid, path=p, label=label, attack=attack, cond=cond,
                        clipped=(r.get("clipped") or "").strip(),
                        padded=(r.get("padded") or "").strip()))
    if no_label:
        print(f"  {len(no_label)} manifest clips are not trials in the protocol "
              f"(expected: the manifest is a superset), e.g. {no_label[:2]}", flush=True)
    if no_file:
        print(f"  WARNING: {len(no_file)} clips missing on disk, e.g. {no_file[:2]}", flush=True)
    if not out:
        sys.exit("nothing to score: manifest, protocol and audio tree do not intersect")
    return out


class Clips(Dataset):
    """Reads, downmixes, resamples to 16 kHz, then fits the model's fixed window."""

    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        x, sr = sf.read(str(self.paths[i]), dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        dur = len(x) / sr
        x = np.asarray(resample(x, sr), dtype="float32")
        n = len(x)
        if n >= MAX_LEN:
            x = x[:MAX_LEN]
        else:                                    # tile-repeat, as the model's extractor does
            x = np.tile(x, int(np.ceil(MAX_LEN / max(n, 1))))[:MAX_LEN]
        return torch.from_numpy(np.ascontiguousarray(x)), i, sr, dur


# ---------------------------------------------------------------- model hooks

def describe_modules(model, limit=5000):
    """Print the module tree with Linear shapes, so hook targets can be pinned by name."""
    print(f"{'module':<70} {'type':<22} shape")
    for i, (name, m) in enumerate(model.named_modules()):
        if i >= limit:
            print("  ... truncated")
            break
        shape = ""
        if isinstance(m, torch.nn.Linear):
            shape = f"{m.in_features} -> {m.out_features}" + ("" if m.bias is not None else " (no bias)")
        elif isinstance(m, torch.nn.Conv1d):
            shape = f"{m.in_channels} -> {m.out_channels} k{m.kernel_size[0]} s{m.stride[0]}"
        print(f"{name or '<root>':<70} {type(m).__name__:<22} {shape}")


class Taps:
    """Forward hooks that stash the classifier's input and the layer-gate output.

    The classifier is found as the last Linear whose out_features equals the number
    of classes; its *input* is the pooled pre-head embedding. The gate module is
    matched by name (fc0 in this backbone) and falls back to nothing rather than
    guessing, since several Linears in this model share the 1280->1 shape.
    """

    def __init__(self, model, num_labels=2, gate_name="fc0", classifier_name=None):
        self.emb = self.gates = None
        self.handles = []
        self.emb_module = self.gate_module = None

        linears = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
        if classifier_name:
            cand = [(n, m) for n, m in linears if n == classifier_name]
            if not cand:
                sys.exit(f"--classifier-name {classifier_name!r} not found; run --inspect")
        else:
            cand = [(n, m) for n, m in linears if m.out_features == num_labels]
        if cand:
            self.emb_module = cand[-1][0]
            self.handles.append(cand[-1][1].register_forward_hook(self._take_emb))

        gate = [(n, m) for n, m in linears if n.split(".")[-1] == gate_name]
        if gate:
            self.gate_module = gate[0][0]
            self.handles.append(gate[0][1].register_forward_hook(self._take_gates))

    def reset(self):
        """Clear before every forward so a hook that does not fire is detected.

        Without this, a missed hook would leave the previous clip's tensor in place
        and it would be recorded against the next utterance -- the counts would
        still line up, so the misalignment would be silent and would corrupt the
        embedding analysis while leaving the scores untouched.
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", nargs="+",
                    help="manifest shard(s); globs allowed, e.g. 'manifest/P0_eval_*.csv'")
    ap.add_argument("--labels", help="ASVspoof2019 CM protocol file for this split")
    ap.add_argument("--nb-root", default=None,
                    help="narrowband dataset root; clip paths are built under it from "
                         "the manifest's cond/subset columns")
    ap.add_argument("--audio-dir", default=None,
                    help="override: take every clip from this one folder instead")
    ap.add_argument("--out", help="output scores CSV")
    ap.add_argument("--save-embeddings", default=None,
                    help="also write an .npz of embeddings, gates, ids and scores here")
    ap.add_argument("--model", default="Speech-Arena-2025/DF_Arena_1B_V_1")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="clips fetched and decoded per DataLoader step. This is an "
                         "I/O prefetch group, NOT a model batch: the model scores one "
                         "clip per forward because backbone.py unsqueezes the batch "
                         "axis itself. Raising it does not use more GPU memory.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N clips")
    ap.add_argument("--precision", choices=["fp32", "tf32", "fp16", "bf16"], default="fp32",
                    help="fp32 for numbers you will publish; fp16/bf16 are 4-6x faster "
                         "but move the logits slightly, so validate before trusting them")
    ap.add_argument("--keep-failed", action="store_true",
                    help="keep manifest rows whose status is not ok")
    ap.add_argument("--gate-name", default="fc0", help="module name holding the layer gates")
    ap.add_argument("--classifier-name", default=None,
                    help="pin the classifier module instead of auto-detecting it")
    ap.add_argument("--inspect", action="store_true",
                    help="print the module tree and exit (CPU only, no data needed)")
    args = ap.parse_args()

    from transformers import AutoModel

    if args.inspect:
        model = AutoModel.from_pretrained(args.model, trust_remote_code=True).eval()
        describe_modules(model)
        taps = Taps(model, gate_name=args.gate_name, classifier_name=args.classifier_name)
        print(f"\nclassifier (embedding tap): {taps.emb_module}")
        print(f"layer-gate tap:             {taps.gate_module}")
        print(f"parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M")
        return

    for req in ("manifest", "labels", "out"):
        if not getattr(args, req):
            sys.exit(f"--{req} is required (or pass --inspect)")
    if not args.nb_root and not args.audio_dir:
        sys.exit("pass --nb-root (or --audio-dir)")

    rows = build(read_manifest(args.manifest, args.keep_failed),
                 read_labels(args.labels), args.nb_root, args.audio_dir)
    if args.limit:
        rows = rows[:args.limit]
    nb = sum(1 for r in rows if r["label"] == "bonafide")
    print(f"scoring {len(rows)} clips ({nb} bonafide / {len(rows) - nb} spoof), "
          f"condition(s) {','.join(sorted({r['cond'] for r in rows}))}, "
          f"resampler {RESAMPLER}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("WARNING: no CUDA device -- this will be very slow", flush=True)
    if args.precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True).to(dev).eval()
    cast = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.precision)
    if cast and dev == "cuda":
        model = model.to(cast)
    taps = Taps(model, gate_name=args.gate_name, classifier_name=args.classifier_name)
    print(f"model on {dev} ({args.precision}); embedding tap {taps.emb_module}, "
          f"gate tap {taps.gate_module}", flush=True)
    if taps.emb_module is None:
        print("WARNING: no classifier Linear found -- embeddings will not be saved; "
              "run --inspect and pass --classifier-name", flush=True)

    loader = DataLoader(Clips([r["path"] for r in rows]), batch_size=args.batch_size,
                        num_workers=args.workers, shuffle=False,
                        pin_memory=(dev == "cuda"))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    embs, gates, rates = [], [], {}
    want_emb = bool(args.save_embeddings) and taps.emb_module is not None
    done, t0 = 0, time.time()
    with open(args.out, "w", newline="") as fh, torch.inference_mode():
        w = csv.writer(fh)
        w.writerow(["utt_id", "cond", "label", "attack_id", "score", "logit_spoof",
                    "logit_bonafide", "orig_sr", "dur_s", "tiled", "clipped", "padded"])
        for batch, idx, srs, durs in loader:
            batch = batch.to(dev, non_blocking=True)
            if cast and dev == "cuda":
                batch = batch.to(cast)
            # One clip per forward: backbone.py does x.unsqueeze(0) itself, so it
            # takes a 1-D waveform and its internal shapes assume a batch of one.
            # Passing a real batch reaches conv1d as a 4-D tensor and raises. The
            # DataLoader still batches, which is what keeps the NFS reads and FLAC
            # decoding overlapped with compute; only the forward is serial.
            for j in range(batch.shape[0]):
                taps.reset()
                lg = model(input_values=batch[j])["logits"].float().cpu().numpy().reshape(-1)
                if want_emb:
                    if taps.emb is None:
                        sys.exit(f"embedding hook did not fire on {rows[int(idx[j])]['utt']}; "
                                 "refusing to continue with misaligned embeddings")
                    embs.append(taps.emb.astype("float16"))
                    if taps.gate_module is not None:
                        if taps.gates is None:
                            sys.exit(f"gate hook did not fire on "
                                     f"{rows[int(idx[j])]['utt']}")
                        gates.append(taps.gates.astype("float32"))
                i, sr, dur = int(idx[j]), int(srs[j]), float(durs[j])
                r = rows[i]
                rates[sr] = rates.get(sr, 0) + 1
                # id2label = {0: spoof, 1: bonafide}; monotone in bonafide-ness
                w.writerow([r["utt"], r["cond"], r["label"], r["attack"],
                            f"{lg[1] - lg[0]:.6f}", f"{lg[0]:.6f}", f"{lg[1]:.6f}",
                            sr, f"{dur:.3f}", int(dur * TARGET_SR < MAX_LEN),
                            r["clipped"], r["padded"]])
            done += batch.shape[0]
            if done % (args.batch_size * 50) < args.batch_size:
                rate = done / max(time.time() - t0, 1e-9)
                eta = (len(rows) - done) / max(rate, 1e-9)
                print(f"  {done}/{len(rows)}  {rate:.1f} clips/s  eta {eta / 60:.1f} min",
                      flush=True)

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
            meta=np.array([json.dumps(dict(model=args.model, precision=args.precision,
                                           resampler=RESAMPLER, emb_module=taps.emb_module,
                                           gate_module=taps.gate_module))]),
        )
        if gates:
            payload["gates"] = np.concatenate(gates, axis=0)
        np.savez_compressed(args.save_embeddings, **payload)
        print(f"wrote embeddings {payload['embedding'].shape}"
              + (f" and gates {payload['gates'].shape}" if gates else "")
              + f" to {args.save_embeddings}")
    taps.close()


if __name__ == "__main__":
    main()
