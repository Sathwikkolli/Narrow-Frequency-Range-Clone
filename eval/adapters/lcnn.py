"""LFCC-LCNN -- the official ASVspoof 2021 baseline (Xin Wang, NII).

Source:   github.com/asvspoof-challenge/2021  ->  LA/Baseline-LFCC-LCNN
Weights:  project/baseline_LA/__pretrained/trained_network.pt (project/00_download.sh),
          or a best.pth from train/train.py --model lcnn
Training: ASVspoof2019 LA train, so 2019 LA eval is in-domain.

This adapter is why adapters/lcnn_NOTES.md no longer has the last word. That note
ruled out an adapter because the baseline's forward() needs the framework's file
plumbing. _lcnn.py bypasses exactly that and nothing else: it subclasses the
baseline's own Model and calls its own _compute_embedding, so the LFCC front end,
the network and the state_dict keys are the baseline's. See _lcnn.py.

  window 64600      NOT the baseline's convention: its own tool scores whole
                    utterances. The harness needs one fixed length to batch and to
                    hold clip length constant across detectors, and 64600 is what
                    train/train.py trains LCNN at. So a released checkpoint scored
                    here is on 4 s windows, and its numbers will differ from
                    results that came through eval/run_lcnn.sh on full clips.
  score z           the baseline's own inference score, the pre-sigmoid logit
                    (model.py _compute_score(..., inference=True)). Higher = more
                    bonafide, as in every other adapter.
  logit_spoof 0     the model has one output. The CSV's two-logit columns hold
                    [0, z], whose softmax is [1 - sigmoid(z), sigmoid(z)].
  16 kHz only       LFCC is computed inside forward() for a fixed 16 kHz rate and
                    the baseline never resamples. The harness resamples every clip
                    to 16 kHz on load, so B1 and C1 need no 16 kHz copies here
                    (run_lcnn.sh did, because it bypasses the harness).

FP32 ONLY: the LFCC front end runs torch.stft and a log on the waveform.

NO EMBEDDINGS. The embedding tap looks for a 2-way Linear and LCNN's output layer
is 1-way, so --save-embeddings stops with a clear error rather than writing nothing.

Needs torch < 2 for the baseline's old-style torch.stft call (no return_complex);
torch 1.8.1 (the ssl_spoof env) works, with the removed torch.rfft supplied during
construction by _lcnn.legacy_rfft.
"""
import sys

import torch

from ._lcnn import build_lcnn


class LcnnAdapter:
    name = "lfcc-lcnn"
    window = 64600
    batch_ok = True
    half_ok = False
    gate_name = None

    def build(self, device, ckpt, repo=None):
        if not repo:
            sys.exit("lcnn needs --repo pointing at "
                     "<asvspoof-challenge/2021>/LA/Baseline-LFCC-LCNN")
        model = build_lcnn(repo).to(device)
        state = torch.load(ckpt, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]       # the framework's training checkpoints
        if any(k.startswith("module.") for k in state):
            state = {k[len("module."):]: v for k, v in state.items()}
        model.load_state_dict(state)          # strict: the keys are the baseline's own
        return model

    def score(self, model, x):
        out = model(x).float().cpu().numpy()
        if out.ndim != 2 or out.shape[1] != 2:
            sys.exit(f"lcnn: expected [B, 2], got {out.shape}")
        logit_spoof, logit_bonafide = out[:, 0], out[:, 1]
        return logit_bonafide, logit_spoof, logit_bonafide
