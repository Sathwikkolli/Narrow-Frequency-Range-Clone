"""ECAPA-TDNN -- trained here by train/train.py --model ecapa / ecapa4k.

Source:   eval/adapters/_ecapa.py (this repo; no external repo, --repo is ignored)
Weights:  best.pth from train/train.py (plain state_dict)
Training: ASVspoof2019 LA train -- wideband (wb), C1 telephone (itu) or both (mix).

  window 64600      the training window (4 s at 16 kHz)
  score out[:, 1]   bonafide logit, as every other adapter
  two variants      ecapa (Mel 20-8000 Hz) and ecapa4k (Mel 20-4000 Hz). The Mel
                    matrix is a buffer in the checkpoint; load is strict and the
                    adapter refuses a checkpoint whose matrix is the other variant's,
                    so a mislabelled run cannot be scored with the wrong front end.

FP32 ONLY: the front end runs torch.stft and a log on the waveform.
Embeddings: the 192-d input of `classifier` (Linear 192 -> 2).
"""
import sys

import torch

from ._ecapa import build_ecapa


class _EcapaAdapter:
    variant = None
    window = 64600
    batch_ok = True
    half_ok = False
    gate_name = None

    @property
    def name(self):
        return self.variant

    def build(self, device, ckpt, repo=None):
        model = build_ecapa(self.variant)
        state = torch.load(ckpt, map_location="cpu")
        if isinstance(state, dict) and "model" in state and "optimizer" in state:
            state = state["model"]            # a last.pth resume checkpoint
        if any(k.startswith("module.") for k in state):
            state = {k[len("module."):]: v for k, v in state.items()}
        want = model.frontend.mel
        got = state.get("frontend.mel")
        if got is None or got.shape != want.shape or not torch.allclose(got, want):
            sys.exit(f"{ckpt}: its Mel front end is not the {self.variant} one -- "
                     f"score it with the other ecapa adapter")
        model.load_state_dict(state)          # strict
        return model.to(device)

    def score(self, model, x):
        out = model(x).float().cpu().numpy()
        if out.ndim != 2 or out.shape[1] != 2:
            sys.exit(f"{self.variant}: expected [B, 2], got {out.shape}")
        logit_spoof, logit_bonafide = out[:, 0], out[:, 1]
        return logit_bonafide, logit_spoof, logit_bonafide


class EcapaAdapter(_EcapaAdapter):
    variant = "ecapa"


class Ecapa4kAdapter(_EcapaAdapter):
    variant = "ecapa4k"
