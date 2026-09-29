"""RawNet2 -- the official ASVspoof 2021 baseline (Tak et al., EURECOM).

Source:   github.com/asvspoof-challenge/2021  ->  LA/Baseline-RawNet2
Weights:  https://www.asvspoof.org/asvspoof2021/pre_trained_DF_RawNet2.zip
Training: ASVspoof2019 LA train, so 2019 LA eval is in-domain.

Everything below is taken from that repo, not guessed:

  window 64600      data_utils.py: `self.cut = 64600  # take ~4 sec audio`
  tile-repeat pad   data_utils.py pad(): np.tile then truncate -- the same rule the
                    harness applies, so short clips are treated identically
  score out[:, 1]   main.py produce_evaluation_file():
                    `batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()`
                    Index 1 is bonafide; the model is trained with CCE over
                    {0: spoof, 1: bonafide}.
  batching          main.py evaluates at batch_size=128, so real batches are fine

The model is constructed from the repo's own YAML rather than a hard-coded arg
dict, so the architecture always matches whatever checkpoint that repo ships.

This adapter needs no fairseq and no SSL front end: RawNet2 is trained from
scratch on raw waveforms and is ~25M parameters, which is why it is the cheapest
model in the suite and the one to run first when validating the pipeline.
"""
import os
import sys

import numpy as np
import torch


class RawNet2Adapter:
    name = "rawnet2"
    window = 64600
    batch_ok = True
    half_ok = True
    gate_name = None          # no SSL front end, so no per-layer gates

    def build(self, device, ckpt, repo=None):
        if not repo:
            sys.exit("rawnet2 needs --repo pointing at "
                     "<asvspoof-challenge/2021>/LA/Baseline-RawNet2")
        repo = os.path.abspath(repo)
        if not os.path.isfile(os.path.join(repo, "model.py")):
            sys.exit(f"model.py not found in {repo} -- point --repo at the "
                     f"Baseline-RawNet2 folder itself")
        sys.path.insert(0, repo)

        import yaml
        from model import RawNet

        cfg_path = os.path.join(repo, "model_config_RawNet.yaml")
        if not os.path.isfile(cfg_path):
            sys.exit(f"{cfg_path} not found")
        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh)
        d_args = cfg["model"] if "model" in cfg else cfg

        model = RawNet(d_args, device).to(device)
        state = torch.load(ckpt, map_location=device)
        # The released checkpoint was saved from a DataParallel wrapper in some
        # copies; strip the prefix rather than loading with strict=False, which
        # would silently leave layers at their random initialisation.
        if any(k.startswith("module.") for k in state):
            state = {k[len("module."):]: v for k, v in state.items()}
        model.load_state_dict(state)
        return model

    def score(self, model, x):
        """x: [B, T] float32 at 16 kHz -> three 1-D arrays of length B."""
        out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        out = out.float().cpu().numpy()
        if out.ndim != 2 or out.shape[1] != 2:
            sys.exit(f"rawnet2: expected [B, 2] logits, got {out.shape}")
        logit_spoof, logit_bonafide = out[:, 0], out[:, 1]
        # the repo's own convention, so the published EER is reproducible
        score = logit_bonafide
        return score, logit_spoof, logit_bonafide
