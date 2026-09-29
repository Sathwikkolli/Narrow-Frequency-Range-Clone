"""Shared plumbing for the XLS-R adapters (AASIST, TCM, SLS, Nes2Net).

All four descend from the SSL_Anti-spoofing training pipeline and share three
awkward properties that are handled once here:

1. They are imported from their own repo, not installed. `sys.path` gets the repo
   prepended, and the entry is kept so a second adapter in the same process does
   not shadow the first.

2. Their SSLModel hardcodes a RELATIVE path to the fairseq checkpoint:

       cp_path = 'xlsr2_300m.pt'

   so the constructor only works if the process's working directory contains that
   file. We chdir into the repo for the duration of the build and restore
   afterwards, rather than requiring the caller to launch from the right place.
   The file must therefore sit in the repo directory (a symlink is fine):

       ln -s /path/to/xlsr2_300m.pt <repo>/xlsr2_300m.pt

   Note the constructor loads those weights and `load_state_dict` then overwrites
   them with the fine-tuned checkpoint -- but the file must still exist, because
   the architecture is built from it.

3. Their `Model(args, device)` takes an argparse Namespace whose fields differ per
   model. Each adapter supplies its own, using that repo's documented defaults.
"""
import contextlib
import os
import sys
from argparse import Namespace


@contextlib.contextmanager
def in_repo(repo):
    """Prepend the repo to sys.path and chdir into it, then restore."""
    repo = os.path.abspath(repo)
    if not os.path.isdir(repo):
        sys.exit(f"--repo not a directory: {repo}")
    cwd = os.getcwd()
    if repo not in sys.path:
        sys.path.insert(0, repo)
    os.chdir(repo)
    try:
        yield repo
    finally:
        os.chdir(cwd)


def require_xlsr(repo):
    p = os.path.join(repo, "xlsr2_300m.pt")
    if not os.path.isfile(p):
        sys.exit(
            f"xlsr2_300m.pt not found in {repo}.\n"
            f"The model's SSLModel hardcodes that relative path and cannot be built "
            f"without it, even though the fine-tuned checkpoint replaces its weights.\n"
            f"  wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt\n"
            f"  ln -s /abs/path/to/xlsr2_300m.pt {p}")


def load_model(repo, module, cls, args, device, ckpt, needs_xlsr=True):
    """Import <module>.<cls> from the repo, build it, load the checkpoint."""
    import importlib
    import torch

    with in_repo(repo) as r:
        if needs_xlsr:
            require_xlsr(r)
        mod = importlib.import_module(module)
        Model = getattr(mod, cls, None)
        if Model is None:
            sys.exit(f"{module}.{cls} not found in {r}")
        model = Model(args, device).to(device)
        state = torch.load(ckpt, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if any(k.startswith("module.") for k in state):
            # saved from DataParallel; strip rather than load with strict=False,
            # which would leave unmatched layers randomly initialised and silent
            state = {k[len("module."):]: v for k, v in state.items()}
        model.load_state_dict(state)
    return model


def logits2(out, who):
    """Normalise a model's output to a [B, 2] float numpy array."""
    import numpy as np

    if isinstance(out, (tuple, list)):
        out = out[0]                      # TCM returns (logits, attention weights)
    out = out.float().cpu().numpy()
    if out.ndim != 2 or out.shape[1] != 2:
        sys.exit(f"{who}: expected [B, 2] logits, got {tuple(out.shape)}")
    return np.asarray(out)


def ns(**kw):
    return Namespace(**kw)
