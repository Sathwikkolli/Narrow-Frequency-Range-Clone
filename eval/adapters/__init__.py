"""Model adapters for harness.py.

One adapter per detector. Each answers only what is model-specific; everything
else -- file lists, decoding, resampling, windowing, CSV, embeddings -- is the
harness's job and is shared, so all six detectors see identical audio samples.

An adapter is a class with:

    name        str, goes into the CSV's `model` column
    window      int, samples at 16 kHz this model expects
    batch_ok    bool, True if forward() takes a real batch
    half_ok     bool, False if the model must stay fp32
    gate_name   str or None, module name holding per-layer gates
    build(device, ckpt, repo) -> nn.Module
    score(model, x) -> (score, logit_spoof, logit_bonafide), each a 1-D array

`score` returns that model's OWN score convention, so its published number can be
reproduced exactly. The two raw logits go into the CSV beside it, so the other
convention is recoverable without rescoring.

Imports are lazy: the six detectors need four different conda environments, and
importing this package must not drag fairseq into an environment that has no
fairseq.
"""
import importlib
import sys

REGISTRY = {
    "rawnet2": ("rawnet2", "RawNet2Adapter"),
    "aasist": ("aasist", "AasistAdapter"),
    "tcm": ("tcm", "TcmAdapter"),
    "sls": ("sls", "SlsAdapter"),
    "nes2net": ("nes2net", "Nes2NetAdapter"),
    # LCNN is not a plain nn.Module: the official LFCC-LCNN baseline is a
    # project-NN-Pytorch-scripts project whose model.py expects that framework's
    # data plumbing and computes LFCC inside forward(). It needs its own approach
    # rather than this adapter shape -- see adapters/lcnn_NOTES.md.
}


def get_adapter(name):
    key = name.strip().lower()
    if key not in REGISTRY:
        sys.exit(f"unknown adapter {name!r}; available: {sorted(REGISTRY)}")
    mod_name, cls_name = REGISTRY[key]
    mod = importlib.import_module(f"adapters.{mod_name}")
    return getattr(mod, cls_name)()
