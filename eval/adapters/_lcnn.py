"""LFCC-LCNN as a plain [B, T] -> [B, 2] module, built from the baseline's own Model.

One builder, used by train/train.py and by the eval adapter (adapters/lcnn.py), so
the network that is trained and the network that is scored cannot drift apart.

The baseline (asvspoof-challenge/2021, LA/Baseline-LFCC-LCNN) lives inside NII's
project-NN framework: Model.forward(x, fileinfo) reads labels from a protocol by
file name and returns a training tuple. Only that plumbing is bypassed. The
subclass below calls the repo's own _compute_embedding -- its LFCC front end and
LCNN-BLSTM network, unchanged -- and adds no parameters, so the state_dict keys
are exactly the baseline's and its released trained_network.pt loads as is.

TWO LOGITS FROM ONE. The baseline emits one logit z and trains sigmoid + BCE.
Returning [0, z] makes softmax([0, z])[1] = sigmoid(z), so unweighted
cross-entropy on it IS the baseline's BCE, and out[:, 1] = z is the baseline's own
inference score (_compute_score(..., inference=True) returns z).

`repo` is the Baseline-LFCC-LCNN folder: sandbox/ and core_scripts/ are imported
from there, model.py from project/baseline_LA/.
"""
import contextlib
import os

from ._common import in_repo, ns


@contextlib.contextmanager
def legacy_rfft():
    """torch.rfft for the baseline's constructor, on torches that removed it.

    sandbox/util_dsp.py builds LFCC's fixed DCT-II matrix (LinearDCT) with
    torch.rfft(v, 1, onesided=False), which torch 1.8 removed -- the baseline pins
    torch 1.6. It is only called while the model is constructed, never in forward.
    This is the old call's exact semantics on torch.fft: a real input's FFT, as a
    trailing [real, imag] axis. Removed again on exit so nothing else sees it.
    """
    import torch

    if hasattr(torch, "rfft"):
        yield
        return

    def rfft(x, signal_ndim, normalized=False, onesided=True):
        if signal_ndim != 1 or normalized:
            raise NotImplementedError("only the 1-D unnormalised form util_dsp.dct uses")
        return torch.view_as_real(torch.fft.rfft(x) if onesided else torch.fft.fft(x))

    torch.rfft = rfft
    try:
        yield
    finally:
        del torch.rfft


def build_lcnn(repo):
    """The baseline's network with freshly initialised weights, on CPU."""
    import importlib.util
    import sys

    import torch

    with in_repo(repo) as r, legacy_rfft():
        path = os.path.join(r, "project", "baseline_LA", "model.py")
        if not os.path.isfile(path):
            sys.exit(f"{path} not found -- point --repo at the Baseline-LFCC-LCNN folder")
        ms = importlib.util.spec_from_file_location("lcnn_baseline_model", path)
        mod = importlib.util.module_from_spec(ms)
        ms.loader.exec_module(mod)

        class LcnnTwoLogit(mod.Model):
            def forward(self, x):
                z = self._compute_embedding(x.unsqueeze(-1), None)[:, 0]   # one sub-model
                return torch.stack([torch.zeros_like(z), z], dim=1)

        # in/out dims 1 and no mean/std: what the baseline's config.py gives it.
        # The protocol path is only read by forward(x, fileinfo), which is unused.
        return LcnnTwoLogit(1, 1, None, ns(optional_argument=[""]), None)
