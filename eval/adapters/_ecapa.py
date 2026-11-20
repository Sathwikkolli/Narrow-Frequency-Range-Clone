"""ECAPA-TDNN as a plain [B, T] -> [B, 2] anti-spoofing module, front end included.

One builder, used by train/train.py and by the eval adapter (adapters/ecapa.py), so
the network that is trained and the network that is scored cannot drift apart --
the same arrangement as _lcnn.py.

Plain torch, no external repo, and no torchaudio: it has to build in ssl_spoof
(torch 1.8.1) as well as in a current torch.

FRONT END (log-Mel filterbank, computed inside forward on 16 kHz input)
    25 ms Hann window (400), 10 ms hop (160), 512-point FFT, power spectrum,
    80 HTK-Mel triangles, log(x + 1e-6), mean subtracted per utterance and per band.
    Two variants differ ONLY in the Mel range:

      ecapa     20-8000 Hz   the usual wideband front end. On telephone audio
                             (C1, upsampled 8 -> 16 kHz by the harness) the filters
                             above ~3.4-4 kHz see an empty band.
      ecapa4k   20-4000 Hz   all 80 filters inside the band an 8 kHz phone channel
                             can carry. Equivalent to an 8 kHz front end, but it
                             takes the same 16 kHz input, so harness, windows and
                             data trees are untouched.

    The Mel matrix is a registered buffer, so it is saved in the checkpoint and a
    checkpoint scored with the wrong variant is overwritten with its own matrix.

BACK END (Desplanques et al., Interspeech 2020)
    Conv1d(80 -> C, k5) -> 3 x SE-Res2Block (k3, dilation 2/3/4, scale 8, SE 128)
    -> multi-layer feature aggregation: concat of the 3 block outputs, Conv1d -> 1536
    -> attentive statistics pooling with global context -> 3072
    -> BN -> Linear 192 (the embedding) -> BN -> Linear 2 (spoof, bonafide).

    C = 512 (the paper's small model, ~6M parameters). The last Linear is 192 -> 2,
    so harness.Taps picks up the 192-d embedding with --save-embeddings.

Output is [B, 2] logits, index 1 = bonafide, as every other model in the suite:
trained with the shared weighted cross-entropy, scored on out[:, 1].
"""
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

SR = 16000
VARIANTS = {
    "ecapa": dict(f_min=20.0, f_max=8000.0),
    "ecapa4k": dict(f_min=20.0, f_max=4000.0),
}


# ---------------------------------------------------------------- front end

def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_matrix(n_mels, n_fft, sr, f_min, f_max):
    """[n_fft // 2 + 1, n_mels] HTK-Mel triangles (unnormalised peak 1)."""
    freqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
    edges = _mel_to_hz(np.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_mels + 2))
    lo, mid, hi = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    up = (freqs[None, :] - lo) / (mid - lo)
    down = (hi - freqs[None, :]) / (hi - mid)
    fb = np.clip(np.minimum(up, down), 0.0, None)          # [n_mels, n_freq]
    if (fb.sum(axis=1) == 0).any():
        raise ValueError(f"a Mel filter covers no FFT bin ({n_mels} mels, {n_fft}-pt FFT, "
                         f"{f_min}-{f_max} Hz)")
    return torch.tensor(fb.T, dtype=torch.float32)


class LogMel(nn.Module):
    def __init__(self, f_min, f_max, n_mels=80, n_fft=512, win=400, hop=160, sr=SR):
        super().__init__()
        self.n_fft, self.win, self.hop = n_fft, win, hop
        self.register_buffer("window", torch.hann_window(win))
        self.register_buffer("mel", mel_matrix(n_mels, n_fft, sr, f_min, f_max))

    def forward(self, x):                                   # [B, T] -> [B, n_mels, frames]
        x = x.float()                                       # fp32 even under autocast
        spec = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win,
                          window=self.window, center=True, return_complex=True)
        power = spec.real ** 2 + spec.imag ** 2              # [B, n_freq, frames]
        feats = torch.log(torch.matmul(power.transpose(1, 2), self.mel) + 1e-6)
        feats = feats - feats.mean(dim=1, keepdim=True)     # per-utterance mean norm
        return feats.transpose(1, 2)


# ---------------------------------------------------------------- back end

class ConvBlock(nn.Module):
    """Conv1d -> ReLU -> BN, 'same' length."""

    def __init__(self, cin, cout, k=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(cin, cout, k, dilation=dilation, padding=dilation * (k - 1) // 2)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


class Res2Conv(nn.Module):
    def __init__(self, channels, k, dilation, scale=8):
        super().__init__()
        if channels % scale:
            raise ValueError(f"channels {channels} not divisible by scale {scale}")
        self.scale = scale
        width = channels // scale
        self.convs = nn.ModuleList(ConvBlock(width, width, k, dilation) for _ in range(scale - 1))

    def forward(self, x):
        parts = torch.chunk(x, self.scale, dim=1)
        out, y = [parts[0]], None
        for i, conv in enumerate(self.convs):
            y = conv(parts[i + 1] if y is None else parts[i + 1] + y)
            out.append(y)
        return torch.cat(out, dim=1)


class SE(nn.Module):
    def __init__(self, channels, bottleneck=128):
        super().__init__()
        self.fc1 = nn.Conv1d(channels, bottleneck, 1)
        self.fc2 = nn.Conv1d(bottleneck, channels, 1)

    def forward(self, x):
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(x.mean(dim=2, keepdim=True)))))
        return x * s


class SERes2Block(nn.Module):
    def __init__(self, channels, k, dilation, scale=8, se=128):
        super().__init__()
        self.pre = ConvBlock(channels, channels, 1)
        self.res2 = Res2Conv(channels, k, dilation, scale)
        self.post = ConvBlock(channels, channels, 1)
        self.se = SE(channels, se)

    def forward(self, x):
        return x + self.se(self.post(self.res2(self.pre(x))))


class AttentiveStatsPool(nn.Module):
    """Channel- and context-dependent attentive statistics pooling (global context)."""

    def __init__(self, channels, attn=128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(channels * 3, attn, 1), nn.ReLU(), nn.BatchNorm1d(attn), nn.Tanh(),
            nn.Conv1d(attn, channels, 1))

    def forward(self, x):                                   # [B, C, T] -> [B, 2C]
        t = x.shape[2]
        mean = x.mean(dim=2, keepdim=True)
        std = torch.sqrt(x.var(dim=2, keepdim=True, unbiased=False).clamp(min=1e-5))
        ctx = torch.cat([x, mean.expand(-1, -1, t), std.expand(-1, -1, t)], dim=1)
        w = torch.softmax(self.attn(ctx), dim=2)
        mu = (w * x).sum(dim=2)
        sg = torch.sqrt(((w * x * x).sum(dim=2) - mu * mu).clamp(min=1e-5))
        return torch.cat([mu, sg], dim=1)


class EcapaTdnn(nn.Module):
    def __init__(self, f_min=20.0, f_max=8000.0, n_mels=80, channels=512, mfa=1536,
                 emb=192, n_classes=2):
        super().__init__()
        self.frontend = LogMel(f_min, f_max, n_mels)
        self.layer0 = ConvBlock(n_mels, channels, 5)
        self.blocks = nn.ModuleList(SERes2Block(channels, 3, d) for d in (2, 3, 4))
        self.mfa = ConvBlock(channels * 3, mfa, 1)
        self.pool = AttentiveStatsPool(mfa)
        self.bn_pool = nn.BatchNorm1d(mfa * 2)
        self.embed = nn.Linear(mfa * 2, emb)
        self.bn_emb = nn.BatchNorm1d(emb)
        self.classifier = nn.Linear(emb, n_classes)

    def embedding(self, wav):
        x = self.layer0(self.frontend(wav))
        outs = []
        for block in self.blocks:
            x = block(x)
            outs.append(x)
        x = self.mfa(torch.cat(outs, dim=1))
        return self.bn_emb(self.embed(self.bn_pool(self.pool(x))))

    def forward(self, wav):                                 # [B, T] -> [B, 2]
        return self.classifier(self.embedding(wav))


def build_ecapa(variant="ecapa"):
    if variant not in VARIANTS:
        raise ValueError(f"unknown ECAPA variant {variant!r}; known: {sorted(VARIANTS)}")
    return EcapaTdnn(**VARIANTS[variant])


def frames(window, hop=160):
    """Feature frames for a window of samples (center=True STFT)."""
    return 1 + window // hop


if __name__ == "__main__":                                 # quick shape / size check
    for v in VARIANTS:
        m = build_ecapa(v).eval()
        n = sum(p.numel() for p in m.parameters())
        with torch.no_grad():
            out = m(torch.randn(3, 64600))
        print(f"{v}: {n / 1e6:.2f}M params, out {tuple(out.shape)}, "
              f"{frames(64600)} frames, Mel {VARIANTS[v]['f_min']:.0f}-{VARIANTS[v]['f_max']:.0f} Hz "
              f"(top filter centre {math.floor(_mel_to_hz(_hz_to_mel(VARIANTS[v]['f_max']) * 80 / 81))} Hz)")
