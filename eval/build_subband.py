#!/usr/bin/env python3
"""Low-pass and high-pass copies of ASVspoof 2019 LA, for a sub-band analysis.

    python eval/build_subband.py --src SRC --out OUT --cond all --headroom-db -6
    python eval/build_subband.py ... --cond "LP3400 HP300" --shard 3 --nshards 20

F1-F5 all cut both edges at once (300-3400 Hz), so they cannot say WHICH missing
edge hurts a detector. These conditions cut one edge at a time:

    LP<hz>   keep 0..hz        e.g. LP3400 removes only what is above 3400 Hz
    HP<hz>   keep hz..Nyquist  e.g. HP300  removes only what is below 300 Hz

LP3400 and HP300 are the two halves of F3: together they split its EER change
into a high-edge part and a low-edge part. The other cutoffs trace where each
detector starts to fail.

Everything except the filter is the pipeline's own nfr_dataset/build_shard.py,
imported and driven from here -- same reader, int16 rounding, headroom, manifest
schema and output layout -- so the control is the existing F0 and the scoring
harness reads these exactly like F1-F5:

    OUT/ASVspoof2019_LA_<subset>/<cond>/flac/<file_id>.flac
    OUT/manifest/<cond>_<subset>_<shard>.csv        lo_hz / hi_hz record the band

THE FILTER is the F3 recipe with one edge: elliptic IIR, zero-phase
(sosfiltfilt), to nfr.SPEC -- 1 dB ripple, 60 dB stopband, 25 Hz transition, as
realised after the double pass. F3 is the tightest of the five (0.10% leakage),
which is what a cutoff sweep wants. Run with --self-test to check the realised
response of every condition against that spec.

Build with the same --headroom-db as F0 (-6), or the control sits at a different
level from the conditions.
"""
import argparse
import os
import re
import sys
from importlib import util as _util
from pathlib import Path

import numpy as np
from scipy import signal

LOWPASS = [2000, 3400, 4000, 5000, 6000, 7000]
HIGHPASS = [100, 300, 500, 1000]
DEFAULT = [f"LP{c}" for c in LOWPASS] + [f"HP{c}" for c in HIGHPASS]
FS = 16000


def load_build_shard(nfr_dir):
    p = Path(nfr_dir) / "nfr_dataset" / "build_shard.py"
    if not p.is_file():
        sys.exit(f"{p} not found -- pass --nfr-dir or set NFR_DIR to the pipeline repo")
    spec = _util.spec_from_file_location("build_shard", p)
    mod = _util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def band_of(cond):
    """LP3400 -> (0, 3400);  HP300 -> (300, Nyquist)."""
    m = re.fullmatch(r"(LP|HP)(\d+)", cond)
    if not m:
        sys.exit(f"unknown condition {cond!r}; expected LP<hz> or HP<hz>")
    hz = float(m.group(2))
    if not 0 < hz < FS / 2:
        sys.exit(f"{cond}: cutoff must be inside 0..{FS // 2} Hz")
    return (0.0, hz) if m.group(1) == "LP" else (hz, FS / 2)


def make_edge_filter(nfr):
    spec = nfr.SPEC

    def edge_filter(x, fs, lo, hi):
        """One-edged band limit: lo <= 0 is a low-pass at hi, hi >= fs/2 a high-pass at lo."""
        # zero-phase runs the filter twice, so design to half the dB figures
        d_stop, d_rip = nfr._halve(True, spec["stop_db"], spec["ripple_db"])
        trans = spec["trans_hz"]
        if lo <= 0:
            wp, ws, btype = hi, hi + trans, "low"
        elif hi >= fs / 2:
            wp, ws, btype = lo, lo - trans, "high"
        else:
            raise ValueError("edge_filter takes one edge; use nfr.elliptic_iir for a band")
        # +1 dB as in nfr._design_ellip: the minimum order can land a fraction short
        n, _ = signal.ellipord(wp, ws, d_rip, d_stop + 1.0, fs=fs)
        sos = signal.ellip(n, d_rip, d_stop + 1.0, wp, btype=btype, output="sos", fs=fs)
        return signal.sosfiltfilt(sos, x)

    return edge_filter


def self_test(nfr, conds):
    """Realised response of each condition against nfr.SPEC; exits 1 on a miss."""
    fn, spec, tol, bad = make_edge_filter(nfr), nfr.SPEC, 0.5, 0
    trans = spec["trans_hz"]
    print(f"{'cond':<8}{'band':>14}{'ripple dB':>11}{'stop dB':>10}{'leak %':>9}")
    for c in conds:
        lo, hi = band_of(c)
        x = np.zeros(65536)
        x[len(x) // 2] = 1.0
        mag = np.abs(np.fft.rfft(fn(x, FS, lo, hi)))
        f = np.fft.rfftfreq(len(x), 1 / FS)
        db = 20 * np.log10(mag + 1e-300)
        pb = db[(f >= max(lo, 0) + (trans / 2 if lo > 0 else 0)) &
                (f <= hi - (trans / 2 if hi < FS / 2 else 0))]
        sb = db[(f <= lo - trans) | (f >= hi + trans)]
        ripple, stop = pb.max() - pb.min(), sb.max() - pb.max()
        leak = nfr.leakage(f, db, lo, hi)
        ok = ripple <= spec["ripple_db"] + tol and stop <= -spec["stop_db"] + tol
        bad += not ok
        print(f"{c:<8}{f'{lo:g}-{hi:g}':>14}{ripple:>11.2f}{stop:>10.1f}{leak:>9.3f}"
              f"{'' if ok else '   MISSES SPEC'}")
    sys.exit(1 if bad else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nfr-dir", default=os.environ.get("NFR_DIR",
                    os.path.expanduser("~/Narrow-Frequency-Range")),
                    help="the pipeline repo (issflab/Narrow-Frequency-Range)")
    ap.add_argument("--cond", default="all",
                    help=f"'all' ({' '.join(DEFAULT)}) or a space-separated list of "
                         f"LP<hz> / HP<hz>")
    ap.add_argument("--self-test", action="store_true",
                    help="check the realised filter responses and exit; needs no audio")
    args, passthrough = ap.parse_known_args()

    bs = load_build_shard(args.nfr_dir)
    conds = DEFAULT if args.cond == "all" else args.cond.split()
    for c in conds:
        band_of(c)
    if args.self_test:
        self_test(bs.nfr, conds)

    fn = make_edge_filter(bs.nfr)
    for c in conds:
        lo, hi = band_of(c)
        kind = "low-pass" if lo <= 0 else "high-pass"
        # build_shard builds whatever is in CONDITIONS; hand it this one condition
        bs.CONDITIONS = {c: ("edge", f"Elliptic IIR {kind}", fn)}
        sys.argv = ["build_shard.py", "--cond", c, "--lo", str(lo), "--hi", str(hi)] + passthrough
        bs.main()


if __name__ == "__main__":
    main()
