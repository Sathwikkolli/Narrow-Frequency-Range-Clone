"""
Fill web/audio/raw with synthetic stand-in clips so the site can be rehearsed
before the real Famous Figures audio is downloaded.

    python tools/make_placeholders.py
    python tools/build_web_assets.py --keep-real --telephone FISHSPEECH
    python tools/serve.py

These are tones, not voices. They exercise every screen, but do not show them to
a class. Clear them before fetching the real thing:

    rm -rf web/audio/raw web/audio/clips web/spectrograms web/clips.json
"""
import shutil
import subprocess
import sys
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "web" / "audio" / "raw"

# source folder, base frequency, duration, sample rate, extra filter
STANDINS = [
    ("-",          175, 3.5, 44100, ""),
    ("F5TTS",      180, 3.4, 44100, ",tremolo=f=5:d=0.7"),
    ("STYLETTS2",  200, 3.1, 22050, ",volume=-12dB"),
    ("FISHSPEECH", 160, 3.6, 16000, ",vibrato=f=6:d=0.4"),
    ("XTTSV2",     220, 3.2, 24000, ",tremolo=f=3:d=0.5"),
    ("COZYVOICE2", 190, 3.3, 24000, ",volume=-6dB"),
]


def main():
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg must be on PATH")
    for source, f0, dur, sr, extra in STANDINS:
        d = RAW / source
        d.mkdir(parents=True, exist_ok=True)
        out = d / "Donald_Trump_00150.wav"
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-f", "lavfi", "-i", "sine=f={0}:d={1}{2}".format(f0, dur, extra),
               "-ar", str(sr), "-ac", "1", "-c:a", "pcm_s16le", str(out)]
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            sys.exit("ffmpeg failed for {0}:\n{1}".format(source, p.stderr[-800:]))
        print("[ok] {0:12s} {1} Hz tone at {2} Hz sample rate".format(source, f0, sr))
    print("\nplaceholders in {0}".format(RAW))
    print("next:  python tools/build_web_assets.py --keep-real --telephone FISHSPEECH")


if __name__ == "__main__":
    main()
