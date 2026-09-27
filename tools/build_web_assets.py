"""
Turn web/audio/raw/<SOURCE>/<clip>.wav into the assets the demo site serves.

    python tools/build_web_assets.py                      # everything it finds
    python tools/build_web_assets.py --pick F5TTS STYLETTS2 FISHSPEECH
    python tools/build_web_assets.py --telephone FISHSPEECH

For each clip: level-match it, resample every variant to one sample rate, write
web/audio/clips/<SOURCE>.wav, render a log-scaled spectrogram to
web/spectrograms/<SOURCE>.png, and write web/clips.json for the site.

Level and sample rate are matched on purpose. Raw variants differ in loudness
and in bandwidth (16 vs 22.05 vs 44.1 kHz), and either one gives the answer
away for reasons that have nothing to do with how good the synthesizer is.

--telephone ATTACK also writes an 8 kHz phone-band version, the same band the
narrowband pipeline studies. Handy as the deliberately easy clip: band-limiting
makes a clone sound obviously machine-like to a first-time listener.

Needs ffmpeg on PATH. No python packages.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "web" / "audio" / "raw"
CLIPS = ROOT / "web" / "audio" / "clips"
SPECS = ROOT / "web" / "spectrograms"
CONFIG = ROOT / "web" / "clips.json"

REAL_DIRS = {"-", "original", "bonafide"}

# What each attack is, and what a listener can actually try to catch it on.
# "rank" is a listening-difficulty guess: 1 = easiest to call out, 5 = hardest.
ATTACKS = {
    "FISHSPEECH": dict(
        name="Fish-Speech",
        kind="LLM text-to-speech over a grouped vector-quantised codec",
        rank=2,
        listen="Word endings that stop a beat too early, and sibilants that hiss flatter than a real mouth.",
    ),
    "XTTSV2": dict(
        name="XTTS-v2",
        kind="Coqui zero-shot voice cloner: a GPT-style token model plus a HiFi-GAN vocoder",
        rank=3,
        listen="Prosody that resets oddly mid-sentence, and a slightly glassy, over-smooth top end.",
    ),
    "COZYVOICE2": dict(
        name="CosyVoice 2",
        kind="Alibaba TTS: supervised semantic tokens, an LLM, then flow matching",
        rank=3,
        listen="Breaths in the wrong places, or none at all, and room tone that never changes.",
    ),
    "STYLETTS2": dict(
        name="StyleTTS 2",
        kind="Style diffusion trained adversarially against large speech language models",
        rank=5,
        listen="Almost nothing on a short clip. Given a long sentence the emphasis pattern gets too even.",
    ),
    "F5TTS": dict(
        name="F5-TTS",
        kind="Non-autoregressive flow-matching diffusion TTS, zero-shot cloning",
        rank=5,
        listen="Very natural rhythm. The tell is usually high-frequency detail too clean for a real microphone.",
    ),
}
FALLBACK = dict(name=None, kind="Text-to-speech voice clone", rank=3,
                listen="Compare the top of the spectrogram against the real clip.")
REAL_META = dict(
    name="Genuine recording",
    kind="The speaker's actual voice, straight from the Famous Figures dataset",
    rank=0,
    listen="This is the baseline. Breaths, room tone and uneven emphasis are what a real mic captures.",
)
PHONE_NOTE = (". Band-limited to roughly 300-3400 Hz at 8 kHz, the telephone band"
              " the narrowband pipeline studies.")

BASE_ID = re.compile(r"_(\d{4,6})")


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit("ffmpeg failed:\n  " + " ".join(cmd) + "\n" + p.stderr[-1500:])
    return p


def probe(path):
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,channels:format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True)
    if p.returncode != 0:
        return {}
    d = json.loads(p.stdout or "{}")
    s = (d.get("streams") or [{}])[0]
    return dict(sample_rate=int(s.get("sample_rate", 0)),
                channels=int(s.get("channels", 0)),
                duration=round(float(d.get("format", {}).get("duration") or 0), 2))


def encode(src, dst, sr, lufs, telephone=False):
    af = ["loudnorm=I={0}:TP=-1.5:LRA=11".format(lufs)]
    if telephone:
        # IRS-ish send band: drop everything outside roughly 300-3400 Hz, then 8 kHz.
        # ffmpeg caps poles at 2, so each section is applied twice for a steeper skirt.
        af += ["highpass=f=300:poles=2", "highpass=f=300:poles=2",
               "lowpass=f=3400:poles=2", "lowpass=f=3400:poles=2"]
        sr = 8000
    dst.parent.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
         "-af", ",".join(af), "-ac", "1", "-ar", str(sr),
         "-c:a", "pcm_s16le", str(dst)])
    return sr


def spectrogram(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
         "-lavfi", "showspectrumpic=s=1100x420:mode=combined:color=magma:"
                   "scale=log:fscale=lin:legend=1:gain=3",
         str(dst)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pick", nargs="*", default=None, metavar="ATTACK",
                    help="attacks to use in the game, in the order you want them played")
    ap.add_argument("--telephone", nargs="*", default=[], metavar="ATTACK",
                    help="also write an 8 kHz phone-band version of these attacks")
    ap.add_argument("--sr", type=int, default=16000, help="common sample rate (default 16000)")
    ap.add_argument("--lufs", type=int, default=-23, help="target loudness in LUFS (default -23)")
    ap.add_argument("--keep-real", action="store_true",
                    help="also publish the genuine clip, for the side-by-side lab panel")
    args = ap.parse_args()

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        sys.exit("ffmpeg and ffprobe must be on PATH")
    if not RAW.exists() or not any(RAW.rglob("*.wav")):
        sys.exit("no wavs under {0}\nrun:  GL_HOST=user@host bash tools/fetch_clips.sh <base-id>".format(RAW))

    found = []
    for wav in sorted(RAW.rglob("*.wav")):
        source = wav.parent.name
        is_real = source.lower() in REAL_DIRS
        m = BASE_ID.search(wav.stem)
        found.append(dict(path=wav, source="REAL" if is_real else source.upper(),
                          is_real=is_real, base_id=m.group(1) if m else "?"))

    ids = sorted({f["base_id"] for f in found})
    if len(ids) > 1:
        print("! raw/ holds more than one base clip: {0}".format(ids))
        print("  the game needs one sentence in several voices."
              " Clear web/audio/raw and fetch a single id.\n")

    fakes = [f for f in found if not f["is_real"]]
    if not fakes:
        sys.exit("only real clips in raw/ -- nothing to demo")

    picked = [a.upper() for a in (args.pick or [])]
    telephone = {a.upper() for a in args.telephone}
    have = {f["source"] for f in fakes}
    for a in picked + sorted(telephone):
        if a not in have:
            sys.exit("--pick/--telephone got {0}, which is not in raw/. present: {1}".format(a, sorted(have)))

    if not picked:
        # one clearly easier clip first, then the two hardest
        by_rank = sorted(fakes, key=lambda f: ATTACKS.get(f["source"], FALLBACK)["rank"])
        picked = list(dict.fromkeys([by_rank[0]["source"]] + [f["source"] for f in by_rank[-2:]]))
        print("no --pick given, going by the built-in difficulty ranking: {0}".format(picked))
        print("  listen to all of them and re-run with --pick if you disagree\n")

    CLIPS.mkdir(parents=True, exist_ok=True)
    SPECS.mkdir(parents=True, exist_ok=True)

    entries = []
    for f in found:
        if f["is_real"] and not args.keep_real:
            continue
        meta = REAL_META if f["is_real"] else ATTACKS.get(f["source"], FALLBACK)
        orig = probe(f["path"])
        for phone in ([False, True] if f["source"] in telephone else [False]):
            slug = f["source"] + ("_PHONE" if phone else "")
            wav = CLIPS / (slug + ".wav")
            png = SPECS / (slug + ".png")
            sr = encode(f["path"], wav, args.sr, args.lufs, telephone=phone)
            spectrogram(wav, png)
            out = probe(wav)
            entries.append(dict(
                id=slug,
                attack=f["source"],
                label="real" if f["is_real"] else "fake",
                system=meta["name"] or f["source"].title(),
                kind=meta["kind"] + (PHONE_NOTE if phone else ""),
                listen=meta["listen"],
                phone=phone,
                use=(not f["is_real"]) and (f["source"] in picked) and not phone,
                order=picked.index(f["source"]) if f["source"] in picked else 99,
                audio="audio/clips/{0}.wav".format(slug),
                spectrogram="spectrograms/{0}.png".format(slug),
                source_file=f["path"].name,
                base_id=f["base_id"],
                original_sample_rate=orig.get("sample_rate"),
                played_sample_rate=sr,
                duration=out.get("duration"),
            ))
            print("[ok] {0:18s} {1} Hz -> {2} Hz  {3}s".format(
                slug, orig.get("sample_rate"), sr, out.get("duration")))

    entries.sort(key=lambda e: (not e["use"], e["order"], e["id"]))
    config = dict(
        base_id=ids[0],
        speaker="Donald Trump",
        dataset="Famous Figures",
        note="Every clip marked use:true is synthetic. Same sentence, different synthesizer."
             " Level and sample rate are matched so those are not the giveaway.",
        matched_sample_rate=args.sr,
        matched_loudness_lufs=args.lufs,
        clips=entries,
    )
    CONFIG.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    used = [e["id"] for e in entries if e["use"]]
    print("\nwrote {0}  ({1} clips, {2} in the game: {3})".format(
        CONFIG.relative_to(ROOT), len(entries), len(used), used))
    print("serve it:  python tools/serve.py")


if __name__ == "__main__":
    main()
