# Narrow-Frequency-Range-Clone

Great Lakes run scripts, dataset paths and sample lists for
[issflab/Narrow-Frequency-Range](https://github.com/issflab/Narrow-Frequency-Range).
The pipeline code lives in that repo. Only what's needed to run it on our data lives here.

| File | What it is |
|---|---|
| `config.sh` | All paths: pipeline repo, STL binaries, Famous Figures root, run folder, conda env |
| `samples/trump_5real_5fake.csv` | 5 real + 5 fake Donald Trump clips (eval split of Deep_SVDD `oc_protocol_eval1000.csv`; fakes from F5TTS, XTTSV2, FISHSPEECH, COZYVOICE2, STYLETTS2) |
| `prepare_samples.py` | Finds those clips under `DATA_ROOT` and symlinks them into `RUN_DIR/input/{real, fake/<ATTACK>}` |
| `run_trump_samples.sbatch` | SLURM job: prepare samples, run `stl_pipeline.py`, then `spectrograms.py` on the narrowband and the original clips |

## First time on Great Lakes

```bash
cd $HOME
git clone https://github.com/issflab/Narrow-Frequency-Range.git
git clone <this repo's URL> Narrow-Frequency-Range-Clone
bash Narrow-Frequency-Range/setup_stl.sh        # login node; builds filter + sv56demo
```

## Each run

```bash
cd $HOME/Narrow-Frequency-Range && git pull
cd $HOME/Narrow-Frequency-Range-Clone && git pull
sbatch run_trump_samples.sbatch
```

The results go to `runs/trump_5real_5fake/output/` (8 kHz .wav files, with the same
`real/` and `fake/<ATTACK>/` layout as the input), along with `pipeline_log.csv`. The input
links, plus `manifest.csv` mapping each one back to its dataset path, go to `runs/trump_5real_5fake/input/`.
One constant-Q log spectrogram per output clip (magma colours) goes to `runs/trump_5real_5fake/spectrograms/`, e.g.
`real_Donald_Trump_01090.png` or `fake_F5TTS_Donald_Trump_00392.png`. The original clips get the
same images, with the same names, in `runs/trump_5real_5fake/spectrograms_original/`, so each pair can be compared side by side.

## Classroom demo: "Spot the Deepfake"

A local, dependency-free web app for showing high-school students that ears are not a
detector. Three clips, **same speaker, same sentence, all three synthetic**, level- and
bandwidth-matched so loudness and audio quality cannot give the answer away. One is
picked to be catchable, two are not, and the reveal at the end is that none of them
were real.

| File | What it is |
|---|---|
| `tools/find_variants.py` | Run on Great Lakes: finds base clips that several attacks all cloned |
| `tools/fetch_clips.sh` | Run locally: pulls every variant of one base clip into `web/audio/raw/` |
| `tools/build_web_assets.py` | Level-matches, resamples, renders spectrograms, writes `web/clips.json` |
| `tools/make_placeholders.py` | Synthetic stand-in tones, to rehearse the site before you have the audio |
| `tools/serve.py` | Static server for `web/` on localhost |
| `web/` | The site: title, three rounds, per-round reveal, the twist, and a lab panel |

Audio and derived images are gitignored; only the code is committed.

### Why the same sentence matters

`samples/trump_5real_5fake.csv` uses a different filename per attack, so those fakes speak
different sentences. The demo needs one sentence in several voices. A fake is named after
the real clip it cloned, so `F5TTS/Donald_Trump_00150.wav` and `FISHSPEECH/Donald_Trump_00150.wav`
are the same content through different synthesizers. `find_variants.py` locates such ids.

### Running it

```bash
# 1. on Great Lakes: pick a sentence several attacks cloned
source config.sh
python tools/find_variants.py --data-root "$DATA_ROOT" --min-attacks 3 --require-real

# 2. locally: pull every variant of that base id (use the id step 1 printed)
GL_HOST=uniqname@greatlakes.arc-ts.umich.edu bash tools/fetch_clips.sh 00150

# 3. locally: build the clips, spectrograms and config
python tools/build_web_assets.py --keep-real --telephone FISHSPEECH

# 4. serve it
python tools/serve.py
```

Listen to everything in the lab panel first. If the built-in difficulty ranking does not
match what you hear, re-run step 3 with your own order, easy clip first:

```bash
python tools/build_web_assets.py --pick FISHSPEECH F5TTS STYLETTS2 --keep-real
```

`--telephone ATTACK` also writes an 8 kHz, 300-3400 Hz version of that attack. Band-limiting
makes a clone noticeably more machine-like, so it is a reliable "easy" clip, and it is the
same band the narrowband pipeline studies. To use it in the game, set `use: true` on the
`*_PHONE` entry in `web/clips.json` and `false` on the one it replaces.

Needs `ffmpeg` on PATH. No Python packages, no CDN: the site runs on school wifi.
