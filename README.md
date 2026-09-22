# Narrow-Frequency-Range-Clone

Great Lakes run scripts, dataset paths and sample lists for
[issflab/Narrow-Frequency-Range](https://github.com/issflab/Narrow-Frequency-Range).
The pipeline code lives in that repo. Only what's needed to run it on our data lives here.

| File | What it is |
|---|---|
| `config.sh` | All paths: pipeline repo, STL binaries, Famous Figures root, run folder, conda env |
| `samples/trump_5real_5fake.csv` | 5 real + 5 fake Donald Trump clips (eval split of Deep_SVDD `oc_protocol_eval1000.csv`; fakes from F5TTS, XTTSV2, FISHSPEECH, COZYVOICE2, STYLETTS2) |
| `prepare_samples.py` | Finds those clips under `DATA_ROOT` and symlinks them into `RUN_DIR/input/{real, fake/<ATTACK>}` |
| `run_trump_samples.sbatch` | SLURM job: prepare samples, then run `stl_pipeline.py` |

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
