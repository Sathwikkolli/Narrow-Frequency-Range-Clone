# LCNN — why it has no adapter

**Official source:** `asvspoof-challenge/2021` → `LA/Baseline-LFCC-LCNN`
(Xin Wang, NII). Pretrained LA and DF models ship with the repo, pulled by
`project/00_download.sh`, and were trained on the ASVspoof2019 LA training set —
so 2019 LA eval is in-domain, same as the other five.

## The problem

The other five detectors are plain `nn.Module`s: import a class, build it, hand it
a `[B, T]` waveform tensor, read `[B, 2]` logits back. LCNN is not. It is a
**project-NN-Pytorch-scripts project**: `model.py` expects that framework's own
data plumbing (`prj_conf`, `fileinfo`, mean/std normalisation state), and the LFCC
front end is computed *inside* `forward()` rather than by the caller.

Wrapping that in the adapter shape would mean reimplementing the framework's call
convention, which is exactly the kind of guesswork that produces numbers that look
plausible and are wrong.

## The plan instead

Run LCNN through its own tool, then convert its output into the harness's CSV
schema. The baseline explicitly supports scoring an arbitrary directory with no
protocol file:

```bash
cd project/baseline_LA
bash 02_eval_alternative.sh  <PATH_TO_WAV_DIR>  <NAME_OF_DATA_SET>  <TRAINED_MODEL>
```

with the pretrained model at `__pretrained/trained_network.pt`. One invocation per
condition, then `eval/lcnn_to_csv.py` maps the result onto
`utt_id,cond,label,attack_id,score,…,model` so `compute_metrics.py` treats it like
every other model.

Consequence: **LCNN has no embeddings and no `analyze_embeddings.py` output.** It
has no SSL front end, so the layer-gate analysis never applied to it anyway, and
the `Taps` classifier hook is not reachable through the framework. It contributes
EER and the other score-derived metrics only.

## The open question — resolve before trusting any LCNN number

**Does LCNN resample its input?**

The harness guarantees every model sees 16 kHz because it resamples on load. LCNN
bypasses the harness entirely and reads audio through project-NN's own data I/O,
which assumes the sampling rate configured for the project. The README documents
accepted *formats* (16/32-bit PCM WAV, 32-bit float WAV, FLAC) but says nothing
about rate conversion.

If it does not resample, feeding it the 8 kHz `B1` and `C1` files yields LFCCs
computed against the wrong frame rate and filter spacing — scores that come out
looking like a narrowband effect but are a configuration bug.

Check it on one file before any full run: score the same clip from `P0` (16 kHz)
and from `B1` (8 kHz) and confirm the frame counts differ in the ratio you would
expect from the rate, not from silent truncation.

**If it does not resample**, LCNN — and only LCNN — needs 16 kHz copies of `B1`
and `C1` written to disk first, produced with the same `soxr`/`resample_poly`
resampler the harness uses, so it still sees the same samples the other five do.
That is 2 conditions × 71,933 files, CPU only. The other eight conditions are
already 16 kHz and need nothing.
