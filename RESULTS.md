# Narrowband detector suite — results

ASVspoof2019 LA eval, 71,237 trials (7,355 bonafide / 63,882 spoof), identical
trial list and labels in every condition. Scored with `eval/harness.py`; one
shared loader, so any difference between conditions is the audio, not the I/O.

78.5% of clips are shorter than the 4.04 s window and are tile-repeated, in every
condition equally.

---

## RawNet2 — official ASVspoof2021 baseline (`pre_trained_DF_RawNet2.pth`)

SincConv front end, ~25M params, trained on ASVspoof2019 LA train.

| Condition | What it is | Rate | EER % | 95% CI | AUC | d′ |
|---|---|---|---|---|---|---|
| `RAW` | untouched | 16k | **4.60** | 4.38–4.76 | 0.9905 | 4.14 |
| `F0` | RAW × −6 dB headroom | 16k | **4.42** | 4.21–4.63 | 0.9908 | 3.55 |
| `P0` | + P.56 level (−26 dBov) | 16k | **6.14** | 5.89–6.43 | 0.9833 | 2.42 |
| `F5` | FFT overlap-add | 16k | 24.16 | 23.70–24.64 | 0.8497 | 1.42 |
| `F2` | Kaiser FIR | 16k | 24.46 | 23.97–24.99 | 0.8469 | 1.39 |
| `F1` | Parks–McClellan | 16k | 24.79 | 24.26–25.36 | 0.8431 | 1.37 |
| `F4` | Polyphase (8k round trip) | 16k | 25.44 | 24.96–26.04 | 0.8364 | 1.28 |
| `F3` | Elliptic IIR | 16k | 26.35 | 25.86–26.93 | 0.8277 | 1.26 |
| `B1` | + IRS8 handset | 8k | **35.73** | 35.19–36.30 | 0.7137 | 0.76 |
| `C1` | + G.711 µ-law | 8k | **35.69** | 35.09–36.26 | 0.7142 | 0.76 |

### 1. P0 moves the results — the pipeline's own warning has fired

`dataset/README.md` says: *"If P0 moves your results, the pipeline itself is a
confound and every other number is suspect."*

    RAW → P0     4.60% → 6.14%      +1.55 pp   (+34% relative)
    d′           4.14  → 2.42       −41%

P0 is the source plus P.56 level normalisation and nothing else. **Level
normalisation alone costs RawNet2 1.5 pp of EER**, and the d′ collapse is larger
still — the classes stayed separable but the distributions moved a long way
together.

This does not invalidate the ITU arm, but it means the `B1`/`C1` degradation is
*partly* a level effect. Every ITU number must be read against P0, never RAW.

### 2. A constant gain is nearly free — so it is normalisation, not loudness

    RAW → F0     4.60% → 4.42%      −0.18 pp

F0 is RAW scaled by a single constant (−6 dB headroom). EER barely moves, which
rules out "the model is just level-sensitive". What hurts in P0 is the *per-file*
P.56 normalisation, not gain as such. Note d′ still drops 4.14 → 3.55 under a
pure constant, so d′ is more fragile here than EER.

### 3. The five methods track LEAKAGE, not algorithm

`nfr_dataset/README.md` instructs: *"if two methods differ, the first thing to
check is whether the difference tracks leakage rather than the algorithm."*

It does, almost perfectly:

| Method | Out-of-band leakage | EER % |
|---|---|---|
| `F3` Elliptic | 0.10% | 26.35 |
| `F4` Polyphase | 0.16% | 25.44 |
| `F2` Kaiser | 0.80% | 24.46 |
| `F1` Parks–McClellan | 0.82% | 24.79 |
| `F5` FFT overlap | 1.51% | 24.16 |

**More surviving out-of-band energy → lower EER**, monotonically, with the single
exception of F1/F2 whose leakage differs by 0.02 pp (a tie within noise).

So the 2.2 pp spread across the five is *not* evidence that the filtering
algorithm matters to a detector. It is evidence that the detector eats whatever
escapes the band. The corpus's stated question — does the choice of algorithm
change what a detector sees — answers **no, once leakage is accounted for**.

### 4. G.711 costs nothing on top of the handset filter

    B1 → C1     35.73% → 35.69%     −0.04 pp (CIs overlap almost entirely)

The µ-law codec adds no measurable further damage. Whatever RawNet2 was using is
already gone by the time the IRS8 filter has run.

### 5. The unexplained gap: ITU 35.7% vs flat band-limiting 24–26%

Both arms target 300–3400 Hz, yet the ITU conditions are ~10 pp worse than the
worst DSP method. Three candidate causes, in the order I would test them:

1. **IRS8 is not a flat bandpass.** It is the ITU-T P.48 handset *send response*,
   a shaped curve, not a brick wall over 300–3400 Hz. It removes more, and
   differently, than F1–F5 do.
2. **Level normalisation** — worth ~1.5 pp on its own (§1).
3. **The 8 kHz round trip** — B1/C1 are upsampled on load, F1–F5 are not.
   `F4` bounds this: it is the only DSP method with a rate round trip, and it
   sits 1.3 pp worse than `F5` while having 9× less leakage, so the round trip
   contributes little.

Ranking them: (1) is the likely bulk, (2) is measured, (3) is small.

### 6. Per-attack

On `RAW`, only **A17 (8.59%)** and **A18 (15.20%)** are hard; the other eleven are
under 3%. Under `B1` the picture inverts — **A13 54.1%, A16 52.7%, A10 52.1%,
A12 47.5%, A19 45.8%** are at or beyond chance, while **A09 stays detectable
(0.23% → 3.24%)** in every condition.

So narrowband does not degrade attacks uniformly: it destroys detection of the
attacks that were easiest in wideband, while the two that were already hard
change least.

### Caveat on the absolute number

`RAW` gives 4.60% where RawNet2's published 2019 LA eval figure is 1.12%. The
checkpoint here is `pre_trained_DF_RawNet2` — the DF-track release, trained on
the same 2019 LA train data but not the model that produced that number. The
magnitude is plausible; the *differences between conditions* are what this table
is for.

---

## W2V2-AASIST — Tak et al., Odyssey 2022 (`LA_model.pth`)

XLS-R 300M front end, 447k AASIST back-end, trained on ASVspoof2019 LA train.

| Condition | EER % | 95% CI | AUC | d′ | n |
|---|---|---|---|---|---|
| `RAW` | **0.228** | 0.16–0.31 | 0.9997 | 13.08 | 71,237 |
| `P0` | **0.203** | 0.15–0.27 | 0.9996 | 11.70 | 71,237 |
| `B1` | **0.244** | 0.19–0.34 | 0.9993 | 11.61 | 71,237 |
| `C1` | **0.247** | 0.19–0.34 | 0.9993 | 11.64 | 71,237 |
| `F0` | 0.202 | 0.16–0.26 | 0.9997 | 12.18 | 67,676 † |
| `F1` | 0.319 | 0.26–0.43 | 0.9994 | 9.90 | 67,676 † |
| `F2` | 0.274 | 0.22–0.36 | 0.9994 | 11.42 | 64,115 † |
| `F3` | 0.287 | 0.22–0.36 | 0.9994 | 11.48 | 64,115 † |
| `F4` | 0.318 | 0.24–0.38 | 0.9994 | 11.19 | 64,115 † |
| `F5` | 0.289 | 0.24–0.38 | 0.9994 | 11.39 | 64,115 † |

† provisional — merged while shards 17–19 were still running. RAW, P0, B1 and C1
are complete and final.

### The result: the front end decides everything

```
P0 -> B1      RawNet2   +29.59 pp      4.60% -> 35.73%
              AASIST     +0.04 pp      0.20% ->  0.24%
```

RawNet2 falls to near-chance; AASIST does not move. Same audio, same loader, same
protocol, same trials — the only difference is the front end.

RawNet2's SincConv filters are laid out across 0–8 kHz, so band-limiting silences
roughly half of them. XLS-R is a learned representation pretrained on wideband
speech, and on this dataset it keeps whatever it relies on inside 300–3400 Hz.

### It also validates the harness

`RAW` at 0.228% sits on the ~0.22% usually reported for W2V2-AASIST on 2019 LA
eval. Reproducing a published figure on unmodified audio means the window, the
score convention, the protocol join and the loader are all correct — which
retroactively supports RawNet2's numbers too.

### Limitation — read this before claiming robustness

AASIST holds **d′ 11–13 and AUC ≈ 0.9993 in every condition**. It has effectively
saturated ASVspoof2019 LA eval, so there is almost no headroom in which to
*observe* degradation: the 0.20% → 0.32% range is a real ordering but only a
handful of trials wide.

The defensible claim is therefore narrow: *narrowband does not meaningfully
degrade W2V2-AASIST on ASVspoof2019 LA eval.* It is **not** evidence that XLS-R
front ends are robust to narrowband in general. A set where AASIST starts around
2–3% — ASVspoof2021 DF, or In-the-Wild — would have the dynamic range to test
that properly, and is the obvious follow-up.

Note also that P0 (0.203%) is very slightly *better* than RAW (0.228%), the
opposite of RawNet2's 1.55 pp penalty. At this EER the difference is within the
confidence intervals, so the honest reading is that level normalisation does
nothing to AASIST either way.

---

## Remaining models

| Model | Front end | Status |
|---|---|---|
| RawNet2 | SincConv | ✅ complete |
| W2V2-AASIST | XLS-R 300M | ✅ complete (F-conditions to re-merge) |
| Nes2Net-X | XLS-R 300M | running |
| W2V2-SLS | XLS-R 300M | running |
| TCM-ADD | XLS-R 300M | running |
| LCNN | LFCC | running |

Nes2Net, SLS and TCM share AASIST's XLS-R front end, so the open question is
whether they land near AASIST's ~0.2% or whether the back-end matters after all.
LCNN is the one to watch: LFCC spreads its resolution evenly across 0–8 kHz, so
if the front end really is what decides this, LCNN should behave like RawNet2
rather than like the SSL models.

LCNN reads through its own tooling with `wav_samp_rate = 16000` hardcoded, so it
cannot read the 8 kHz conditions; it needs resampled copies of `B1` and `C1`
first. See `eval/adapters/lcnn_NOTES.md`.
