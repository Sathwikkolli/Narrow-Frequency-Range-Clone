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
| `F0` | 0.204 | 0.16–0.26 | 0.9997 | 12.18 | 71,237 |
| `F1` | 0.316 | 0.27–0.41 | 0.9994 | 9.94 | 71,237 |
| `F2` | 0.285 | 0.23–0.37 | 0.9993 | 11.47 | 71,237 |
| `F3` | 0.288 | 0.22–0.36 | 0.9993 | 11.52 | 71,237 |
| `F4` | 0.313 | 0.24–0.38 | 0.9993 | 11.24 | 71,237 |
| `F5` | 0.296 | 0.24–0.37 | 0.9994 | 11.43 | 71,237 |

All ten conditions complete at the full trial count.

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

### The leakage relationship disappears — and that is the point

RawNet2's five DSP conditions tracked out-of-band leakage almost perfectly
(Spearman −0.90). AASIST's do not:

| | `F3` | `F4` | `F2` | `F1` | `F5` |
|---|---|---|---|---|---|
| leakage | 0.10% | 0.16% | 0.80% | 0.82% | 1.51% |
| RawNet2 EER | 26.35 | 25.44 | 24.46 | 24.79 | 24.16 |
| AASIST EER | 0.288 | 0.313 | 0.285 | 0.316 | 0.296 |

Spearman(leakage, EER) for AASIST is **+0.30** — no relationship, and the five
sit inside a 0.03 pp band, well within their own confidence intervals. They are
statistically indistinguishable.

The two halves fit one explanation. RawNet2 **depends** on energy above the
telephone band: more leakage gives it more to work with (ρ = −0.90), and removing
the band destroys it (+29.6 pp). AASIST does not depend on that energy: extra
leakage buys it nothing, and removing the band costs it nothing (+0.04 pp).

So the leakage correlation is not a nuisance to be controlled for — it is a
*measurement* of how much a detector was relying on the discarded band.

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

---

## FAR / FRR at a fixed operating point

EER picks a new threshold for every condition, which a deployed detector never
gets to do. Here each model gets **one** threshold, its EER threshold on
ASVspoof2019 LA `RAW`, frozen and applied unchanged to every condition on all
three datasets. FAR = spoofs accepted, FRR = bonafide rejected; bonafide is
accepted when score > threshold. Exact counts are in `results/far_frr.json`
(generated by `eval/far_frr.py`).

Checks: trial counts match every EER run; on ASVspoof2019 `RAW`,
FAR = FRR = the published EER for every model (by construction); the
thresholds equal the `eer_threshold` values in `results/*.json`.

#### ASVspoof2019 LA — FAR / FRR (%)

| Condition | What | RawNet2 | LCNN | AASIST | Nes2Net | TCM | SLS |
|---|---|---|---|---|---|---|---|
| RAW | original | 4.60 / 4.60 | 6.35 / 6.35 | 0.23 / 0.23 | 0.14 / 0.14 | 0.15 / 0.15 | 0.23 / 0.23 |
| P0 | level only | 3.17 / 12.43 | 9.43 / 2.94 | 0.28 / 0.16 | 0.12 / 0.23 | 0.19 / 0.30 | 0.20 / 0.76 |
| B1 | phone filter | 34.96 / 36.66 | 22.87 / 1.82 | 0.14 / 0.58 | 0.10 / 16.41 | 0.98 / 14.33 | 0.34 / 39.76 |
| C1 | + codec | 34.93 / 36.46 | 12.21 / 76.61 | 0.14 / 0.61 | 0.09 / 16.67 | 0.99 / 14.41 | 0.34 / 39.69 |
| F0 | control | 2.78 / 8.27 | 8.44 / 3.81 | 0.26 / 0.15 | 0.12 / 0.14 | 0.19 / 0.24 | 0.22 / 0.52 |
| F1 | Parks–McClellan | 3.61 / 60.31 | 18.22 / 18.29 | 0.09 / 4.68 | 0.09 / 15.47 | 0.35 / 18.22 | 0.36 / 35.23 |
| F2 | Kaiser FIR | 3.60 / 59.90 | 18.86 / 16.07 | 0.10 / 2.56 | 0.10 / 10.81 | 0.35 / 15.76 | 0.36 / 32.70 |
| F3 | Elliptic IIR | 4.28 / 61.36 | 17.52 / 20.86 | 0.09 / 2.39 | 0.11 / 10.89 | 0.51 / 11.28 | 0.39 / 31.38 |
| F4 | Polyphase | 5.17 / 56.64 | 17.57 / 21.36 | 0.10 / 2.68 | 0.10 / 11.54 | 0.48 / 11.57 | 0.39 / 32.22 |
| F5 | FFT overlap | 3.32 / 60.72 | 19.11 / 15.45 | 0.10 / 2.58 | 0.10 / 10.78 | 0.36 / 15.38 | 0.36 / 33.11 |

#### FakeOrReal (FF) — FAR / FRR (%)

| Condition | What | RawNet2 | LCNN | AASIST | Nes2Net | TCM | SLS |
|---|---|---|---|---|---|---|---|
| RAW | original | 14.26 / 72.74 | 1.09 / 83.76 | 10.08 / 95.67 | 11.26 / 85.97 | 10.50 / 89.34 | 18.47 / 72.63 |
| P0 | level only | 12.18 / 79.34 | 1.03 / 84.94 | 9.73 / 96.36 | 13.26 / 81.62 | 11.78 / 86.01 | 18.47 / 64.28 |
| B1 | phone filter | 30.25 / 47.01 | 2.57 / 81.91 | 12.06 / 94.28 | 4.27 / 96.68 | 5.05 / 93.58 | 11.38 / 86.50 |
| C1 | + codec | 30.09 / 47.45 | 4.83 / 73.47 | 11.92 / 94.34 | 4.25 / 96.70 | 5.13 / 93.69 | 11.37 / 86.47 |
| F0 | control | 23.65 / 66.72 | 1.29 / 81.32 | 9.53 / 95.56 | 4.19 / 92.97 | 6.17 / 93.44 | 11.92 / 85.09 |
| F1 | Parks–McClellan | 0.99 / 89.38 | 0.49 / 94.99 | 6.47 / 97.01 | 2.34 / 97.81 | 4.82 / 94.32 | 6.88 / 90.60 |
| F2 | Kaiser FIR | 0.98 / 89.55 | 0.50 / 95.04 | 7.07 / 96.87 | 2.42 / 97.75 | 4.70 / 94.24 | 7.28 / 90.22 |
| F3 | Elliptic IIR | 0.95 / 89.80 | 0.38 / 95.12 | 6.36 / 97.01 | 2.36 / 97.88 | 5.13 / 93.65 | 7.34 / 90.30 |
| F4 | Polyphase | 1.04 / 88.62 | 0.44 / 95.25 | 6.35 / 96.99 | 2.26 / 97.98 | 5.02 / 93.63 | 7.15 / 90.68 |
| F5 | FFT overlap | 1.04 / 89.86 | 0.55 / 94.74 | 7.07 / 96.89 | 2.34 / 97.69 | 4.70 / 94.34 | 7.41 / 90.30 |

#### In-the-Wild (ITW) — FAR / FRR (%)

| Condition | What | RawNet2 | LCNN | AASIST | Nes2Net | TCM | SLS |
|---|---|---|---|---|---|---|---|
| RAW | original | 5.92 / 88.21 | 9.13 / 64.43 | 0.29 / 77.69 | 0.17 / 64.88 | 0.72 / 53.04 | 0.30 / 60.54 |
| P0 | level only | 2.49 / 88.62 | 8.99 / 63.95 | 0.27 / 74.08 | 0.10 / 66.24 | 0.64 / 49.92 | 0.25 / 59.06 |
| B1 | phone filter | 17.62 / 65.47 | 13.07 / 40.50 | 0.32 / 82.40 | 0.15 / 94.89 | 1.67 / 80.56 | 0.43 / 90.39 |
| C1 | + codec | 17.69 / 65.44 | 26.03 / 32.37 | 0.32 / 82.51 | 0.15 / 94.97 | 1.68 / 80.71 | 0.42 / 90.45 |
| F0 | control | 18.77 / 80.15 | 13.64 / 56.27 | 0.29 / 82.12 | 0.27 / 81.00 | 1.11 / 65.59 | 0.36 / 75.07 |
| F1 | Parks–McClellan | 5.21 / 88.53 | 7.41 / 84.33 | 0.25 / 93.44 | 0.56 / 94.00 | 5.39 / 84.51 | 2.24 / 92.14 |
| F2 | Kaiser FIR | 5.23 / 88.53 | 7.33 / 84.40 | 0.25 / 92.80 | 0.54 / 93.68 | 5.32 / 84.29 | 2.23 / 91.69 |
| F3 | Elliptic IIR | 4.97 / 88.25 | 7.69 / 84.73 | 0.27 / 93.52 | 0.66 / 93.44 | 7.11 / 83.60 | 2.76 / 91.61 |
| F4 | Polyphase | 5.69 / 87.44 | 7.26 / 85.98 | 0.25 / 93.73 | 0.63 / 94.24 | 7.00 / 83.82 | 2.67 / 92.28 |
| F5 | FFT overlap | 5.13 / 88.98 | 7.82 / 83.18 | 0.25 / 92.87 | 0.60 / 93.61 | 5.12 / 84.43 | 2.22 / 91.79 |

| Model | Threshold |
|---|---|
| RawNet2 | -0.003148 |
| LCNN | +4.633137 |
| AASIST | +1.996360 |
| Nes2Net | +3.130621 |
| TCM | +0.433996 |
| SLS | -0.856500 |

**Read-out**

- **Narrowband costs are paid almost entirely in FRR, not FAR.** On ASVspoof2019
  the XLS-R models keep FAR ≤ 1.2% in every condition, while FRR goes up to
  11–18% (Nes2Net, TCM) and 31–40% (SLS) under F1–F5/B1/C1. The detector does
  not start accepting fakes. It starts rejecting real, band-limited speech.
  AASIST is the exception: FRR stays under 5% throughout.
- **LCNN fails the other way.** Its FAR rises from 6.3% to 17–23% under F1–F5
  and B1, so it begins letting spoofs through. C1 is its outlier (FRR 76.6%).
- **RawNet2**: FRR is about 60% under F1–F5, and B1/C1 split about 35/36.
- **Out of domain, the ASVspoof threshold does not carry over.** On FF, FRR is
  47–98% for every model in every condition. On ITW it is 53–88% at `RAW` and
  83–94% for every model under F1–F5. The EER tables hide this, because they
  re-fit the threshold each time. At a fixed threshold, most real speech from
  these datasets is flagged as fake.
