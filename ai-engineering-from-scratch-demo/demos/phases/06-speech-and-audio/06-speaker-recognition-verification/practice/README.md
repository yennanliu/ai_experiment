<!-- generated:start -->
# 06-speech-and-audio / 06-speaker-recognition-verification

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/06-speech-and-audio/06-speaker-recognition-verification/) · upstream spec
`phases/06-speech-and-audio/06-speaker-recognition-verification/docs/en.md`

```bash
uv run demo practice run 06-speaker-recognition-verification --ex 1
uv run demo explain 06-speaker-recognition-verification --ex 1
uv run pytest demos/phases/06-speech-and-audio/06-speaker-recognition-verification
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run `code/main.py`. Builds synthetic "speakers" (different tone profiles), enrolls, com… | code | T1 | `ex01_three_hundred_pairs_not_one_hundred.py` |
| 2 | Medium. Use SpeechBrain ECAPA on 30 VoxCeleb1 utterances (5 speakers × 6 each). Compute EER w… | code | T1 | `ex02_plda_has_four_dimensions_to_learn_from.py` |
| 3 | Hard. Build the full enroll → diarize → verify pipeline with `pyannote.audio`. Evaluate DER o… | code | T1 | `ex03_the_whole_error_fits_inside_the_collar.py` |
<!-- generated:end -->

## Answers

`main()` prints **EER 0.00%** and explains it in one line: "synthetic speakers are
near-orthogonal". Four lines above, the same output puts the different-speaker
mean cosine at **0.558**. All three exercises are about that gap — between what
the demo measures and what it says it measures — and each of the three runs into
a sample size the exercise itself chose.

All three run at **T1**: each rebuilds embeddings through the lesson's own
pure-Python DFT.

### 1 — three hundred pairs, not one hundred

Five speakers × five utterances give `5·C(5,2) = 50` same-speaker and
`C(5,2)·25 = 250` different-speaker trials — **300, not the 100 the exercise
names**. EER is **0.00% at threshold 0.9888**.

**FINDING: this trial list cannot express one row of the lesson's own
leaderboard.** False rejects move in steps of `1/50` = 2.00 pp and false accepts
in `1/250` = 0.40 pp, so the reachable EERs are the multiples of **0.20 pp**. Not
one of 0.39, 0.42, 0.65, 0.87, 3.10 lands on that grid.

**FINDING: "near-orthogonal" is not what the same file prints.** 0.558 is 56°
from orthogonal. What gives 0% is the **margin**: the lowest same-speaker score is
**0.9888** and the highest different-speaker score is **0.7732**. A speaker's five
utterances are one deterministic waveform plus independent Gaussian noise — same
frequencies, same phases, same amplitude. There is no session or channel
variation, which is the entire difficulty of real verification.

**MECHANISM: the printed means do not predict the EER.**

| noise | same-speaker mean | different-speaker mean | EER |
|---:|---:|---:|---:|
| 0.04 (the lesson's) | 0.995 | 0.558 | **0.00%** |
| 0.2 | 0.999 | 0.989 | **0.00%** |
| 0.5 | 1.000 | 0.999 | 11.60% |

At 0.2 the two means are 0.011 apart and the separation is still perfect.
Separation lives in the tails.

**CONTROL: the coefficient carrying most of the cosine carries none of the
decision.** Dropping MFCC `c0` from both halves leaves EER at 0.00% and *lowers*
the different-speaker mean to 0.397. Keeping only `c0` gives **49.60%** — chance.

### 2 — PLDA has four dimensions to learn from, and twenty-five to fill

`speechbrain`, `torch`, `torchaudio` and `pyannote` are absent and VoxCeleb1 is
not here, so the lesson's own 26-dimensional `embed_mfcc_stats` stands in. The
ranks below are set by `N` and `S` alone, so they carry to ECAPA unchanged.

| 5 speakers × | N | rank(within) / dim | cond(within) | cosine EER | PLDA EER |
|---|---:|---:|---:|---:|---:|
| **6 (the exercise)** | 30 | **25 / 26** | **6.0e+16** | **1.22%** | **69.67%** |
| 20 | 100 | 26 / 26 | 2.2e+03 | 0.96% | 2.12% |

**ANSWER: PLDA cannot be estimated at the sample size the exercise names.** A
two-covariance model gets `N − S` = **25** independent residuals for a matrix it
must invert in 26 dimensions. For ECAPA's **192**-dimensional embedding that is
rank 25 in 192 — 13% of the space.

**`numpy.linalg.inv` does not raise.** It returns a matrix for a singular input,
and the scores it produces give **69.67% EER** — worse than a coin. A rank
deficiency has no other way to fail.

**FINDING: the speaker subspace has four directions at any sample size.**
`rank(between) ≤ S − 1 = 4` whether there are 30 utterances or 100, because it
counts *speakers*, not recordings. PLDA's entire model is a speaker subspace, so
collecting more audio cannot help it — only more speakers can, and the exercise
fixes that number at five.

### 3 — the whole error fits inside the collar

`pyannote`, `torch`, `torchaudio` and `speechbrain` are absent and AMI is not
here, so this is the `DESIGN D11` scaled-down run, built to the exercise's own
three stages. **Enroll**: one clean utterance per voice through
`embed_mfcc_stats`. **Diarize**: an energy VAD, then 0.5 s windows slid inside the
speech regions only. **Verify**: every window scored against every enrolled
speaker with the lesson's `cosine`. DER at 10 ms, NIST definition.

| | DER | missed | false alarm | confusion |
|---|---:|---:|---:|---:|
| as scored | **0.0134** | 0.0013 | 0.0121 | **0.0000** |
| with the standard 0.25 s collar | **0.0000** | 0 | 0 | 0 |

**FINDING: the collar removes all of it.** Every point of error lies within a
quarter-second of a reference boundary — exactly where a sliding window must
straddle two speakers. "Report DER" has two answers, **1.34%** and **0.00%**, for
one segmentation, and AMI numbers in the literature are quoted both ways.

**FINDING: the label mapping moves DER further than the pipeline does.** Enrolment
is what makes these labels *named*; a clustering diarizer — which is what
`pyannote` is — produces anonymous clusters, so DER has to search the assignment.
Over the six permutations of three labels this same hypothesis scores from
**0.0134 to 1.0121**, a range wider than the metric itself, for a step nothing in
the name "diarization error rate" announces.

**CONTROL: of enroll → diarize → verify, `code/main.py` ships the last stage.** It
has embeddings and a trial list. No VAD, no change detection, no clustering, no
overlap handling, and no DER — the metric the exercise asks you to report is not
in the module.
