<!-- generated:start -->
# 06-speech-and-audio / 17-audio-evaluation-metrics

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/06-speech-and-audio/17-audio-evaluation-metrics/) · upstream spec
`phases/06-speech-and-audio/17-audio-evaluation-metrics/docs/en.md`

```bash
uv run demo practice run 17-audio-evaluation-metrics --ex 1
uv run demo explain 17-audio-evaluation-metrics --ex 1
uv run pytest demos/phases/06-speech-and-audio/17-audio-evaluation-metrics
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run `code/main.py`. Compute WER / CER / EER / SECS / FAD-ish / MMAU-ish on toy inputs. | code | T0 | `ex01_the_one_metric_that_is_not_normalised.py` |
| 2 | Medium. Build a TTS round-trip WER harness. Run your Kokoro or F5-TTS output through Whisper.… | code | T0 | `ex02_the_ten_per_cent_flag_is_a_length_threshold.py` |
| 3 | Hard. Score your Lesson 10 LALM choice on MMAU-Pro speech + multi-audio subsets (50 items eac… | code | T0 | `ex03_fifty_items_is_wider_than_the_whole_table.py` |
<!-- generated:end -->

## Answers

This is the phase's metrics lesson, so the metrics themselves are the subject.
Replaying `main()`'s random stream reproduces every figure it prints — **EER
0.10%, SECS 0.644, FAD-like 1.748** — which means each divergence below is a
divergence from the published definition rather than from the demo. Four of the
six diverge, the doc quotes its own worked example with the wrong answer, and
both harder exercises pick a sample size that cannot reach the number it is asked
to compare against.

All three run at **T0**; the lesson's own code imports `math`, `random` and `re`.

### 1 — the one metric that is not normalised is CER

**FINDING: `cer` never normalises, and the file's own takeaway says it must.**
`wer` runs both strings through `normalize`; `cer` does not, and divides by the
raw reference length. On the doc's own Step 1 example:

| `"Please turn on the lights."` vs `"please turn on the light"` | |
|---|---:|
| CER, raw (as shipped) | **0.1154** |
| CER, normalised | **0.0400** |
| ratio | **2.9×** |

The capital `P` and the trailing full stop are both scored as character errors,
and the stop is in the denominator as well.

**FINDING: Step 1 quotes its own example with the wrong answer.** It says jiwer
returns "~0.17". One substitution over a five-word reference is **0.2000**, which
is exactly what the lesson's own `wer` returns. No normalisation reaches 0.17.

**FINDING: `embedding_fad_like` is the square root of a *diagonal* FAD.** It
keeps only per-dimension variance, so every off-diagonal covariance is discarded:

| two blocks, identical per-dimension moments, different correlation | |
|---|---:|
| per-dimension mean gap | 4.6e-17 |
| per-dimension variance gap | 1.3e-15 |
| mean \|off-diagonal correlation\| | **0.036 vs 0.896** |
| `embedding_fad_like` | **0.000000** |
| published Fréchet formula | **6.07** |

On `main()`'s own data it prints **1.748** where the real FAD is **15.608** — and
the line beside it compares that number to a published **4.5**.

**FINDING: SECS is pinned below the target printed next to it.** The clone is
`ref + noise` at equal variance, so its expected cosine is exactly
`1/√2 = 0.7071`, under the "> 0.75 for recognizable clone" the same line states.
Reaching 0.75 needs the noise at most **0.8819×** the signal; raising the
dimension only tightens the estimate around 0.7071.

**CONTROL: 100 same and 500 different pairs put EER on a 0.10 pp grid.** Of the
three EER figures in the file's own benchmark table, **ECAPA at 0.87%** and
**ASVspoof 5 at 7.23%** fall between the rungs.

### 2 — the ten per cent flag is a prompt-length threshold

`kokoro`, `f5_tts`, `whisper`, `torch`, `transformers`, `soundfile` and `jiwer`
are all absent and no audio ships anywhere in the reference tree. The harness
around the round trip runs, scored with the lesson's own `wer`.

**ANSWER: `WER > 10%` is a length threshold, not a quality threshold.** A prompt
of `N` reference words has WER on the grid `{0, 1/N, 2/N, …}`, so the flag fires
on a single word error whenever `1/N > 0.10`. Over 50 prompts of 4–12 words,
**31 of 50** are short enough that "over 10%" and "not perfect" are the same
predicate; for the other 19 it quietly becomes "at least two errors".

| per-word error rate | corpus (micro) WER | mean of per-prompt WER | flagged | any error |
|---:|---:|---:|---:|---:|
| 0.02 | **0.0167** | 0.0163 | **3** | **6** |
| 0.05 | **0.0646** | **0.0700** | 14 | 20 |

**FINDING: two different numbers are both "WER over 50 prompts".** Pooling edits
over all words and averaging the per-prompt rates are not the same statistic —
the macro average weights a four-word prompt like a twelve-word one. The exercise
does not say which it wants.

**FINDING: 50 prompts is 418 reference words.** At the measured corpus WER that
is a binomial standard deviation of **0.63 pp** and a 95% interval of
**±1.23 pp**. Two systems a point apart are one sample.

### 3 — fifty items is three times wider than the whole table

No LALM is installed and MMAU-Pro is not here, but "compare with the published
number" is arithmetic. The published numbers are parsed straight out of Lesson
10's own benchmark table rather than retyped.

| | |
|---|---:|
| published anchor (Qwen2.5-Omni-7B) | 52.2% |
| binomial sd over 50 items | **7.06 pp** |
| 95% interval | **±13.85 pp** → [38.4%, 66.0%] |
| full spread of Lesson 10's table | **7.8 pp** |
| interval ÷ spread | **3.6×** |

**ANSWER: a 50-item accuracy cannot separate any two rows of that table.**
Sampling 50 items for each of the five models at their published rates, 5,000
times: the published order returns **5.2%** of the time (0.83% for a shuffle) and
the best model is picked **47.2%** of the time against 20% for a coin.

**FINDING: no precisely published figure lies on the grid a 50-item score can
reach.** 50 items puts accuracy on multiples of **2.0 pp**, and all **11** of the
exactly reported figures in Lesson 10's table fall between the rungs. The only
entries that do land on it are the four written with a tilde — which were already
rounded.

**FINDING: three of the four published multi-audio numbers are below chance.**
The multi-audio subset is 4-way multiple choice, so **25.0%** is the floor:
Gemini 2.5 Pro at 22.0, Gemini 2.5 Flash at 21.2 and Qwen2.5-Omni-7B at 20.0 all
sit under it. Comparing a noisy 50-item score against a below-chance number
compares nothing.

**CONTROL: `mmau_accuracy` reads a length mismatch two ways, and neither raises.**
It zips the two lists and divides by `len(predictions)`, so ten predictions
against five golds score **0.5** while five predictions against ten golds score
**1.0** — a truncated run and a truncated key are the same bug with opposite
signs.
