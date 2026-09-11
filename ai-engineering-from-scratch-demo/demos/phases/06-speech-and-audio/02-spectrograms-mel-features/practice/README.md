<!-- generated:start -->
# 06-speech-and-audio / 02-spectrograms-mel-features

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/06-speech-and-audio/02-spectrograms-mel-features/) · upstream spec
`phases/06-speech-and-audio/02-spectrograms-mel-features/docs/en.md`

```bash
uv run demo practice run 02-spectrograms-mel-features --ex 1
uv run demo explain 02-spectrograms-mel-features --ex 1
uv run pytest demos/phases/06-speech-and-audio/02-spectrograms-mel-features
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run `code/main.py`. It synthesizes a chirp (frequency swept 200 → 4000 Hz) and prints t… | code | T0 | `ex01_the_track_the_lesson_prints_turns_around.py` |
| 2 | Medium. Re-run with `n_mels` in `{40, 80, 128}` and `frame_len` in `{200, 400, 800}`. Measure… | code | T0 | `ex02_a_third_of_the_finest_filterbank_is_empty.py` |
| 3 | Hard. Implement `power_to_db` and compare ASR accuracy of a tiny CNN classifier on AudioMNIST… | code | T1 | `ex03_the_three_feature_sets_differ_by_one_number.py` |
<!-- generated:end -->

## Answers

All three exercises turn on the same 24 lines of `main()`'s Step 5, and the first
one settles the other two: **the chirp the lesson ships is not the chirp it
describes.** `chirp` puts the instantaneous frequency into the phase, so the
sweep runs at double rate and crosses Nyquist halfway through the clip. The
argmax mel track the exercise asks you to "confirm" is a V, printed in the
lesson's own output.

Exercises 1 and 2 run at **T0** on stdlib alone. Exercise 3 is **T1**: it builds
120 utterances through the lesson's pure-Python STFT and fits four classifiers.

### 1 — the track the lesson prints turns around

`main()`'s Step 5, all 24 frames:

```text
11 15 20 23 26 28 30 33 35 36 38 39 | 39 38 36 35 33 31 28 26 23 20 15 11
```

It falls on **11 of 23** transitions and is its own mirror to within one bin at
one frame. A linear sweep is monotone; this is a fold.

**MECHANISM: `chirp` puts the instantaneous frequency into the phase.**
`sin(2π·f(t)·t)` with `f(t) = f0 + (f1−f0)·t/T` differentiates to
`f0 + 2(f1−f0)·t/T` — **twice** the requested rate, ending at **7800 Hz** rather
than 4000. A linear sweep needs quadratic phase.

**MECHANISM: at `sr=8000` the doubled sweep crosses Nyquist at the midpoint.**
4000 Hz is reached at `t = 0.200 s`, exactly half the 0.4 s clip, so the second
half folds back down — and the printed track turns at frame 11 of 23.

**CONTROL: fixing only the phase makes the same pipeline monotone.** The same
`stft_magnitude`, the same `mel_filterbank`, the same `log_transform`, driven by
a quadratic-phase chirp: `8 11 13 15 18 20 … 37 38 39 39`, **zero descents**,
ending at the top mel bin because the corrected sweep ends exactly at Nyquist.

### 2 — a third of the finest filterbank is empty

Half-max ridge width per frame, averaged over frames, at `main()`'s fixed hop of
128 with `n_fft` tied to `frame_len`:

| mels \ frame_len | 200 | 400 | 800 |
|---|---:|---:|---:|
| **40** | **116 Hz** / 2.17 bins | 336 Hz / 3.82 | 739 Hz / 6.42 |
| **80** | 178 Hz / 3.92 | 385 Hz / 7.09 | 791 Hz / 12.26 |
| **128** | 181 Hz / 5.00 | 381 Hz / 10.59 | 804 Hz / 19.11 |

**ANSWER: 40 mels at `frame_len=200`** — narrowest on both readings. Frame length
dominates `n_mels` by a wide margin: 200 → 800 costs **6.4×** in Hz, 40 → 128
mels costs at most 1.6×.

**FINDING: the doc's resolution trade runs backwards here.** "Larger FFT = better
frequency resolution" is a statement about stationary tones. This chirp sweeps at
**19,000 Hz/s**, so an 800-sample frame spans **1900 Hz** of sweep against its own
**10 Hz** bin spacing — a 190-bin smear the FFT's resolution never touches.

**FINDING: mel bins and Hz rank the grid in opposite orders.** 40/400 beats
128/200 in bins (3.82 vs 5.00) and loses to it in Hz (336 vs 181). A mel bin is
`range/(n_mels+1)`, not a fixed width, so "bandwidth" measured in bins is partly
a measurement of `n_mels`. The exercise does not say which unit it wants.

**FINDING: 128 mels over a 200-sample frame leaves 40 of 128 filters empty.**
`mel_filterbank` rounds triangle edges onto `n_fft//2+1 = 101` FFT bins; adjacent
edges collide and the row stays all zero — 31% of the bank. 80/200 loses 12 and
128/400 loses 12. `apply_filterbank` returns `0.0`, `log_transform` maps that to a
constant `log(1e-10) = −23.026`, and Step 6's DCT then transforms it. Nothing
warns.

**CONTROL: the three columns do not share a time axis** — 24, 22 and 19 frames.
The nine cells are not nine measurements of one thing.

### 3 — the three feature sets differ by one number

AudioMNIST and torch are both absent, so the dataset is a 10-class synthetic
spoken-digit stand-in (two formants per class, jittered F0, seeded) and the "tiny
CNN" is a multinomial logistic classifier over mean+std-pooled frames. Both
substitutions are shared by all arms.

`power_to_db` is implemented as librosa defines it — `10·log10(S/max S)`,
`amin=1e-10`, 80 dB floor — and the first result is that it is not a new
representation:

```text
power_to_db(S) == (10/ln 10) · log_mel(S) − max(…)        to 1.4e-14
```

(b) is (a) rescaled by 4.3429 and shifted by **one scalar per utterance**.

| features | matched | 20–40 dB quieter |
|---|---:|---:|
| (a) raw log-mel | 1.000 | **0.383** |
| (b) dB-mel, `ref=max` | 1.000 | **1.000** |
| (c) MFCC-13 + Δ + ΔΔ | 1.000 | **0.150** |
| (c′) the same, c0 dropped | 1.000 | **1.000** |

**ANSWER: on matched material the comparison measures nothing** — all three arms
score 1.000 on 60 held-out utterances. The exercise's question only has an answer
once the test set stops resembling the training set.

**FINDING: one dimension of 39 sinks the MFCC arm, and the lesson names it.**
Deltas difference out any constant, so 26 of the 39 dimensions are
level-invariant by construction and 12 of the remaining 13 are too. `c0` is the
sum of all 40 log-mels, so it moves **40× as far** as any single log-mel bin —
which is why (c) lands *below* (a). `main()`'s Step 6 prints "coef 0 encodes
overall energy; typically dropped downstream"; the exercise's feature spec keeps
it, and dropping it recovers 1.000.
