<!-- generated:start -->
# 07-transformers-deep-dive / 10-audio-transformers-whisper

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/07-transformers-deep-dive/10-audio-transformers-whisper/) · upstream spec
`phases/07-transformers-deep-dive/10-audio-transformers-whisper/docs/en.md`

```bash
uv run demo practice run 10-audio-transformers-whisper --ex 1
uv run demo explain 10-audio-transformers-whisper --ex 1
uv run pytest demos/phases/07-transformers-deep-dive/10-audio-transformers-whisper
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run `code/main.py`. Confirm the frame count for a 1-second signal at 16 kHz with 10 ms… | code | T0 | `ex01_two_frames_short_at_every_length.py` |
| 2 | Medium. Build the full log-mel spectrogram using `numpy.fft`. Verify 80 mel bins match `libro… | code | T1 | `ex02_nineteen_filters_are_narrower_than_a_bin.py` |
| 3 | Hard. Implement streaming inference: chunk audio into 10 s windows with 2 s overlap, run Whis… | code | T0 | `ex03_the_overlap_costs_twenty_five_points_of_wer.py` |
<!-- generated:end -->

## Answers

The lesson is a 107-line framing pipeline that stops short of the FFT, plus a
task-prompt builder and a size table. Exercise 1 asks you to confirm two numbers
that are both wrong by exactly 2. Exercise 2 asks for a comparison against a
library that is not installed. Exercise 3 asks for a model that does not exist —
and both of the things it *does* specify turn out to be where the answer lives.

Exercises 1 and 3 are **T0**; Exercise 2 is **T1** (numpy).

### 1 — 98, not 100, and two short at every length

| duration | samples | frames | `n / hop` | short by |
|---|---:|---:|---:|---:|
| 1 s | 16,000 | **98** | 100 | 2 |
| 5 s | 80,000 | **498** | 500 | 2 |
| 30 s | 480,000 | **2,998** | 3,000 | 2 |

**ANSWER: `frame_signal` emits `floor((n − 400)/160) + 1` frames** — exactly
`n/160 − 2` whenever the hop divides the length. The deficit is
`(frame − hop)/hop = 1.5` rounded up: a constant **2 frames**, not a percentage.

**FINDING: the lesson's own `TARGET_FRAMES` is unreachable from audio.** It is
3,000; 30 seconds of real signal gives 2,998. `pad_or_clip` therefore appends
**2 zero frames** to every full-length Whisper window this pipeline builds — the
last 20 ms of every input is silence that was never recorded. Whisper's own front
end centre-pads the waveform, which is what gets it to 3,000.

**FINDING: every frame is 12.5 ms late.** Frame `k` covers `[160k, 160k + 400)`,
so its centre is `160k + 200` — a constant **200 samples** after the hop position
it is named for, at every frame.

**CONTROL: `pad_or_clip` truncates silently too.** 60 seconds frames to 5,998 and
comes back as 3,000, with the second half dropped and nothing said.

### 2 — nineteen of the eighty filters are narrower than an FFT bin

`librosa`, `torch`, `torchaudio` and `soundfile` are all absent, so there is
nothing to compare against. `numpy` is present, so the spectrogram is built at
Whisper's own settings and checked against properties a correct filterbank must
have:

| property | value |
|---|---|
| spectrogram / bank shape | (80, 98) from (80, 201) |
| frames vs `frame_signal` | identical, 98 |
| Σ filters, interior bins | **1.0000** (partition of unity) |
| total power conserved | **1.0000** |
| 440 Hz tone peaks at | mel bin **15** — where 440 Hz falls |

**FINDING: 19 of the 80 filters span fewer than two FFT bins.** `n_fft=400` at
16 kHz gives bins of **40 Hz**; the first mel triangle spans **44.9 Hz = 1.12
bins**. A quarter of Whisper's mel resolution is finer than the spectrogram it is
computed from — a property of Whisper's published settings, not of this
implementation.

**FINDING: 28 of the 80 bins cover the first eighth of the band.** That matches
`mel(1000)/mel(8000)` = **35.2%**: 35% of the bins describe 12.5% of the
spectrum. The mel scale working as designed, and the reason the bottom filters
are unresolvable.

**FINDING: `frame_energy` keeps the one number that does not matter.** It tracks
the log total mel power at **r = 0.9976** — an excellent measure of *how loud*,
carrying nothing about *which frequencies*. The sequence here is `(3000, 1)`;
Whisper's is `(3000, 80)`.

### 3 — the prescribed overlap costs 25 points of WER before ASR runs

There is no Whisper, no audio and no network, so the recogniser is replaced by a
**perfect** one — each chunk returns exactly the words spoken inside it. That is
the right control: every error left is one the protocol created.

| | words | WER |
|---|---:|---:|
| reference (5 min @ 180 wpm) | 900 | — |
| naive concatenation of 38 chunks | **1,122** | **24.7%** |
| longest-suffix-prefix merge | 900 | **0.0%** |
| naive, one word clipped per edge | — | 16.7% |
| merged, one word clipped per edge | — | 0.2% |

**ANSWER: 10-second windows at 2-second overlap make 38 chunks covering 374 s of
a 300 s file.** The 222 extra words are the overlap, and they are insertions.

**FINDING: the chunk is smaller than Whisper's window, so streaming costs 3.8×.**
Whisper's encoder takes a fixed 30-second input and pads anything shorter, so a
10-second chunk costs a full 30-second forward pass. Single-pass: 10 windows =
300 s of encoder input. Streaming: `38 × 30 = 1,140` s. The exercise's own chunk
length turns a latency optimisation into **3.8× the compute**.

**FINDING: the merge is the difficulty, and nothing in the exercise says to do
it.** The 2 seconds of overlap exists precisely to make a suffix-prefix merge
possible; skipping it is the single largest error source in the pipeline, larger
than anything a real recogniser would contribute.
