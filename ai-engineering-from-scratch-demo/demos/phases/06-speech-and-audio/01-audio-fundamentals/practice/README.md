<!-- generated:start -->
# 06-speech-and-audio / 01-audio-fundamentals

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/06-speech-and-audio/01-audio-fundamentals/) · upstream spec
`phases/06-speech-and-audio/01-audio-fundamentals/docs/en.md`

```bash
uv run demo practice run 01-audio-fundamentals --ex 1
uv run demo explain 01-audio-fundamentals --ex 1
uv run pytest demos/phases/06-speech-and-audio/01-audio-fundamentals
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Synthesize a 1-second mix of 220 Hz + 440 Hz + 880 Hz at 16 kHz. Run DFT. Confirm three… | code | T1 | `ex01_the_easy_exercise_is_the_expensive_one.py` |
| 2 | Medium. Record a 3-second WAV of your voice at 48 kHz. Downsample to 16 kHz using `torchaudio… | code | T0 | `ex02_the_alias_lands_between_the_harmonics.py` |
| 3 | Hard. Build the STFT from scratch using only `math` and the DFT from Step 3. Frame size 400,… | code | T0 | `ex03_the_hard_one_is_sixteen_times_cheaper.py` |
<!-- generated:end -->

## Answers

The lesson is three functions long and every exercise turns out to be a question
about one of them. The arithmetic all works; what does not survive contact is the
labelling. The **Easy** exercise is the most expensive computation in the lesson —
24 s of pure-Python DFT — and the **Hard** one is 16× cheaper. The **Medium** one
cannot be set up at all, because nothing here can record audio and `torchaudio`
is absent, so what it is really asking about is a `mod`.

Exercise 1 runs at **T1** because it spends 24 s inside the lesson's own `dft`
and compares against `numpy`. Exercises 2 and 3 are **T0**; Exercise 3 imports
nothing but `math`, as its own wording demands.

### 1 — the Easy exercise is the expensive one

One second of 220 + 440 + 880 Hz at 16 kHz, through `code/main.py`'s own `dft`:

| | measured | closed form |
|---|---:|---:|
| peak bins | **220, 440, 880** | `k · sr/N = k`, since `sr/N` = 1 Hz |
| peak magnitude | **1333.3333** ×3 | `(0.5/3) · 16000/2` = 1333.3333 |
| worst deviation | 9.6e-12 | — |
| lesson `dft` | **23.9 s** | `16000²` = 256M inner steps |
| `numpy.fft.rfft` | **0.07 ms** | agrees to **3.7e-13** relative |

**MECHANISM: `mix` averages, it does not sum.** `sum(s[i] for s in signals) /
len(signals)` — so three tones at `amp=0.5` arrive at 0.1667 each and the mixed
signal peaks at **0.372**, not 1.5. Read `mix` as a sum and every predicted peak
height is 3× too big.

**FINDING: the Easy exercise is the lesson's most expensive computation.** Step 3
says the O(N²) DFT is "fine for N=256 to confirm correctness, useless for real
audio". Two sections later the exercise asks for N=16000 — **977× the work of
`main()`'s largest call**, and the doc gives no warning that the size changed
anything. Measured: 23.9 s against numpy's 0.07 ms, a factor of 3.5e5.

**FINDING: "the expected bins" are integers only at these parameters.** One second
at 16 kHz gives a bin width of exactly 1 Hz, so bin index *is* frequency in Hz.
That property belongs to the exercise, not to the lesson: `main()` runs 512
samples at 8 kHz, where the same three tones sit at bins **14.08, 28.16, 56.32**
and the printed peaks — 218.8, 437.5, 875.0 Hz — are low by 1.25, 2.5 and 5.0 Hz.

**CONTROL: each peak ties with its mirror above Nyquist** to 4.4e-11. A top-3
taken over the full transform returns the right answer only because `sorted`
breaks the exact tie by index. `main()` halves the spectrum first, which is what
makes it safe.

### 2 — the alias lands between the harmonics, never on them

Neither half of the setup exists: `torchaudio`, `torch`, `librosa`, `soundfile`
and `sounddevice` all return `None` from `find_spec`, and the whole reference tree
ships no `.wav`, `.flac` or `.mp3`. So the 3 s at 48 kHz is synthesised the way a
voice is built — a 120 Hz glottal comb, `1/k` voiced rolloff to 6 kHz, then a flat
sibilant shelf out to 22.9 kHz, 191 harmonics. The naive arm is the lesson's own
`downsample_naive(x, 3)`; the anti-aliased arm is a 193-tap windowed sinc built in
the solution, measuring **0.00 dB at 4 kHz** and **−113 dB at 12 kHz**.

**ANSWER: mirrored about the new Nyquist.** All **125** components above 8 kHz
reappear at `|f − 16000·round(f/16000)|` — every one of them — and the loudest
land at **7240–7960 Hz**. Aliasing arrives at the *top* of the retained band and
works downward, because the partials just above Nyquist fold just below it.

**ANSWER: 40 or 80 Hz off the comb, never on it.** `16000 mod 120 = 40`, so a
partial at `120k` folds to a frequency exactly 40 Hz (from 8–16 kHz) or 80 Hz
(from 16–24 kHz) away from the 120 Hz grid. The measured offsets across all 125
are `{40, 80}` and **never 0**. No aliased partial can reinforce a real harmonic;
it triples the comb density with an inharmonic ghost instead. That is the
difference between sounding bright and sounding metallic, and it is why "naive
decimation just adds some high-end crud" is the wrong intuition.

**MECHANISM: nothing is lost, it is moved.** The source carries **5.45%** of its
energy above 8 kHz; the naive arm's band carries **6.1% more** energy than the
filtered arm's, and the two signals differ by **24.5% relative RMS**. Decimation
is energy-preserving — the low-pass filter is the only thing that discards
anything.

**CONTROL: the lesson ships the broken half only.** `downsample_naive` is
`x[::factor]` and `main()`'s Step 6 prints "always low-pass filter before
decimating" beside a module that contains nothing which does.

### 3 — the Hard one is sixteen times cheaper

Frame 400, hop 160, periodic Hann, over a 1 s 200 → 4000 Hz chirp, using nothing
but `math` and the lesson's `dft`: **98 frames × 201 bins** at 40 Hz per bin, and
the per-frame argmax tracks the sweep to **19.5 Hz** worst case — inside half a
bin. `matplotlib` is absent, so `imshow` becomes a raster of the same matrix
(0–4 kHz bottom to top, 1 s left to right, 60 dB of shading):

```text
                                                                                        .:=*%@@@@@
                                                                                   .:=*%@@@@@@%*=:
                                                                             .:-+#@@@@@@@#+-:.
                                                                        .:-+#@@@@@@@*=:.
                                                                   .:=*%@@@@@@%*=:.
                                                              .:=*%@@@@@@#+=:.
                                                        .:-+#@@@@@@@#+-:.
                                                   .:=+#@@@@@@%*=:.
                                              .:=*%@@@@@@%*=:.
                                         .:=*@@@@@@@#+-:.
                                   .:-+#@@@@@@@#+-:.
                              .:=+%@@@@@@%*=:.
                         .:=*%@@@@@@%*=:.
                    .-=#@@@@@@@#+-:.
              .:-+#@@@@@@@#=-..
         .:=*%@@@@@@%*=:.
    .:=*%@@@@@@%*=:.
-=#@@@@@@@#+-:.
@@@@@#=-.
*=:.
```

**FINDING: the Hard exercise is 16× cheaper than the Easy one.**

| exercise | inner DFT steps | measured |
|---|---:|---:|
| 1 — "Easy", one transform of 16000 | 256M | 23.9 s |
| 3 — "Hard", 98 transforms of 400 | **15.7M** | **1.5 s** |

Framing is what makes an O(N²) DFT survivable, and that — not the window or the
plot — is the actual content of the exercise.

**FINDING: 400 / 160 is not a COLA pair.** Overlap-adding Hann(400) at hop 160
sums to 1.250 with **9.44% ripple**, so this spectrogram cannot be inverted by
plain overlap-add. Hop 200 (`N/2`) and hop 100 (`N/4`) both sum flat, to 7e-16.
The prescribed hop is the one in its own neighbourhood that fails, and 400/160 at
16 kHz is exactly the doc's own "25 ms with 10 ms hop".

**FINDING: it is not the spectrogram of Lesson 02.** That lesson's `main` ships
`frame_len=256, hop=128` at `sr=8000` — a power-of-two frame at an `N/2` hop — and
its `hann` divides by `N - 1` where the periodic window divides by `N`. Every
parameter in the sentence differs from the lesson it points at.

**CONTROL: 400 is not a power of two.** The doc defines the FFT as "an O(N log N)
algorithm requiring `N` = power of 2" and then sets the frame to 400 = 2⁴·5².
Its own fast path is excluded by its own parameter; real libraries factor 400
without complaint, and this lesson has no FFT to factor it with.
