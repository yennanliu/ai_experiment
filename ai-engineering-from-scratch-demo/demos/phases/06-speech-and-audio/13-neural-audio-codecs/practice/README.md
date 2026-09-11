<!-- generated:start -->
# 06-speech-and-audio / 13-neural-audio-codecs

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/06-speech-and-audio/13-neural-audio-codecs/) · upstream spec
`phases/06-speech-and-audio/13-neural-audio-codecs/docs/en.md`

```bash
uv run demo practice run 13-neural-audio-codecs --ex 1
uv run demo explain 13-neural-audio-codecs --ex 1
uv run pytest demos/phases/06-speech-and-audio/13-neural-audio-codecs
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run `code/main.py`. It implements a toy scalar + residual quantizer and measures recons… | code | T0 | `ex01_the_table_goes_blind_at_the_fifth_codebook.py` |
| 2 | Medium. Install `encodec` and compare 1, 4, 8, 32 codebooks on a held-out speech clip. Plot P… | code | T0 | `ex02_three_of_the_four_points_plot_at_zero.py` |
| 3 | Hard. Load Mimi. Encode a clip. Replace codebook 0 with random integers; decode. Then replace… | code | T0 | `ex03_a_codec_with_no_semantic_codebook_does_it_too.py` |
<!-- generated:end -->

## Answers

`code/main.py` implements a real residual vector quantizer, and it works — the
error falls geometrically, 13.22 dB per codebook, all the way down to double
precision. What the three exercises turn up is that the table reporting it cannot
hold the result: `%.6f` erases the bottom two rows, and both the Medium and Hard
exercises then ask for measurements whose answers are already sitting in those
erased rows.

All three run at **T0** on stdlib alone; the lesson imports `math` and `random`.

### 1 — the table goes blind at the fifth codebook

| # codebooks | printed MSE | measured MSE |
|---:|---:|---:|
| 1 | 0.013647 | 1.364684e-02 |
| 2 | 0.000521 | 5.205277e-04 |
| 4 | 0.000001 | 1.073862e-06 |
| **8** | **0.000000** | **9.157033e-12** |
| **12** | **0.000000** | **3.886251e-17** |

**ANSWER: the two zero rows differ by a factor of 235,626.** `%.6f` runs out at
the fifth codebook, and the rows it erases are exactly the ones the doc's
Pitfalls section is about when it says "stop at 8-12".

**MECHANISM: the error is geometric.** Each added codebook is worth **13.22 dB**
on average — between 10.03 and 14.23 dB across the twelve, sd 1.12 — a mean ratio
of **21.0×**. Linear MSE cannot hold that range in six decimal places.

**FINDING: the doc calls this linear, and only the dB reading is.** "Adding
codebooks increases fidelity linearly but LM sequence length linearly too" is
true of decibels, where the gain really is additive. The column the lesson prints
is linear MSE, where the same relationship falls off the bottom of the format
after four rows.

**FINDING: `bits_per_frame = n_cb * 3` bills for entropy that is not there.**

| | charged | measured |
|---|---:|---:|
| bits per sample, 12 codebooks | 36 | **29.136** |
| bits per codebook | 3 | **2.43** |

23.6% high — and one codebook runs a dead centroid, cb8 using 7 of its 8
codewords.

**CONTROL: the codebooks appear in no column.** Eight `float64` centroids per
codebook is 6,144 bits of side information against 36,000 bits of indices at
twelve codebooks — **17.1%**, the same share at every row of the table.

### 2 — three of the four points plot at zero

`encodec`, `torch`, `torchaudio`, `pesq` and `soundfile` are all absent, and a
recursive scan of the reference checkout finds **zero** `.wav`, `.flac`, `.mp3`
or `.ogg` files — so there is no held-out speech clip either. Both axes are still
computable.

**MECHANISM: the bitrate axis comes out of the doc's own constants.** Step 1 says
8 codebooks at 6 kbps with 10-bit codes; Step 3 gives EnCodec-24k a 75 Hz frame
rate. `8 × 75 × 10 = 6000` bps exactly, so one codebook is **0.75 kbps**.

| codebooks | kbps | measured MSE | as `code/main.py` prints it | dB |
|---:|---:|---:|---:|---:|
| 1 | 0.75 | 1.364684e-02 | 0.013647 | −18.6 |
| 4 | 3.00 | 1.073862e-06 | 0.000001 | −59.7 |
| 8 | 6.00 | 9.157033e-12 | **0.000000** | −110.4 |
| 32 | 24.00 | 4.119887e-32 | **0.000000** | −313.9 |

**ANSWER: three of the four points are at or below the printed floor**, and the
two zeros stand **2.2e+20** apart. On a linear MSE axis the plot is one visible
point and three on the bottom. It only becomes a plot in dB, where the four
points fall **9.5 dB per codebook** on a line — the axis the exercise asks for is
the one that hides the result.

**FINDING: past codebook 26 the quantizer is encoding float64 rounding error.**
The signal's RMS is 0.7759, so double precision resolves it to **1.72e-16**; from
codebook 26 onward every centroid is smaller than that. The `4.12e-32` at 32
codebooks is a property of the arithmetic, not of the codec — the exercise's
widest data point is the one that means least.

### 3 — a codec with no semantic codebook does it too

`moshi`, `torch`, `torchaudio` and `transformers` are absent and there is no
clip, so this is the `DESIGN D11` scaled-down run: the same corruption, level by
level, on the plain residual quantizer `code/main.py` already implements. Clean
MSE at 8 codebooks is 9.157e-12.

| corrupted codebook | MSE | vs clean | mean-square contribution |
|---:|---:|---:|---:|
| **0** | **1.301458** | 1.42e+11 | 5.859e-01 |
| 1 | 7.445039e-02 | 8.13e+09 | 1.314e-02 |
| 3 | 1.856689e-04 | 2.03e+07 | 2.446e-05 |
| 5 | 5.289699e-07 | 5.78e+04 | 4.140e-08 |
| **7** | **1.262134e-09** | 1.38e+02 | 1.977e-10 |

**ANSWER: the predicted asymmetry reproduces, at a factor of 1.03e+09**, monotone
at every step between.

**FINDING: corrupting codebook 0 is worse than emitting silence.** MSE **1.3015**
against the signal's own variance of **0.6020** — 2.16× — so the decode sits more
than twice as far from the original as a zero signal does. That is the "destroys
intelligibility" end of the comparison, quantified.

**MECHANISM: what orders the codebooks is residual energy.** Mean-square
contribution falls `5.859e-01 → 1.977e-10`, a ratio of **2.96e+09**, monotone
across all eight levels beside the corruption column's 1.03e+09. Corrupting a
level costs roughly twice what that level contributes.

**FINDING: this quantizer has no semantic codebook, so the test proves nothing.**
`rvq_encode` builds all eight with **one** call to
`learn_codebook(residuals, codebook_size, seed=cb_i)`, differing only in the
residual handed to it and an integer seed. None of `wavlm`, `semantic` or
`distill` appears anywhere in `rvq_encode`, `rvq_decode` or `learn_codebook`; all
three appear only inside `main`'s Step 4 print statements.

So the experiment as designed cannot distinguish the hypothesis it is testing.
Any residual quantizer — semantic split or not — gives codebook 0 the largest
share of the signal and the last codebook a share near the noise floor, and the
asymmetry follows. Testing the semantic claim needs the control the exercise does
not ask for: a codec whose codebooks carry comparable energy, which residual
quantization by construction cannot produce.
