<!-- generated:start -->
# 01-math-foundations / 20-fourier-transform

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/01-math-foundations/20-fourier-transform/) · upstream spec
`phases/01-math-foundations/20-fourier-transform/docs/en.md`

```bash
uv run demo practice run 20-fourier-transform --ex 1
uv run demo explain 20-fourier-transform --ex 1
uv run pytest demos/phases/01-math-foundations/20-fourier-transform
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Pure tone identification. Create a signal with a single sine wave at an unknown frequency (be… | code | T0 | `ex01_pure_tone_identification.py` |
| 2 | FFT vs DFT verification. Generate a random signal of length 64. Compute both DFT (O(N^2)) and… | code | T1 | `ex02_fft_vs_dft.py` |
| 3 | Convolution theorem proof by example. Create signal x = [1, 2, 3, 4, 0, 0, 0, 0] and filter h… | code | T0 | `ex03_convolution_theorem.py` |
| 4 | Windowing effects. Create a signal that is the sum of two sine waves at 10 Hz and 12 Hz (very… | code | T0 | `ex04_windowing_effects.py` |
| 5 | Positional encoding analysis. Generate the sinusoidal positional encodings for d_model = 128… | code | T0 | `ex05_positional_encoding.py` |
<!-- generated:end -->

## Answers

**1 — the tone is identified exactly, and the noise answer is quantitative.**
128 samples over 1 second gives 1 Hz bins, so an integer frequency lands in one
bin with nothing to interpolate. σ=0.5 noise leaves the peak **unmoved**; what it
does is raise the floor from 6e-14 to 13.8, cutting the peak-to-floor ratio to
4.2×.

Identification then breaks at **σ = 2**, sooner than "the DFT averages noise away"
suggests. The arithmetic predicts it: a unit sine gives a peak of N/2 = 64 while
white noise gives ≈ σ√(N/2) = 8σ per bin, and the largest of 64 such bins is a few
times that — so the crossover lands near σ ≈ 3. The √N ≈ 11.3 coherent gain is
real but modest.

**2 — the ratio grows like N/log₂N, measured 5.8× against a predicted 5.8×.**

| N | DFT | FFT | ratio |
|---:|---:|---:|---:|
| 256 | 33 ms | 0.7 ms | 46× |
| 512 | 132 ms | 1.6 ms | 81× |
| 1024 | 541 ms | 3.6 ms | 152× |
| 2048 | 2136 ms | 7.9 ms | 270× |

The exponents are fitted **separately** — DFT at N^2.01, FFT at N^1.16 — because a
ratio can grow correctly while both terms are wrong. Coefficients match to
1.3e-13, about 800× inside the exercise's 1e-10 bar.

**3 — the exercise's zero-padding step changes nothing as given.** x has 4
non-zero taps and h has 3, so linear convolution needs 4+3−1 = 6 and the signals
are already 8 long: circular and linear results coincide exactly. To see what the
padding is *for*, the solution runs the same convolution at N=4, where taps 5 and
6 have nowhere to go and fold onto positions 1 and 2 — `[8, 7, 6, 9]` against the
correct `[1, 3, 6, 9]`.

**4 — the expected answer is a window, and the measurement says the opposite.**
Scored by valley depth between the peaks:

| treatment | valley depth |
|---|---:|
| **none** | numerically infinite |
| Hamming | 1.36× |
| Hann | 0.98× |

Hann's trough is *higher* than the lower peak — it merges the two tones. A window
trades main-lobe width for sidelobe suppression, and with exactly integer
frequencies there are no sidelobes to suppress: each tone already occupies one
bin, so the trade is all cost. Hamming beats Hann because its main lobe is
narrower.

At 10 vs 11 Hz — one bin apart — nothing resolves them, windows included. Only a
longer observation narrows a bin.

**5 — the distance-only claim is exactly true; the decay claim needs care.**
Across 530 pairs at 9 distances the spread within a distance is 2e-13. That is
algebraic, not approximate: sin(ωp₁)sin(ωp₂) + cos(ωp₁)cos(ωp₂) = cos(ω(p₁−p₂)),
term by term, so absolute position cancels.

What happens as distance grows depends on how you sample. At the sparse distances
(0, 1, 2, 5, 10, 50, 100, 250, 500) it looks monotonically decreasing. Sampled
every 8 positions, **28 of 63 steps rise** — it falls sharply then oscillates. And
it never approaches zero: the floor is 11.5 against a self dot product of 64,
about 18%, because the lowest frequency has a period near 57,000 positions and
barely moves across 512. Distant positions are not close to orthogonal.
