<!-- generated:start -->
# 08-generative-ai / 11-audio-generation

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/08-generative-ai/11-audio-generation/) · upstream spec
`phases/08-generative-ai/11-audio-generation/docs/en.md`

```bash
uv run demo practice run 11-audio-generation --ex 1
uv run demo explain 11-audio-generation --ex 1
uv run pytest demos/phases/08-generative-ai/11-audio-generation
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run `code/main.py` and set style explicitly. Verify the generated sequences match the s… | code | T0 | `ex01_the_alternating_style_is_a_ramp.py` |
| 2 | Medium. Add delayed parallel decoding: simulate 2 streams of tokens that must stay offset by… | code | T0 | `ex02_the_delay_is_what_carries_the_coupling.py` |
| 3 | Hard. Use HuggingFace transformers to run MusicGen-small locally. Generate a 10-second clip w… | code | T0 | `ex03_ab_is_won_in_five_tokens_and_every_clip_glitches.py` |
<!-- generated:end -->

## Answers

The lesson is 99 lines: a per-style bigram count table over 16 "codec tokens",
add-one smoothed, trained on two synthetic styles and sampled with temperature.
All three exercises are **T0**; exercise 3 names MusicGen-small, which cannot run
here, so it ships the scaled-down runnable `DESIGN D11` requires.

Two facts about the code decide most of what follows: `init_counts` starts every
cell at **1.0**, so every row leaks mass to tokens training never reached; and
both "styles" are forward ramps, not the alternating pattern the labels describe.

### 1 — the sequences match their style, and the "alternating" style is a ramp

A step outside the style's training step set (`{0,1,2}` for style 0, `{2,3,4}`
for style 1, mod 16) counts as a glitch.

| | style 0 | style 1 |
|---|---:|---:|
| off-pattern steps, T=1.0 | 2.3% | 2.2% |
| clean 20-token sequences, T=1.0 | 64% | 65% |
| off-pattern steps, T=0.7 | 0.28% | 0.24% |

**ANSWER: yes, up to the smoothing's glitch rate.** The add-one prior leaves
**2.6%** of each row on unseen tokens; the training data itself is **0.0%**
off-pattern (CONTROL).

**FINDING: style 0 is not alternating.** `main()` calls it "speech-like
(alternating)", but **100%** of its 9500 training steps go forward by 0, 1 or 2 —
a +1 ramp with a stutter. The +2 step is shared with style 1, so no single step
identifies the style.

**FINDING: `style=2` raises IndexError, `style=-1` silently returns style 1** —
`counts[-1]` is the last table.

### 2 — the one-step delay is what carries the coupling

Two streams: `a` from `make_tokens`, and `b = a + 8 mod 16` as a dependent second
codebook. Two 16-way heads built from the lesson's own count functions.

| decoding | coupled frames |
|---|---:|
| delayed, `b_t` conditioned on `a_t` | **97.3%** |
| no delay, `b_t` conditioned on `a_{t-1}` | 36.5% |
| no delay, one 256-way joint head | 27.1% |

**ANSWER: delayed decoding keeps the streams coupled 97.3% of the time.**

**FINDING: without the delay, coupling is the closed-form 0.375** — two
independent draws from `{1/4, 1/2, 1/4}` agree that often. One extra frame buys
**2.7x**.

**FINDING: the joint 256-way head is worse, not better.** The lesson's add-one
prior scales with the output vocabulary: 256 ones against ~600 real counts leave
**71%** of a seen row on valid pairs, and only **16** of 256 contexts are ever
seen, so one leak lands in a uniform row and errors compound.

### 3 — the A/B is won in five tokens, and every 10-second clip still glitches

`transformers`, `torch` and `torchaudio` are absent. A 10-second clip is 500
tokens at MusicGen's 50 Hz; the judge prefers the clip with the higher
log-likelihood under the prompt's own model.

**ANSWER: both trained prompts win 100% of their 10-second A/Bs.**

**FINDING: 10 seconds is about 100x more than the A/B needs.** Win rates are
**90% / 94%** at 2 tokens and **100%** at 5 — a tenth of a second.

**FINDING: yet no 10-second clip is clean.** At T=1.0 a clip carries **11.0**
off-pattern tokens on average and **0 of 200** are clean; at T=0.7, 1.2 per clip
and 63 of 200 clean. A perfect A/B and a glitch in every clip are the same length
effect read two ways.

**FINDING: the third prompt cannot be A/B'd.** `style=2` raises IndexError; an
untrained slot is uniform, so **100%** of its pairs tie. Adherence to a prompt
the model never learned is undefined, not low.
