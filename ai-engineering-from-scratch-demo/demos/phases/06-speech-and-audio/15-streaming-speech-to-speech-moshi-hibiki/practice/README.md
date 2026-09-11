<!-- generated:start -->
# 06-speech-and-audio / 15-streaming-speech-to-speech-moshi-hibiki

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/06-speech-and-audio/15-streaming-speech-to-speech-moshi-hibiki/) · upstream spec
`phases/06-speech-and-audio/15-streaming-speech-to-speech-moshi-hibiki/docs/en.md`

```bash
uv run demo practice run 15-streaming-speech-to-speech-moshi-hibiki --ex 1
uv run demo explain 15-streaming-speech-to-speech-moshi-hibiki --ex 1
uv run pytest demos/phases/06-speech-and-audio/15-streaming-speech-to-speech-moshi-hibiki
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run `code/main.py`. It simulates the two-stream + inner-monologue architecture symbolic… | code | T0 | `ex01_two_streams_that_never_touch.py` |
| 2 | Medium. Pull Moshi from HuggingFace, run the server, test one conversation. Measure wall-cloc… | code | T1 | `ex02_the_stopwatch_has_no_start.py` |
| 3 | Hard. Take your Lesson 12 pipeline agent and compare P50 latency vs Moshi on 20 matched test… | code | T1 | `ex03_twenty_utterances_and_two_different_quantities.py` |
<!-- generated:end -->

## Answers

The doc describes a three-stream architecture whose whole point is that the
streams are coupled — Moshi hears the user while it speaks, and its text stream
conditions its audio. `code/main.py` builds all three streams and **couples none
of them**: the frame index is the only input either generator reads. That fact
settles the first exercise and shapes the other two, because the quantity
Exercise 2 asks to time does not exist in a full-duplex model, and the number
Exercise 3 asks to compare is not the number Lesson 12 prints.

Exercise 1 runs at **T0**. Exercises 2 and 3 are **T1** — both measure wall-clock
time, and every check they make asserts a ratio rather than an absolute
millisecond figure.

### 1 — two streams that never touch

**ANSWER: the loop runs.** 25 frames, `25 × 8` user codebooks, `25 × 8` Moshi
codebooks, 25 inner-monologue tokens, over 2000 ms of simulated audio at 12.5 Hz.

**FINDING: all 25 user frames encode to one token vector.** `fake_mimi_encode`
seeds its RNG on `int(mean|x| · 1000)`, and the mean absolute value of a
0.15-amplitude sine is `0.15 · 2/π = 0.09549` whatever its frequency:

| | |
|---|---|
| distinct encoder seeds across 25 frames | **1** (`95`) |
| distinct user token vectors | **1** |

The 220 Hz sweep rising 20 Hz a frame that `simulate_user_speech` goes to the
trouble of building is erased by the encoder.

**FINDING: silence and noise produce the same Moshi streams as speech.**

| user audio | `user_mimi` | `moshi_mimi` | `moshi_text` |
|---|---|---|---|
| the swept sine | — | baseline | baseline |
| digital silence | **differs** | **identical** | **identical** |
| Gaussian noise | **differs** | **identical** | **identical** |

`depth_transformer` seeds on `len(user) + len(moshi)`; `inner_monologue_next_token`
returns `f"tok_{len(text_so_far)}"`. Both read lengths, neither reads a value.

**FINDING: the inner monologue is a parameter that is never read.**
`context_text` is the first argument of `depth_transformer` and does not appear in
its body. The doc's central claim for the design — *"force it to emit text tokens
alongside audio … this improves semantic coherence"* — is the one arrow the
simulation drops.

**CONTROL: the depth transformer predicts the codebooks independently.** The doc
says the 8 codebooks "have inter-codebook dependencies" and are predicted
"sequentially"; the body draws them in one list comprehension from a single RNG.
The same `main()` prints its target as `target: &lt; 80 ms` — an HTML escape that
survived into a Python string literal.

### 2 — the stopwatch has no start

`moshi`, `torch`, `transformers`, `sounddevice` and `websockets` are all absent,
so the server cannot run. What can be settled without it is that the quantity the
exercise names is one this architecture does not have.

**ANSWER: there is no end of user speech.** A word-boundary search across every
function body finds no `vad`, no `silence`, no `turn`, no `endpoint`, no `energy`
and no `threshold`. The loop emits a Moshi frame on *every* iteration,
unconditionally, so the response starts at frame 0 while the user still has
2000 ms to speak:

```text
end-of-user-speech  −  start-of-Moshi-response  =  −1920 ms
```

Full duplex removes the turn boundary, and the stopwatch the exercise describes
needs one.

**FINDING: what is printed is compute, not latency.**

| | |
|---|---|
| two `time.sleep` constants in the generators | **5 ms**, about ⅔ of a frame |
| measured per-frame cost | single-digit milliseconds |
| the doc's own architectural floor | **160 ms** (80 ms frame + 80 ms acoustic) |
| ratio | the demo runs **more than 20× below** the floor it quotes |

The floor is a property of the frame grid, not of the hardware, so no amount of
compute speed reaches it. Printing a single-digit millisecond figure beside a
`target: < 80 ms per frame` line invites the opposite conclusion.

**FINDING: the per-frame cost does not grow with context.** Over 200 frames the
median of the last 25 is within a few percent of the median of the first 25 —
where attention over a KV cache growing one frame at a time would put it near
**15×**. `depth_transformer` seeds an RNG on two `len()` calls and does constant
work, so the one thing that makes streaming inference hard is absent.

**CONTROL: widening the depth transformer costs nothing either.** Taking
`CODEBOOKS` from 8 to 128 is sixteen times the tokens to predict and moves the
frame cost by a couple of percent, because `time.sleep(0.003)` dominates both.

### 3 — twenty utterances, and two different quantities

Both modules exist and both print milliseconds, so the comparison can be run —
and running it is how you find that the two numbers are not the same quantity.

| | P50 over 20 matched utterances |
|---|---:|
| Lesson 12, user-perceived delay to first audio | **~804 ms** |
| Lesson 15, cost of one 80 ms frame | **~7 ms** |
| ratio | **of order 100×** |

One number contains a 400 ms end-of-speech wait and two model calls; the other
contains a `sleep(0.003)`.

**MECHANISM: the pipeline's latency is constants.** `streaming_stt` returns the
literal `"set a timer for five minutes"` for a 10-sample buffer and a
40,000-sample buffer alike; `llm_with_tools` sleeps 0.12 s and `streaming_tts`
0.10 s regardless of their arguments; the end-of-speech wait is the literal
`400`. The only input-dependent term in the whole pipeline is `streaming_stt`'s
`len(utterance)/sr * 0.05`, which is why 20 utterances of 1–2 s spread about
**1.5%** relative standard deviation. Twenty matched utterances are twenty
measurements of one constant.

**FINDING: at n=20 a P50 is a band, not a point.** A median has no standard error
to quote, but it has an exact distribution-free interval: for `n = 20` the
tightest symmetric 95% confidence interval for the median runs from the **6th to
the 15th** order statistic, coverage **0.9586**.

**FINDING: twenty utterances cannot resolve a ten percent difference.** Simulated
on log-normal latencies:

| true ratio between the two systems | detected at n=20 vs 20, σ=0.20 | σ=0.35 |
|---:|---:|---:|
| 1.10× | **30%** | **13%** |
| 1.25× | 91% | **51%** |
| 1.50× | 100% | 92% |

The doc's own cheatsheet puts Moshi at 200–300 ms and the managed pipelines at
300–500 ms — overlapping ranges whose true ratio could be anywhere from 1.0× to
2.5×. At the low end of that, 20 utterances is a coin flip.

As for *when a pipeline architecturally wins anyway*: the doc answers it in Step
4 and the Pitfalls, and nothing measured here contradicts it. Tool calling is the
case, and it is structural rather than a matter of latency — Lesson 12's turn
spends 120 ms in `llm_with_tools` and 10 ms dispatching `set_timer`, a branch
Moshi has no equivalent of, because a full-duplex acoustic model has no point in
the loop where a tool result could be injected between hearing and speaking.
