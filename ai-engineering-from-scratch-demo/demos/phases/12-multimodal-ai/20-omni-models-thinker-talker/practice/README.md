<!-- generated:start -->
# 12-multimodal-ai / 20-omni-models-thinker-talker

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/12-multimodal-ai/20-omni-models-thinker-talker/) · upstream spec
`phases/12-multimodal-ai/20-omni-models-thinker-talker/docs/en.md`

```bash
uv run demo practice run 20-omni-models-thinker-talker --ex 1
uv run demo explain 20-omni-models-thinker-talker --ex 1
uv run pytest demos/phases/12-multimodal-ai/20-omni-models-thinker-talker
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Your target TTFAB is 300ms. On a 7B Thinker and 300M Talker, write out every component's late… | code | T0 | `ex01_the_floor_is_fifty_five_percent_of_the_target.py` |
| 2 | Qwen2.5-Omni uses TMRoPE. Describe what the model sees for a prompt where the user starts spe… | explain | T0 | prose, below |
| 3 | Full-duplex support requires the model to emit audio while listening. Propose a training data… | code | T0 | `ex03_the_format_is_two_streams_and_a_quarter_of_it_is_silence.py` |
| 4 | Read Moshi's paper Section 4. Describe the "inner monologue" separation and why it avoids the… | explain | T0 | prose, below |
| 5 | Compute the throughput budget: how fast must a Talker emit tokens to keep up with 16kHz speec… | code | T0 | `ex05_the_rate_is_fifty_for_the_transformer_and_four_hundred_for_the_head.py` |
<!-- generated:end -->

## Answers

Three code solutions and two prose ones. Exercise 1 and exercise 5 both find the
lesson's code and its prose disagreeing about the same quantity — the latency
budget in one case, the token rate in the other.

### 1 — the floor is 55% of the target

| component | ms |
|---|---:|
| mic → speech tokens | 50 |
| Thinker prefill | 100 |
| Thinker first text token | 40 |
| Talker first speech tokens | 20 |
| residual-VQ decode | 30 |
| waveform decoder | 70 |
| **TTFAB** | **310** |

**ANSWER: 310 ms — ten over the 300 ms target**, and **390** with vision on.

**FINDING: 165 ms takes neither size parameter.** Mic tokenisation, the Talker's
`max(15, …)` floor, the residual decode and the waveform decoder are constants —
**53.2%** of the total and **55%** of the target. A free Thinker and a free Talker
still cost 165 ms.

**FINDING: the target admits a 6B Thinker and no more.** Both model-scaled terms
carry `thinker_b / 7`, so the total is `165 + 20 × thinker_b`:

| Thinker | 1B | 5B | **6B** | 7B | 8B |
|---|---:|---:|---:|---:|---:|
| TTFAB | 190 | 270 | **290** | 310 | 330 |

The Talker is nearly irrelevant by comparison — 100M to 1,000M spans **52 ms**.

**FINDING: the code disagrees with its own prose at both ends.** The docs state
"320–510 ms at 7B" and "600–900 ms at 70B"; `ttfab` returns **310** at 7B (below
its own floor) and **1,690** at 72B (**1.9×** its own ceiling).

### 2 — what TMRoPE sees at t = 1 s and t = 1.2 s

Drawing on **TMRoPE — time-aligned multimodal positions**, which states the
mechanism and gives the "he waved while saying hello" example this exercise makes
concrete.

**The model sees two tokens 0.2 s apart on one clock, not two tokens in two
streams.** Every token carries an absolute timestamp, and the rotation is a
function of that timestamp, so:

```text
t = 1.00 s   audio token   (user's first speech frame)
t = 1.02 s   audio token
t = 1.04 s   audio token           <- audio arrives at 50 Hz
...
t = 1.20 s   vision token x N      <- the gesture frame, at 4 FPS
t = 1.22 s   audio token
```

**Three consequences, and the third is the one the architecture exists for:**

1. **The two modalities interleave by time, not by block.** A naive layout —
   all vision, then all audio — would put the gesture either before all the
   speech or after it, and 0.2 s of separation would be indistinguishable from
   20 s. Under TMRoPE the gesture sits *inside* the speech, ten audio frames in.
2. **Their sampling rates do not have to match.** Audio at 50 Hz and vision at
   4 FPS means the sequence has roughly 12 audio tokens between consecutive
   vision tokens, unevenly spaced whenever the frame sampler is dynamic. An index
   table cannot express that — the same argument Lesson 12.17's exercise 2 makes
   for video alone.
3. **"While" becomes representable.** The gesture at 1.2 s and the word being
   spoken at 1.2 s have *the same rotation*, so attention between them is not
   attenuated by distance. That is what makes "he waved while saying hello" a
   single event rather than two adjacent ones — and there is no other mechanism
   in the architecture that could express simultaneity, because the sequence is
   one-dimensional and time is not.

**What it does not give you.** A shared clock is necessary and not sufficient: the
model still has to be *trained* on aligned data for the co-occurrence to mean
anything, and Lesson 12.20's exercise 3 prices exactly how expensive that data is
to record. TMRoPE makes simultaneity representable; the corpus is what makes it
learnable.

### 3 — the format is two streams, and a quarter of it is silence

**ANSWER: two aligned token streams at one frame rate, with silence made
explicit.**

```text
frame  user_tokens        assistant_tokens   user_text  assistant_text
  0    [c0..c7]           [SILENCE x8]       "so"       -
  1    [c0..c7]           [SILENCE x8]       "I"        -
  2    [c0..c7]           [c0..c7]           "think"    "mm-hmm"    <- the signal
```

Both parties are present at every instant, so speaking-while-listening is
representable rather than special-cased, and a turn boundary is a *run of
silence ending* rather than a delimiter.

**FINDING: it costs 62× a transcript.** 7,500 frames × 2 streams × 8 codebooks =
**120,000** tokens for ten minutes — 200 a second — against **1,950** text
tokens. The ratio is fixed by the frame rate, not by the conversation.

**FINDING: a quarter of it is (silence, silence).**

| frames | share |
|---|---:|
| both silent | **25%** |
| exactly one speaking | 50% |
| **both speaking** | **25%** |

A quarter teaches nothing, half teaches turn-taking, and the full-duplex signal
is the smallest bucket.

**FINDING: and that bucket is 0% of any turn-based corpus.** Half-duplex
recordings are segmented by turn, so overlap is discarded or was never captured
on separate channels. "Propose a format" is really "propose a recording setup".

### 4 — why the inner monologue avoids the split

Drawing on **Thinker and Talker**, which states the split's rationale — "Thinker
has to be big for good reasoning; Talker can be small because its job is local".

**The split exists to solve a throughput problem, and Moshi solves the same
problem a different way.** Lesson 20's exercise 5 puts the number on it: the
speech stream needs **50** base tokens a second and the cited LLM throughput is
30–80, so a 7B model generating speech tokens directly is a coin flip on whether
it streams. Qwen's answer is to put the fast job in a small model. Moshi's answer
is to make the fast job *cheaper* rather than to move it.

**How the monologue does that.** Moshi predicts, in one transformer, several
parallel streams per frame: its own audio codes, the user's audio codes, and a
**text** stream that runs a fixed delay ahead of its own audio. The text is not
an output — nobody reads it — it is a plan. Per frame the model emits one text
token and a set of audio codes, so the transformer's own rate is the **frame**
rate (12.5 Hz), not the token rate, and a small depth head expands each frame
into its codebooks. One model, one pass per frame.

**So the three things the split buys, and how the monologue gets them
otherwise:**

| what Thinker–Talker buys | how the monologue gets it |
|---|---|
| reasoning capacity in the big model | the *same* model reasons — via the text stream |
| a fast path for the speech tokens | the frame rate *is* the pass rate; the depth head does the rest |
| decoupled sizing | nothing to size: one model |

**What it costs instead.** The monologue's text stream is a plan, so its planning
capacity is whatever the single model has — you cannot swap in a 70B reasoner
without also paying 70B per *frame*, at 12.5 Hz. The split can. So the two designs
trade the same quantity in opposite directions: Thinker–Talker buys reasoning
headroom and pays an interface; the monologue buys a single coherent model and
pays a ceiling on how big it can be.

**And the interface is not free.** The Talker sees text tokens, so everything the
Thinker knew and did not say — hesitation, emphasis, the fact that it is
uncertain — is gone at the boundary. The monologue has no boundary, which is why
its prosody is generally reported as the better half of the comparison, and why
Lesson 12.16's exercise 5 makes the same point about a redundant encoding being
checkable.

### 5 — the rate is 50 for the transformer and 400 for the head

**ANSWER: 50 a second for the base layer, 400 for all eight.** One token every
**2.5 ms** against the base layer's every 20 ms. The lesson states the 50 and not
the 400.

**FINDING: the 400 is not a transformer rate.** Lesson 12.16 measures the
residual levels at **4.3 ms** each against a **40 ms** first-token cost —
**9.3×** cheaper, so a small depth head. The transformer runs at 50 and the head
supplies the other 350.

**FINDING: the lesson's own throughput range fails its own requirement.**

| cited throughput | real-time factor |
|---|---:|
| 30 tok/s | **1.67** — falls 0.67 s behind per second |
| 50 tok/s | 1.00 |
| 80 tok/s | **0.62** |

It cites "30–80 tok/s" and calls a small Talker "fast enough"; only **67%** of
the cited points keep up.

**FINDING: and the margin at the top is thin.** The whole range spans RTF
**0.62 to 1.67** and straddles 1. A Talker sized by that range is a coin flip on
whether it streams — which is the real argument for a dedicated small model,
rather than "small is fast enough".
