<!-- generated:start -->
# 12-multimodal-ai / 18-long-video-million-token

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/12-multimodal-ai/18-long-video-million-token/) · upstream spec
`phases/12-multimodal-ai/18-long-video-million-token/docs/en.md`

```bash
uv run demo practice run 18-long-video-million-token --ex 1
uv run demo explain 18-long-video-million-token --ex 1
uv run pytest demos/phases/12-multimodal-ai/18-long-video-million-token
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | A 45-minute lecture at 1 FPS, 81 tokens per frame. Total tokens? Fits in which models' contexts? | code | T0 | `ex01_the_agentic_row_is_cheaper_than_the_one_million_row.py` |
| 2 | Design a needle-in-a-haystack test: at what minute do you inject the marker, and what is the… | code | T0 | `ex02_the_simulator_puts_the_hardest_position_at_the_end.py` |
| 3 | Compare brute-context Qwen2.5-VL-72B (80k context) to VideoAgent (Claude 3.5 + retrieval) on… | code | T0 | `ex03_the_brute_arm_does_not_fit_so_there_is_no_recall_to_compare.py` |
| 4 | Ring attention's memory cost scales linearly in sequence length and linearly in device count.… | explain | T0 | prose, below |
| 5 | Read Gemini 1.5 Section 5 on needle-in-a-haystack. What did the paper find about recall at th… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

Three code solutions and two prose ones. The code ones all check the lesson's own
budget table against arithmetic it already contains, and each finds the table's
recommendation column disagreeing with its token column.

### 1 — the agentic row is cheaper than the one-million row

**ANSWER: 218,700 tokens.** Fits a **256k** context at **83.4%**; **1.67×** over
128k and **1.09×** over a 200k one — a 45-minute lecture misses Claude-class
context by 9%.

| row | tokens | smallest context that fits | the lesson's label |
|---|---:|---|---|
| 1 min @ 1 FPS | 4,860 | 8k | 32k+ |
| 5 min @ 1 FPS | 24,300 | 32k | 32k |
| 5 min @ 2 FPS | 48,600 | **64k** | 128k |
| 30 min @ 1 FPS | 145,800 | 256k | 256k |
| 60 min @ 1 FPS | **291,600** | **512k** | **1M / LongVILA** |
| 120 min @ 1 FPS | 583,200 | 1024k | Gemini 2.5 only |
| 120 min @ 32/frame | **230,400** | 256k | **agentic retrieval** |

**FINDING: three of the five sized labels are loose**, the 1-hour row by 2×.

**FINDING: the row labelled "agentic retrieval" is the cheaper of the two.**
230,400 against 291,600 — the table routes the *smaller* one to an agent and the
larger to a million-token model, so its recommendation column is not ordered by
its own token column.

**FINDING: the 45-minute case takes the lower of its two neighbours' labels**,
because the 60-minute label was already 2× loose.

### 2 — the simulator puts the hardest position at the end

**ANSWER: sweep eleven depths, N trials at each, with the middle as the
separator**, and a query that admits exactly one string:

```text
At what time does the red umbrella appear?
Answer with only the timestamp in seconds, e.g. 123.4
```

The marker has to be lexically unique *and* semantically out of place, so that a
model which never found it cannot recover it from the prior.

**FINDING: the lesson's recall model is monotone, so it has no middle.**

| depth | 0.0 | 0.2 | 0.4 | **0.5** | 0.6 | 0.8 | **1.0** |
|---|---:|---:|---:|---:|---:|---:|---:|
| lesson's curve | 0.95 | 0.85 | 0.85 | 0.85 | **0.75** | 0.75 | **0.75** |
| lost-in-the-middle | 0.98 | 0.88 | 0.78 | **0.73** | 0.78 | 0.88 | **0.98** |

Its minimum is first reached at **0.6** and never recovers; the alternative
bottoms at **0.5** and climbs back. The two disagree most at depth **1.0**, by
**0.23** — which is exactly where the lesson is most confident and the literature
is least.

**FINDING: `nih_trial` returns a probability and never samples it.** No draw
against `recall_prob` anywhere, so a "trial" produces no hit and no miss, and N
of them produce no recall.

**FINDING: the demo runs one trial per model at a random depth.** Four printed
probabilities, four different depths, four different curves — not comparable to
each other and not a measurement of any of them.

### 3 — the brute arm does not fit, so there is no recall to compare

**ANSWER: VideoAgent wins recall by forfeit.**

| | |
|---|---:|
| 1 hour @ 1 FPS, 81 tokens/frame | **291,600** |
| brute context | 80,000 |
| overflow | **3.65×** |
| the context holds | 16.4 minutes |
| FPS that would fit an hour | **0.274** — one frame per 3.65 s |

At one frame every 3.65 seconds, a one-second event appears in no frame at all —
so the needle question the exercise is really asking has no answer to find.

**ANSWER: brute context wins latency, and by more than the token count
suggests.** The agent spends **7,490** tokens (38.9× fewer, 97.4% cheaper) but
pays **3 sequential** tool round trips. Prefill is parallel and a round trip is
not, so the agent wins only if *each* retrieval costs less than **24,170** tokens
of prefill time — 24 ms at a microsecond per token, and real retrieval is not
that fast.

**FINDING: flat against linear, so they cross at 92 seconds** of video. The
lesson's 2-hour claim is really a claim about the crossover being early.

**FINDING: and that claim is 98.7%, not 99%** — and **97.4%** at the one hour the
exercise names, because the agent's numerator is constant and the denominator
halves.

### 4 — why ring attention scales, and what the rotation buys

Drawing on **Path 2: Ring attention (LWM, LongVILA)**, which states the linear
scaling claim this exercise asks to explain.

**Memory is linear in sequence length because no device ever holds the matrix.**
Split a sequence of N tokens across P devices, c = N/P each. A device stores its
own K and V — that is **c × d** per device, **N × d** across the ring, linear in
N. What it never stores is the N × N attention matrix; it stores one c × c block
at a time, and c × c is `(N/P)²`. So the quadratic term still exists, but it is
divided by P² per device and the ring only ever has one of it resident.

**Linear in device count is the other half, and it is the same statement.** Each
added device takes a proportional share of K and V, so per-device memory falls as
1/P while total memory stays N × d. That is what "scales linearly in device
count" means: adding hardware buys length, not a smaller constant.

**What the rotation is doing.** Query block *i* has to attend to *every* key
block, including the P−1 held elsewhere. The ring solves this by passing K and V
one hop per step: at step *s*, device *i* holds the K/V of device `i−s`, computes
the partial attention of its own queries against them, and folds the result into
a running accumulator. After P steps every query has seen every key, and no
device ever held more than two blocks at once. Communication overlaps with
compute, which is why the cost is a constant factor rather than a serialisation.

**Drop the rotation and three things happen, in increasing order of how badly.**

1. **Each device attends only to its own chunk.** The result is *block-diagonal*
   attention — exactly the mask Lesson 12.06 measures for patch-n'-pack, where
   the density was `(1 + CV²)/k`. There the block structure was correct because
   the blocks were different images. Here it is wrong, because the blocks are one
   sequence.
2. **The model silently becomes a chunked one.** Nothing errors. The forward pass
   completes, the loss goes down, and the context length reported is N while the
   effective receptive field is N/P. A 1M-token model on 8 devices is a
   125k-token model that says 1M.
3. **And needle-in-a-haystack is exactly the test that catches it.** A marker in
   chunk 3 is invisible to a query in chunk 7, so recall collapses to roughly 1/P
   — the chance the needle happened to land in the querying device's own block.
   Which is the connection between this exercise and exercise 2: the benchmark
   exists because this failure is otherwise invisible.

**The accumulator is the subtle part.** Softmax does not decompose across blocks
for free — you cannot average partial softmaxes. The ring carries a running
maximum and a running denominator and rescales as each block arrives, which is
the same online-softmax trick FlashAttention uses, applied across devices instead
of across SRAM tiles. Drop *that* and the rotation still runs but the
normalisation is wrong, which is a quieter bug than dropping the rotation
outright.

### 5 — what Gemini found at the one-million boundary

Drawing on **Needle-in-a-haystack benchmarks**, which is where the lesson gives
its own recall figures — >99% for Gemini 2.5 Pro at up to 90-minute videos,
~85–90% for open 72B models at 30 minutes, degrading past 60.

**The headline finding was that recall stayed near-perfect across the whole
window, and that this was the surprising part.** The expectation going in was a
degradation curve — recall high at the start, sagging in the middle, recovering
at the end, and worsening overall as the context grew. What the paper reported
instead was recall above 99% held essentially flat across the full 1M-token
context, and in the extended 10M experiments it remained high rather than
collapsing at some boundary. The interesting result was the *absence* of the
curve everyone was looking for.

**Three qualifications that matter more than the headline, and which the lesson's
own numbers illustrate:**

1. **Retrieval is not comprehension.** A single-needle test asks whether a unique
   string can be located. It does not ask whether the model can reason over two
   facts a million tokens apart, count occurrences, or notice an absence.
   Multi-needle variants degrade far earlier than single-needle ones, and the gap
   between the two is the actual measure of long-context capability.
2. **Near-perfect recall at 1M is compatible with the lesson's own 85–90% at 30
   minutes for open models.** Those are the same test at different scales of
   model, and the spread between 99% and 87% is far larger than the spread within
   either across position. Which model matters more than where the needle is —
   the opposite of what a lost-in-the-middle framing predicts.
3. **A video needle is not a text needle.** The lesson's own exercise 3
   arithmetic shows why: at the sampling rates a long video forces, a short event
   may appear in *no frame at all*. A text needle is always present in the
   context; a visual one may never have been encoded. Recall at 1M tokens of text
   and recall at 1M tokens of video are measuring different things, and only one
   of them has a sampling step in front of it.

**Which is the honest summary.** The 1M result was real and was about retrieval;
the inference people drew from it — that long context is solved — did not follow,
and the benchmarks that would have caught the gap (multi-needle, aggregation,
absence-detection) were built afterward.
