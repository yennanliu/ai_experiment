<!-- generated:start -->
# 12-multimodal-ai / 16-mio-any-to-any-streaming

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/12-multimodal-ai/16-mio-any-to-any-streaming/) · upstream spec
`phases/12-multimodal-ai/16-mio-any-to-any-streaming/docs/en.md`

```bash
uv run demo practice run 16-mio-any-to-any-streaming --ex 1
uv run demo explain 16-mio-any-to-any-streaming --ex 1
uv run pytest demos/phases/12-multimodal-ai/16-mio-any-to-any-streaming
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Your product accepts speech input and returns speech output. What's the end-to-end latency bu… | code | T0 | `ex01_the_first_line_of_the_streaming_model_is_not_streaming.py` |
| 2 | SpeechTokenizer residual-VQ uses 8 codebooks. Propose why parallel-decoding the residual leve… | code | T0 | `ex02_sequential_residual_decoding_runs_at_four_times_real_time.py` |
| 3 | Your vocabulary has 32k text + 4k image + 4k speech. Add 8k music and ~10 separators. What is… | code | T0 | `ex03_the_seven_residual_levels_share_one_codebook_slot.py` |
| 4 | Chain-of-visual-thought emits an intermediate image. What kinds of questions benefit? What ki… | explain | T0 | prose, below |
| 5 | Read Moshi (arXiv:2410.00037). Describe its "inner monologue" technique and compare to MIO's… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

Three code solutions and two prose ones. The three code ones all measure the
lesson's own two artefacts — a five-term latency model and an eleven-slot
vocabulary table — and each finds a design decision hidden inside a label:
"streaming" in a term that scales with the whole utterance, "layers 1..7" in a
cost that is not a model pass, and seven residual codebooks in one slot.

### 1 — the first line of the streaming model is not streaming

| component | ms |
|---|---:|
| mic audio → speech tokens | 40 |
| prefill prompt tokens | 80 |
| first output token | 40 |
| residual-VQ layers 1..7 | 30 |
| speech decoder | 80 |
| **total** | **270** |

**ANSWER: under 500 ms TTFAB**, which the lesson's own threshold table calls
"conversational"; its default configuration lands at **270**.

**FINDING: the tokenization term scales with the whole utterance.** It is
`prompt_audio_seconds * 20`, so the total goes 230 / 270 / **830** ms across 0-,
2- and 30-second turns — the last one **sluggish** by the lesson's own scale. A
streaming tokenizer contributes one frame, not one utterance.

**FINDING: only two of the five terms move with model size.** 165 / 270 /
**1,200** ms at 1B / 8B / 70B, and only **44.4%** of the 8B total is
model-scaled, so halving the model buys less than a quarter of the latency.

**FINDING: there is a 110 ms floor nothing can touch** — the residual levels and
the speech decoder, spent *after* the model has decided what to say. **41%** of
the default budget.

### 2 — sequential residual decoding runs at four times real time

**ANSWER: parallel decoding saves 250 ms, 48.1%.** Seven residual levels priced
together at 30 ms, against one pass each at the 40 ms first-token cost:
**270 → 520 ms**, the difference between *conversational* and *acceptable* on the
lesson's own scale.

**ANSWER: necessity is a different argument, and it is about real-time factor.**

| frame rate | parallel RTF | sequential RTF |
|---|---:|---:|
| 12.5 Hz (Moshi-class) | **0.50** | **4.00** |
| 50 Hz (SpeechTokenizer's own) | 2.00 | 16.00 |

At 12.5 Hz, sequential decoding falls **three seconds behind for every second
spoken**, and the gap never closes. Latency you can amortise; a real-time factor
above 1 you cannot.

**FINDING: the frame rate decides it and the lesson names none.** At 50 Hz even
*parallel* decoding cannot stream. The architecture question has a threshold, and
the threshold is the one number the latency model leaves out.

**FINDING: the 30 ms is not a model pass.** Seven levels in 30 ms is **4.3 ms**
each — **9.3×** cheaper than the first-token cost — so the figure prices a small
depth head run over all levels at once, not the main transformer. That is the
actual design, and the label "layers 1..7" hides it.

### 3 — the seven residual levels share one codebook slot

**ANSWER: 48,394 × 4,096 = 198,221,824 parameters.** 198.2M tied, **396.4M**
untied — **2.83%** and **5.66%** of a 7B model.

**FINDING: the lesson's own vocabulary is 52,486, not 48,394.** It allocates
`speech L0` *and* `speech L1..L7` at 4,096 each and six separators rather than
ten: **215.0M**, **8.5%** above the exercise's figure.

**FINDING: "speech L1..L7" is one 4,096-entry slot for seven codebooks.**
Residual-VQ levels each have their own codebook, so either the seven are tied or
the slot is seven times too small. Allocating all eight separately:

| | entries | embedding |
|---|---:|---:|
| exercise as stated | 48,394 | 198.2M |
| lesson's `build_vocab` | 52,486 | 215.0M |
| all 8 levels untied | **77,062** | **315.6M** (+46.8%) |

A design detail the table cannot express, worth nearly half the embedding.

**FINDING: the music slot alone outweighs a whole projector.** 8,192 × 4,096 =
**33.6M**, which is **1.49×** LLaVA's entire two-layer projector from Lesson
12.05 — and it is one row of a vocabulary table.

### 4 — which questions chain-of-visual-thought helps

Drawing on **Chain-of-visual-thought**, which is where the lesson describes the
four-step trace and notes that it "wins on spatial-reasoning benchmarks; hurts
latency".

**Benefits: questions whose answer is a spatial relation that the input does not
lexicalise.** Occlusion order ("is the mug in front of the laptop"), containment,
route and reachability ("could the cat get to that branch"), counting under
overlap, and fit questions ("would this box go through that door"). The common
property is that the *relation* is not a property of any one object, so no
caption of the scene contains it — and the scratchpad is in the same
representation as the evidence, so a relation that is hard to say is easy to
draw.

**Hurt: everything whose answer is a lookup.** Recognition, OCR, attribute
questions ("what colour is the car"), counting without overlap. The answer is
already in the input; the intermediate image adds tokens and a chance to be
wrong.

**The cost, in this phase's own numbers.** An emitted image is **1,024** tokens
at Lesson 12.14's rate and **4,096** at Lesson 12.12's. Against a twenty-token
answer that is a **51×** to **205×** expansion, and at Lesson 12.12's measured
30 tokens per second it is **34 seconds to 2:16** of scratchpad before the answer
starts. Exercise 1 of this lesson prices a whole spoken turn at 270 ms. A
reasoning step three orders of magnitude more expensive than the response is not
a latency cost, it is a different product.

**And there is a third category, which is the one worth watching.** Questions
where the scratchpad can *introduce* the error. The intermediate image is
**generated**, so it can place the cat on the wrong side of the tree — and the
text step will then faithfully analyse the wrong sketch and produce a confident,
internally consistent, wrong answer. Text chain-of-thought has the same failure,
but its intermediate is legible to whoever reads the output. An image scratchpad
compounds a generation error into a reasoning error with nothing checking the
scratchpad against the input it was drawn from.

**So the decision rule is narrow:** use it when the question is relational, the
input is genuinely ambiguous in text, and the latency budget is measured in
seconds rather than milliseconds. That is a small set, and none of it is a
spoken-dialogue product.

### 5 — Moshi's inner monologue against chain-of-visual-thought

Drawing on **Competitors in any-to-any**, which is where the lesson places MIO
among the other four-modality systems.

**Moshi's inner monologue: the model predicts text alongside its own audio.**
Two streams in one sequence — audio tokens and a text transcript of what the
model is about to say — with the text running a fixed delay *ahead* of the audio.
The text is not shown to anyone; it is a plan that the audio then realises. The
effect is to ground speech generation in the language model's own text
competence, which is the thing a purely acoustic stream loses.

**The two techniques are the same trick pointed in opposite directions.**

| | Moshi's inner monologue | MIO's chain-of-visual-thought |
|---|---|---|
| auxiliary modality | text | image |
| relative to the output | **cheaper** | **more expensive** |
| timing | concurrent, one delayed stream | sequential, a prefix |
| latency cost | ≈ none | the scratchpad's full token count |
| content | the *same* content, re-expressed | *new* content, generated |
| divergence is | detectable — text and audio disagree | a hallucination with nothing to check |

**The cost asymmetry is the heart of it.** Moshi's text stream is a handful of
tokens per second against roughly a hundred audio tokens per second (8 codebooks
at a 12.5 Hz frame rate) — call it a few percent of the stream, emitted in
parallel, costing no extra step. MIO's image scratchpad is 1,024–4,096 tokens
against a twenty-token answer, emitted in series. One plans in the cheap modality
to produce the expensive one; the other produces the expensive modality to reason
about a cheap answer.

**And the verifiability asymmetry is what makes the first safe.** Because the
monologue is the same content in another form, the two streams are a redundant
encoding — a mismatch is a detectable fault, and the model is being asked to say
the thing it already decided to say. Because the sketch is new content, there is
no second copy to compare it with, and the model is being asked to invent
evidence and then trust it.

**Which suggests the interesting variant neither does.** Emit the scratchpad as a
*structured* intermediate — boxes, depths, a scene graph — rather than as pixels.
It keeps the relational content that exercise 4 says is the whole benefit, costs
tens of tokens rather than thousands, and is checkable against the input the way
Moshi's transcript is checkable against its audio. That is closer to what
Unified-IO 2 does with its depth and normal outputs than to what either of these
two does.
