<!-- generated:start -->
# 14-agent-engineering / 22-voice-agents-pipecat-livekit

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/22-voice-agents-pipecat-livekit/) · upstream spec
`phases/14-agent-engineering/22-voice-agents-pipecat-livekit/docs/en.md`

```bash
uv run demo practice run 22-voice-agents-pipecat-livekit --ex 1
uv run demo explain 22-voice-agents-pipecat-livekit --ex 1
uv run pytest demos/phases/14-agent-engineering/22-voice-agents-pipecat-livekit
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a metrics observer to your toy pipeline: count frames per stage per second. Where does la… | code | T0 | `ex01_frame_counts_are_flat_and_the_latency_is_not.py` |
| 2 | Implement confidence-gated STT: below threshold, request "could you repeat that?" | code | T0 | `ex02_the_payload_is_the_transcript_so_there_is_nothing_to_doubt.py` |
| 3 | Add semantic turn detection: simple rule — if transcript ends with "?", end of turn. | code | T0 | `ex03_the_rule_fires_on_punctuation_the_stt_never_emits.py` |
| 4 | Read Pipecat's transport docs. Swap the stdlib transport for the SmallWebRTCTransport config… | explain | T0 | prose, below |
| 5 | Measure an OpenAI Realtime vs STT+LLM+TTS cascade on the same query. What latency cost does t… | code | T0 | `ex05_text_level_control_costs_the_stt_and_tts_bands.py` |
<!-- generated:end -->

## Answers

### 1 — the observer answers the first question and refutes the second

A Pipecat-style observer wrapping every processor's `process` records `[1, 1, 1,
1, 1]` frames for one utterance. That is the honest answer to "count frames per
stage per second", and it is useless for the question that follows it. On a
serial chain each processor forwards exactly one frame, so the counts are
identical by construction and carry no information about where the time went. A
dashboard of five flat frame rates would sit next to an end-to-end latency of
695ms and explain none of it.

Latency comes from the lesson's own published bands, which is what a budget is
for — a clock here would measure this laptop. Summed, the chain is 400–990ms, and
LLM first token (150–400ms) is 37.5% of the floor and 40.4% of the ceiling: the
largest single contributor by both measures, and the only stage whose band is
wider than 150ms. That is where latency accumulates, and it is also the stage a
voice team has the least control over.

Two things make this an *addition* rather than an extraction. The lesson's module
references a clock zero times, `Processor.trace` holds plain strings, and `Frame`
carries `kind`, `payload`, `direction` with no identity among them — so one
frame's journey appears as five unrelated log lines that cannot be subtracted
from each other. Give `Frame` an id and a birth timestamp and every question
above becomes a query; without them, no amount of log parsing recovers a span.

And the observer has to read `frame.direction`. A barge-in cancel is seen at all
five stages travelling upstream, taking the observed frame total from 5 to 10 — a
100% traffic increase for an event that makes the user-perceived turn *shorter*.
An observer blind to direction reports load where there is relief.

### 2 — the payload is the transcript, so the confidence has to be invented first

`STT.process` does `transcript = str(frame.payload)`. The audio is the answer,
exactly, every time, so there is nothing to be unconfident about. The port widens
the payload to `(text, confidence)` and leaves the rest of the pipeline alone;
below threshold the STT emits a `text` frame carrying the clarification instead
of a `transcript`, so the LLM is never called on a guess.

A gate at 0.60 re-asks on 3 of 10 utterances and takes wrong transcripts from 3
to 0. Where the threshold lands is a cost question, and sweeping it against a
cost model shows the optimum sitting at 0.60 whether a wrong answer is worth 2,
5 or 10 re-asks. That stability is not a lucky choice of ratio — every error this
recogniser makes is at confidence 0.55 or below, so a single cut separates them
at any price. It is a statement about calibration, and it is the property to
check before trusting any threshold: if the optimum moves with the cost ratio,
the scores are not calibrated and the threshold is fitting noise.

The cost of gating is easy to underestimate from a per-stage budget. A
clarification is not a stage, it is a whole turn — the user hears it, answers,
and the chain runs again. At the lesson's 450–600ms premium band, three gates add
1350–1800ms to the conversation. Gating is cheap per frame and expensive per
dialogue.

One implementation detail is worth keeping. There is no route for the
clarification that skips the LLM by design: sent as a `transcript` it hits
`LLM.replies`, which has no entry for it and answers `'[no canned reply]'` — the
user is asked to repeat and then told nothing. Emitting a `text` frame directly
from the STT reaches TTS in one hop, which works only because `Processor`
forwards by frame kind rather than by a declared contract. That is convenient
here and is the same looseness that lets a malformed frame traverse the whole
chain untouched.

### 3 — the rule keys on punctuation this pipeline never produces

Zero of twelve raw transcripts end in a question mark, because `STT.process`
emits `str(frame.payload)` verbatim and real recognisers do not punctuate partial
transcripts. So `endswith("?")` fires zero times and scores 4/12 purely by
agreeing with the utterances that were not turn ends. A rule that reads the words
instead — end the turn unless it trails off in a filler — scores 11/12.

Restoring punctuation is instructive rather than decisive: the same "?" rule goes
from 4/12 to 10/12, still behind the word rule, because two of the turn-ending
utterances are not questions at all ("tell me the status", "hello"). Whether the
rule works is a fact about the STT's formatting policy, not about the speaker —
which is precisely why LiveKit ships a transformer for semantic turn detection
rather than a regex.

The deeper problem is that the pipeline has no turn to detect. Every `vad_speech`
frame produces a `transcript` and an LLM call on the spot, so an utterance
arriving as three audio chunks makes three LLM calls and delivers three replies
before the speaker has finished. A detector needs somewhere to accumulate, and
`Processor` carries `name`, `next`, `prev`, `trace` — no buffer. Turn detection
is not a predicate you bolt onto the STT; it is a buffering stage that has to
exist first.

Finally, the two rules fail in opposite directions and those failures do not cost
the same. The word rule makes one error and it is a false end; the punctuated
rule makes two and both are missed ends. A missed end is merely slow — one extra
turn, 450–600ms. A false end interrupts the speaker, and recovering from it needs
a barge-in the shipped `TTS` cannot perform (exercise 5). Choosing a turn rule
means choosing which of those two you would rather have, and on this pipeline the
answer is "missed ends", because the machinery to recover from a false end is not
there.

### 4 — swapping the transport is a config change, and the pipeline is built for it

**Pipecat (pipecat-ai/pipecat)** describes a frame-based pipeline where `Frame`
travels a `FrameProcessor` chain in two directions — DOWNSTREAM for audio in and
TTS out, UPSTREAM for cancellation, metrics and barge-in — with `PipelineTask`
managing lifecycle events (`on_pipeline_started`, `on_pipeline_finished`,
`on_idle_timeout`) and observers for metrics, tracing and RTVI. The listed
transports are Daily, LiveKit, SmallWebRTCTransport, FastAPI WebSocket and
WhatsApp.

The stub is small, and that is the point:

```python
transport = SmallWebRTCTransport(
    params=TransportParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_analyzer=SileroVADAnalyzer(),
    ),
)
pipeline = Pipeline([transport.input(), stt, llm, tts, transport.output()])
task = PipelineTask(pipeline, observers=[MetricsObserver()])
```

Three things are worth noticing about that shape against the toy.

The transport appears **twice** — `transport.input()` at the head and
`transport.output()` at the tail — where the toy has a single `Transport` at the
sink end and nothing at the source. That is not cosmetic. It is what makes the
chain a loop rather than a line, and it is where UPSTREAM frames originate: a
barge-in is detected at the input side and travels back up through TTS. The toy's
`main()` fakes this by calling `transport.process(Frame("cancel",
direction="upstream"))` by hand.

The VAD moves **into** the transport params as a `vad_analyzer`, rather than
being a processor in the chain. In the toy, `VAD` is the first `Processor` and
`bool(frame.payload)` is the entire speech decision. Real transports own the
audio clock, so VAD belongs where the frames are minted.

And the observer is a constructor argument to `PipelineTask`, not a wrapper
around each processor. Exercise 1 builds the wrapper version because that is all
the toy allows — `Processor.process` is the only hook — and the result works but
is a monkey-patch. The framework's answer is that observation is a first-class
lifecycle concern, which is also why it can see `on_idle_timeout`, an event the
toy has no concept of at all.

What does *not* change when the transport is swapped is the part worth trusting:
STT, LLM and TTS see the same frames whichever transport is under them. That is
the actual payoff of the frame abstraction, and it is why the exercise is phrased
as a swap.

### 5 — 200–450ms, which is exactly the STT and TTS bands

Summing the lesson's published bands, the cascade is VAD + STT + LLM + TTS +
transport = 400–990ms, midpoint 695ms. A Realtime path carries audio straight
into the model: VAD + model + transport = 200–540ms, midpoint 370ms. The
difference is 200–450ms, which is STT's 100–250 plus TTS's 100–200 with nothing
else changing. That is the latency cost of text-level control, stated exactly
rather than sampled.

The two land on opposite sides of the lesson's own grading. Realtime's 370ms
midpoint is below the 450–600ms premium band; the cascade's 695ms is above it,
and the cascade's 990ms ceiling sits squarely inside the 800–1200ms "common"
band. Neither approaches the 1500ms that feels broken. So this is not a
functioning-versus-broken choice, it is better-than-premium against ordinary —
which is exactly the kind of margin that gets traded away for control.

What the 200–450ms buys is the input to three of the five stages. `stt`, `llm`
and `tts` all name a text frame kind in their own source, so a Realtime path
leaves three of five processors with no frame they recognise. Everything built in
this lesson keys on that text: the confidence gate of exercise 2 reads a
transcript, the turn rule of exercise 3 reads a transcript, `LLM.replies` is a
dictionary keyed on one. Text-level control is not a nicety here; it is the only
surface any of these features can attach to. That is LiveKit's
`VoicePipelineAgent`-versus-`MultimodalAgent` split, priced.

The uncomfortable finding is that the shipped cascade pays the cost and does not
deliver the headline benefit. `TTS.process` sets `self.cancelled = False` and
then loops over words checking it, with nothing able to mutate it in between — so
a TTS pre-set to cancelled still emits all 14 words and logs zero cuts. The
branch that makes a text pipeline interruptible is unreachable. Barge-in is the
one thing a cascade is supposed to do better than direct audio, and making it
real needs the emit loop to yield between words so an UPSTREAM frame can land —
which is why Pipecat's processors are async and this toy's are not.
