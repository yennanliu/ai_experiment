<!-- generated:start -->
# 12-multimodal-ai / 19-audio-language-whisper-to-af3

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/12-multimodal-ai/19-audio-language-whisper-to-af3/) · upstream spec
`phases/12-multimodal-ai/19-audio-language-whisper-to-af3/docs/en.md`

```bash
uv run demo practice run 19-audio-language-whisper-to-af3 --ex 1
uv run demo explain 19-audio-language-whisper-to-af3 --ex 1
uv run pytest demos/phases/12-multimodal-ai/19-audio-language-whisper-to-af3
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Compute the log-Mel spectrogram dimension for a 30-second clip at 16kHz, 25ms window, 10ms ho… | code | T0 | `ex01_the_dimension_does_not_change_at_forty_eight_kilohertz.py` |
| 2 | Why does Whisper underperform on music? What audio features does BEATs capture that Whisper d… | explain | T0 | prose, below |
| 3 | Audio Q-former with 64 queries vs 32: at what task complexity does 64 pay off? 32 save comput… | code | T0 | `ex03_at_initialisation_every_query_returns_the_frame_mean.py` |
| 4 | Read AF3 Section 4 on on-demand thinking. Propose three audio tasks where chain-of-thought he… | explain | T0 | prose, below |
| 5 | Implement a minimal diarization pipeline using AF3's output. How do you signal speaker changes? | code | T0 | `ex05_swapping_two_labels_costs_a_hundred_percent_or_nothing.py` |
<!-- generated:end -->

## Answers

Three code solutions and two prose ones. Exercise 1's answer is the one that
surprises: the log-Mel tensor is the same shape at 16 kHz and 48 kHz, because the
window and hop are stated in milliseconds and a millisecond is not a sample.

### 1 — the dimension does not change at 48 kHz

| | 16 kHz | 48 kHz |
|---|---:|---:|
| window / hop, samples | 400 / 160 | 1,200 / 480 |
| **frames × Mel bins** | **2998 × 80** | **2998 × 80** |
| Nyquist | 8 kHz | **24 kHz** |
| **bandwidth per bin** | **100 Hz** | **300 Hz** |

**ANSWER: 2998 × 80 either way.** Tripling the sample rate triples both the
window and the hop and leaves the ratio that sets the frame count untouched.

**FINDING: what changes is the bandwidth each Mel bin covers** — **3×** the span
over the same 80 bins. Same tensor, a third of the frequency resolution, which is
exactly why Whisper resamples everything to 16 kHz rather than accepting whatever
it is handed.

**FINDING: `window_frames` drops the tail.** Its loop runs
`while i + win <= len(x)`, so it gives **2998** where Whisper pads to **3,000** —
0.07% of the tensor and the last **20 ms** of every clip.

**FINDING: the demo contradicts its own comment.** It prints `frames : 98` beside
"(should be ~99 at 1s)". `(16000 - 400) // 160 + 1 = 98` — the comment rounds up
and the code rounds down.

### 2 — why Whisper underperforms on music

Drawing on **BEATs and audio-specific encoders**, which states the weakness and
names the fix without saying what the two encoders actually differ in.

**Whisper's objective is the answer, not its architecture.** It was trained to
map audio to *text* — 680k hours of weakly supervised transcription. That
objective is a filter: a representation is rewarded exactly to the extent it
predicts the words, and everything that does not predict the words is free to be
discarded. The encoder is not bad at music; it was trained to be **invariant** to
most of what music is.

**What that invariance covers, concretely:**

- **Timbre.** Two instruments playing the same note differ in spectral envelope
  and almost nothing else. For transcription, timbre is speaker identity — a
  nuisance variable to normalise away. For music it is the instrument.
- **Harmonic structure above the speech band.** Whisper resamples to 16 kHz,
  so everything above **8 kHz** is gone before the encoder sees it. Exercise 1
  puts a number on the rest: at 16 kHz a Mel bin spans 100 Hz, which is wider
  than a semitone anywhere below about 1.7 kHz. Pitch relations inside a chord
  fall *inside* single bins.
- **Long-range temporal structure.** Whisper's window is 30 seconds and its
  training target is a transcript, which has no notion of a repeat, a phrase
  boundary or a section. Rhythm at the bar level is not something a transcript
  rewards.
- **Non-speech events entirely.** A door slam, a bird, an engine — the
  transcription target for all three is the empty string.

**What BEATs captures instead.** It is self-supervised on AudioSet — masked
prediction over *acoustic tokens*, with no text anywhere. The objective is to
reconstruct the audio's own discrete structure, so nothing is a nuisance
variable: timbre, texture, event onsets and ambience all have to be encoded
because all of them are part of what gets predicted. That is the whole
difference, and it is a difference in supervision rather than in capacity —
"at the same parameter count", as the lesson says.

**Which is why AF-Whisper concatenates rather than chooses.** The two encoders
are not competing at the same task; they are discarding complementary things.
Whisper throws away everything that does not carry words; BEATs has no special
purchase on words. Concatenation is the cheapest way to stop throwing away
either, and Lesson 12.07's exercise 2 measures the same pattern in vision: a
DINOv2 + SigLIP concat beats the mean of its parts everywhere and the *better*
part only where the two genuinely disagree.

### 3 — at initialisation every query returns the frame mean

**ANSWER: 32 saves exactly half.**

| | 32 queries | 64 queries |
|---|---:|---:|
| multiply-adds | 126,720 | **253,440** |
| tokens into the LLM | 32 | 64 |
| mean attention entropy | 4.409 | **4.411** (uniform: 4.595) |
| mean pairwise cosine | 0.3381 | **0.3299** |

**FINDING: at initialisation the queries barely differentiate.** Attention
entropy is **96.0%** of uniform, so every query is very nearly averaging the whole
clip and every output token is very nearly the same vector.

**FINDING: doubling the queries changes that by 0.008.** The extra 32 are as
redundant as the first 32 — **2×** the compute and 2× the LLM tokens for a
difference in the fourth decimal place.

**ANSWER: so the answer is about the training signal, not the task.** 64 pays off
when the supervision distinguishes things a single pooled summary cannot —
overlapping speakers, an event at a named second, a genre *and* an instrument at
once — because that is what forces the queries apart. 32 saves compute whenever a
clip-level embedding already answers the question. The exercise's framing, "at
what task complexity", has the dependency backwards: complexity is necessary and
not sufficient, and an untrained 64 is a strictly worse 32.

### 4 — three audio tasks where thinking helps most

Drawing on **The arc — SALMONN, Qwen-Audio, AF3**, which records the on-demand
chain-of-thought and its "3–5 points on complex reasoning tasks".

**The selection rule first, because it is what makes the three non-arbitrary.**
Chain-of-thought buys nothing when the answer is a single classification the
encoder either supports or does not. It buys something when the answer is a
*composition* of intermediate judgements, each of which the model can make and
none of which the final token can hold. So: tasks that decompose, where the
decomposition is in the audio and not in world knowledge.

**1. Multi-source scene understanding — "how many distinct sound sources, and
what is each?"** The decomposition is explicit: isolate, identify, count. Each
step is a judgement the encoder supports and the composition is what fails
without it. This is also the task where a wrong intermediate is *visible* in the
trace, which matters for a modality where the user cannot check the input.

**2. Temporal ordering and causation — "did the door close before or after the
phone rang?"** Two events, two timestamps, one comparison. The comparison is
trivial *once the timestamps exist*, and the failure without thinking is that the
model answers from the prior over event orderings rather than from the clip.
Lesson 12.17's exercise 3 shows the same shape in video: a grounding answer that
cannot be checked is indistinguishable from a guess at the prior.

**3. Music-theoretic questions — "what key is this in?"** The chain is chord
identification, then root relations, then key. The last step is deterministic
given the first two and almost impossible directly, which is the clearest case
where the thinking tokens are doing work the encoder cannot.

**And the one it does not help, which is worth naming.** Straight classification
— genre, language ID, speaker verification, deepfake detection. There is no
decomposition; the answer is a property of the embedding. Thinking here adds
tokens, adds latency, and adds a surface for the model to talk itself out of a
correct first instinct. "On-demand" is the important word in the feature's name:
a 3–5 point average lift over complex tasks is compatible with a loss on simple
ones, and nothing in the headline number separates them.

### 5 — swapping two labels costs 100% or nothing

**ANSWER: contiguous turns with a local speaker id, scored under the best label
permutation.**

```json
[{"speaker": "S1", "start": 0.0, "end": 4.0, "text": "..."},
 {"speaker": "S2", "start": 4.0, "end": 7.0, "text": "..."}]
```

Turns tile the timeline with no gaps and no overlaps, and consecutive turns carry
*different* speakers — which is what makes a boundary a **change** rather than a
repetition.

**FINDING: without permutation matching, a perfect transcript scores 100%
error.**

| hypothesis | naive DER | permutation-matched DER |
|---|---:|---:|
| exact | 0.0% | 0.0% |
| S1 and S2 swapped | **100.0%** | **0.0%** |
| one boundary moved 0.5 s | 5.0% | **5.0%** |

Speaker ids are local to a clip, so a scorer that compares them literally is
measuring a naming convention.

**FINDING: the metric is boundary-sensitive in seconds, not in turns.** A single
boundary off by 0.5 s in a 10-second clip costs **5.0%** with the turn count, the
speaker count and the order all correct. "3 of 3 turns found" would call that
perfect.

**FINDING: this is why the lesson's table calls cascaded diarization
"partial".** Whisper supplies `start`, `end` and `text` for free and no `speaker`
field at all — three of the four fields come from the cascade and the fourth is
the entire task.
