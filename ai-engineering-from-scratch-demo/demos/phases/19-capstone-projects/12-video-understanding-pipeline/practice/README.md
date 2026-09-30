<!-- generated:start -->
# 19-capstone-projects / 12-video-understanding-pipeline

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/12-video-understanding-pipeline/) · upstream spec
`phases/19-capstone-projects/12-video-understanding-pipeline/docs/en.md`

```bash
uv run demo practice run 12-video-understanding-pipeline --ex 1
uv run demo explain 12-video-understanding-pipeline --ex 1
uv run pytest demos/phases/19-capstone-projects/12-video-understanding-pipeline
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Swap Gemini 2.5 Pro for Qwen3-VL-Max on the captioning pass. Report caption quality delta on… | code | T0 | `ex01_the_lower_rated_captioner_retrieves_better_and_fake_embed_cannot_see_action_order.py` |
| 2 | Reduce per-scene frame embedding to one pooled vector instead of multi-vector. Measure the re… | code | T0 | `ex02_pooling_the_three_vectors_raises_mrr_from_0_259_to_0_391_because_rrf_drowns_the_matched_stream.py` |
| 3 | Build a "counting strict" mode: the synthesizer extracts each counted instance with a timesta… | code | T0 | `ex03_verification_cuts_hallucinated_instances_58_to_69pct_but_cannot_fix_a_miss.py` |
| 4 | Benchmark ingest cost: hours-of-video-per-dollar across three VLM choices. Pick the sweet spot. | code | T0 | `ex04_gemini_2_5_pro_indexes_4_8_video_hours_per_dollar_and_caption_length_picks_between_molmo_2_and_qwen.py` |
| 5 | Add speaker-diarized transcript: run pyannote speaker diarization on the audio and embed per-… | code | T0 | `ex05_per_speaker_turns_answer_78pct_of_what_did_alice_say_queries_and_the_shipped_index_0.py` |
<!-- generated:end -->

## Answers

The lesson's `code/main.py` is a small version of the scene index. Each
`Scene` holds three strings (caption, frame tags, transcript). `fake_embed`
turns each string into a 24-dim bag of hashed words. `multi_vector_search`
ranks every scene once per stream and merges the three rankings with RRF
(k = 60). `ground_window` narrows the top scene to the span between the
first and last query word found in its transcript. There is no video, no
VLM, no ASR and no Qdrant. Every solution runs this code as shipped, on
fixtures built in the solution file.

`fake_embed` uses Python's builtin `hash()`, which is salted per process,
so the lesson's own output changes from run to run. The solutions swap in a
salted blake2b and average over 10-20 salts. External facts were read on
2026-09-29: Gemini API pricing (https://ai.google.dev/gemini-api/docs/pricing)
and token rules (https://ai.google.dev/gemini-api/docs/tokens), Alibaba
Model Studio pricing (https://www.alibabacloud.com/help/en/model-studio/model-pricing),
Molmo 2 8B pricing (https://pricepertoken.com/pricing-page/model/allenai-molmo-2-8b),
the Qwen3-VL and Molmo 2 preprocessor configs on Hugging Face
(https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct, https://huggingface.co/allenai/Molmo2-8B),
the Gemini 2.5 Pro thinking-budget floor as reported in
https://github.com/cline/cline/issues/7735, and pyannote's README
(https://github.com/pyannote/pyannote-audio).

### 1 — the lower-rated captioner retrieves better, and `fake_embed` cannot see action order

**The rating harness reports a delta of +0.34, 95% interval [-0.02, 0.71],
so this 50-scene sample cannot call the swap.** No VLM runs in the lesson,
so the two captioners are stand-ins. Each writes a gold caption and then
makes errors at a placeholder rate: a wrong count, the two actions swapped,
an object dropped. The rating is a 1-5 rubric seen by two noisy raters.

| | Gemini stand-in | Qwen stand-in |
|---|---:|---:|
| mean rating (two raters) | 4.17 | 3.83 |
| hit@1 through the lesson's index | 0.418 | 0.433 |

The raters agree exactly on 56% of scenes. Over 200 fresh 50-scene studies
at the same rates, 89% find the gap, so about one study in nine would report
no difference.

**The rating gap does not reach retrieval.** The lower-rated captions
retrieve slightly better. Gold captions score 0.432, and blanking every
caption only drops hit@1 to 0.395. The caption is one of three RRF streams,
and a "how many <object>" query never contains the count the captioner got
wrong.

**Action-order errors are invisible to the embedder.** `fake_embed` is a bag
of words, so "pours then stirs" and "stirs then pours" are the same vector.
All 6 captions whose only error was a swapped order embed exactly like the
gold caption. The lesson's own demo asks "what happened first pour or stir".

### 2 — pooling the three vectors raises MRR from 0.259 to 0.391, because RRF drowns the stream that matched

**There is no regression. Pooling helps.** 50 scenes, each with its
caption, frame tags and transcript drawn from separate word pools; 150
queries, each two words from one stream; 10 salts.

| MRR | RRF (lesson) | pooled vector | max cosine of 3 |
|---|---:|---:|---:|
| EMB_DIM 24 (shipped) | 0.259 | 0.391 | 0.608 |
| EMB_DIM 256 | 0.220 | 0.801 | not run |

Keeping three vectors pays off only without RRF. RRF ranks every scene in
every stream. With k = 60 and 50 scenes, first place in the matching stream
is worth 0.0073 more than last place, and the two streams that did not match
add random ranks worth as much. With fewer hash collisions (dim 256) most
cosines are exactly 0, `sorted` breaks the ties by list order, and fixture
scene 0 becomes RRF's top hit for 21.0% of queries (2% would be uniform).
The skill file lists pooling as a hard reject. On this index pooling is the
better choice.

On the lesson's own 6 scenes the two tie, 31 of 40 each. Most misses are the
counting demo query "how many cars pass through the intersection": the word
"intersection" belongs to scene 1 and the cars are in scene 2, so it finds
its scene in only 4 (RRF) or 5 (pooled) of 10 salts.

### 3 — verification cuts hallucinated instances by 58-69% but cannot fix a miss

**User verification reduces hallucination in every setting. It improves the
count only when the model over-counts.** 400 counting questions. A stand-in
synthesizer misses events, double-counts them and invents some. The
simulated user rejects instances with no event in the frame (5% slip) and
catches half of the double counts. "Hallucinated" means instances beyond
one per real event.

| setting | exact count, free | exact count, strict | hallucinated, free | hallucinated, strict |
|---|---:|---:|---:|---:|
| balanced | 33.0% | 52.5% | 19.4% | 7.3% |
| over-counting | 14.2% | 52.5% | 33.3% | 14.0% |
| under-counting | 34.5% | 33.8% | 8.6% | 2.7% |

Verification only removes instances. When the model under-counts, 85.9% of
the wrong counts are too low before verification and 96.6% after. For
scale, the Molmo 2 authors report that on their video counting benchmark
"no model yet reaches 40% accuracy" (https://allenai.org/blog/molmo2).

**The lesson grounds its counting demo query on the word "the".** In scene 2
("let me count the vehicles approaching") the only query word present is
"the". `ground_window` returns 01:36-01:53, 26.7% of the scene. A strict mode
that counted inside that window would miss the cars in the other 73%.

**The doc's sample output is not something the code can produce.** "Use It"
prints top scene 3 at [01:32-01:54] and then a refined window [00:12-00:58],
which is outside that scene. `ground_window` always stays inside its scene,
on all 6 lesson scenes x 5 queries.

### 4 — Gemini 2.5 Pro indexes 4.8 video hours per dollar, and caption length picks between Molmo 2 and Qwen

**The sweet spot is Molmo 2 8B or Qwen3-VL-Plus, at 9-10x the hours per
dollar of Gemini 2.5 Pro.** One 1280x720 keyframe per scene, a 50-token
prompt, 80 scenes per hour (the doc's upper budget), list prices.

| model | image tokens | $/1M in / out | hours per $, 30-token caption | 60 | 120 |
|---|---:|---:|---:|---:|---:|
| Gemini 2.5 Pro | 516 | 1.25 / 10.00 | 5.5 | 4.8 | 3.9 |
| qwen3-vl-plus | 880 | 0.20 / 1.60 | 53.4 | 44.3 | 33.1 |
| Molmo 2 8B | 1,162 | 0.20 / 0.20 | 50.3 | 49.1 | 46.9 |

Qwen wins at 30-token captions and Molmo at 60 and 120. The crossover is
a 40-token caption: Molmo spends more tokens on the image, and Qwen's output
tokens cost 8x more. There is no "Qwen3-VL-Max" on Alibaba's price list; the
nearest are qwen3-vl-plus and the older qwen-vl-max.

**Most of Gemini's bill is output nobody reads.** Gemini 2.5 Pro cannot turn
thinking off, and its minimum budget is 128 tokens. With that minimum,
output is 72.7% of the per-scene cost. Sending the whole hour as native
video (263 tokens per second) instead of keyframes gives 0.7 hours per
dollar, 7x worse.

**The lesson's sample cuts more scenes than the doc budgets.** vid_001 has
5 scenes in 210 s (85.7 per hour) and vid_002 1 in 40 s (90 per hour),
against the doc's 60-80 per hour (6k-8k scenes per 100 h). At the lesson's
rate every figure above is 7-50% too optimistic.

### 5 — per-speaker turns answer 78% of "what did Alice say" queries, and the shipped index 0%

**With diarized per-speaker turns the query works. The shipped index cannot
answer it.** pyannote needs audio and a Hugging Face token, and the lesson
has neither, so the diarizer is replaced by its output: turns with
anonymous labels (`SPEAKER_00` ...), with whole turns relabelled until the
error share matches pyannote community-1's published DER. Alice, Bob and
Carol each introduce themselves, then take turns on six topics, two
speakers per topic. A label takes its name from the turn with the
introduction. 12 queries x 20 seeds, and an answer is right at IoU >= 0.5
with the gold turn.

| DER | 0 | 11.2% (VoxConverse) | 17.0% (AMI) | 20.2% (DIHARD 3) |
|---|---:|---:|---:|---:|
| per-speaker turns | 77.9% | 52.1% | 46.3% | 44.6% |

At DER 0 the misses are hash collisions in the 24-dim `fake_embed`. At
`EMB_DIM = 256` the same run scores 99.6%.

**A wrong label on an introduction costs a speaker's whole set of answers.**
That speaker is then unnamed, or two labels get the same name. At DER 20.2%
this happens in 45% of runs, and those runs hold 49% of the misses.

**The shipped grounding keys on "the".** Its transcripts carry no speaker,
and all 12 topic turns and every query contain "the". `ground_window`
therefore stretches from the first "the" to the last: 22.5 s windows
against 6 s turns, and 0 of 240 answers right.
