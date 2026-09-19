<!-- generated:start -->
# 12-multimodal-ai / 24-multimodal-rag-cross-modal

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/12-multimodal-ai/24-multimodal-rag-cross-modal/) · upstream spec
`phases/12-multimodal-ai/24-multimodal-rag-cross-modal/docs/en.md`

```bash
uv run demo practice run 24-multimodal-rag-cross-modal --ex 1
uv run demo explain 24-multimodal-rag-cross-modal --ex 1
uv run pytest demos/phases/12-multimodal-ai/24-multimodal-rag-cross-modal
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Propose a medical-triage multimodal RAG: query = photo of injury + text symptoms. What modali… | explain | T0 | prose, below |
| 2 | Score fusion is a simple weighted sum. What failure mode does it have that MoE fusion avoids? | code | T0 | `ex02_a_modality_with_no_opinion_still_takes_its_weight.py` |
| 3 | Read Abootorabi et al.'s taxonomy (Section 3). What are the three canonical sub-problems and… | explain | T0 | prose, below |
| 4 | Design an eval spec for a trip-planner multimodal RAG. What metrics cover image recall, audio… | code | T0 | `ex04_the_weakest_retriever_carries_thirty_percent_of_the_weight.py` |
| 5 | Agentic multi-hop RAG has a latency tax per round-trip. At what query difficulty does the acc… | code | T0 | `ex05_the_second_round_changes_no_ranking_so_the_tax_buys_nothing.py` |
<!-- generated:end -->

## Answers

Three code solutions and two prose ones. The three code ones take apart the same
weighted sum from three directions, and each finds a number the lesson reports
that does not mean what it appears to.

### 2 — a modality with no opinion still takes its weight

**ANSWER: a constant score is indistinguishable from a low one.** On "find me a
vegan brunch" the lesson's own retrievers return:

| retriever | scores |
|---|---|
| text | varied |
| image | **0.0 × 5** |
| audio | **0.5 × 5** |

A constant shifts every fused score and changes no ranking, so two of the three
modalities are mute and the fusion is still a three-term weighted average
consuming **70%** of its budget on them. A gate that reads the query routes that
weight elsewhere; a fixed weight cannot — which is the whole of what MoE fusion
adds here.

**FINDING: the three scores are on three different scales.**

| | span on the demo query | nominal weight | **effective** |
|---|---:|---:|---:|
| text | 0.4286 | 0.3 | 0.1286 |
| image | **1.0000** | 0.4 | **0.4000** |
| audio | 0.4250 | 0.3 | 0.1275 |

Effective weighting is the nominal weight times the spread, and nobody computes
the product — so the 0.4 on image is worth three times the 0.3 on text.

**FINDING: which is why the agentic trigger is meaningless.** Doubling every
weight cannot change a ranking, and does not — `['r1','r4','r3']` either way —
but takes the top fused score from **0.686** to **1.372**. `agentic_loop`
compares that number against a floor, so its trigger measures the weights rather
than the retrieval.

### 4 — the weakest retriever carries 30% of the weight

**ANSWER: four metrics, three of them per-modality** — per-retriever recall@k,
fused recall@k and precision@k, and citation coverage. Gold is re-derived from
the corpus (< 45 dB, vegan brunch, natural light) as `{r1, r4}`, so the spec is
checkable rather than asserted.

| | recall@2 |
|---|---:|
| **fused** | **1.00** (precision@3 = 0.67) |
| image alone | 1.00 |
| audio alone | 1.00 |
| **text alone** | **0.50** |

**FINDING: the weakest retriever carries 30% of the weight** — and it is weak
because `r1` wins outright and the second slot is a **three-way** tie at
**0.2857** between `r2`, `r3` and `r4` that `sorted` breaks by position rather
than relevance, so gold `r4` loses it to `r2`. No eval in the lesson would have
shown it.

**FINDING: citation coverage is already satisfied and still worth having.**
**3 of 3** modalities on every result. It is worth measuring because it degrades
*silently*: drop the audio retriever from the fusion and the dB line is still
printed, so the answer keeps citing evidence that no longer scored anything.

### 5 — the second round changes no ranking, so the tax buys nothing

**ANSWER: a round pays when its recall gain beats the latency cost — and the
lesson's own round gains 0.00.**

| | ranking | top score | recall@2 |
|---|---|---:|---:|
| round 1 | `['r1','r4','r3']` | 0.6861 | **1.00** |
| round 2 | `['r1','r4','r3']` | **0.7336** | **1.00** |

Every score moved and no answer did.

**FINDING: the trigger fires on a number the reformulation is guaranteed to
raise.** Round 2 shifts 0.1 of weight into the image retriever, whose span is
**1.0** against 0.43 and 0.43 — the widest of the three. Moving weight into the
widest-spread retriever raises the top fused score *by construction*. A
thermostat wired to its own heater.

**FINDING: and it stops below its own floor without saying so.** At a floor of
0.99 the loop runs two rounds, reaches 0.734 and returns. No third round, no
signal that the threshold was never met — a caller reading the answer cannot tell
a satisfied loop from an exhausted one.

**ANSWER: so the threshold is a property of the retrieval, not of the query.**
Exercise 4 measures first-round fused recall@2 at **1.00**, so no reformulation
has anything to add and every extra round is pure tax. The condition to test
*before* building an agentic loop is first-round recall; if it is already 1.0,
query difficulty is not the variable.

### 1 — a medical-triage multimodal RAG

Drawing on **Cross-modal retrieval**, which lists the three patterns — shared
embedding space, per-modality encoder plus translator, and VLM-as-encoder — this
design has to choose between.

**The query is a photo plus symptom text; the knowledge bases are not
symmetrical, and that asymmetry is the design.**

| query modality | retrieves from | with | why |
|---|---|---|---|
| **photo of injury** | a labelled **image** atlas (wound types, rashes, fracture presentations) | image→image, a medical vision encoder | CLIP-style text→image fails here: "erythematous plaque" is not in a web-image-caption distribution |
| **photo of injury** | **text** triage protocols | VLM-as-encoder | the photo has to reach *text* rules, and only a VLM's hidden states can express "this looks like a second-degree burn" as a retrieval key |
| **symptom text** | **text** protocols and drug interactions | ordinary text retrieval | no cross-modal step needed, and this is the leg that must never be degraded by the other two |
| **symptom text** | the image atlas | text→image | weak and worth keeping anyway: it is the leg that catches "the photo is uninformative but the words are not" |

**Three things this domain forces that the lesson's restaurant example does
not.**

1. **Retrieval must be recall-first and abstention-capable.** A missed
   differential is not a worse answer, it is a wrong one. That argues for a wide
   first stage and an explicit "insufficient evidence — escalate" output, which is
   Lesson 12.22's exercise 5 in a different domain: the value of the second
   system is that it can *disagree*, and disagreement is a signal to stop.
2. **Fusion cannot be a fixed weighted sum.** Exercise 2 shows what happens when
   a modality has nothing to say: a constant takes its share of the weight. In
   triage the photo is sometimes uninformative (bad lighting, wrong angle,
   clothing in the way) and the correct behaviour is to fall back to the text leg
   entirely — which requires a gate that can read *how much* signal the image
   retriever found, not just its scores.
3. **Grounding is the deliverable, not the answer.** The output is a ranked
   differential with a citation per item and a stated confidence, because a
   clinician is the consumer and the system's job is to surface candidates rather
   than to conclude. Exercise 4's citation-coverage metric is the one that
   matters most here, and it is the one that degrades silently.

**And the thing not to build.** A single shared embedding space across photo,
symptom text and protocol text — pattern 1. It is the cheapest option and it
fails precisely on the medical vocabulary that matters, because CLIP-family
training pairs are web captions and the distinctions that separate two
differentials were never in them.

### 3 — the three canonical sub-problems

Drawing on **The 2025 surveys**, which names Abootorabi et al. as the broadest
taxonomy and says it "covers retrieval, fusion, generation".

**The three are retrieval, fusion and generation, and the lesson's own section
ordering follows them.** What each one actually contains:

1. **Cross-modal retrieval** — getting documents of modality B from a query of
   modality A. The open problem is that a shared space (CLIP, CLAP) only spans
   the pairs it was trained on, so every modality outside those pairs needs
   either a translator or a VLM's hidden states, at a cost the lesson's own
   §"Cross-modal retrieval" lays out as three patterns.
2. **Fusion** — merging *k* results of mixed modality into one ranked set. This
   is the sub-problem exercise 2 takes apart: a weighted sum cannot represent a
   modality with no opinion, cannot normalise across scales it never sees, and
   produces a score that is not comparable to anything.
3. **Generation grounding** — producing an answer that cites the retrieved
   evidence rather than the model's prior, across modalities where "cite" means
   different things (a text span, an image region, a timestamp).

**How they map to the medical triage product of exercise 1:**

| sub-problem | what it is in triage | hardest part |
|---|---|---|
| retrieval | photo → atlas, photo → protocol text | the photo→text leg; no shared space covers it |
| fusion | combining an image match with a symptom match | knowing when the photo is uninformative |
| generation | a ranked differential with per-item citations | citing an image *region*, not just an image |

**And the honest observation the lesson makes and is worth repeating.** "Most of
the sub-problems are still open" — which is visible in its own code: three
retrievers on incomparable scales, a fusion with fixed weights, and an agentic
loop whose confidence is an unnormalised sum. The taxonomy is useful because it
names where the gaps are, not because the gaps are closed.
