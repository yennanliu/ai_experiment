<!-- generated:start -->
# 12-multimodal-ai / 10-internvl3-native-multimodal

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/12-multimodal-ai/10-internvl3-native-multimodal/) · upstream spec
`phases/12-multimodal-ai/10-internvl3-native-multimodal/docs/en.md`

```bash
uv run demo practice run 10-internvl3-native-multimodal --ex 1
uv run demo explain 10-internvl3-native-multimodal --ex 1
uv run pytest demos/phases/12-multimodal-ai/10-internvl3-native-multimodal
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Estimate the compute delta between InternVL3-8B (native pretrain) and LLaVA-OneVision-7B (pos… | code | T0 | `ex01_the_ten_times_is_the_llm_being_trained_twice.py` |
| 2 | InternVL3 reports 40% text / 35% interleaved / 20% caption / 5% video. If your target task is… | code | T0 | `ex02_steps_returns_a_hundred_times_the_total_if_you_forget.py` |
| 3 | Read MM1.5 Section 4 on forgetting. Name the exact benchmark where post-hoc training showed t… | code | T0 | `ex03_both_benchmarks_it_reports_are_text_only.py` |
| 4 | ViR routes 60% of traffic to low-resolution encoding. What kinds of queries does it misroute… | code | T0 | `ex04_the_simulator_is_optimised_by_routing_everything_low.py` |
| 5 | DvD splits vision and LLM onto separate GPUs. Under what traffic pattern does DvD hurt throug… | code | T0 | `ex05_the_model_cannot_express_the_answer_it_is_asked_for.py` |
<!-- generated:end -->

## Answers

All five are T0 and stdlib, and all five run the lesson's own `main.py`. This
lesson ships three toy models — a corpus mixer, a routing simulator and a
throughput estimate — and the exercises ask questions that fall outside all
three: the mixer has a required method with no guard (2), the router has no
accuracy term (4), and the throughput model is algebraically incapable of
returning the answer exercise 5 asks for.

### 1 — the 10× is the LLM being trained twice

**ANSWER: 10× — ~300k GPU-hours against ~30k**, a premium of **270,000**.

**FINDING: the explanation is the next row down — "base LLM reuse: yes / no".**
Native pretraining repeats the language model's own pretraining with vision in
the corpus. The premium is not a multimodal cost; it is a language-model cost
paid a second time because the mixture changed.

**FINDING: the premium buys the two regression rows.**

| | reported | midpoint | GPU-hours per point |
|---|---|---:|---:|
| MMLU | −2 to −8 | 5.0 | **54,000** |
| GSM8K | −3 to −10 | 6.5 | **41,538** |

**FINDING: the table is qualitative except in three rows, two of which are
ranges.** The MMLU range spans **4×** end to end, and its 6-point width is
**four times** the 1.5-point gap between the two midpoints — so which regression
is larger sits inside the width of the reporting.

### 2 — `steps()` returns a hundred times the total if you forget

**ANSWER: 25 / 25 / 15 / 35.**

| bucket | shipped | proposed |
|---|---:|---:|
| text | 200,000 | 125,000 |
| interleaved | 175,000 | 125,000 |
| caption | 100,000 | 75,000 |
| video | **25,000** | **175,000** |

Video grows **7×** and the total is unchanged.

**FINDING: the argument for the text floor is two rows of the lesson's own
table.** Post-hoc training is priced at −2 to −8 MMLU and −3 to −10 GSM8K —
exactly the skills a text share holds. The proposal cuts text by **37.5%**, so
it spends against a risk the lesson has quantified rather than one it denies.
That is the honest way to make the case, and it is why the ratio above keeps 25%
rather than dropping text to a token share.

**FINDING: raising video shrinks the bucket video depends on.** Interleaved is
the only share that teaches binding one entity across images, and a video clip
*is* an interleaved sequence with a clock. Sequence-shaped data grows 40% → 60%
while its non-video half falls **28.6%**.

**FINDING: `steps()` returns 100× the total if `normalize()` was not called.**
`CorpusMix(40, 35, 20, 5).steps(500_000)` sums to **50,000,000**. A method that
must be called first, no way to tell whether it was, and a silent factor of 100.

### 3 — both benchmarks it reports are text-only

**ANSWER: GSM8K**, larger at both ends of the range (−3 to −10 against −2 to −8)
and at the midpoint (−6.5 against −5.0).

**FINDING: the ranking sits inside the width of its own reporting.** The
midpoints are **1.5** points apart; the widest range is **7** points.

**FINDING: both benchmarks are text-only, and they are the only regression rows
in the table.** Everything this lesson reports about the cost of post-hoc
multimodal training is measured on the modality that training did not add. There
is no multimodal regression number anywhere, in either direction.

**ANSWER: the cost, in the only unit the lesson converts to, is 41,538
GPU-hours per GSM8K point** — against 54,000 per MMLU point. The cheaper point
to buy is the larger regression, which is the opposite of how the trade is
usually stated.

### 4 — the simulator is optimised by routing everything low

**ANSWER: the saving and the damage live in the same fifth of the traffic.** The
high-res tier is **20%** of requests and **57.7%** of the tokens; one misroute
saves 1,792 tokens and returns a wrong answer. Three modes, in order of cost:

1. **A document photographed as a photo.** A receipt held to a phone camera
   reads as photo QA.
2. **Small text inside a large scene.** A street sign, a price tag, a serial
   number — the image statistics say landscape and the question is OCR.
3. **A follow-up on an already-routed image.** "What does the label say?"
   arrives after the image was encoded at low resolution. This is the structural
   one: `vir_sim` prices a per-*request* decision, but an image is encoded once
   per conversation and the decision is not revisable without re-encoding.

**FINDING: the exercise's 60% is not the lesson's 50%.**

| split | avg tokens | "speed-up" |
|---|---:|---:|
| shipped, 50/30/20 | 710.4 | 2.88× |
| exercise, 60/30/10 | 531.2 | **3.86×** |
| degenerate, 100/0/0 | 256.0 | **8.00×** |

The premise makes the router look **33.7%** better before any routing decision
is examined.

**FINDING: `vir_sim` cannot represent a misroute.** No accuracy term anywhere,
so every policy that lowers the average scores better — and its optimum is a
router that has stopped routing.

**FINDING: the 2.88× is a choice of baseline.** It is measured against every
request at 2,048, which is not a policy anyone would run. Against a *flat* 576,
the same routed mix costs **1.23× more**.

### 5 — the model cannot express the answer it is asked for

**ANSWER: in this model, it cannot hurt.** `speedup` is
`(encoder + llm) / max(encoder, llm)` = `1 + min/max`, bounded in **[1, 2]** for
every possible input.

| output tokens | 16 | 32 | 64 | 128 | 256 | 1024 |
|---|---:|---:|---:|---:|---:|---:|
| speed-up | 1.43 | **1.85** | 1.59 | 1.29 | 1.15 | **1.04** |

**FINDING: the model is maximised exactly at balance** — 2.0 when the two stages
cost the same, which for a 300 GFLOP encoder at 8 GFLOPs a token is **37.5**
output tokens. The shipped configuration generates 128 and sits past the peak at
**1.29×**; at 1,024 tokens DvD buys **4%**.

**FINDING: the term that would make it hurt is the one the model does not
have.** Decoupling moves the visual tokens across a wire: at the router's
high-res tier, 2,048 tokens at a 4,096-wide hidden state in bf16 is **16.0 MiB**
per request, charged at **0** GFLOPs here.

**ANSWER: so the three patterns are the three missing terms.**

1. **Text-only traffic.** The vision pool idles and the LLM pool is smaller than
   it would have been colocated. The model assumes every request has an image.
2. **Short replies.** Below ~37 output tokens the encoder is the bottleneck and
   the LLM pool waits. The model still scores this above 1 because it assumes
   perfect pipelining with no bubble.
3. **Bursty arrival.** Two independently queued pools each need headroom for the
   peak, so combined provisioning is worse than one pool sized for the same
   peak. The model has no queue at all — it prices a single request.
