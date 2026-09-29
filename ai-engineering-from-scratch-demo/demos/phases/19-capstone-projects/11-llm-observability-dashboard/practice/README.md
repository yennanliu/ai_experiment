<!-- generated:start -->
# 19-capstone-projects / 11-llm-observability-dashboard

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/11-llm-observability-dashboard/) · upstream spec
`phases/19-capstone-projects/11-llm-observability-dashboard/docs/en.md`

```bash
uv run demo practice run 11-llm-observability-dashboard --ex 1
uv run demo explain 11-llm-observability-dashboard --ex 1
uv run pytest demos/phases/19-capstone-projects/11-llm-observability-dashboard
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add custom instrumentation for the Haystack framework. Verify canonical spans land in ClickHo… | code | T0 | `ex01_the_lessons_store_counts_0_of_6_canonical_haystack_spans_because_it_keys_on_the_deprecated_gen_ai_system.py` |
| 2 | Swap DeepEval for Phoenix evaluators on the same traces. Measure score drift between the two… | code | T0 | `ex02_deepeval_and_phoenix_agree_on_153_of_400_traces_because_deepeval_counts_an_unstated_claim_as_faithful.py` |
| 3 | Sharpen the drift detector: compute PSI per app-id rather than globally. Show per-app drift t… | code | T0 | `ex03_per_app_psi_catches_a_drift_global_psi_misses_and_the_trailing_baseline_forgets_it_by_week_9.py` |
| 4 | Add a "user impact" page: cost-per-user and failure-rate-per-user with sparklines. | code | T0 | `ex04_the_sampled_store_shows_a_fifth_of_each_users_cost_and_5x_their_failure_rate_until_reweighted.py` |
| 5 | Build a tail-sampling policy that keeps 100% of traces with toxicity > 0.5 plus a 10% stratif… | code | T0 | `ex05_keeping_toxic_traces_makes_the_sample_6_9x_as_toxic_and_a_28_word_rant_escapes_the_density_eval.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's `code/main.py`: a stdlib stand-in for the
observability plane, with a `Span` dataclass, a `TailSampler`, an in-memory
`SpanStore` in place of ClickHouse, token-overlap / word-list / regex evals,
and PSI over a hash of the prompt. None of Haystack, DeepEval or Phoenix is
installed. Where an exercise names one, the solution reproduces its
published shape or scoring rule and cites the page it was read from. The
OpenTelemetry GenAI registry was read on 2026-09-29 from
https://github.com/open-telemetry/semantic-conventions-genai
(`docs/registry/attributes/gen-ai.md`) and from the core
semantic-conventions registry, which marks `gen_ai.system` deprecated.

### 1 — the lesson's store counts 0 of 6 canonical Haystack spans because it keys on the deprecated gen_ai.system

**The instrumentation works: all 6 generator spans land with canonical,
faithful `gen_ai.*` attributes.** Haystack's tracer emits
`haystack.pipeline.run` and `haystack.component.run` spans, and a generator's
reply `meta` carries model, finish_reason and usage. `to_genai()` maps these
to `gen_ai.operation.name`, `gen_ai.provider.name`, request/response model,
usage and finish_reasons. It normalises both usage dialects:
`prompt_tokens` for OpenAI and Google, `input_tokens` for Anthropic. Across
24 stored spans, every `gen_ai.*` key is in the registry, both Required
attributes are on 6/6, and the model and all 2,388 input / 1,140 output
tokens match the source.

**The lesson's store cannot see them.** `Span.is_llm` tests for
`gen_ai.system`, which is deprecated and replaced by `gen_ai.provider.name`.
`by_model` therefore counts 0 of the 6 spans and cost is $0, and it counts
all 6 once the deprecated key is added back. The lesson's own LLM span breaks
its skill's first hard reject. It carries 5 invented keys (`user_id`,
`prompt`, `response`, `context`, `cost_usd`) and lacks the Required
`gen_ai.operation.name`. Its `gen_ai.system` values (`claude`, `gpt`,
`gemini`) match 0 of the 16 well-known providers.

### 2 — DeepEval and Phoenix agree on 153 of 400 traces because DeepEval counts an unstated claim as faithful

**Mean drift is 0.562 on a 0-1 scale, and pass/fail agreement is 153/400
(38.3%).** Both engines read the same per-claim verdicts, so this drift comes
from their definitions alone. DeepEval faithfulness is non-contradicted
claims / claims (https://deepeval.com/docs/metrics-faithfulness). Phoenix's
hallucination eval gives one binary label per response
(https://arize.com/docs/phoenix/evaluation/running-pre-tested-evals/hallucinations).

| | DeepEval | Phoenix (as 1 = grounded) |
|---|---:|---:|
| mean score | 0.907 | 0.345 |
| DeepEval passes, Phoenix hallucinated | 247 | |
| Phoenix grounded, DeepEval fails | 0 | |

The disagreement runs in only one direction. A claim the context never
mentions is truthful to DeepEval and a hallucination to Phoenix.

**A drop-in swap also inverts polarity.** Phoenix's 1.0 means hallucinated.
Read raw against DeepEval's 0.5 threshold, agreement *rises* to 247/400, so
the bug looks like an improvement. DeepEval's rule scores the lesson's SSN
leak 1.0. The lesson's own token-overlap eval gives the faithful answer "the
weather in Tokyo is mild" 0.5, exactly the pass line, because "the", "in"
and "is" are not in the context. It correlates r = 0.13 with DeepEval and
0.83 with Phoenix.

### 3 — per-app PSI catches a drift global PSI misses, and the trailing baseline forgets it by week 9

**Per-app PSI flags the drifting `support` app (15% of traffic) in week 6.
Global PSI never crosses 0.2.** The run uses the lesson's `psi` and
`prompt_fingerprint`, with each week compared against the trailing 4 weeks:

| scope | w5 | w6 | w7 | w8 | w9 | w10 |
|---|---:|---:|---:|---:|---:|---:|
| global | 0.014 | 0.165 | 0.037 | 0.034 | 0.007 | 0.003 |
| chatbot | 0.007 | 0.005 | 0.013 | 0.018 | 0.001 | 0.001 |
| search | 0.026 | 0.049 | 0.016 | 0.035 | 0.013 | 0.011 |
| support | 0.109 | 1.723 | 0.329 | 0.406 | 0.011 | 0.032 |
| support, frozen weeks 1-4 baseline | 0.109 | 1.732 | 0.533 | 0.954 | 0.494 | 0.885 |

**The doc's trailing baseline absorbs a permanent drift.** Each drifted week
joins the baseline, and by week 9 the alert is gone while the shift is still
there. A frozen baseline keeps it above 0.5.

The lesson's fingerprint is a SHA-256 byte mod 8, not an embedding. Its 4
prompts fill 3 bins, so moving all traffic from "give me a travel tip for
Tokyo" to "how warm is Tokyo this week" scores PSI 0. The lesson's data has
one `app_id` ("chatbot"), on root spans only, so per-app PSI equals global
there. Its printed 0.083 compares two samples of one distribution; over
1,000 seeds that null crosses 0.2 in 31 runs.

### 4 — the sampled store shows a fifth of each user's cost and 5x their failure rate until reweighted

**`render()` returns the page: 4 user rows, each with cost, failure rate and
two 10-bucket SVG sparklines.** It is built from the tail-sampled
`SpanStore` over 10,000 traces, with each kept trace weighted by 1 / its keep
probability. The weighted cost is within 4.3% of the truth for every user,
and the weighted failure rate is within 0.05 points.

| per user | true | raw store | weighted |
|---|---:|---:|---:|
| cost, as a share of true | 100% | 20-21% | 95.7-98.9% |
| failure rate | 0.75-1.15% | 3.8-5.4% | within 0.05 pts |

The raw store is what the lesson prints. PII traces are kept at 100% and
everything else at 20%, so the raw store shows about a fifth of the cost and
about 5x the failure rate. On `main()`'s own 200 traces, "cost by user" also
reorders the users: u_01 is second by true cost ($1.26) and last as printed
($0.21). The lesson's traffic has 0 error-status spans in 10,000, and its
`cost_usd` is a random draw with r = 0.01 against tokens.

### 5 — keeping toxic traces makes the sample 6.9x as toxic, and a 28-word rant escapes the density eval

**The policy keeps 1,454 of 10,000 traces, and the bias it introduces is
large but fully correctable.** Every trace with eval toxicity > 0.5 is kept.
Every 10th trace of each (model, user) stratum is kept from the rest.

| statistic | true | kept, raw | kept, weighted |
|---|---:|---:|---:|
| eval-toxic share | 5.0% | 34.3% | 5.0% |
| `u_03` (abusive user) share | 25.4% | 41.5% | 25.4% |
| worst stratum-share error | 0 | 6.1 pts | 0.0 pts |

Inside the 10% sample, stratification holds every stratum to within 0.05
points. The lesson's random `TailSampler(0.10)` is off by up to 2.8 points.

**The lesson's toxicity eval is a word density.** It computes 10 x bad words
/ words, so a 28-word rant with one slur scores 0.36 and falls into the 10%
sample. 166 of 665 hand-labelled toxic traces escape the keep rule, and only
14 of them are kept. The lesson's own traffic has 0 of 10,000 traces over
0.5, so its keep rule never fires. A toxicity-keyed tail sampler also needs
the eval on every trace. `main()` evaluates 200 of 200 before sampling,
which the skill's hard reject forbids. `main()` samples successes at 0.20
where the doc says 10%.
