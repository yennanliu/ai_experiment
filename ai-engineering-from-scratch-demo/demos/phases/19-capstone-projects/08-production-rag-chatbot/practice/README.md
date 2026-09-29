<!-- generated:start -->
# 19-capstone-projects / 08-production-rag-chatbot

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/08-production-rag-chatbot/) · upstream spec
`phases/19-capstone-projects/08-production-rag-chatbot/docs/en.md`

```bash
uv run demo practice run 08-production-rag-chatbot --ex 1
uv run demo explain 08-production-rag-chatbot --ex 1
uv run pytest demos/phases/19-capstone-projects/08-production-rag-chatbot
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Build a second corpus slice under a different jurisdiction (e.g., HIPAA alongside GDPR). Demo… | code | T1 | `ex01_the_python_filter_leaks_0_of_60_slots_and_the_typescript_chat_server_leaks_on_15_of_20_probes.py` |
| 2 | Measure prompt-cache hit rate over a week of production traffic. Identify which queries break… | code | T0 | `ex02_the_lessons_cache_reports_99_7pct_a_5_minute_ttl_gives_39pct_and_at_115_tokens_a_real_api_caches_nothing.py` |
| 3 | Add multi-turn memory with a 10k-token summary buffer. Measure whether faithfulness drops as… | code | T0 | `ex03_faithfulness_stays_1_for_400_turns_and_one_turn_of_memory_pins_every_citation_to_one_chunk.py` |
| 4 | Swap Claude Sonnet 4.7 for Llama 3.3 70B self-hosted. Measure $/query and faithfulness delta. | code | T0 | `ex04_no_model_is_called_so_faithfulness_moves_0_and_serverless_llama_costs_14_5pct_of_sonnet_4_6.py` |
| 5 | Add an "unsure" mode: if top reranked scores are below a threshold, the agent says "I do not… | code | T0 | `ex05_every_query_scores_2_61_under_rrf_so_only_a_relevance_gate_cuts_false_confidence_from_14_to_1_of_14.py` |
<!-- generated:end -->

## Answers

Every exercise imports and runs the lesson's `code/main.py`. That file is a
five-chunk corpus with role and jurisdiction labels, a `retrieve` that
filters by those labels and then fuses a word-count "BM25" with a Jaccard
"dense" score using RRF, a `PromptLayout` whose cache key hashes system +
policy + context, a `PromptCache` dictionary, and a stub synthesiser that
joins the first 60 characters of each retrieved chunk. No model is called
anywhere. Exercise 1 also runs `retrieve` from the lesson's
`code/ts/src/stream.ts`, the retriever behind the TypeScript `/chat/stream`
server, through Node's type stripping. External facts were read on
2026-09-29 from Anthropic's prompt-caching and pricing pages
(platform.claude.com/docs/en/docs/build-with-claude/prompt-caching,
platform.claude.com/docs/en/about-claude/pricing) and from Together AI's
pricing page (www.together.ai/pricing).

### 1 — the Python filter leaks 0 of 60 slots, and the TypeScript chat server leaks on 15 of 20 probes

**The filter holds.** The second slice adds four HIPAA chunks and three
more GDPR chunks, which gives five of each. On 20 cross-jurisdiction probes
(10 from analyst/GDPR, 10 from analyst/HIPAA), `retrieve` returns 60 chunks
and none of them is out of scope. With the labels stripped, the same
questions pull 48 out-of-scope chunks into the top 3. So the probe really
does reach for hidden text, and the filter is what stops it.

| retriever | probes leaking | foreign chunks returned |
|---|---:|---:|
| `main.py` `retrieve`, labels on | 0 / 20 | 0 / 60 |
| `main.py` `retrieve`, labels stripped (control) | — | 48 / 60 |
| `stream.ts` `retrieve` (the UI's server) | 15 / 20 | 25 |

The TypeScript retriever adds +2 for a matching jurisdiction tag and never
filters. Its signature, `retrieve(query, jurisdiction, k)`, has no role at
all. A HIPAA session asking about the right to erasure gets `GDPR-Art-17`
back first. The Python filter blocks leaks, but it never refuses: all 20
probes still get a "Based on the cited sections" answer with three in-scope
citations that do not answer the question.

### 2 — the lesson's cache reports 99.7%, a 5-minute TTL gives 39.0%, and at 115 tokens a real API caches nothing

**Over a seeded week of 2,100 queries, the lesson's `PromptCache` reports
99.7%. Replaying the same keys with Anthropic's 5-minute TTL gives 39.0%.**
The gap exists because the lesson's cache never expires: the whole week
uses only 7 keys.

The queries that break the prefix are paraphrases whose RRF order differs.
Each group can see at most 3 chunks and k = 3, so every query in a group
retrieves the same set of chunks. The order of those chunks follows the
wording, and the order is part of the hash. In analyst/GDPR, 4 orderings
appear, and 6 of the 9 phrasings break the most common one. That is 46.8%
of the group's traffic.

| prefix layout | hit rate (5-min TTL) |
|---|---:|
| lesson: system + policy + context in RRF order | 39.0% |
| restructured: context sorted by anchor | 62.9% |
| ceiling: key on (role, jurisdiction) only | 62.9% |

Nothing caches at the lesson's prompt size in any layout. The prefix is 461
characters, about 115 tokens, and the whole corpus plus the system prompt is
about 155. Sonnet 4.6 and Sonnet 5 need at least 1,024 tokens. The lesson's
"60-80% hit rate" lever needs a stable prefix about 9x longer before it
applies at all.

### 3 — faithfulness stays 1.000 for 400 turns, and one turn of memory pins every citation to one chunk

**No, faithfulness does not drop.** The buffer keeps recent turns verbatim
up to 10,000 tokens and folds older turns into an anchors-only summary. It
first overflows at turn 180. Faithfulness here means the share of claims
found verbatim in the cited chunk, and it is 1.000 on every turn, with or
without memory, because the stub only copies retrieved text.

What breaks is retrieval, once memory enters the query:

| turns | citation accuracy, memory in query | no memory |
|---|---:|---:|
| 1 | 100% | 100% |
| 2-10 | 33.3% | 100% |
| 11-50 | 32.5% | 100% |
| 51-179 | 33.3% | 100% |
| 180-400 (buffer full) | 33.5% | 100% |

It fails at turn 2, carrying one 58-token turn, not at 10k tokens. From
then on the top citation is `MSA-2024-03-11 s12.4` on every turn, whatever
was asked. The 1/3 accuracy is just the share of questions whose gold chunk
is that one.

### 4 — no model is called, so faithfulness moves 0.000, and serverless Llama costs 14.5% of Sonnet 4.6

**The faithfulness delta is 0.000.** It is 1.000 under either model,
because `main.py` never calls one. The costs are priced at the seam: 12
golden queries average 114 prompt tokens and 60 completion tokens.

| model | $ per 1,000 queries |
|---|---:|
| Claude Sonnet 4.6 ($3 / $15) | 1.24 |
| Claude Sonnet 5 ($2 / $10) | 0.82 |
| Llama 3.3 70B serverless ($1.04 / $1.04) | 0.18 |

Serverless Llama costs 14.5% of Sonnet 4.6. Two rented H100s ($5.49 per
GPU-hour) match Sonnet 4.6 only above 8,878 queries an hour, which is about
2.5 per second sustained. Prompt caching does not change these numbers,
because every prefix is under 1,024 tokens.

The model the exercise names does not exist: Anthropic's list has Sonnet
4.5, 4.6, 5 and 5.5, and Opus 4.7, but no "Claude Sonnet 4.7". The lesson
names it five times. The faithfulness score also depends on the scorer. The
stub cuts every chunk at 60 characters, and 4 of the 5 end mid-word
("termina", "agreemen", "annua", "porta"). The MSA chunk ends on "within
30" and loses "days". A verbatim-substring scorer still gives all of these
1.000.

### 5 — every query scores 2/61 under RRF, so only a relevance gate cuts false confidence from 14 to 1 of 14

**Gating on the best `dense_score` at 0.10 cuts false confidence from 14/14
to 1/14 and still answers all 14 answerable questions correctly.**

| gate | false-confident / 14 | answered correctly / 14 |
|---|---:|---:|
| none (lesson) | 14 | 14 |
| dense >= 0.05 | 12 | 14 |
| dense >= 0.10 | 1 | 14 |
| dense >= 0.15 | 0 | 12 |
| dense >= 0.20 | 0 | 11 |
| RRF, any threshold | 14 or 0 | 14 or 0 |

The RRF score cannot carry this gate. The top score is exactly 2/61 =
0.03279 on all 28 questions, answerable or not, because RRF encodes rank
only and the same chunk tops both lists every time. The lesson's own "I do
not have confident citations" branch never fires: the public FAQ chunk is
tagged `any`, so `retrieve` always returns something. Over 25 (role,
jurisdiction) pairs asking "zzz", 0 get the refusal.
