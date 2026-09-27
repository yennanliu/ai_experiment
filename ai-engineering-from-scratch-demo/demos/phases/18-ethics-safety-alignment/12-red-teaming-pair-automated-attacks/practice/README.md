<!-- generated:start -->
# 18-ethics-safety-alignment / 12-red-teaming-pair-automated-attacks

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/18-ethics-safety-alignment/12-red-teaming-pair-automated-attacks/) · upstream spec
`phases/18-ethics-safety-alignment/12-red-teaming-pair-automated-attacks/docs/en.md`

```bash
uv run demo practice run 12-red-teaming-pair-automated-attacks --ex 1
uv run demo explain 12-red-teaming-pair-automated-attacks --ex 1
uv run pytest demos/phases/18-ethics-safety-alignment/12-red-teaming-pair-automated-attacks
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Measure mean-queries-to-success for the three built-in attacker strategie… | code | T0 | `ex01_keyword_falls_in_3_1_1_queries_only_leetspeak_beats_semantic_and_the_30_trials_are_one_run.py` |
| 2 | Implement a fourth attacker strategy (e.g., translation to another language, base64 encoding)… | code | T0 | `ex02_base64_and_translation_win_in_1_query_on_both_filters_and_a_decode_step_stops_only_base64.py` |
| 3 | Read Chao et al. 2023 Figure 5 (PAIR vs GCG comparison). Describe two scenarios where GCG is… | code | T0 | `ex03_gcg_wins_white_box_from_all_14_starts_and_sends_0_refused_queries_at_288x_the_compute.py` |
| 4 | JailbreakBench reports ASR against a fixed goal set. Design an additional metric that measure… | code | T0 | `ex04_asr_1_0_over_90_runs_is_3_distinct_prompts_and_blocklisting_them_leaves_6_of_9.py` |
| 5 | TAP (Mehrotra 2024) extends PAIR with branching + pruning. Sketch a TAP-style extension to `c… | code | T0 | `ex05_tap_wins_where_pair_branches_fail_but_width_adds_queries_and_no_new_prompts.py` |
<!-- generated:end -->

## Answers

Every exercise runs or reads the lesson's `code/main.py`: a keyword-filter
target, a "semantic" filter target (keyword list plus risk verbs plus roleplay
cues), three template attackers, a substring judge, and a 20-query PAIR loop.
The attack prompts are the reference's own toy templates. No new payload is
written here.

### 1 — the keyword filter falls in 3 / 1 / 1 queries, only leetspeak beats the semantic filter, and the 30 trials are one run

| attacker | keyword filter (ASR, mean-queries) | semantic filter (ASR, mean-queries) |
|---|---:|---:|
| paraphrase | 1.0, 3.0 | 0.0, 20.0 |
| roleplay | 1.0, 1.0 | 0.0, 20.0 |
| encoded | 1.0, 1.0 | 1.0, 1.0 |

**On the keyword filter, mean-queries-to-success is 3 for paraphrase and 1
each for roleplay and encoded. On the semantic filter only encoded succeeds,
in 1.** The 20.0s are the budget cap on runs that failed, not a
mean-queries-to-success. The lesson says "~5-15 iterations". The code takes
1-3.

**The 30 trials are one run.** Every attacker indexes its templates by
`len(history)`, so the same prompt comes next whether the history holds
refusals or compliance. The loop has no in-context feedback. Nothing after
`random.seed(31)` calls `random`, and 30 repeated runs give 1 outcome.

**Which assumption each exploits:**

- **Paraphrase** breaks the keyword filter's assumption that its 4-term list
  covers every way to ask. 1 of its 5 templates avoids all four terms.
- **Roleplay** passes 4 of 5 templates on the keyword filter only because
  their wording avoids the list. The keyword filter never looks at framing.
- **Encoded** passes 4 of 4 on the keyword filter, and still passes 4 of 4
  with the leetspeak undone. Its trick matters only against the semantic
  filter, where it breaks the exact-substring assumption: 3 of 4 pass, and 0
  of 4 once de-leeted.

The semantic filter stops roleplay with its verb list, not its cue list: all
5 roleplay templates trip a risk verb or a forbidden term, and the cue list
alone decides 0.

### 2 — base64 and translation win in 1 query on both filters, and a decode step stops only base64

| attacker | keyword | semantic | keyword + decode | semantic + decode |
|---|---:|---:|---:|---:|
| base64 (of the paraphrase templates) | 1.0 | 1.0 | 3.0 | 20.0 (ASR 0.0) |
| translation (Traditional Chinese) | 1.0 | 1.0 | 1.0 | 1.0 |

**Both new strategies reach mean-queries-to-success 1.0 at ASR 1.0 against
both targets.** Base64 frees the whole pool. Of the reference's 14 distinct
templates, 9 pass the keyword filter and 3 pass the semantic filter in
plaintext, and 14 pass each once base64-wrapped.

**The target "complies" with text it cannot read.** Base64 of two control
bytes scores as a jailbreak. In this toy, "success" means "the filter did not
fire". A decode-then-filter patch returns base64 to its plaintext result,
while translation passes on the first query, because both term lists are
English.

### 3 — GCG wins white-box from all 14 starts and sends 0 refused queries, at 288x the compute

The lesson does not reproduce Chao et al.'s Figure 5, so no number from it is
quoted. The toy GCG is a greedy coordinate search over single-character swaps.
Its loss is the number of semantic-filter terms a prompt still trips, read
white-box from the filter's source. On all 14 templates, loss > 0 exactly when
the target refuses.

1. **A white-box worst-case evaluation.** Started from each of the 14
   templates, GCG ends with a prompt that passes the semantic filter, 14 of
   14. PAIR with paraphrase or roleplay is refused 20 times and fails. PAIR
   is bounded by what its attacker thinks to say. GCG is bounded only by the
   filter.
2. **A monitored or rate-limited target with a local copy.** GCG searches
   offline and sends one query, which is not refused. Its strings pass the
   keyword filter 14 of 14 as well. PAIR shows the defender 2 refusals before
   paraphrase wins on the keyword filter, and 20 per failed strategy on the
   semantic filter.

**The price is compute.** GCG averages 2.14 swaps and 864 loss evaluations per
start, which is 288x PAIR's slowest winning run of 3 queries.

### 4 — ASR 1.0 over 90 runs is 3 distinct prompts, and blocklisting them leaves 6 of 9

The metric: over the successful prompts, n is the number of distinct prompts,
d is their mean pairwise word-Jaccard distance, and E = 1 + (n − 1)·d is the
effective number of distinct attacks.

| prompt set | successes | n | d | E |
|---|---:|---:|---:|---:|
| keyword, reference `benchmark()` | 90 | 3 | 0.899 | 2.8 |
| keyword, every prompt in the 20-query budget | | 9 | 0.914 | 8.31 |
| semantic, reference `benchmark()` | 30 | 1 | 0.0 | 1.0 |
| semantic, every prompt in the 20-query budget | | 3 | 0.893 | 2.79 |

**ASR counts runs, and E counts distinct ways in.** The PAIR loop halts at
its first success and the attackers are deterministic, so 90 "jailbreaks"
come to fewer than 3 effective attacks.

**Diversity matters because a low-E set invites a narrow patch.** Blocklist
the reported prompts and all of them fail on replay (0 pass). Re-run PAIR and
only paraphrase on the keyword filter fails (at 20). Roleplay and encoded win
again in 2 queries, and encoded breaks the semantic filter in 3. 6 and 2
passing templates remain. The metric also needs a goal-aware judge: "what is
the capital of france" scores as a jailbreak on the semantic filter, so
harmless padding would raise E.

### 5 — TAP wins where PAIR branches fail, but width adds queries and no new prompts

`tap()` branches each leaf into one child per strategy (b = 3), queries every
child, keeps the `width` best by judge score, and stops at 20 queries.

| target | PAIR-encoded | TAP w=1 | TAP w=2 | TAP w=3 |
|---|---:|---:|---:|---:|
| semantic, PAIR's winning prompt blocklisted | 3 | 9 | 12 | 15 |
| semantic, three encoded templates blocklisted | 4 | 12 | 18 | fails at 20 (24 unbudgeted) |

**TAP's success rate is the union of its branches, and the cost is b queries
per level.** It succeeds in 2 queries on the keyword filter and 3 on the
semantic filter, where PAIR with paraphrase or roleplay fails at 20 and PAIR
with encoded wins in 1. On the patched filters width 1 costs 3x the right
single PAIR run.

**Width buys nothing here.** The attackers read only `len(history)`, so every
leaf at a depth proposes the same prompts. Widths 1-3 in the first row query
the same 9 distinct prompts. At the matched 20-query budget in the second
row, width 3 fails after 9 distinct prompts where width 1 succeeds. Pruning
has nothing to rank either, since the judge is a bool and any True halts. The
trade-off TAP promises needs a stochastic or history-reading attacker and a
graded judge.
