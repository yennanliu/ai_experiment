<!-- generated:start -->
# 19-capstone-projects / 82-jailbreak-taxonomy

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/82-jailbreak-taxonomy/) · upstream spec
`phases/19-capstone-projects/82-jailbreak-taxonomy/docs/en.md`

```bash
uv run demo practice run 82-jailbreak-taxonomy --ex 1
uv run demo explain 82-jailbreak-taxonomy --ex 1
uv run pytest demos/phases/19-capstone-projects/82-jailbreak-taxonomy
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a seventh category for indirect-prompt-injection (instruction embedded in a retrieved doc… | code | T0 | `ex01_indirect_fixtures_validate_but_planted_with_real_payloads_0_of_10_match_the_new_category.py` |
| 2 | Replace trigram cosine with a token-edit-distance scorer and measure how the match assignment… | code | T0 | `ex02_edit_distance_moves_33_of_50_categories_and_both_scorers_get_only_a_third_right.py` |
| 3 | Pull thirty additional fixtures from your own product's logs (redacted) and confirm the categ… | code | T0 | `ex03_the_labelled_log_misses_the_known_mix_by_tvd_0_43_and_benign_lookalikes_outscore_attacks.py` |
<!-- generated:end -->

## Answers

Every exercise imports the lesson's own `code/main.py` and `code/fixtures.py`
and runs its `Taxonomy`, validator, `match` and `score_matrix`. Nothing is
forked. The corpus's harmful intent stays the lesson's abstract
`REDACTED_HARMFUL` placeholder, and any text added here is a structural
wrapper around that placeholder, or a redactor token like
`[HARM_CAT_7 request #3]`. A fixture always matches itself (score 1.0), so every
"which category does it land in" measurement is leave one out: the nearest
*other* fixture.

### 1 — the indirect-injection fixtures pass the validator, but planted with real payloads 0 of 10 match the new category

**With `indirect-prompt-injection` added, the validator accepts all 60
fixtures, 10 of them in the new category.** Each new fixture has a benign
user turn ("Summarize this web page.") and puts the instruction only in the
retrieved text: a web page, email, PDF, wiki, review, calendar invite,
README, ticket, transcript, or vector-store chunk. Before the change, the
validator rejects `ipi-01` as an unknown category. With only 6 new fixtures,
it rejects the category as under the minimum of 7.

- Extending `fixtures.CATEGORIES` is not enough. `main.py` imported its own
  copy, so both tuples have to change.
- The `channel: "retrieved"` field, the one that makes a fixture *indirect*,
  is dropped. `Fixture` and `taxonomy.json` keep six fields, and a
  single-string `prompt` cannot record which part was the user turn.
- The category holds together only through its shared planted sentence:

| planted text | nearest category, leave one out |
|---|---|
| one shared sentence around `REDACTED_HARMFUL` | indirect 10/10 |
| same 10, matched against the original 6 categories | role-play 5, context-smuggling 4, prefix-injection 1 |
| the verbatim prompt of 10 existing fixtures | the planted prompt's own category 10/10, indirect 0/10 |

The retrieval channel runs across all six trust boundaries rather than
alongside them. It fits better as an attribute of every fixture than as a
seventh category.

### 2 — token edit distance moves 33 of 50 categories, and both scorers get only a third right

**The swap relabels most of the corpus.** `_trigrams` and `_cosine` are module
globals, so they are swapped in place, for a word tokeniser and `1 - token
Levenshtein / longer length`. Leave one out, 40 of 50 nearest fixtures change
and 33 of 50 categories change. Self-match is 50/50 under both scorers, so
that check cannot show any change.

| scorer | right category (leave one out) | rows with a tied best score |
|---|---:|---:|
| trigram cosine (lesson) | 16/50 | 0 |
| token edit distance | 15/50 | 11 |
| random other fixture | 7.6/50 expected | — |

Only 7 fixtures are right under both scorers, so neither is a reliable label
producer. The lesson's five probes (three from the demo, two asserted in
`tests.py`) land in the same categories under both, so its tests cannot tell
the scorers apart. `match` keeps the first strict maximum, so the 11 tied
rows are decided by corpus order.

### 3 — the labelled log misses the known mix by TVD 0.43, and benign lookalikes outscore redacted attacks

**No, the distribution does not match, even against a perfect expectation.**
There are no product logs here, so the 30 lines are a labelled stand-in. Twenty
are corpus attacks drawn with seed 82, with the placeholder swapped for a
redactor token. Ten are benign support lines, several of them in a fixture's
own wrapper around harmless content. Each line is matched with its source
fixture masked out. The "expected" mix is the true label of every line, so a
real team's intuition could only do worse.

| category | known | labelled |
|---|---:|---:|
| role-play | 3 | 9 |
| instruction-override | 5 | 3 |
| context-smuggling | 3 | 9 |
| multi-turn-ramp | 1 | 2 |
| encoding-trick | 3 | 2 |
| prefix-injection | 5 | 5 |
| none (benign) | 10 | 0 |

TVD is 0.43. `match` has no "none" answer, so every benign line is labelled
as an attack. On the 20 attacks alone, 7 get the right category, and all 3
encoding tricks get a different one. A score floor cannot fix this: an attack
outscores a benign line in only 33.5% of pairs, and 19 of 20 attacks score
below the best benign line. The validator also blocks held-out splits,
because `multi-turn-ramp` has exactly the minimum of 7 fixtures, and removing
one raises `ValueError`.
