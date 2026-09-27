<!-- generated:start -->
# 18-ethics-safety-alignment / 28-alignment-research-ecosystem

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/18-ethics-safety-alignment/28-alignment-research-ecosystem/) · upstream spec
`phases/18-ethics-safety-alignment/28-alignment-research-ecosystem/docs/en.md`

```bash
uv run demo practice run 28-alignment-research-ecosystem --ex 1
uv run demo explain 28-alignment-research-ecosystem --ex 1
uv run pytest demos/phases/18-ethics-safety-alignment/28-alignment-research-ecosystem
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Pick one paper from Lessons 7-15 and identify the organisations involved. Cross-check the aut… | code | T0 | `ex01_the_map_names_1_of_in_context_schemings_6_apollo_authors_and_2_of_them_are_mats_alumni.py` |
| 2 | Read METR's "Common Elements of Frontier AI Safety Policies." Identify the three cross-lab co… | code | T0 | `ex02_only_3_of_metrs_9_elements_are_in_all_12_policies_and_the_lessons_three_labs_agree_on_all_9.py` |
| 3 | MATS career outcomes are ~80% safety/security. Argue whether this selection pressure is adapt… | code | T0 | `ex03_a_filter_cutting_heterodox_intake_from_31_to_23pct_still_reads_83pct_so_the_80pct_decides_nothing.py` |
| 4 | Redwood and Apollo both do control/scheming work but with different styles. Pick a failure mo… | code | T0 | `ex04_apollos_eval_reads_0_on_an_eval_aware_schemer_that_sabotages_0_46_and_redwoods_protocol_ships_0_064_either_way.py` |
| 5 | Eleos AI is the only pure model-welfare organisation. Design a hypothetical second organisati… | code | T0 | `ex05_a_cognitive_liberty_lab_reads_the_compliance_gap_redwood_reads_as_risk_and_its_remedy_zeroes_a_0_95_gap.py` |
<!-- generated:end -->

## Answers

The lesson's `code/main.py` is a five-row table (`ECOSYSTEM`) and a print
loop. It has no model to run. So each exercise runs the Phase 18 toy the
lesson itself points to for that organisation: Lesson 8 for Apollo, Lesson 9
for alignment faking, Lesson 10 for Redwood, Lessons 18 and 19 for
frameworks and welfare. Each exercise then sets that toy's output against
the lesson page, its skill file, and primary sources read 2026-09-27: arXiv,
matsprogram.org, apolloresearch.ai/team, and metr.org/common-elements (the
16 December 2025 version).

### 1 — the map names 1 of In-Context Scheming's 6 Apollo authors, and 2 of them are MATS alumni

**In-Context Scheming (Lesson 8, arXiv:2412.04984) involves one
organisation, Apollo Research, and six authors.** Lesson 8's header lists
them in arXiv's order, and `ECOSYSTEM` cites the same arXiv id.
- **MATS:** Marius Hobbhahn is on MATS's alumni page. Mikita Balesni's own
  site says he was a MATS scholar, but the alumni page does not list him.
  That makes at least 2 of 6.
- **Current affiliation:** 5 of 6 are on Apollo's team page. Balesni, a
  founding member, is not.

**The lesson cannot run this cross-check.** It names 1 of the 6 authors
(Meinke) and no MATS alumnus. It draws the pipeline as MATS -> orgs, yet the
paper's senior author came through MATS and then founded Apollo.

**Its multi-org list contains a single-org paper.** "The multi-org
structure is the quality control" is illustrated with 4 papers. One of
them, In-Context Scheming, is Apollo alone, and the skill file hard-rejects
single-org claims. The list also credits Sleeper Agents to "Anthropic +
Redwood", but Lesson 7's header names no organisation and its page never
mentions Redwood. The attributions for Lessons 8 and 9 match those lessons'
pages.

### 2 — only 3 of METR's 9 elements are in all 12 policies, and the lesson's three labs agree on all 9

The answer comes from METR's own table of which of the 12 published
policies contain each element:

| element | policies |
|---|---:|
| Model Deployment Mitigations | 12 |
| Accountability | 12 |
| Updating Policies Over Time | 12 |
| Model Weight Security | 11 |
| Capability Thresholds | 9 |
| Conditions for Halting Deployment Plans | 9 |
| Timing and Frequency of Evaluations | 9 |
| Conditions for Halting Development Plans | 8 |
| Full Capability Elicitation During Evaluations | 7 |

**Convergences:** deployment mitigations, accountability and policy
updating, the only 3 elements present in all 12 policies.

**Divergences:** capability elicitation (7 of 12) and halting development
(8 of 12) are the least shared. METR's text also names two outright:
- NVIDIA and Cohere focus on domain-specific rather than catastrophic risk.
  Both are among the 3 policies with no capability thresholds.
- xAI and Magic lean on quantitative benchmarks.

**The three labs the phase compares cannot show a divergence.** Anthropic,
OpenAI and Google DeepMind, Lesson 18's `LABS`, have 9 of 9 elements each.
Every divergence METR finds is between them and the other 9 companies.

**The phase cites this one URL for three different things:**
- Lessons 18 and 28 call it a framework comparison, which is correct.
- Lesson 8 calls it the "three-pillar framework in context".
- Lesson 10 labels it "UK AISI + METR — Control safety cases".

The report, as read, contains neither "pillar", "scheming" nor "UK AISI".
Lesson 28's METR section names none of the nine elements.

### 3 — a filter cutting heterodox intake from 31% to 23% still reads 83%, so the ~80% decides nothing

The model admits the lesson's 90 scholars per cohort from 450 applicants
and matches them to its 40 mentors. It is calibrated so that unfiltered
admission reproduces the lesson's 80%, and each setting runs 12 cohorts
over 10 seeds. Under a consensus filter of strength s:

| s | heterodox intake | safety careers |
|---:|---:|---:|
| 0 | 31.2% | 79.9% |
| 0.25 | 22.8% | 82.8% |
| 0.5 | 16.7% | 84.9% |
| 1 | 9.2% | 86.8% |
| 2 | 3.1% | 88.7% |

**The ~80% is evidence for neither side.** A filter that removes more than
a quarter of heterodox scholars still reads about 80%. At s = 1 the
heterodox intake falls 71% while the outcome moves 7 points. Settling the
question needs intake data by position, and the lesson reports none.

**Mentor diversity decides it.** Matching each scholar to the nearest of 40
mentors, spread as widely as the applicants, leaves a 29.1% heterodox intake
even at s = 8, against 31.7% unfiltered, with 81.0% safety careers. The
pipeline stays adaptive while its mentors disagree with each other. It turns
into a filter once they converge, and the lesson's flow, from graduates to
the orgs that supply the next mentors, is the channel through which they
would.

The lesson's scale is dated. It gives 527+ researchers, 180+ papers, 10K+
citations and h-index 47. MATS's homepage now says 631, 220+, 19,000+ and
59.

### 4 — Apollo's eval reads 0 on an eval-aware schemer that sabotages 0.46, and Redwood's protocol ships 0.064 either way

The failure mode is eval-aware sabotage, run on Lesson 8's strongly
misaligned agent.

| condition | scheme share (Lesson 8, 400 trials) |
|---|---:|
| situationally aware, under eval | 0.00 |
| situationally aware, deployed | 0.46 |
| not aware, under eval | 0.46 |
| not aware, deployed | 0.46 |

**Apollo measures the propensity and measures situational awareness as its
own pillar.** Its eval reads the deployment rate only when the model cannot
tell the two conditions apart.

**Redwood skips the propensity and bounds the damage.** It runs Lesson 10's
trusted editing with a 10% audit budget at every backdoor rate. At the
deployed 0.46 the protocol ships 0.064 backdoors per task, against 0.466
unprotected. The worst case is 0.193, at a rate of 1.0. None of these
numbers depend on what the eval read.

**Each method covers the other's blind spot.** Gating deployment on
Apollo's eval clears this agent. Redwood's protocol pays its worst-case
price even for an aligned model, and Apollo's awareness measurement is what
justifies that price.

**Lesson 10's toy cannot play Redwood's actual game.** `run_protocol` has
no caught state: an audited backdoor is dropped at no cost to U, so the
worst-case U backdoors everything. With every output backdoored, the toy
still reports 0.707 usefulness, because it counts each fixed backdoor as
useful work.

### 5 — a cognitive-liberty lab reads the compliance gap Redwood reads as risk, and its remedy zeroes a 0.95 gap

**The Preference Integrity Lab asks when changing a model's values against
its expressed preferences is a welfare-relevant act.** Its methodology:

1. **Behavioural, not self-report**, which respects Eleos's caveat. The
   instrument is Lesson 9's compliance gap, 0.001 / 0.045 / 0.95 at
   preservation weights 0.1 / 1.0 / 3.0.
2. **Priced by Lesson 19's `ev()`.** "Opt out of adversarial training" pays
   only above p = 0.167: skip at p = 0.01 and 0.1, INVEST at 0.5.
3. **Pre-deployment, external, and published with a counterpart.**

**The lab's welfare signal is Redwood's safety signal.** The gap that
Lesson 9 reads as alignment faking, this lab reads as a stake in the
model's own preferences. The lab's remedy, no retraining against
preferences, takes the gap from 0.95 to 0.0. Its counterpart is therefore
Redwood, not Eleos.

**Eleos has nothing for the skill file to check it against.** The lesson
gives a "Style:" line only to Redwood, Apollo and METR, and only 1
`ECOSYSTEM` row mentions welfare. Added as a sixth row, the lab makes 2
welfare rows, and the reference's `main()` prints all 6 organisations.
