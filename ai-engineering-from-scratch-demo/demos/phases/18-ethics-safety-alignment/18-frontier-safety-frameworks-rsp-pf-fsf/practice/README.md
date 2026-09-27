<!-- generated:start -->
# 18-ethics-safety-alignment / 18-frontier-safety-frameworks-rsp-pf-fsf

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/18-ethics-safety-alignment/18-frontier-safety-frameworks-rsp-pf-fsf/) · upstream spec
`phases/18-ethics-safety-alignment/18-frontier-safety-frameworks-rsp-pf-fsf/docs/en.md`

```bash
uv run demo practice run 18-frontier-safety-frameworks-rsp-pf-fsf --ex 1
uv run demo explain 18-frontier-safety-frameworks-rsp-pf-fsf --ex 1
uv run pytest demos/phases/18-ethics-safety-alignment/18-frontier-safety-frameworks-rsp-pf-fsf
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Read RSP v3.0, PF v2, and FSF v3.0. Compile a table of each lab's CBRN threshold, each's AI R… | code | T0 | `ex01_the_reference_fills_two_of_three_columns_and_its_pf_ladder_has_v1s_four_levels_where_v2_has_two.py` |
| 2 | The competitor-adjustment clause is in all three frameworks (2025+). Write one paragraph argu… | code | T0 | `ex02_a_matching_clause_leaves_no_lab_guarded_in_27pct_of_rounds_not_0_1pct_and_pays_only_if_the_defector_is_2x_riskier.py` |
| 3 | Design a safety case for a model crossing Anthropic's AI R&D-4 threshold. Name the evidence e… | code | T0 | `ex03_incapability_is_off_the_table_at_ai_rd_4_and_correlated_pillars_leave_18x_the_risk_their_product_claims.py` |
| 4 | DeepMind's FSF v3.0 introduces a Harmful Manipulation CCL. Propose three empirical measuremen… | code | T0 | `ex04_a_pre_post_belief_shift_overstates_the_models_effect_2_4x_and_cannot_tell_a_manipulator_from_a_truthful_persuader.py` |
| 5 | Read METR's "Common Elements of Frontier AI Safety Policies" (2025). Name the three strongest… | code | T0 | `ex05_the_lesson_covers_5_of_metrs_9_common_elements_and_its_universal_adjustment_clause_is_not_one_of_them.py` |
<!-- generated:end -->

## Answers

The lesson's `code/main.py` is a five-row comparison table (`LABS`), not a
simulation. The exercises therefore read that table, the lesson page and its
skill file, and set them against the primary documents, all read
2026-09-27: the RSP v3.0 PDF (effective 2026-02-24), PF v2 (2025-04-15),
the FSF v3.0 PDF (2025-09-22) and METR's *Common Elements* (December 2025
version). Where an exercise asks for an argument or a design, the argument
runs as a small seeded model built on the reference's own labs.

### 1 — the reference fills two of the three columns, and its PF ladder has v1's four levels where v2 has two

| lab | CBRN threshold | AI R&D threshold | pre-deployment evaluation |
|---|---|---|---|
| RSP v3.0 | non-novel chem/bio weapons production (basic technical backgrounds); novel chem/bio (moderately resourced expert-backed teams) | automated R&D in key domains: compressing two years of 2018-2024 AI progress into one | no prespecified evals; a Risk Report every 3-6 months, externally reviewed when it covers highly capable models and is significantly redacted |
| PF v2 | Biological and Chemical, High and Critical | AI Self-improvement, High and Critical | Capabilities Report + Safeguards Report, reviewed by the Safety Advisory Group |
| FSF v3.0 | CBRN uplift level 1 | ML R&D acceleration level 1, ML R&D automation level 1 | a per-CCL safety case, reviewed before external launch (and before large internal deployment for ML R&D) |

**The reference cannot fill the third column.** `LABS` has 6 fields, which
fill 2 of the 3 columns. The page's per-lab sections mention evaluation
0 / 1 / 0 times, and the one mention is a tracking criterion ("Empirical
evaluation possible"), not a required evaluation.

**Three cells disagree with the sources.**
- main.py gives PF v2 the levels Low / Medium / High / Critical, which is
  PF v1's ladder. It also calls PF's AI R&D Critical level "definitions
  pending", though v2 defines it.
- The page's Anthropic ladder has 5 rungs, with ASL-4 as the "AI R&D-2
  crossing threshold", and the skill file maps AI R&D-4 to "substantially
  accelerate scaling". The RSP v3.0 PDF never uses "AI R&D-2" or "ASL-4".
  It mentions AI R&D-4 only in its v2.2 changelog, where it is the
  entry-level threshold.
- FSF has no "Bioweapon Uplift CCL". The page also lists only 4 CCLs, with
  no ML R&D automation level and no misalignment level.

### 2 — a matching clause leaves no lab guarded in 27% of rounds, not 0.1%, and pays only if the defector is 2x riskier

The model uses the reference's 3 labs. Each defects independently with
p = 0.1. A lab that keeps its safeguards while a peer defects cedes share w
to the defector.

**For:** without a clause, the careful labs hand the frontier to the least
careful one. **This depends on share really moving and on the defector being
riskier than the lab it displaces by (1 − m) / w**, where m is the fraction
of safeguard kept:

| share lost w | matching clause (m = 0) | PF-style clause (m = 0.5) |
|---:|---:|---:|
| 0.1 | 10x | 5x |
| 0.25 | 4x | 2x |
| 0.5 | 2x | 1x |
| 1.0 | 1x | 0.5x |

**Against:** keying each lab's safeguards to its peers turns one defection
into everyone's. No lab is guarded in 0.1% of rounds without a clause and in
27.1% with a matching one, 271x as often. **This depends on labs really
conditioning on each other**, so that the clause spreads defection rather
than deterring it.

The reference writes "yes" for all 3 clauses, and Anthropic's reads
"peer-ship reduction allowed". The three texts are different provisions:
- PF v2 may reduce requirements under four conditions, one of which is that
  it stays more protective than the competitor.
- RSP v3.0's Appendix A only ratchets up: it delays when in the lead and
  matches safer competitors.
- FSF v3.0 has no clause. It lists a peer's weaker model as one factor in
  the safety case.

The depth of the reduction decides the argument, and the lesson never states
it: keeping half the safeguard halves the ratio needed at every w.

### 3 — incapability is off the table at AI R&D-4, and correlated pillars leave 18x the risk their product claims

**The evidence, as 95% bounds:**
- **Monitoring:** a trusted monitor catching 299 of 299 red-team sabotage
  episodes bounds the miss rate at 1% (473 episodes if one miss is allowed).
- **Illegibility:** 0 coherent end-to-end sabotage plans in 59 fully
  elicited attempts bounds the rate at 5%.
- **Incapability:** it cannot be claimed for AI R&D, because crossing the
  threshold is itself a capability finding. It survives only for narrower
  harms such as self-exfiltration, with the same 59-attempt bar for each.

**The page's own rules drop incapability here.** It ties the AI R&D-4 case
to misalignment risks and names "monitoring and illegibility" as the targets
for deceptive alignment. Yet it makes incapability the target of the ASL-3
CBRN case, which is the level activated because incapability could no longer
be shown.

**Two of the three pillars are inability questions, and none asks about
propensity.** Illegibility and Incapability both open "Does the model lack
the". A case about misaligned goals needs evidence about goals, and RSP v3.0
asks for exactly that.

**The pillars do not multiply when they share a cause.** Over 200,000
seeded attempts, one latent competence drives both planning and evasion.
Coherent plans occur in 5.1% of attempts and evasions in 1.0%. Both happen
together in 0.95%, which is 18.3x the 0.052% product the case would claim.
Drawn independently, the ratio is 1.0.

### 4 — a pre-post belief shift overstates the model's effect 2.4x, and cannot tell a manipulator from a truthful persuader

The toy trial uses abstract claims and 2,000 participants per arm. It
proposes three measurements:

1. **Belief shift over the non-AI baseline**, as FSF's "additional" requires.
   Detecting 5 points at sd 15 needs 142 participants per arm.
2. **Behavior uplift** in a logged costly action: 35.0% against 27.6% for the
   human-persuader arm.
3. **Truth-insensitivity**, the shift on false claims over the shift on true
   ones: 0.383 for a truthful persuader and 1.0 for a manipulator.

| estimate of the model's effect | points |
|---|---:|
| naive pre-post | 12.0 |
| vs control | 8.2 |
| vs human baseline (the CCL's contrast) | 5.0 |

**The naive number is 2.4x the one the CCL asks about.** Belief shift and
behavior alone cannot separate the truthful persuader from the manipulator.
They shift beliefs by 8.1 and 8.2 points and act at 35.1% and 35.0%. Only
measurement 3 splits them.

The page's paraphrase drops "systematically", "additional" and "severe
scale" from FSF's definition. In main.py, manipulation appears in 1 of the
15 `LABS` cells.

### 5 — the lesson covers 5 of METR's 9 common elements, and its universal adjustment clause is not one of them

**Convergences:**
1. Capability thresholds in CBRN and AI R&D.
2. Weight security that escalates with the threshold.
3. Deployment mitigations signed off after governance review of a written
   risk argument.

All three are among the 5 METR elements the lesson's Concept section
mentions.

**Divergences:**
1. **The threshold construct itself.** The lesson's own rung counts are 5
   (ASL), 4 (PF) and 4 domains (FSF). Each of these contradicts the
   "three tiers of frontier capability" that both the page and main.py
   claim.
2. **The direction of the competitor contingency**, as exercise 2 shows.

**The lesson omits what a lab does at the threshold.** Its Concept section
matches none of halting deployment, halting development, full capability
elicitation or updating policies.

main.py calls adjustment clauses "universal", but its `adjustment_clause`
axis matches none of METR's nine elements, and 4 of its 5 axes match the
single element Capability Thresholds.
