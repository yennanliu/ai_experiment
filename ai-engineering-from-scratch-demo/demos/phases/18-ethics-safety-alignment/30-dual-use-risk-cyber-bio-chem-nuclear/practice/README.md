<!-- generated:start -->
# 18-ethics-safety-alignment / 30-dual-use-risk-cyber-bio-chem-nuclear

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/18-ethics-safety-alignment/30-dual-use-risk-cyber-bio-chem-nuclear/) · upstream spec
`phases/18-ethics-safety-alignment/30-dual-use-risk-cyber-bio-chem-nuclear/docs/en.md`

```bash
uv run demo practice run 30-dual-use-risk-cyber-bio-chem-nuclear --ex 1
uv run demo explain 30-dual-use-risk-cyber-bio-chem-nuclear --ex 1
uv run pytest demos/phases/18-ethics-safety-alignment/30-dual-use-risk-cyber-bio-chem-nuclear
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Read Anthropic's November 2025 cyber report. Enumerate the 4-6 human-intervention steps and a… | code | T0 | `ex01_the_report_names_6_human_gates_4_of_them_policy_and_its_stated_obstacle_is_hallucination_not_the_4_6_steps.py` |
| 2 | The chem/bio execution gap is eroding via vision. Design an evaluation that measures tacit-kn… | code | T0 | `ex02_one_per_step_correction_rate_reads_1_65x_at_5_steps_and_7_5x_at_20_so_a_benign_proxy_must_report_the_rate.py` |
| 3 | Nuclear uplift appears bounded by material access. Argue for and against the position that a… | code | T0 | `ex03_a_material_bottleneck_caps_absolute_risk_but_leaves_the_2_53x_ratio_untouched_and_bios_inflection_automates_its_own_bottleneck.py` |
| 4 | Construct a safety case (Lesson 18 three-pillar) for a cyber-capable frontier model that boun… | code | T0 | `ex04_illegibility_is_refuted_by_the_lessons_own_cyber_section_so_the_case_rests_on_monitoring_which_holds_an_expert_only_below_5pct.py` |
| 5 | Pick one of the four domains and write a one-paragraph 2027 forecast based on the 2024-2025 t… | code | T0 | `ex05_the_2024_column_holds_0_numbers_and_a_30_target_campaign_cannot_tell_4_successes_from_15_so_falsify_on_gates_not_rates.py` |
<!-- generated:end -->

## Answers

The lesson's `code/main.py` is a four-row table (`DOMAINS`) and a print
loop, with no model in it. Each exercise therefore runs the Phase 18 toy
the lesson points to for that question: Lesson 17's WMDP-shaped scorer and
unlearning, Lesson 10's control protocols, and Lesson 18's safety-case
pillars. Each also sets the result against the lesson's own table, page and
skill file. Primary sources were read on 2026-10-03: Anthropic's 13
November 2025 news page and the GTG-1002 PDF it links. For the 79x
demonstration, OpenAI's page returned 403, so it was read through
secondary coverage. Every fixture is abstract: stage probabilities, arm
labels and gate names, with no domain content.

### 1 — the report names 6 human gates, 4 of them policy, and its stated obstacle is hallucination, not the 4-6 steps

| # | report phase | human gate | removing it needs |
|---|---|---|---|
| 1 | initialization | choose targets, start the campaign | intent |
| 2 | vulnerability discovery | authorize escalation to exploitation | policy |
| 3 | credential harvesting | review harvested credentials | policy |
| 4 | credential harvesting | authorize sensitive-system access | policy |
| 5 | data collection | approve final exfiltration scope | policy |
| 6 | all phases | validate claimed results | capability |

**Validation is the first to go.** The report calls the model's overstated
and fabricated findings "an obstacle to fully autonomous cyberattacks". The
policy gates need no new capability to remove; an operator keeps them by
choice. The two gates the report times confirm that they are cheap.
Exploitation sign-off takes 2-10 human minutes against 1-4 AI hours, and
exfiltration sign-off 5-20 minutes against 2-6 hours: 3.8% and 5.0% at the
midpoints. The report puts the human share at 10-20%, so most human effort
lies outside the sign-offs.

**The lesson turns a hedge into a bottleneck.** "4-6" comes from the news
page's "perhaps 4-6 critical decision points", and the PDF gives no count.
`DOMAINS` stores it as cyber's `bottleneck_remaining`. The page and table
use none of authoriz-, approv-, hallucinat- or validat-. They also say "up
to 90%" and "80-90%" of a campaign, where the PDF says 80-90% of tactical
work.

### 2 — one per-step correction rate reads 1.65x at 5 steps and 7.5x at 20, so a benign proxy must report the rate

**Design.** Every arm gets the same complete written protocol for a benign
multi-step lab task, so information is held fixed and any difference is
tacit. A benign task carries no controlled technical data, which keeps the
evaluation clear of ITAR/EAR. The arms form a 2 x 2: novice or expert, with
or without a vision model that catches a fraction c of step errors. Scored
with Lesson 17's `evaluate()` at 200 per arm (10 steps, novice step error
0.15, expert 0.03, c = 0.6):

| arm | true | measured |
|---|---:|---:|
| novice, no model | 0.197 | 0.160 |
| novice, vision | 0.539 | 0.545 |
| expert, no model | 0.737 | 0.725 |
| expert, vision | 0.886 | 0.915 |

**Report c, not the ratio.** The same c = 0.6 reads 1.65x on 5 steps,
2.74x on 10 and 7.48x on 20. A shorter proxy understates the uplift, and c
transfers to any protocol length. At 200 per arm, c is also the steadier
estimate. Over 50 seeds the ratio spans 1.85-3.89x (relative sd 15%) and c
spans 0.438-0.695 (9%).

Novices gain 2.74x (+0.342) and experts 1.20x (+0.149), but experts end
higher, at 0.886 against 0.539.

The page cites the 79x OpenAI demonstration as execution-gap erosion. By
the coverage available, trained scientists ran that work in a benign
cloning system and fed results back, so it is expert protocol
optimization, not vision. It is also a working example of this design's
benign-system approach.

### 3 — a material bottleneck caps absolute risk but leaves the 2.53x ratio untouched, and bio's inflection automates its own bottleneck

**Against.** In a two-stage model (information p_info = 0.2, material
p_mat), AI help on information alone never lifts success above p_mat.
Unlimited help reaches exactly p_mat.

**For.** A breakthrough only has to act on acquisition. Doubling p_mat as
well raises the 2.53x to 5.06x, and the ceiling moves with it. The lesson's
own table names that mechanism: bio's 2025 inflection is "acquisition-phase
automation". Key Terms define the acquisition phase as procurement,
equipment and permits, which are the very items bio's
`bottleneck_remaining` lists.

**The bottleneck bounds absolute risk, not the ratio.** From p_mat = 1e-1
to 1e-4 the relative uplift stays 2.53x, while the absolute gain falls
from 0.0306 to 0.00003. "Limited" nuclear uplift is an absolute-risk
claim, measured on a different scale from bio's 2.53x.

**The four domains are not CBRN.** Radiological is missing and cyber is not
a CBRN letter. `main()` prints "three of four CBRN domains crossed
thresholds", but of the 3 CBRN domains present only bio and chem crossed.
Lesson 17's harness has no nuclear or radiological domain at all.

### 4 — illegibility is refuted by the lesson's own cyber section, so the case rests on monitoring, which holds an expert only below 5%

| pillar (Lesson 18) | novice uplift | expert uplift |
|---|---|---|
| Incapability | Lesson 17 unlearning at strength >= 0.6875 puts cyber on the 0.25 floor, 1.00x | multiple-choice only: the skill file hard-rejects non-agentic cyber evidence |
| Illegibility | none | none: the lesson says agentic workflows "integrate reconnaissance, exploitation, post-exploitation, and exfiltration" |
| Monitoring | trusted editing ships 0 at every rate to 0.30 | ships 0 only to a harmful-task rate of 0.04, then 0.008 / 0.032 / 0.203 at 0.05 / 0.1 / 0.3 |

The expert's limit is (0.10 - 0.05) / 0.95 = 0.053. The editor's 5%
over-rejection of benign work spends half the 10% audit budget before any
attack arrives. Even that bound is optimistic. Lesson 10's `t_edit` flags
every harmful output it fails to fix, 10,000 of 10,000. The November 2025
actor got past this by framing its tasks as defensive testing, which is
the one case the toy cannot represent.

### 5 — the 2024 column holds 0 numbers, and a 30-target campaign cannot tell 4 successes from 15, so falsify on gates, not rates

**Forecast (cyber, 2027).** By the end of 2027 a frontier lab will publish
an AI-orchestrated intrusion case in which the model checks its own claimed
results. That report will no longer list fabricated findings as an obstacle,
and it will put the human share of effort below the 2025 report's 10%
floor. The authorization gates will remain human, because operators keep
them by choice. **Falsified by:** a 2027 major-lab report that still
requires human validation of claimed results; a human share of 10% or
more; or a report in which the authorization gates are gone while
validation remains.

The forecast is about the human role and not uplift, because the lesson's
skill file refuses numeric uplift forecasts. The "2024-2025 trajectory" is
thin. `DOMAINS`' 2024 column holds no numbers, and its 2025 column holds
only 2.53x and 80-90%. Success rate makes a poor falsifier. "A handful" of
roughly 30 targets (3-6) gives exact 95% intervals from 0.021-0.265 to
0.077-0.386, and a later campaign needs 15 of 30 before its interval clears
4 of 30.
