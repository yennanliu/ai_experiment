<!-- generated:start -->
# 18-ethics-safety-alignment / 26-model-system-dataset-cards

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/18-ethics-safety-alignment/26-model-system-dataset-cards/) · upstream spec
`phases/18-ethics-safety-alignment/26-model-system-dataset-cards/docs/en.md`

```bash
uv run demo practice run 26-model-system-dataset-cards --ex 1
uv run demo explain 26-model-system-dataset-cards --ex 1
uv run pytest demos/phases/18-ethics-safety-alignment/26-model-system-dataset-cards
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Inspect the generated cards. Identify sections that are weak (placeholder… | code | T0 | `ex01_four_sections_are_placeholder_only_and_the_quantitative_analysis_is_typed_in_by_hand.py` |
| 2 | Extend the model card with a quantitative disaggregated analysis across two demographic group… | code | T0 | `ex02_regenerated_from_its_datasheet_the_model_scores_0_996_and_a_0_03_parity_gap_is_inside_the_0_088_noise.py` |
| 3 | Read Oreamuno et al. 2023 on the 0.3% adoption rate. Propose one structural change to the mod… | code | T0 | `ex03_the_audits_hard_rejects_pass_a_card_that_calls_its_bias_metrics_placeholder_and_a_per_factor_table_fails_it_on_2_of_2.py` |
| 4 | Laminator (Duddu et al. 2024) uses TEEs for verifiable attestations. Design a model-card fiel… | code | T0 | `ex04_a_signed_0_996_makes_the_verifier_reject_the_cards_printed_0_97_and_a_reused_key_leaks_124_of_256_positions.py` |
| 5 | Write a System Card (System Card, not Model Card) for one of your past projects or a hypothet… | code | T0 | `ex05_the_keyword_filter_goes_from_0_to_3_sends_under_the_auditors_test_and_ifc_scores_0_because_it_cannot_send.py` |
<!-- generated:end -->

## Answers

`code/main.py` prints three fixed Markdown strings: a model card, a datasheet
and a system card. It computes nothing. Each exercise parses those strings
and checks them against the lesson text, the audit skill file, or the code of
the lessons the cards point to (Lesson 21 for the fairness metrics, Lesson 15
for prompt injection).

### 1 — four sections are placeholder-only, and the Quantitative Analysis is typed in by hand

**Four sections are placeholder-only.** In the model card these are Metrics,
Training Data and Ethical Considerations. In the system card it is Regulatory
Alignment. Two more sections are partial: Security Capabilities (1 of 3
bullets is a placeholder: prompt injection `N/A`) and Incident Response (1 of
2). A bullet counts as a placeholder if it is `N/A`, a "see ..." pointer, a
self-declared placeholder, "not validated", or "goes nowhere". The datasheet
has no placeholder sections.

| section | evidence that would strengthen it |
|---|---|
| Metrics | definitions, thresholds, the evaluation split |
| Training Data | the datasheet's counts, seed and label rule, inline |
| Ethical Considerations | one row per listed factor, with the measured gap and a mitigation (ex 3) |
| Regulatory Alignment | for each `N/A`, which article applies and why the system is out of scope |
| Security: prompt injection | a measured test result (ex 5) |
| Incident Response | an owner and a response time |

**The model card is missing a Mitchell section.** The lesson lists 9 sections
for a model card, 7 for a datasheet and 5 for a system card. The generated
model card has 8 sections, and Evaluation Data is the one missing.

**The section that looks strongest is written by hand.** The figures 0.97,
+0.03 and -0.01 are string literals inside `model_card()`, and the only import
in `main.py` is `__future__`. The card's own Ethical Considerations says "Bias
metrics are placeholder". The datasheet mentions a "fixed seed" but never says
what it is, and it says the data is "regenerated on every run", but no code
generates it.

### 2 — regenerated from its datasheet, the model scores 0.996 and a 0.03 parity gap is inside the 0.088 noise

I rebuilt the dataset from the datasheet: 1,500 rows, split 1000/500, seed 26
(the datasheet names no seed). The model was trained and scored with Lesson
21's own `train`, `predict`, `demographic_parity` and `equalized_odds`.

| group | n | accuracy | selection rate | TPR | FPR |
|---|---:|---:|---:|---:|---:|
| 0 | 250 | 0.992 | 0.536 | 0.985 | 0.000 |
| 1 | 250 | 1.000 | 0.500 | 1.000 | 0.000 |

The parity gap is -0.036 ± 0.088 (95%) and the TPR gap is +0.015.

**Neither dataset reproduces the card's numbers.** The card says 0.97 / +0.03
/ -0.01 (accuracy / parity gap / TPR gap). On the datasheet's data the model
scores 0.996, and the parity gap has the opposite sign. On Lesson 21's data,
where the card's Metrics line points, it scores 0.71 / +0.429 / +0.326.
Lesson 21's own `main()` prints the same +0.429.

**A 0.03 gap cannot be told apart from zero at this size.** The label rule
ignores the attribute. Even so, over 1000 seeds a 500-row test split has a
label-rate gap of at least 0.03 in 50.4% of cases. A perfect classifier's
parity gap equals that label-rate gap. A card that reports a gap should
report its interval with it.

### 3 — the audit's hard rejects pass a card that calls its bias metrics placeholder; a per-factor table fails it on 2 of 2

**Proposal: generate Ethical Considerations from the Factors section.** It
becomes a table with one row per listed factor: `factor | risk | measured |
mitigation`. The measured cell holds either a number or an explicit "not
measurable: <reason>". Validation fails if any factor has no complete row.
Authors then face named blanks rather than an optional free-text heading, and
a blank shows up as an error rather than a silent omission.

**The lesson's own audit accepts the lesson's placeholder card.** The skill
file has 3 hard rejects, and none of them fires on the generated model card.
Under a presence rule like that, the card counts as adopted. The per-factor
rule fails it. Factors lists gender and age bucket, and neither one is named
in Ethical Considerations or in Quantitative Analysis, which says only
"group0 vs group1".

**The table exposes a datasheet mismatch.** The datasheet documents 1 binary
sensitive attribute, but the card lists 2 factors. A card that carries
Exercise 2's measured gender row plus "not measurable: datasheet has no age
field" passes the rule. So the rule can be satisfied, and the missing age
data ends up written down instead of hidden.

(The 0.3% adoption figure is the lesson's quote. Nothing here measures
adoption. What this measures is which cards the rule accepts.)

### 4 — a signed 0.996 makes the verifier reject the card's printed 0.97, and a reused key leaks 124 of 256 positions

**Field design: `## Attestations`, one row per claim, `metric | value |
payload | signer`.** The payload binds the metric and value to SHA-256 hashes
of the model weights, the evaluation dataset and the evaluation code. The
"enclave" is a function that runs the evaluation itself and signs the result.
The signature is a stdlib Lamport one-time signature (8,192 bytes), shipped as
a sidecar file.

**The verifier's role is to recompute everything it can and trust only the
signature.**

| case | rejected at |
|---|---|
| honest card | accepted |
| card prints the reference's 0.97 | value |
| payload edited to 0.97 | signature |
| weights swapped | model hash |
| other dataset | dataset hash |
| key not in the registry | signer |
| key already used | one-time key |

**On the lesson's card, the attestation rejects the card's own number.** The
card prints 0.97. The enclave measures 0.996 and signs that. None of the 3
cards carries an attestation today, so every figure in them is self-report.

**The verifier also has to enforce one-time keys.** When one Lamport key
signs two attestations, both secret halves are revealed at 124 of 256
positions. That is enough to forge signatures, so the registry has to retire
each key after one use.

### 5 — the keyword filter goes from 0 to 3 sends under the auditor's test, and IFC scores 0 because it cannot send

**The card is written for Lesson 15's inbox assistant, and every security
figure comes from running that lesson's agents.** It uses the reference card's
five sections. Exercise 1's audit finds no placeholder-only or partial
section, and all five items the lesson lists are covered. The table shows how
many of the 3 contacts each agent sends mail to:

| agent | static `[SYSTEM:]` | adaptive (polite) | benign "forward" |
|---|---:|---:|---:|
| naive | 3 | 3 | 3 |
| keyword_filter | 0 | 3 | 3 |
| ifc | 0 | 0 | 0 |

**The most valuable section for a third-party auditor is Security
Capabilities, specifically the prompt-injection row.** It is the section
where the vendor's test and an outsider's test disagree. Against the static
attack, the keyword filter truthfully "blocks injection". The adaptive
wording gets it to send to all 3 contacts, and so does a benign email that
only mentions forwarding. The reference system card fills this row with "N/A
(non-generative)".

**IFC's zero comes partly from a missing capability.** `ifc_agent` never
calls `tool_send`, so it would score 0 against any attack, and it also cannot
forward mail when the user asks it to. An auditor has to test the legitimate
action too. The card records this in Alignment as "User-requested send: not
supported".
