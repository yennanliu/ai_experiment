<!-- generated:start -->
# 17-infrastructure-and-production / 26-compliance-frameworks

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/17-infrastructure-and-production/26-compliance-frameworks/) · upstream spec
`phases/17-infrastructure-and-production/26-compliance-frameworks/docs/en.md`

```bash
uv run demo practice run 26-compliance-frameworks --ex 1
uv run demo explain 26-compliance-frameworks --ex 1
uv run pytest demos/phases/17-infrastructure-and-production/26-compliance-frameworks
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Your first enterprise customer requires SOC 2 Type II, HIPAA BAA, EU AI Act statement. What i… | code | T0 | `ex01_seven_of_the_maps_ten_controls_two_baas_and_an_ai_act_statement_the_map_has_no_row_for.py` |
| 2 | Classify three hypothetical LLM products under EU AI Act risk tiers. What changes at high-risk? | code | T0 | `ex02_only_the_cv_screener_is_high_risk_and_the_map_holds_neither_its_logging_nor_its_documentation_article.py` |
| 3 | You accidentally sent PHI to a provider without BAA. Walk through the incident response. | code | T0 | `ex03_the_lesson_25_scrubber_passes_the_name_dates_and_address_and_the_log_cannot_say_whose_phi_left.py` |
| 4 | Argue whether ISO 42001 is "necessary in 2026" for a mid-market AI vendor. | code | T0 | `ex04_not_by_the_lessons_own_map_which_requires_it_for_2_of_7_profiles_and_credits_it_with_0_of_10_controls.py` |
| 5 | Map your LLM audit log fields (Phase 17 · 25) to at least three framework controls. | code | T0 | `ex05_seven_fields_reach_five_frameworks_but_the_logging_clauses_sit_under_retention_and_nothing_is_written.py` |
<!-- generated:end -->

## Answers

The lesson's `code/main.py` is two dicts and a `main()` that prints them:
CONTROL_MAP (10 controls → framework clauses) and PROFILE_MAP (7
geography/segment pairs → required frameworks). There is no lookup function,
so every exercise reads the dicts directly. Exercises 3 and 5 also run Lesson
25's scrubber and audit log. External facts were checked on 2026-09-26: the AI
Act articles on artificialintelligenceact.eu, the HIPAA sections on the
Cornell LII copy of 45 CFR, and the Digital Omnibus, Colorado SB 26-189 and
ISO 42001 through law-firm and vendor write-ups. iso.org returned 403, and I
did not read the Official Journal text of the Omnibus.

### 1 — seven of the map's ten controls, two BAAs, and an AI Act statement the map has no row for

The product is read as ordinary B2B LLM SaaS, which the lesson calls
limited-risk. The team is the one in The Problem, with Type I in hand and six
months from Type II.

| ask | minimum deliverable |
|---|---|
| SOC 2 Type II | Type I report, observation window already running, committed report date. Type II itself needs 6-12 months of operated controls. |
| HIPAA BAA | BAA with the customer, **and** a BAA with every model provider on the PHI path |
| EU AI Act statement | written risk-tier classification: limited-risk, Art. 50 disclosure that users are talking to an AI |

The three frameworks touch **9 of the 10** CONTROL_MAP rows. Drop conformity
assessment and impact assessment, which only apply at high risk, and **7**
remain: access logging, change management, encryption in transit, secrets
management, inference-time PII redaction, audit log retention, BAA signed.

- **By the map, 2 controls "cover" all three frameworks.** 10 pairs of rows
  each touch SOC 2, HIPAA and the EU AI Act, for example access logging plus
  impact assessment. A row touching a framework means one clause of it, so
  counting frameworks touched says nothing about readiness.
- **No profile is this customer, and the code has no lookup.** No PROFILE_MAP
  row is exactly these three. The smallest that contains them is ("EU",
  "healthcare") with 4 frameworks, including "HIPAA (global)", although HIPAA
  is a US law. `main` is the module's only function. All 7 profiles require
  SOC 2 Type II.
- **Every EU AI Act citation in the map is a high-risk article** (Art. 10, 27,
  43). Art. 50, which is the whole obligation of a limited-risk chatbot, has no
  row. The statement this customer asks for has nothing in the map to cite.

### 2 — only the CV screener is high-risk, and the map holds neither its logging nor its documentation article

| product | tier | why | CONTROL_MAP rows that bind |
|---|---|---|---:|
| customer-support chatbot | limited | interacts with people → Art. 50 disclosure | 0 |
| CV-screening ranker for recruiters | **high** | employment, Annex III point 4 | 3 |
| internal code-review assistant | minimal | no listed domain, no public interaction | 0 |

At high risk, one disclosure becomes the Chapter III Section 2 set: Art. 9-15
(risk management, data governance, technical documentation, record-keeping,
transparency to deployers, human oversight, accuracy and robustness), plus
Art. 43 conformity assessment.

- **The map holds neither article the lesson names.** The lesson's high-risk
  line is "conformity assessment, documentation, logging". The map has Art. 43,
  but no row cites Art. 11 (documentation) or Art. 12 (record-keeping). Of
  Art. 9-15 it cites only Art. 10.
- **`main()` prints the wrong fine next to the high-risk date.** It closes
  with "high-risk enforcement August 2, 2026" and then "Fines up to €35M or
  7%". That is Art. 99(3), the ceiling for prohibited practices. High-risk
  breaches fall under Art. 99(4), €15M or 3%. The lesson's "whichever higher
  applies" also leaves out Art. 99(6): for SMEs the *lower* applies. A
  screener vendor with €40M turnover and under 250 staff faces up to **€1.2M**,
  not €15M.
- **Both dates the lesson drills have moved.** Per Gibson Dunn (27 May 2026)
  and Usercentrics, the Digital Omnibus on AI entered into force on 27 July
  2026. It defers Annex III high-risk obligations to **2 December 2027** and
  Annex I products to 2 August 2028. I did not verify this against the
  Official Journal. Colorado SB 26-189, signed 14 May 2026 (McDermott), moves
  that law to 1 January 2027 and removes the impact assessment that CONTROL_MAP
  cites SB24-205 for.

### 3 — the Lesson 25 scrubber passes the name, dates and address, and the log cannot say whose PHI left

"You" are the vendor, which makes you a business associate of the covered
entity. The provider had no subcontractor BAA, so the disclosure was
impermissible.

1. **Contain.** Cut the route and repoint traffic at a provider under BAA.
2. **Mitigate.** Get deletion and a written attestation from the provider.
   This is 164.402 factor (iv).
3. **Scope.** Establish the window, the calls and the individuals.
4. **Assess.** Run the four factors. The disclosure is presumed a breach
   unless the assessment shows a low probability of compromise.
5. **Notify.** For a discovery on 2 March 2026:

| who | rule | deadline |
|---|---|---|
| covered entity | 164.410, ≤ 60 days from discovery | 1 May 2026 |
| individuals | 164.404, ≤ 60 days from the entity's discovery; chains unless you are its agent | 30 June 2026 |
| HHS | 164.408: at the same time if ≥ 500, else 60 days after year end | 30 June 2026 / 1 March 2027 |
| media | 164.406: > 500 residents of one state | with individual notice |

Running this phase's own tools against the incident gives three findings:

- **Lesson 25's scrubber in front would not have prevented it.** On a
  one-line clinical note it masks 4 values: SSN, phone, email, and the 10-digit
  MRN, which it catches only because it matches the phone pattern
  (`[PHONE_001]`). The name, dates, street address and ZIP, and health-plan ID
  pass. That is 4 of the note's 8 Safe Harbor categories, plus the diagnosis.
- **The audit log cannot answer factor (i).** No field names a patient, and
  `prompt_hash` is taken over the *scrubbed* prompt. Two patients' refill
  requests scrub to the same text and log the same hash. The log bounds the
  window. The number of people affected has to be presumed from call volume.
- **The lesson's map has no incident-response row.** Its only breach citation
  is filed under change management, and no row cites 164.402-414.

### 4 — not by the lesson's own map, which requires it for 2 of 7 profiles and credits it with 0 of 10 controls

**No, not as a certificate. It is necessary where a buyer's questionnaire
names it.** The lesson's PROFILE_MAP requires ISO 42001 for ("US", "B2B
SaaS") and ("Global", "enterprise"). It leaves it out of healthcare, fintech,
Colorado and both EU profiles. Of the 3 profiles that carry the EU AI Act,
only the global one lists it. The map has no mid-market profile.

- **The map gives 42001 zero reuse.** No CONTROL_MAP row cites it; ISO 27001
  appears in 5 of 10. The impact-assessment row cites Colorado and Art. 27, not
  42001, although secondary sources list an AI impact assessment among 42001's
  Annex A objectives (A.2-A.10, 38 controls).
- **The real argument for it is structural, and the map cannot show it.**
  42001 shares ISO's Harmonized Structure (clauses 4-10) with 27001, which is
  why a 27001 holder's extra effort is small. All 5 of the map's ISO 27001
  citations are Annex A controls, none a clause.
- **The 2026 regulatory push got weaker.** The Annex III deadline moved to
  December 2027 (exercise 2).

The 2026 posture for a mid-market vendor: hold ISO 27001, document the
AI-specific practices 42001 asks for (impact assessment, data quality, human
oversight), and certify when a contract requires it.

### 5 — seven fields reach five frameworks, but the logging clauses sit under retention and nothing is written

| Lesson 25 field | controls |
|---|---|
| timestamp, user, tenant | HIPAA §164.312(b), ISO 27001 A.8.15, SOC 2 CC7, EU AI Act Art. 12 |
| model | SOC 2 CC8, EU AI Act Art. 12 |
| prompt_hash, response_hash | GDPR Art. 25 (kept in place of raw text) |
| guardrail_trips | SOC 2 CC7, ISO 27001 A.8.16, EU AI Act Art. 12 |
| input_tokens, output_tokens, cost_usd | none |

7 of the 10 fields map to 7 clauses across 5 frameworks. In `main()`'s own
output, cost_usd is 0.0012 on all 3 entries, while the entries carry 11, 6 and
7 input tokens.

- **The map files the logging clauses under retention.** A.8.15, 164.312(b)
  and CC7 all sit in CONTROL_MAP's "audit log retention" row. The "access
  logging" row cites ISO A.5.15-5.18 and HIPAA 164.312(a), whose CFR heading
  is "Standard: Access control". None of those clauses is in this mapping.
- **The log is not appended, retained or tamper-evident.** Lesson 25 says it
  "Appends to an immutable audit log". `audit_log_call` returns a string, and
  the module opens no file. Edit the user on entry 1 and nothing notices. Add
  a `prev` hash chain and the same edit is caught at entry 1.
- **"Model + version" is logged as a name.** Every entry says
  `anthropic/claude-3.7-sonnet`, with no snapshot, so an Art. 12 or CC8 record
  cannot say which build served a call.
