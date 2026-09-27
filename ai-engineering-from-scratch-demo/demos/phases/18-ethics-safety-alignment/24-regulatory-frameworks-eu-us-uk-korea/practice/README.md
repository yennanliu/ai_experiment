<!-- generated:start -->
# 18-ethics-safety-alignment / 24-regulatory-frameworks-eu-us-uk-korea

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/18-ethics-safety-alignment/24-regulatory-frameworks-eu-us-uk-korea/) · upstream spec
`phases/18-ethics-safety-alignment/24-regulatory-frameworks-eu-us-uk-korea/docs/en.md`

```bash
uv run demo practice run 24-regulatory-frameworks-eu-us-uk-korea --ex 1
uv run demo explain 24-regulatory-frameworks-eu-us-uk-korea --ex 1
uv run pytest demos/phases/18-ethics-safety-alignment/24-regulatory-frameworks-eu-us-uk-korea
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Read the EU AI Act (regulation 2024/1689) and the GPAI Code of Practice (10 July 2025). Ident… | code | T0 | `ex01_open_source_gpai_skips_the_transparency_chapter_the_lesson_says_binds_everyone_and_10_of_12_commitments_are_systemic_only.py` |
| 2 | A deployment is made by a US company, runs on EU infrastructure, and serves Korean users. Whi… | code | T0 | `ex02_eu_hosting_does_not_trigger_the_ai_act_so_korea_binds_3_of_4_questions_and_the_skill_file_misroutes_18_of_64_placements.py` |
| 3 | The UK AI Security Institute's rename narrows scope. Argue for and against the narrower frami… | code | T0 | `ex03_the_rename_keeps_4_of_16_named_harms_and_drops_all_6_societal_ones_though_free_speech_was_never_in_the_remit.py` |
| 4 | CAISI's "pro-growth" framing is a departure from the 2022-2024 AI safety institute model. Ide… | code | T0 | `ex04_caisi_keeps_the_testing_contact_role_but_deletes_pre_deployment_and_safety_and_3_of_6_duties_name_foreign_actors.py` |
| 5 | Korea's AI Framework Act requires local representatives for foreign providers. Describe the o… | code | T0 | `ex05_a_korean_representative_is_conditional_a_chatbot_over_the_bar_gets_one_with_0_mandatory_tasks_and_a_hiring_tool_under_it_gets_none.py` |
<!-- generated:end -->

## Answers

The lesson's `code/main.py` prints a 14-entry `TIMELINE` and is not a
simulation. The exercises therefore set the lesson page, its skill file and
`TIMELINE` against the primary texts, all read 2026-09-27:
- Regulation 2024/1689 and the GPAI Code of Practice (10 July 2025).
- The AI Omnibus, in force 27 July 2026.
- gov.uk's 2023 *Introducing the AI Safety Institute* and its 14 February
  2025 rename announcement.
- NIST's 2024 AISI *Strategic Vision* and Commerce's 3 June 2025 CAISI
  statement.
- CSET's translation of Korea's Framework Act.

Each argument or design question runs as a small model coded from those
texts.

### 1 — open-source GPAI skips the Transparency chapter the lesson says binds everyone, and 10 of 12 commitments are systemic-only

**Three obligations bind every GPAI provider:**
- a copyright policy that respects text-and-data-mining opt-outs
  (Art 53(1)(c));
- a public summary of training content (Art 53(1)(d));
- cooperation with the Commission and national authorities (Art 53(3)).

These are the only duties that hold for all four profiles the Act separates:
open-source or not, systemic-risk or not. Technical documentation,
Art 53(1)(a)-(b), skips open-source models without systemic risk (Art 53(2)).

**Four obligations bind only systemic-risk providers (Art 55(1)):**
- model evaluation with adversarial testing;
- assessment and mitigation of systemic risk;
- serious-incident reporting;
- cybersecurity for the model and its infrastructure.

| chapter | commitments | profiles bound, of 4 | lesson says |
|---|---:|---:|---|
| Transparency | 1 | 3 | All GPAI providers |
| Copyright | 1 | 4 | All GPAI providers |
| Safety and Security | 10 | 2 | Systemic-risk GPAI providers |

The lesson's figures check out. Its 12 commitments (page and main.py) are
1 + 1 + 10, and its 1e25 FLOP threshold is Art 51(2)'s. Its Transparency
row is the one that is wrong.

### 2 — EU hosting does not trigger the AI Act, so Korea binds 3 of 4 questions and the skill file misroutes 18 of 64 placements

**Only Korea reaches this deployment with an AI statute.** The scope tests
come out as follows:
- **EU:** Art 2(1) keys on the Union market, EU establishment or output used
  in the Union, and never on where the servers are, so the Act does not apply.
- **Korea:** Art 4(1) reaches acts abroad that affect Korean users, so the
  Act applies.
- **US:** there is no federal AI statute, and no US users bring in a state law.

| question | binding rule |
|---|---|
| transparency | Korea Art 31 |
| risk assessment | Korea Art 34(1), Art 32 |
| local representative | Korea Art 36 |
| copyright | none: the Korean Act never mentions it, and EU Art 53(1)(c) does not reach |

**The skill file sends all 4 questions to the EU.** Its trigger is "touches
EU users or infrastructure". Across all 64 provider × infrastructure × user
placements over EU / US / UK / KR, it disagrees with Art 2(1) on 18:
- 9 where only the hosting is in the EU;
- 9 where an EU company serves no EU users, which the trigger misses.

**The lesson's penalty cap is the middle tier, and its date is a year late.**
"15M EUR / 3%" is Art 99(4). Prohibited practices go up to 35M EUR / 7%
(Art 99(3)), 2.33x that. Chapter XII has applied since 2025-08-02, 365 days
before the 2026-08-02 that main.py gives. Korea's only fine is KRW 30M.

### 3 — the rename keeps 4 of 16 named harms and drops all 6 societal ones, though free speech was never in the remit

**For the narrower framing:** the kept harms (chemical and biological
weapons, cyber attacks) are ones no other UK body can evaluate on frontier
models before release. *Assumption:* the dropped harms have other owners,
such as the equality, data-protection and online-safety regulators, and the
outside bodies the 2023 remit already relied on.

**Against it:** the 2023 remit said societal harms need "both pre and
post-deployment evaluations", and only the institute has pre-deployment
access. *Assumption:* manipulation and disinformation can be separated from
security, although the 2023 remit filed both under dual-use.

| 2023 area | harms named | kept in the 2025 focus sentence |
|---|---:|---:|
| dual-use | 5 | 3 |
| societal impacts | 6 | 0 |
| system safety and security | 2 | 1 |
| loss of control | 3 | 0 |
| **total** | **16** | **4** |

The 2025 sentence adds 2 harms the 2023 remit never named: fraud and child
sexual abuse. The lesson says the rename "drops algorithmic bias and
free-speech framings". None of the 2023 remit's named harms is speech, so
the cut was the societal area plus persuasion and disinformation.

`TIMELINE` cannot place the rename:
- 7 of its 14 dates have day "00".
- A string sort swaps the rename (actually 14 February) with the 2 February
  EU prohibitions.
- The Paris summit the rename followed is on the page but not in `TIMELINE`.
- The AI Omnibus moved two EU dates: `TIMELINE`'s 2026-08-02 high-risk date
  by 16 months, to 2 Dec 2027 for Annex III, and its 2027-08-02 embedded
  high-risk date by 12 months.

### 4 — CAISI keeps the testing-contact role but deletes "pre-deployment" and "safety", and 3 of 6 duties name foreign actors

Two shifts, each measured first on the mandates themselves (2024 AISI
*Strategic Vision* against the 2025 CAISI statement):

| measure | 2024 | 2025 |
|---|---:|---:|
| items naming foreign or adversary actors | 0 of 7 | 3 of 6 |
| risk classes in the scope sentence | 3 | 1 (national security) |
| "rights" | 1 | 0 |
| items naming a network | 1 of 7 | 0 of 6 |
| items naming regulation | 0 of 7 | 1 of 6 |
| items naming evaluation | 4 of 7 | 4 of 6 |
| items naming safety | 4 of 7 | 0 of 6 |

1. **Evaluations move to national-security risks and foreign models.**
   Track the share of CAISI's published evaluations that cover
   foreign-developed models, and the share that cover rights harms.
2. **The international role flips from building a safety network to
   resisting foreign regulation.** Track US sign-ons to multilateral safety
   statements and US positions in standards bodies. This is the page's
   "domestic counterweight to EU AI Act's regulatory posture", now written
   into the mandate.

**The lesson's "reduced emphasis on pre-deployment evaluation" is a
deletion inside a role that survives.** The point-of-contact sentence stays.
It loses "pre-deployment", "post-deployment" and "safety", and gains
"industry's" and "commercial". Evaluation itself is not cut.

### 5 — a Korean representative is conditional: a chatbot over the bar gets one with 0 mandatory tasks, and a hiring tool under it gets none

**Operationally:**
1. Check the thresholds: KRW 1 trillion total revenue, KRW 10 billion
   AI-service revenue, or 1 million daily Korean users.
2. If any is crossed, appoint a representative with a Korean address in
   writing and report the appointment to MSIT.
3. The company stays liable for what the representative does (Art 36(3)).

Failing to appoint costs at most KRW 30M. That is 0.3% of the smallest
revenue that triggers the duty, and a grace period runs to at least
2027-01-22. The representative has 3 statutory tasks, and one of them, the
Art 33 confirmation request, is optional. The Art 31 notice and labelling
duties stay with the company.

| Bay Area archetype | representative | mandatory tasks | Art 34 duties |
|---|---|---:|---|
| consumer chatbot on a licensed model (1.5M daily users) | yes | 0 | no |
| high-impact hiring-screening SaaS (under every bar) | no | 0 | yes |
| frontier-model API (3e26 FLOP) | yes | 1 | no |

**The representative and the duties it carries are triggered separately.**

**The lesson states two conditional provisions as unconditional:**
- **Representative:** the page has it as a flat mandate, but Art 36(1)
  applies only above thresholds set by decree.
- **Institute:** the page says "Article 12 establishes an AISI", but the
  Minister "may operate" one.

`TIMELINE`'s "2026-01-00" is 2026-01-22.
