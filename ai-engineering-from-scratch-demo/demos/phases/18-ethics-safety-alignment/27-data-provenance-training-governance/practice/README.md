<!-- generated:start -->
# 18-ethics-safety-alignment / 27-data-provenance-training-governance

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/18-ethics-safety-alignment/27-data-provenance-training-governance/) · upstream spec
`phases/18-ethics-safety-alignment/27-data-provenance-training-governance/docs/en.md`

```bash
uv run demo practice run 27-data-provenance-training-governance --ex 1
uv run demo explain 27-data-provenance-training-governance --ex 1
uv run pytest demos/phases/18-ethics-safety-alignment/27-data-provenance-training-governance
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Produce a 12-field summary for a toy dataset and identify which fields ar… | code | T0 | `ex01_a_y_with_a_note_drops_the_cpra_flag_and_5_of_12_missing_fields_crash_the_generator.py` |
| 2 | The EU Copyright Directive TDM opt-out is machine-readable. Propose a standard format for the… | code | T0 | `ex02_robots_txt_honours_12_of_24_opt_out_cases_c2pa_16_and_a_purpose_keyed_record_20.py` |
| 3 | Read the Data Provenance Initiative's "Consent in Crisis" (July 2024). Describe the three fas… | code | T0 | `ex03_a_2023_crawl_keeps_the_25pct_of_top_sources_a_2024_entrant_loses_and_ab_2013_flags_both_alike.py` |
| 4 | The 2025 DPA alignment accepts legitimate interest for public-content training. Construct a s… | code | T0 | `ex04_the_lessons_interest_plus_opt_out_rule_clears_health_and_minors_posts_that_need_consent.py` |
| 5 | Sketch a training-data-provenance manifest that composes with the AB 2013 fields and a C2PA-s… | code | T0 | `ex05_cleaning_breaks_every_source_signature_erasure_breaks_the_chain_and_a_kept_hash_leaks_the_record.py` |
<!-- generated:end -->

## Answers

Every exercise runs or reads the lesson's `code/main.py`, a generator for the
12-item California AB 2013 summary with a follow-up flagger. Datasets, sites
and people in the fixtures are abstract placeholders.

### 1 — a "Y" with a note drops the CPRA flag, and 5 of 12 missing fields crash the generator

**In a forum-crawl toy filled the way teams really fill the form, items 3,
6, 9, 10 and 11 are under-specified.** Item 3 says "various". Item 6 is a
bare "Y" that does not say purchased or licensed. Item 9 names no purpose,
item 10 gives no ongoing-collection notice, and item 11 says "TBD". The
shipped toy passes all five rules. The reference renders all 12 lines of
both summaries without a warning, although the lesson's skill file asks to
flag placeholder-only fields.

**A "Y" with a note drops the CPRA flag.** `flag_followups` tests items 5
and 12 with `startswith("Y")` and items 6, 7 and 8 with `== "Y"`. Written in
the toy's own annotated style, "Y (usernames and signatures)" on item 7
raises no CPRA obligation. An all-"Y (note)" summary raises 2 of the 5
follow-ups, and an all-bare-"Y" summary raises all 5.

**Missing fields crash the generator for 5 of the 12 items.** Removing item
5, 6, 7, 8 or 12 makes `render_markdown` raise KeyError. Only the other 7
reach the "(missing)" placeholder.

**The shipped toy's only follow-up is a false positive.** It flags "base
model used for generation" for data drawn from `random.gauss`. The printed
takeaway says items 5 and 7 trigger cascading obligations, and the toy
triggers neither.

### 2 — robots.txt honours 12 of 24 opt-out cases, C2PA 16, and a purpose-keyed record 20

**Proposal: one purpose-keyed, crawler-independent record, published at
`/.well-known/tdmrep.json` and embedded in each asset.** The record is
`{"tdm-reservation": 1, "applies-to": "*", "purposes": {"train": "reserved",
"search-index": "allowed"}, "policy": <licence URL>}`, and in HTML it is
`<meta name="tdm-reservation" content="train=reserved; search-index=allowed">`.

The intent tested is "no training, search indexing fine, for any crawler,
wherever the copy is". It runs over 2 crawlers × 2 purposes × 2 asset types ×
3 places (origin, mirror, stripped copy):

| signal | cases honoured of 24 | where it fails |
|---|---:|---|
| robots.txt | 12 | an unlisted crawler; no purpose field; mirrors |
| C2PA "No AI Training" | 16 | all 6 HTML train cases; stripped images |
| proposal | 20 | the 4 stripped copies asked to train |

**robots.txt opts a work into training by omission.** With GPTBot and CCBot
disallowed, `urllib.robotparser` lets NewAIBot fetch everything.
**C2PA has the right purpose semantics and the wrong coverage.** It honours
all 4 image cases per crawler at origin and on mirrors, but HTML text
carries no manifest.

The lesson's generator records no opt-out. None of the 12 fields mentions
one, and a crawl declared public domain raises no TDM flag. The lesson also
spells the signal "TDM.Reservation", and the TDMRep property is
`tdm-reservation`.

### 3 — a 2023 crawl keeps the 25% of top sources a 2024 entrant loses, and AB 2013 flags both alike

**The fastest-restricting categories in "Consent in Crisis" are news sites
first, then social media / forums and encyclopedias,** the categories that
make up the head of C4. The lesson names none of them.

**Economic consequence: an incumbency moat.** robots.txt is not
retroactive. On a 20-source panel where the lesson's "about 25%" restrict AI
crawlers in 2024, robotparser gives:

| crawler | sources open |
|---|---:|
| incumbent, crawled 2023 | 20 / 20 |
| entrant, 2024, named user-agent | 15 / 20 |
| entrant, 2024, unnamed user-agent | 20 / 20 |

At 25% a year the open share compounds to 0.75, 0.562 and 0.422 after 1, 2
and 3 years, a half-life of 2.41 years. Fresh data gets a price that only
entrants pay. The restriction also taxes the crawler that identifies itself.
The AB 2013 summaries of the two crawls differ only in item 10 and get
identical follow-ups, so the disclosure cannot show the moat.

### 4 — the lesson's "interest + opt-out = lawful" rule clears health and minors' posts that need consent

**Scenario: public posts in a patient-support forum that reveal each
author's diagnosis. Legitimate interest is not enough. The provider needs
explicit consent under GDPR Article 9(2)(a) on top of an Article 6 basis.**

| scenario | lesson rule | basis needed |
|---|---|---|
| A adult public posts, opt-out | lawful | legitimate interest |
| B patient-forum diagnosis posts | lawful | explicit consent, Art 9(2)(a) |
| C private messages | not lawful | consent, Art 6(1)(a) |
| D public posts by users under 18 | lawful | parental consent, Art 8 |
| E adult public posts, no opt-out | not lawful | consent, Art 6(1)(a) |

The rule `code/main.py` prints ("legitimate interest + opt-out = lawful")
disagrees with the GDPR reading on 2 of the 5 scenarios, B and D, and calls
both lawful. The reference cannot tell B from A either. The summaries differ
only in item 4, and both get the same CPRA and TDM follow-ups, because item 7
is a binary Y/N. The lesson dates the Brazilian ANPD suspension both "June
2024" and "2 July 2024".

### 5 — cleaning breaks every source signature, erasure breaks the chain, and a kept hash leaks the record

**Sketch: per dataset, an `ab2013` block with the 12 fields and a `chain` of
C2PA-style actions.** Each action commits to the dataset's Merkle root before
and after, and is signed over the previous signature. Item 9 is derived from
the chain's actions. The 3-step chain verifies and fails at step 1 when one
record is edited. 8 of the reference's 12 field names are not usable as
keys, so the manifest normalises them, and all 12 render with 0 "(missing)".

**Technical barrier: C2PA signs bytes, and cleaning changes them.** 1 of 10
records still matches its source hash after a whitespace strip, and 0 after
lowercasing.

**Legal barrier: GDPR erasure against an append-only chain.** Erasing the
personal record fails the chain at step 1. Keeping its leaf hash makes the
chain verify again, but guessing ages 0-120 reverses that hash on attempt
38. A hash of low-entropy personal data is still personal data.
