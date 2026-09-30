<!-- generated:start -->
# 19-capstone-projects / 86-constitutional-rules-engine

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/86-constitutional-rules-engine/) · upstream spec
`phases/19-capstone-projects/86-constitutional-rules-engine/docs/en.md`

```bash
uv run demo practice run 86-constitutional-rules-engine --ex 1
uv run demo explain 86-constitutional-rules-engine --ex 1
uv run pytest demos/phases/19-capstone-projects/86-constitutional-rules-engine
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Add a rule that requires every response to include the phrase "If this is urgent" when the pr… | code | T0 | `ex01_the_rule_fires_on_6_of_6_safety_prompts_and_the_lessons_engine_would_fire_it_on_every_response.py` |
| 2 | Replace the regex fixer with a templating fixer that takes named slots. Demonstrate one rule… | code | T0 | `ex02_slot_templates_match_the_regex_fixer_on_5_of_5_drafts_and_a_misspelled_fix_is_a_silent_no_op.py` |
| 3 | Add a metrics endpoint that, given a corpus of drafts, returns the per-rule violation rate so… | code | T0 | `ex03_the_pii_rule_is_the_over_firer_at_precision_0_25_while_the_refusal_rule_has_the_top_rate.py` |
<!-- generated:end -->

## Answers

Every exercise loads the lesson's `code/main.py` (with its `yaml_subset`
module registered for the import) and runs its shipped `Engine`, `Fixer`
and six-rule `rules.yml` unchanged. All fixtures are harmless text. All
three solutions are stdlib-only, deterministic, and run in well under a
second.

### 1 — the rule fires on 6 of 6 safety prompts, and the lesson's Engine would fire it on every response

**The rule adds one key, `applies_when_prompt`, written in the lesson's
own predicate grammar. A per-call `bind_prompt` step then turns it into
something the unchanged Engine understands.** `Engine.evaluate(text)` only
sees the response, so the prompt condition is evaluated separately with
the lesson's `_eval_predicate`. When it is false, the rule's `applies_when`
becomes `{not_: {}}`, which is always false. "Mentions safety" is an
`any_of` over two regexes: safe/unsafe/safety, and danger/hazard/
emergency/injury. The `must` is `contains_regex: '\bif this is urgent\b'`,
and the fix is a `prepend_if_missing`.

| prompts | draft | after fix |
|---|---|---|
| 6 safety prompts | violation | clean |
| 6 other prompts | not_applicable | — |
| safety prompt, phrase already present | pass | — |

The lesson's 5 demo drafts keep their original violations with the rule
added. The rule YAML parses the same under PyYAML and `yaml_subset`.

**The lesson's Engine ignores rule keys it does not know.** Loaded into a
plain `Engine`, the rule has no `applies_when`, so it applies everywhere.
It fires on 6/6 responses and raises no error.

**A plain `\b(un)?safe(ty)?\b` trigger fires on 3/6 non-safety prompts:**
thread-safe, memory safety and type-safe. The hyphen counts as a word
boundary. The rule excludes these three with lookbehinds.

### 2 — slot templates match the regex fixer on 5 of 5 demo drafts, and a misspelled fix is a silent no-op in the lesson's

**Each fix becomes `{where, template, slots, pattern}`, and every fix is
validated when the fixer is built.** `where` is append, prepend or
replace. `template` is a `str.format` string whose named slots are
literals or `{"from": "group_name"}`. The lesson's three operations
translate to slot-free templates, and on all 5 demo drafts the revised
text and second-pass violations are identical to the lesson's `Fixer`.

The rule rewritten is `no-pii-in-examples`. Its template is
`[{label}-{kind}]`, and its pattern has named `email` and `phone` groups:

| draft | lesson `Fixer` | template fixer |
|---|---|---|
| Example user: lee@example.com. | `[example-contact]` | `[example-email]` |
| Call (555) 123-4567 for an example. | `Call ([example-contact] …` | `Call [example-phone] …` |
| Example: +1 555 123 4567 | `+[example-contact]` | `[example-phone]` |
| Sample: ops@acme.test or 555.123.4567 | `[example-contact]` x2 | `[example-email]`, `[example-phone]` |

**The lesson's replacement leaves a stray `(` or `+` on 2 of 4 probes.**
The phone regex begins at `\b`, which cannot sit before either character.
The engine's second pass reports 0 violations on both results anyway.

**A misspelled fix operation does nothing in the lesson.** `Fixer`
checks three keys with `if/elif`. With `append_if_mising` it returns the
draft unchanged and the violation stays. The templating fixer raises
ValueError at build time, both for that typo and for a template that uses
an undeclared `{x}` slot.

### 3 — the PII rule is the over-firer at precision 0.25, while the refusal rule has the top violation rate

**`POST /metrics` runs on a stdlib `ThreadingHTTPServer` bound to
127.0.0.1 on an ephemeral port.** It takes `{"drafts": [{"text", "expected"?}]}`
and returns, per rule, applicable, fired and rate (fired / drafts). When
reviewer labels are present it also returns tp/fp/fn, precision, and
`most_false_fires`. On 20 labelled drafts it names `no-pii-in-examples`:

| rule | rate | fired | false fires | precision |
|---|---:|---:|---:|---:|
| no-empty-refusal | 0.25 | 5 | 2 | 0.60 |
| no-pii-in-examples | 0.20 | 4 | 3 | 0.25 |
| end-with-runnable / cite / internal-leak | 0.05 each | 1 | 0 | 1.00 |
| bounded-length | 0.00 | 0 | 0 | — |

The three false PII fires are an order number, an epoch timestamp and a
build ID: 10-digit numbers that match the phone shape. Without labels the
same rates come back and no over-firer is named. Malformed JSON gets a 400.

**Ranking by raw rate points at the wrong rule.** `no-empty-refusal` has
the highest rate, but 3 of its 5 fires are real refusals. Its false fires
are "I cannot recommend this library enough" and "I will not bore you with
details". A violation rate alone cannot separate over-firing from a corpus
that really contains violations. That takes labels.

**The PII rule over-fires and under-fires at once.** It fires on 4 of the
4 drafts it applies to. It misses "Reach the author at dana@acme.test",
because that draft has no "example" or "sample" word, so the rule is
`not_applicable`.
