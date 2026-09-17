<!-- generated:start -->
# 11-llm-engineering / 03-structured-outputs

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/11-llm-engineering/03-structured-outputs/) · upstream spec
`phases/11-llm-engineering/03-structured-outputs/docs/en.md`

```bash
uv run demo practice run 03-structured-outputs --ex 1
uv run demo explain 03-structured-outputs --ex 1
uv run pytest demos/phases/11-llm-engineering/03-structured-outputs
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Extend the schema validator to support `oneOf` (the data must match exactly one of several sc… | code | T0 | `ex01_a_schema_with_no_type_validates_everything.py` |
| 2 | Build a "schema diff" tool that compares two schemas and identifies breaking changes (removed… | code | T0 | `ex02_three_breaking_changes_the_validator_cannot_feel.py` |
| 3 | Implement a more realistic constrained decoding simulator. Given a JSON Schema and a vocabula… | code | T0 | `ex03_the_decoder_stops_where_the_validator_fails.py` |
| 4 | Build an extraction eval suite. Create 50 product descriptions with hand-labeled JSON outputs… | code | T0 | `ex04_type_compliance_is_100_percent_and_exact_match_is_not.py` |
| 5 | Add "confidence scores" to your extraction pipeline. For each extracted field, estimate how c… | code | T0 | `ex05_resampling_measures_the_branch_table_not_the_model.py` |
<!-- generated:end -->

## Answers

The lesson is pure stdlib — `import json` and nothing else — so all five
exercises are **T0** and run in CI. What they find is one shape repeated: the
schema machinery is checked in four places and each place trusts a different
subset of it. `_validate` dispatches on `type` and drops everything it does not
recognise; `next_valid_tokens` ignores the schema entirely; the extraction
pipeline's fallback is schema-valid; and the confidence estimate measures a
branch table. Type-correct and correct are different properties, and the lesson
has an instrument for only one of them.

### 1 — a schema without `type` validates everything

```python
validate_schema("anything", {"oneOf": [PRODUCT, SERVICE]})   ->  []
```

**ANSWER: the state the extension replaces is worse than "unsupported".**
`_validate` dispatches on `schema.get("type")` and has no `else` branch, so a
schema carrying only `oneOf` matches nothing and appends no error. It accepts
all 12 labelled documents — including the two that match neither branch — and
the bare string `"anything"`.

| | shipped | with `oneOf` | with a discriminator |
|---|---:|---:|---:|
| accepted, of 12 | **12** | 7 | 10 |
| matching both branches | — | **3** | **0** |
| matching neither | — | 2 | 2 |

**FINDING: "exactly one" is the hard half, because the objects are open.** The
lesson's validator ignores unknown keys, so anything carrying both a `price` and
a `rate` satisfies Product *and* Service, and `oneOf` rejects it. Three of the
twelve are rejected for being too informative rather than malformed. A Product
carrying a stray `duration_days` survives only because Service also wants a
`rate`.

**CONTROL: a required `kind` with a one-value `enum` per branch** takes ambiguity
3 → 0 and acceptance 7 → 10, because a document can then satisfy only the branch
it names. The other standard fix is not available: `additionalProperties` is not
implemented by `_validate` at all, so the objects cannot be closed.

### 2 — three breaking changes the validator cannot feel

Breaking is read in the **producer** direction — a document v1 accepted that v2
rejects — because that is the direction a versioned extraction schema breaks in,
and the only one with a mechanical witness. Every labelled change ships one.

| change | diff says | validator feels it? |
|---|---|---|
| added required `sku` | breaking | ✅ |
| removed required `price` | non-breaking | ✅ |
| `price` string → number | breaking | ✅ |
| added optional `rating` | non-breaking | ✅ |
| `minimum` 0 → 10 on a **number** | breaking | ✅ |
| `minimum` 0 → 10 on an **integer** | breaking | ❌ |
| `enum` added on a **string** | breaking | ✅ |
| `enum` added on an **integer** | breaking | ❌ |
| `maximum` 5 → 10 | non-breaking | ✅ |
| number → integer | breaking | ✅ |

**ANSWER: the diff is right 10 of 10. The validator agrees 8 of 10.** For those
two, deploying the new schema changes nothing it was meant to change — and
rolling it back changes nothing either.

**MECHANISM: the constraints live inside the type branches.** `minimum` and
`maximum` are read only where `schema_type == "number"`, `enum` only where it is
`"string"`. So:

```python
validate_schema({"qty": -5},  {"type": "integer", "minimum": 0})   ->  []
validate_schema({"lvl": 99},  {"type": "integer", "enum": [1, 2]}) ->  []
```

**FINDING: narrowing `number` to `integer` loosens the schema.** `-5` is
rejected by `{"type": "number", "minimum": 0}` and accepted by
`{"type": "integer", "minimum": 0}`, because the range check does not exist in
the integer branch. A change the diff correctly calls breaking is, in this
validator, a relaxation.

**CONTROL: hoisting the two checks out of the type dispatch** takes agreement
8 → 10 of 10 with no other verdict moving. The diff was never the unreliable
half.

### 3 — the decoder stops where the validator fails

The mask is a prefix automaton over an enumerated language: every serialisation
of `PRODUCT_SCHEMA` over a fixed value set and every key ordering — 30
documents. A token is valid after a prefix exactly when some document starts with
`prefix + token`, and `<EOS>` exactly when the prefix *is* a document.

**ANSWER: 1 token wide at 21 of 27 positions**, over a 100-token vocabulary:

```text
{   "   product   "   :   ␣   "   Sony   "   ,   ␣   "   price   "   :  …
1%  1%     7%     1%  1%  1%  1%   2%    1%  1%  1%  1%    6%    1%  1%
```

Mean **1.63%**. The 7% is the only position where more than one property name
could still begin. Constrained decoding against a closed schema is not a soft
preference — it is a single legal token nearly every step.

**FINDING: `next_valid_tokens` never reads its `schema` argument.** Its output
is identical for `PRODUCT_SCHEMA` and `{}` at every state on the path. It is a
JSON-*grammar* masker, so it cannot know that `"produkt"` is not a property.

**FINDING: it stops where the validator fails.**

```text
next_valid_tokens('{"product": "Sony"}', PRODUCT_SCHEMA)  ->  ['<EOS>']
validate_schema(  {"product": "Sony"},   PRODUCT_SCHEMA)  ->  ['.price: required field
                                                                 missing',
                                                               '.in_stock: required field
                                                                 missing']
```

The schema-aware mask refuses the end token there, because no document in the
language stops at that prefix. Required-field completeness is a property of the
language, not of the parser.

**FINDING: five states on the path return `['any']`** — 100% of the vocabulary,
which is no mask at all — and its class labels `"0-9"` and `"a-z"` expand to 10
and 26 tokens. Mean permitted fraction **30.46%** against the schema-aware
**1.63%**: nineteen times as wide.

### 4 — type compliance is 100%, and the fallback is what guarantees it

Fifty descriptions generated from a labelled template table, so the label is the
generator's input and cannot drift from the text.

| metric | result |
|---|---:|
| type compliance | **50 / 50** |
| exact match | **0 / 50** |
| distinct outputs the extractor can produce | **4** |
| descriptions hitting the fallback | 20 / 50 |

**ANSWER: one metric is at its ceiling and the other at its floor.** Every
output validates; every output is wrong.

**FINDING: the fallback is schema-valid.**

```json
{"product": "Unknown", "price": 0, "in_stock": false}   ->  validate_schema: []
```

Every required field present, a non-negative price. The pipeline's failure mode
is a valid document — exactly the failure a schema cannot catch.

**FINDING: the hardest field is `price`, at 0 of 50.**

| field | hits / 50 | why |
|---|---:|---|
| `price` | **0** | no digit in any description reaches the output |
| `product` | 15 | the five descriptions each of three hard-coded names |
| `categories` | 20 | two families whose fixed category list happens to match |
| `in_stock` | 25 | a coin flip scored against a constant |

**FINDING: the retry path is dead.** All four branch documents validate, so
`extract_with_retry` returns on the first attempt every time — the only
`attempt` value requested across 50 runs is `0`, and the `attempt >= 1` Sony
variant that drops `categories` is unreachable.

### 5 — resampling measures the branch table, not the model

The exercise's first option is closed: `simulate_llm_extraction` returns a `str`
and nothing in the lesson carries logprobs. So the second option, exactly as
written — three extractions, per-field majority agreement, flag anything below
1.00.

| what varies across the three runs | fields flagged | of which wrong |
|---|---:|---:|
| nothing (three identical runs) | **0** | 0 |
| `attempt` 0, 1, 2 | 4 | 4 |
| the input (three paraphrases) | **32** | **28** |

*(80 fields total; 50 of them are wrong.)*

**ANSWER: nothing is ever flagged.** `simulate_llm_extraction` is a pure
function of `(text, attempt)`, so the three runs collapse to one document and
every field scores 1.00. The mechanism the exercise offers as a fallback
measures determinism, and this extractor is deterministic.

**FINDING: sweeping `attempt` moves one field on one family.** The Sony branch
drops `categories` at `attempt >= 1`, scoring it 0.67; everything else stays
1.00. The confidence is a measurement of the branch table.

**FINDING: consistency cannot see a confidently wrong field.** On the
descriptions the branch table misses, the fallback returns `"Unknown"` three
times of three — confidence 1.00 on a field that is certainly wrong. Flagged
and wrong have **zero** overlap.

**CONTROL: perturb the input, not the model.** Lowercase it, remove the brand
token, append a distractor category. That flags 32 fields, 28 of them wrong —
four false positives against resampling's zero true positives. Consistency is
informative when what varies is the input the extractor actually reads.
