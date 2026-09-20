<!-- generated:start -->
# 13-tools-and-protocols / 04-structured-output

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/13-tools-and-protocols/04-structured-output/) · upstream spec
`phases/13-tools-and-protocols/04-structured-output/docs/en.md`

```bash
uv run demo practice run 04-structured-output --ex 1
uv run demo explain 04-structured-output --ex 1
uv run pytest demos/phases/13-tools-and-protocols/04-structured-output
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Add a fourth test case whose `total_usd` is a negative number. Confirm th… | code | T0 | `ex01_the_minimum_holds_and_three_constraints_beside_it_do_not.py` |
| 2 | Extend the validator to support `oneOf` with a discriminator. The common case: `line_item` is… | code | T0 | `ex02_an_unsupported_keyword_accepts_everything_instead_of_failing.py` |
| 3 | Write the same Invoice schema as a Pydantic BaseModel and compare `model_json_schema()` outpu… | explain | T0 | prose, below |
| 4 | Measure refusal rates. Construct ten inputs that should not be extractable (a song lyric, a m… | code | T0 | `ex04_none_of_the_ten_reaches_the_refusal_branch.py` |
| 5 | Read OpenAI's structured outputs guide top to bottom. Identify the one construct it explicitl… | code | T0 | `ex05_the_forbidden_construct_is_the_optional_property.py` |
<!-- generated:end -->

## Answers

All five are T0 and stdlib. Four ship code; exercise 3 is prose, below, citing
the lesson section it rests on.

The lesson's closing line is that "strict-mode eliminates parse_error and
violation branches at the provider level". Four of the five exercises end up
measuring the gap in that sentence: **strict mode constrains the shape of a
document, and every interesting property of an invoice is not a shape.**

### 1 — the minimum holds, and three constraints beside it do not

**ANSWER: rejected at `$.total_usd` with `below minimum 0`** — exactly one
error. The `number` branch returns early only on a type mismatch, so the shared
`minimum` check below it is reached for every well-typed number.

Three sibling constraints on the same schema do not hold up:

| perturbation | errors |
|---|---:|
| `total_usd: -5.0` | **1** ✓ |
| `sku: "ABC-1\n"` | 0 |
| `line_items: []` | 0 |
| `total_usd: 999.0` against a 10.00 item | 0 |

**FINDING: `pattern` lets a SKU end in a newline.** `validate` uses `re.match`,
and Python's `$` matches *before* a trailing newline. It is `re.fullmatch` that
means what the anchors look like they mean.

**FINDING: `line_items` has no `minItems`**, so an invoice with nothing on it is
valid — and the lesson's own parse-error fixture is built on that shape.

**FINDING: and `total_usd` is never checked against the line items.** The
schema's keywords are `enum, items, maxLength, minLength, minimum` — all of them
describe one field in isolation. The invariant that makes an invoice an invoice
is not expressible in JSON Schema at all.

### 2 — an unsupported keyword accepts everything instead of failing

**FINDING: the lesson's validator ignores `oneOf` entirely, which is worse than
rejecting it.** `validate({"oneOf": [...]}, 5)` returns **no errors**. There is
no `type` key, so every branch of the dispatch is skipped and the shared
constraint checks find nothing to check. A schema built on `oneOf` validates
*every* input, and the failure is silent — an unsupported keyword reads as an
unconstrained one.

**ANSWER: `oneOf` plus a `kind` discriminator, dispatching on the tag before
validating.** Both well-formed documents validate clean; a service whose `hours`
is a string gives one error at `$.hours`.

**FINDING: dispatching on the tag is what makes the errors usable.** The same
bad service checked against *both* branches gives **6** errors, **5** of them
about the product branch the document never claimed to be — a missing `sku`, a
missing `unit_usd`, two properties "not allowed", and a tag outside `product`'s
enum. With the tag it is **1**.

**FINDING: the strict-mode subtlety is that the tag must be an `enum` of one.**
A union is only decidable if the tag selects exactly one branch;
`"kind": {"type": "string"}` would let both branches accept any tag. Pinning it
makes the branch selectable by a grammar, which is what constrained decoding
needs.

### 3 — the field Pydantic adds is `title`

*Cites "Pydantic, the Python binding".*

**Pydantic sets `title` by default, on the model and on every field, and the
hand-rolled schema omits it throughout.** For the lesson's `Invoice`,
`model_json_schema()` returns `"title": "Invoice"` at the root and
`"title": "Customer"`, `"title": "Line Items"`, `"title": "Total Usd"` on the
three properties — each derived from the Python identifier by replacing
underscores and title-casing. `INVOICE_SCHEMA` has no `title` anywhere.

Two things follow, and the second is the one worth keeping.

The mechanical one: `title` is annotation, not constraint. No validator branches
on it, so its presence changes nothing about which documents are accepted —
which is why the hand-rolled version can omit it without changing behaviour.

The one that matters: **`title` is where the field's name reaches the model.**
The three provider dialects in Lesson 13.02 serialize a tool's schema into the
prompt, and a property called `total_usd` with `"title": "Total Usd"` gives the
model two chances to understand what it is. A hand-rolled schema that drops the
title is relying on the key name alone. That is usually fine and is exactly the
kind of thing that stops being fine on a field named `amt` or `dt`.

The corollary is that Pydantic's default is the wrong default for a *tool*
schema, and right for a documentation one: it derives `title` from the
identifier, so it can only restate the key it sits beside. `"Total Usd"` adds
nothing to `total_usd`. The field that would earn its place is `description`,
which Pydantic sets only from a docstring or `Field(description=...)` — and
which neither schema here uses at all. Exercise 1's finding is the sharp end of
this: nothing in either schema says the total should equal the sum of the line
items, and `description` is the only field in JSON Schema where that sentence
could go.

### 4 — none of the ten reaches the refusal branch

**ANSWER: 0 of 10 are classified as refusals** — 7 `parse_error`, 3 `violation`.
The branch is reachable only through the literal prefix `__REFUSAL__`, a
sentinel the *provider* emits. A song lyric arriving as a song lyric is not a
refusal to this code; it is malformed JSON.

**FINDING: three of the ten are worse than the seven.** `{}`, `null` and `[]` are
valid JSON, so they clear the parse branch and land in `violation` with **4, 1
and 1** errors — indistinguishable at this layer from a model that tried and got
the shape wrong, which is the case the retry logic is for. A retry on `{}` is
reasonable; a retry on a song lyric is a loop.

**FINDING: the sentinel is the whole mechanism, and it is a string prefix.**
`process_model_output` checks `startswith` before it parses anything, so an
output carrying a full JSON body still classifies as a refusal if it happens to
begin with those eleven characters. Real providers signal this out of band — a
typed `refusal` field, a `stop_reason` — precisely so it cannot collide with
content.

**FINDING: so the exercise needs a provider, for a reason it does not give.** The
ratio it asks for is refusals against *hallucinated outputs*, and a
hallucination is a well-formed document with invented values. The invoice
claiming **999.0** against a 10.00 line item classifies as **`ok`, 0 errors**.
The measurement needs a provider not because refusals are hard to simulate, but
because the other arm of the ratio is invisible to the validator.

### 5 — the forbidden construct is the optional property

**ANSWER: a field in `properties` and absent from `required`.** Plain JSON Schema
treats absence as permitted; strict mode requires every declared property to be
listed, because a grammar that may or may not emit a key cannot be built from the
schema alone. The refactor is mechanical: add the key to `required`, widen its
type to include `null`.

| document | loose | strict |
|---|---:|---:|
| no `po_number` | 0 errors | **1** |
| `po_number: null` | **1** | 0 |
| `po_number: "PO-42"` | 0 | 0 |

The refactor does not preserve the accepted set. It moves optionality out of the
schema's *shape* and into its *values*, which is the whole point.

**FINDING: the lesson's own Invoice schema is already strict-compatible on this
axis** — 4 of 4 top-level properties required, 3 of 3 line-item properties
required, `additionalProperties: False` at both levels. Which is why the exercise
has to supply its own example and says "non-essentially".

**FINDING: the refactor is correct as a schema and unenforceable by this
validator.** Widening `po_number` to `["null", "string"]` makes its type a
*list*, and `validate` dispatches with `== "string"` and `== "number"` — so a
list matches no branch and falls through. **0 of 3** nonsense values (`42`, `[]`,
`{"a": 1}`) are caught. The clean result on `po_number: null` above is therefore
not evidence the refactor works; it is evidence the validator stopped looking —
the same hole Lesson 13.02 finds on its own union-typed field.
