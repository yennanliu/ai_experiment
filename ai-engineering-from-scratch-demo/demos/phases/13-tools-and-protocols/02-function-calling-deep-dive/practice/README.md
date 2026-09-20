<!-- generated:start -->
# 13-tools-and-protocols / 02-function-calling-deep-dive

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/13-tools-and-protocols/02-function-calling-deep-dive/) · upstream spec
`phases/13-tools-and-protocols/02-function-calling-deep-dive/docs/en.md`

```bash
uv run demo practice run 02-function-calling-deep-dive --ex 1
uv run demo explain 02-function-calling-deep-dive --ex 1
uv run pytest demos/phases/13-tools-and-protocols/02-function-calling-deep-dive
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py` and verify that the three provider declaration JSONs all serialize the sam… | code | T0 | `ex01_the_one_quirk_it_names_is_the_one_the_translator_misses.py` |
| 2 | Add a `ListToolsResponse` parser for each provider that extracts the tool list a model return… | code | T0 | `ex02_the_asymmetry_is_three_ways_not_one.py` |
| 3 | Implement `tool_choice` conversion: map a canonical `ToolChoice(mode="force", tool_name="x")`… | code | T0 | `ex03_the_mode_it_asks_for_is_a_wire_name_not_a_canonical_one.py` |
| 4 | Pick one of the three providers and read its function-calling guide end to end. Find one fiel… | explain | T0 | prose, below |
| 5 | Write a test vector: a tool call whose arguments violate the declared schema. Run it through… | code | T0 | `ex05_the_canonical_tool_declares_a_value_it_cannot_accept.py` |
<!-- generated:end -->

## Answers

All five are T0 and stdlib. Four ship code; exercise 4 is prose, below, citing
the lesson section it rests on.

The lesson's thesis is that one canonical `Tool` serializes cleanly into three
provider dialects. Running the translators finds the cost of that claim:
**one of the canonical `Tool`'s four fields survives only one of the three
translations, and the one schema quirk the exercises single out is the one the
translator does not handle.**

### 1 — the one quirk it names is the one the translator misses

**ANSWER: they do not all serialize the same object.** The canonical `Tool`
carries `description, input_schema, name, strict`, and `strict` reaches
`{openai: True, anthropic: False, gemini: False}` — **1 of 3**. `to_anthropic`
passes three fields and `to_gemini` passes three; the field OpenAI uses to
guarantee schema compliance is dropped by both without comment.

**ANSWER: the enum parameter is already there.** `units` is declared
`enum: ["celsius", "fahrenheit"]` in `WEATHER.input_schema`, so the exercise's
"modify the canonical tool to add" step is already done by the lesson.

**FINDING: Gemini needs the OpenAPI quirk handled and does not handle it.**
`_gemini_schema` uppercases `type` only `if isinstance(v, str)`. `units` is
declared `["string", "null"]` — a *list* — so it falls through untouched:

| | emitted `type` |
|---|---|
| `properties.city` | `STRING` |
| `properties.units` | **`["string", "null"]`** |
| root | `OBJECT` |

OpenAPI 3.0, the dialect Gemini follows, has no union types at all; the correct
emission is `type: "STRING"` with `nullable: true`. The exercise is right that
only Gemini needs the quirk handled. The translator is what is wrong.

**FINDING: `additionalProperties` is the field Gemini drops on purpose.**
Stripped by name, so Gemini sees a schema permitting undeclared keys while the
other two see one forbidding them — a correct translation of an untranslatable
field, and the only place the three are *meant* to disagree. That makes it the
control for the `strict` loss above.

### 2 — the asymmetry is three ways, not one

**ANSWER: none of the three has a tool-list response, so all three parsers are
the same parser.** Searching the module for `list_tools`, `tools/list`,
`ListTools`, `listTools` finds **0** shapes. In all three APIs the tool list
travels in the *request*. A model cannot return a tool list because it never
held one; it was handed one.

**FINDING: the asymmetry the exercise names is real and points the wrong way.**
"OpenAI does not have one natively" implies the other two do. The three
`parse_*` functions the lesson ships all read *calls* — keyed off `tool_calls`,
`tool_use`, `functionCall` — never declarations. The asymmetry is not OpenAI
against Anthropic and Gemini; it is all three against MCP.

**FINDING: which is the whole reason MCP exists.** MCP's `tools/list` is a
response shape, so the registry can change between turns without the host
redeploying. The three provider APIs bind the registry at request time. That is
the difference Lesson 06 opens with, visible here as a parser with nothing to
parse.

**FINDING: the round trip recovers every name and only two of three schemas.**
Reading each provider's declaration back recovers `get_weather` from **3 of 3**,
so the parser is writable — it just has to read the request the host sent. But
only **2 of 3** schemas come back identical: OpenAI's and Anthropic's are
byte-for-byte and Gemini's is not. Discovery over these APIs would be lossy for
one provider even once the shape existed.

### 3 — the mode it asks for is a wire name, not a canonical one

**ANSWER: `force` and `none` convert cleanly; `any` raises in all three.**

| mode | OpenAI | Anthropic | Gemini |
|---|---|---|---|
| `force` | `{"type":"function","function":{"name":…}}` | `{"type":"tool","name":…}` | `ANY` + `allowed_function_names` |
| `none` | `"none"` | `{"type":"none"}` | `{"mode":"NONE"}` |
| `any` | `ValueError: any` | `ValueError: any` | `ValueError: any` |

**FINDING: "any" is Anthropic's wire name for the mode the lesson calls
`required`.** The canonical vocabulary is `auto / none / required / force`, and
`tool_choice_anthropic` maps `required` **to** `{"type": "any"}`. The exercise
asks you to feed a provider's output back in as canonical input: the mode it
means exists, but the name it uses is the one on the other side of the
translation.

**FINDING: the lesson's own diff table is where the vocabulary leaks.** Reading
it down the Anthropic column gives `auto, none, any, tool` — two are also
canonical names and two are not. `required` is the only mode whose Anthropic
spelling differs from its canonical name, and it is the one the exercise gets
wrong.

**FINDING: and `force` is the only mode that needs the tool name.** Supplying a
`tool_name` changes nothing for `auto`, `none` or `required`. Gemini is the only
provider whose `force` is a *modifier* on another mode — its `force` and
`required` carry the same `mode` and differ only by `allowed_function_names`,
while the other two change shape entirely.

### 4 — `strict` is the field with no equivalent

*Cites "Shape diffs, field by field".*

**The field is OpenAI's `strict`, and the reason it has no equivalent is that it
is not a schema field at all — it is a decoding instruction.**

The lesson's own table names all three in its last row:

| Strict schema | OpenAI | Anthropic | Gemini |
|---|---|---|---|
| | `strict: true` | schema-is-schema (always enforced) | `responseSchema` at request level |

The other two candidates the exercise offers are genuinely provider-specific but
are *controls*, not schema fields. Anthropic's `disable_parallel_tool_use` sits
in `tool_choice`, and Gemini's `allowed_function_names` sits in
`function_calling_config` — both constrain which calls may be emitted, and both
have workarounds a host can implement itself by filtering the tool list it
declares. `strict` has none, because it changes *how the tokens are sampled*.

That is the asymmetry worth recording. With `strict: true` the provider
constrains decoding so an invalid call cannot be generated. Anthropic's
"schema-is-schema" is a statement about model compliance, not a guarantee — the
lesson's own limits section says "no strict-mode flag; schema is a contract and
the model tends to comply." Gemini's `responseSchema` is request-level and
governs the *response* format rather than tool arguments.

So the three rows in that table are not three spellings of one feature. They are
a guarantee, a tendency, and a different feature. Exercise 1 measures what
happens to the guarantee when it crosses the translator: `strict` survives
**1 of 3**, and the canonical layer that was supposed to make the providers
interchangeable is exactly where it is lost.

### 5 — the canonical tool declares a value it cannot accept

| vector | `validate` returns |
|---|---|
| `{"city": "Bengaluru"}` | `missing required field 'units'` |
| `{"city": 42, …}` | `expected string, got int` |
| `units: "kelvin"` | `value 'kelvin' not in enum […]` |
| `units: null` | **`value None not in enum […]`** |
| `…, "evil": 1` | `[]` |

**ANSWER: four vectors fire four errors, and the extra field fires none** —
although the canonical schema sets `additionalProperties: false`. Lesson 01's
validator has no support for the field.

**FINDING: `units: null` is declared valid by its type and rejected by its
enum.** `"type": ["string", "null"]` permits null, `enum` does not contain it,
and `required` lists `units`. No value of `units` exercises the null branch: the
schema declares a nullability it then forbids.

**FINDING: the union type is never type-checked at all.** `validate` dispatches
with `== "string"` and `== "number"`, so a *list* of types matches neither branch
and falls through. Removing the enum as a control, `units` accepts `7`, `None`,
`"celsius"` and `{"a": 1}` — **4 of 4**, all clean. The only thing rejecting a
bad `units` today is the enum; the type is decorative.

**ANSWER: OpenAI, for strictness — and the reason is the field the other two
drop.** `strict: true` is enforced by constrained decoding, so a violating call
is never generated rather than caught afterwards. Anthropic and Gemini validate
server-side and return an error, which still costs a turn. Strictness enforced
before generation is the only kind that costs nothing — and per exercise 1, it
survives 1 of 3 translations.
