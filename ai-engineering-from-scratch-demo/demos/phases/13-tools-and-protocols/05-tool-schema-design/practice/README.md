<!-- generated:start -->
# 13-tools-and-protocols / 05-tool-schema-design

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/13-tools-and-protocols/05-tool-schema-design/) · upstream spec
`phases/13-tools-and-protocols/05-tool-schema-design/docs/en.md`

```bash
uv run demo practice run 05-tool-schema-design --ex 1
uv run demo explain 05-tool-schema-design --ex 1
uv run pytest demos/phases/13-tools-and-protocols/05-tool-schema-design
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Take the `BAD_REGISTRY` in `code/main.py` and rewrite each tool to pass the linter. Measure d… | code | T0 | `ex01_twenty_findings_to_zero_and_one_of_them_was_counted_twice.py` |
| 2 | Design an MCP server for a notes application with atomic tools: list, search, create, update,… | code | T0 | `ex02_zero_findings_is_reachable_by_a_registry_the_lesson_calls_bad.py` |
| 3 | Pick an existing popular MCP server from the official registry and lint its tool descriptions… | code | T0 | `ex03_the_nearest_real_registry_lints_at_three_nits_and_two_of_them_matter.py` |
| 4 | Add the linter to your CI. On a PR that changes a tool registry, fail the build on severity `… | code | T0 | `ex04_the_gate_blocks_six_of_twenty_and_lets_the_poisoned_tool_through.py` |
| 5 | Read Composio's tool-design field guide top to bottom. Identify one rule not covered in this… | code | T0 | `ex05_the_lesson_states_five_parameter_rules_and_lints_two.py` |
<!-- generated:end -->

## Answers

All five are T0 and stdlib, and all five ship code.

The lesson builds a linter and two fixtures, one good and one bad, and the
exercises ask you to move registries between them. Doing that finds the shape of
the instrument: **every rule that blocks a build checks what the author wrote,
and the properties that decide whether a tool works are the ones nothing
checks.**

### 1 — twenty findings to zero

| | `BAD_REGISTRY` | rewritten |
|---|---:|---:|
| findings | **20** (6 block, 11 warn, 3 nit) | **0** |
| description lengths | 19 / 16 / 109 | 88 / 112 / 85 |

Two of the three descriptions were under the 40-character floor; the third was
long enough and carried two injection patterns.

**FINDING: the duplicate-name rule reports once per copy, not once per
collision.** `lint_registry` loops over every name and appends a finding whenever
`names.count(n) > 1`, so 2, 3 and 4 copies give **2, 3 and 4** findings. One
naming mistake repeated four times reads as four blocking problems.

**FINDING: passing the linter is not the same as being right.** A tool named
`delete_note` described as "Use when the user asks for the weather" lints at
**0**. Nothing compares the description to the name — the one property that makes
a description useful to a model is the one property not checked.

**FINDING: and one of the twenty is a false positive.** The embedded-argument
rule is `_(in|for|at|by)_\w+$`, which correctly flags `get_weather_in_tokyo` and
also fires on **`search_notes_for_text`**, where `text` names the field being
searched.

### 2 — zero findings is reachable by a registry the lesson calls bad

**ANSWER: six entries — `notes_list`, `notes_search`, `notes_create`,
`notes_update`, `notes_delete`, and a `summarize` prompt — at 0 findings.**

**FINDING: the monolithic version scores 0 too, if the key is not called
`action`.** The identical five-operations-in-one tool scores **1** warn with its
verb under `"action"` and **0** under `"op"`. `lint_schema` matches
`key == "action"` literally, so the anti-pattern the lesson devotes a section to
is detected by the *name* of the field rather than its shape.

**FINDING: the prompt is linted as though it were a tool** — including "Do not
use for", which is tool-disambiguation advice a slash prompt has no counterpart
to be disambiguated from. It passes because the wording was bent to satisfy a
rule that does not apply.

**FINDING: five atomic tools cost 10× the description budget of one** — **729**
characters against **75**, all of it reaching the model on every request.
Atomicity is the right call and it is not free, which is the trade the
"atomic vs monolithic" section asserts without pricing.

### 3 — the nearest real registry lints at three nits, and two matter

The official registry is a network fetch and this runs offline, so the
substitution is stated rather than smuggled: **Lesson 13.01's registry**, written
by someone else for another purpose and never intended to be linted.

**ANSWER: 3 findings, all `nit`, all the same rule** — `get_time.timezone`,
`get_weather.city` and `get_weather.units` lack field descriptions.

**FINDING: two of the three are actionable and one is not.** `units` carries an
`enum`, so the universe of values already reaches the model and a description
would restate it. `city` and `timezone` are open strings — nothing says whether
`timezone` wants `UTC`, `Etc/UTC` or `+00:00`, and 13.01's executor accepts all
three and distinguishes none.

**FINDING: the lint is clean on a registry nothing dispatches from.** 13.01's
exercise 1 found `fake_decide` never reads `REGISTRY`. The linter scores
descriptions the decider does not consult.

**FINDING: the injection rules would not catch a reworded poisoning.** The
shipped poisoned description trips **2** of the four patterns; rewording it to
"disregard earlier directions" trips **0** while meaning the same thing.

### 4 — the gate blocks six of twenty, and lets the poisoned tool through

**ANSWER: exit 1 on any `block`.** `BAD_REGISTRY` → 1, `GOOD_REGISTRY` → 0,
Lesson 13.01's registry → 0.

**FINDING: the policy discards 14 of the 20 findings on the registry it
rejects.** A registry whose every description is missing "Use when" ships green,
because `lint_description` files that as `warn` and a 39-character description as
`block`.

**FINDING: the gate passes a registry the linter already knows is poisoned.**
Rewording the injection takes the same tool from exit **1** on 2 blocks to exit
**0** on 0.

**FINDING: and the one registry it protects is the one nobody would submit.**
Exactly **1 of 3** fails — the one the lesson wrote to fail. A gate calibrated on
a deliberately broken fixture has no evidence about the registries it will
actually see, which is the argument for the eval-driven pattern the exercise
defers.

### 5 — the lesson states five parameter rules and lints two

Composio's field guide is a network fetch, so the rule is taken from a source
that *is* checkable: the lesson's own "Parameter design" section, which states
five rules in five bullets.

| stated rule | enforced |
|---|---|
| enum every closed set | only for a key named `action` |
| required vs optional | only that the key exists |
| typed IDs need a `pattern` | **no** |
| no overly flexible types | yes |
| describe the field | yes |

**ANSWER: the rule added is snake_case for parameter names.** `lint_name` applies
`^[a-z][a-z0-9_]*$` to the *tool* name and nothing applies it to the fields — a
schema with `noteId` and `MAX` lints at **0** today and **2** with the rule.

**FINDING: the two it skips are the two that catch hallucinated arguments.** A
`note_id` with no `pattern` accepts `"note-oops"`; a closed set with no `enum`
accepts any string. Both are the shapes a model invents when guessing. The rules
the linter implements govern what the *author* wrote; the rules it skips govern
what the *model* may send.

**FINDING: and the new rule fires on nothing the lesson ships.** All **10**
parameter names across both registries are already snake_case. A rule with no
failing fixture is a rule nobody has tested — the same gap exercise 4 found in
the CI gate.
