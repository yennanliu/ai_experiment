<!-- generated:start -->
# 13-tools-and-protocols / 22-skills-and-agent-sdks

Solutions to all 6 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/13-tools-and-protocols/22-skills-and-agent-sdks/) · upstream spec
`phases/13-tools-and-protocols/22-skills-and-agent-sdks/docs/en.md`

```bash
uv run demo practice run 22-skills-and-agent-sdks --ex 1
uv run demo explain 22-skills-and-agent-sdks --ex 1
uv run pytest demos/phases/13-tools-and-protocols/22-skills-and-agent-sdks
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Classify five workflows from your own team using `TaskShape`. Defend every case where you cho… | code | T0 | `ex01_more_than_one_primitive_is_arithmetic_not_a_judgement.py` |
| 2 | Add boundary tests proving that a 500-character `compatibility` value passes and a 501-charac… | code | T0 | `ex02_the_limit_is_characters_and_three_fields_have_three_of_them.py` |
| 3 | Add one runtime extension to the allowlist. Write a test proving the same file is still disti… | code | T0 | `ex03_allowlisting_makes_it_valid_and_the_report_still_names_it.py` |
| 4 | Split a 400-line prompt into `SKILL.md`, one reference, one script contract, and one output t… | code | T0 | `ex04_only_one_of_the_four_files_is_validated.py` |
| 5 | Design a failure response for a skill that references an unavailable MCP tool. Do not silentl… | code | T0 | `ex05_the_substitute_is_tempting_because_it_is_a_superset.py` |
| 6 | Review an existing skill and label every sentence as routing, procedure, policy, reference po… | code | T0 | `ex06_a_sentence_that_belongs_in_two_places_belongs_in_neither.py` |
<!-- generated:end -->

## Answers

All six are T0 and stdlib, and all six ship code.

A skill is a file with a validator, and the recurring finding is **how little
of the file the validator sees**: the frontmatter is checked, the body is
checked for existing, and everything the other five exercises care about —
companion files, tool availability, sentence placement — is a convention with
no checker at all. Two of the six therefore write the checker as part of the
answer.

### 1 — more than one primitive is arithmetic, not a judgement

**ANSWER: five workflows; each multi-primitive case loses exactly one
primitive when a flag is dropped.**

**FINDING: the count is the number of true flags, with no interaction.** Six
flags → six primitives; none → `("prompt",)`. Setting a flag *is* choosing a
primitive.

**FINDING: the output order is the function's, not the task's.** A reader
cannot tell which primitive is primary.

**FINDING: the default is the weakest primitive, reached by silence — and
contradictions are admitted.** `deterministic_logic` + `repeatable_method`
returns ordinary code *and* an Agent Skill. **64** shapes, all accepted.

### 2 — the limit is characters, and three fields have three of them

**ANSWER: 500 passes, 501 fails with `compatibility-too-long`.**

**FINDING: absent, empty and over-length are three outcomes, not two.** A
length-only test would treat the optional field as required.

**FINDING: three bounded fields, three limits, stated nowhere.**

| field | limit |
|---|---:|
| `name` | 64 |
| `description` | 1024 |
| `compatibility` | 500 |

**FINDING: the limit counts characters, so the byte length is unbounded.** A
500-character multibyte value is **1500** bytes and passes; a 501-character
ASCII value is **501** bytes and fails.

### 3 — allowlisting makes it valid, and the report still names it

**ANSWER: allowlisting flips `valid` and leaves `runtime_extensions`
unchanged.** That is what keeps the two files distinguishable.

**FINDING: validity is a fact about the pair; the extension list is a fact
about the file.** "Runs here" reads `valid`; "runs anywhere" reads
`runtime_extensions` and ignores the host.

**FINDING: the allowlist is per-field.** One issue per unpermitted field.

**FINDING: `core_fields` is the complement of a 6-name hard-coded set, and
there is no version on it** — which is what would let a skill declare which
contract it was written against.

### 4 — only one of the four files is validated

**ANSWER: 400 lines into 4 files, reconciling exactly, SKILL.md at 36
lines.**

**FINDING: the validator sees one of the four.** Three parameters, none a
file list. The one-responsibility rule is a convention with no checker —
which is why this solution writes one.

**FINDING: the frontmatter is the only machine-checked part.** A monolithic
SKILL.md validates identically.

**FINDING: the description is what makes progressive disclosure work.** It
survives at 60 characters while the body behind it shrank **91%**.

### 5 — the substitute is tempting because it is a superset

**ANSWER: refuse, naming the skill, the missing tool and the host; perform no
work.**

**FINDING: the substitute is a strict superset, which is the whole problem.**
Binding `notes_export` to `files_write` completes the task — so an output
test passes — while adding `fs:write`, `fs:delete`, `net:post`. The skill is
granted filesystem write by a *resolver*, not by its `allowed-tools`.

**FINDING: the validator checks the field's shape and never its contents.** A
tool that never existed validates. Availability is a runtime fact; validity
is a static one.

**FINDING: refusing needs the same parse the substitution would have used.**
The cost of refusing is zero, so the choice is entirely policy — which is why
it has to be written where a reviewer can see it.

### 6 — a sentence that belongs in two places belongs in neither

**ANSWER: 14 sentences, 5 labels, 4 files — and **2** split before they could
be placed**, which is why 16 pieces come out of 14 sentences.

**FINDING: the mixed sentences are mixed in one direction.** Both carry
procedure *and* policy — a constraint written where the action is. None mixes
routing, because routing is one sentence and everything else is body.

**FINDING: the description is the one sentence the validator reads.** The
other **13** could be in any order in any file.

**FINDING: moving a sentence changes no verdict.** The review improves what a
model reads and is invisible to every check the module ships — exercise 4's
finding reached from the other direction.
