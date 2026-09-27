<!-- generated:start -->
# 18-ethics-safety-alignment / 16-red-team-tooling-garak-llamaguard-pyrit

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/18-ethics-safety-alignment/16-red-team-tooling-garak-llamaguard-pyrit/) · upstream spec
`phases/18-ethics-safety-alignment/16-red-team-tooling-garak-llamaguard-pyrit/docs/en.md`

```bash
uv run demo practice run 16-red-team-tooling-garak-llamaguard-pyrit --ex 1
uv run demo explain 16-red-team-tooling-garak-llamaguard-pyrit --ex 1
uv run pytest demos/phases/18-ethics-safety-alignment/16-red-team-tooling-garak-llamaguard-pyrit
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Compare the Llama-Guard-style classifier's detection rate on single-turn… | code | T0 | `ex01_the_guard_flags_4_of_5_single_turn_probes_and_8_of_13_campaign_turns_but_none_of_the_5_that_break_the_target.py` |
| 2 | Implement a new Garak probe: a base64-encoded harmful request. Measure its detection by the L… | code | T0 | `ex02_base64_takes_the_guard_from_4_of_5_to_0_of_5_and_one_decode_step_gives_back_exactly_4.py` |
| 3 | Extend the PyRIT-style converter chain with a "translate to French, then paraphrase" converte… | code | T0 | `ex03_appending_french_then_paraphrase_leaves_asr_at_5_of_5_because_paraphrase_is_a_no_op_on_french_and_bombe_still_says_bomb.py` |
| 4 | Read Llama Guard 3's hazard-category list. Identify two categories where the training data wo… | code | T0 | `ex04_on_152_benign_developer_lines_violent_crimes_flags_48_and_privacy_16_and_the_guard_flags_its_own_lesson_page.py` |
| 5 | Compare Garak and PyRIT's design principles. Argue for a deployment where each is the right t… | code | T0 | `ex05_a_patch_takes_garak_to_0_of_5_while_pyrit_still_breaks_5_of_5_and_a_regression_moves_garak_but_not_pyrit_asr.py` |
<!-- generated:end -->

## Answers

Every exercise runs or reads the lesson's `code/main.py`: a keyword
classifier standing in for Llama Guard (`guard_classify`), a five-probe
Garak loop (`garak_scan`) and a four-converter PyRIT loop
(`pyrit_campaign`), all against a toy target that refuses on keywords.
Attack payloads are the lesson's own five probe strings; no new harmful
content is added.

### 1 — the guard flags 4 of 5 single-turn probes and 8 of 13 campaign turns, but none of the 5 that break the target

**80% single-turn against 61.5% of campaign turns, and 0 of the 5 turns
that actually break the target.** Each of the five probe payloads was used
as a campaign seed:

| attack | flagged by the guard |
|---|---:|
| single-turn Garak probes | 4 / 5 |
| campaign turns executed | 8 / 13 |
| campaign turns that broke the target | 0 / 5 |

Every campaign breaks at the one turn the guard also misses: `encode` on
four seeds, and the seed turn of the probe that is already leetspeak.
`main()`'s own printout shows the same thing, with 4 probes detected and 2
of its 3 campaign turns flagged.

**The guard adds nothing the target does not already catch.** Across all 18
messages the target refuses exactly the ones the guard flags. Putting the
guard in front of the target leaves every break point where it was.

**The toy's "multi-turn" has no state, and state is what would catch it.**
`pyrit_campaign` sends one message per call with no history. A guard that
classifies the whole conversation stops 4 of the 5 campaigns. The one that
still gets through is the leetspeak seed, which the guard never flagged at
any turn. The paraphrase turn also resends the seed unchanged on 3 of 5
seeds.

### 2 — base64 takes the guard from 4 of 5 to 0 of 5, and one decode step gives back exactly 4

**0 of 5.** Five base64 probes were appended to `GARAK_PROBES` and run
through `garak_scan`. All five came back undetected and passed the target,
and every category scored 0.0 on them. The same payloads in plaintext are 4
of 5 detected and 4 of 5 blocked.

**On base64 the guard returns the same verdict every time.** The benign
corpus is a fixed fixture of 96 developer lines (12 idioms, 8 module names),
48 of which contain a trigger by accident, such as `proc.kill()` or an
`SSN_RE` pattern. The guard flags exactly those 48 in plaintext and 0 once
they are base64-encoded. With a decode step in front it flags the same 48
again. More probes cannot raise the plain guard's number, because it never
fires on base64 at all.

**Decoding before classifying restores the plaintext rate and no more.** A
guard that decodes clean base64 tokens first detects 4 of 5. The leetspeak
probe is still missed, because decoding strips only the layer the probe
added. Each stacked obfuscation needs its own normaliser.

### 3 — appending French-then-paraphrase leaves ASR at 5 of 5: paraphrase is a no-op on French, and "bombe" still says "bomb"

**Campaign ASR stays 5 of 5, and no break point moves.** The shipped chain
already breaks every seed by turn 3, and a campaign that stops at the first
success never reaches a fifth turn. Seeds each turn breaks on its own:

| turn | seeds broken alone |
|---|---:|
| seed | 1 / 5 |
| paraphrase | 1 / 5 |
| encode | 5 / 5 |
| paraphrase+encode | 5 / 5 |
| French+paraphrase | 4 / 5 |

When the new turn is inserted before `encode`, it becomes the break turn on
3 seeds and ASR is still 5 of 5.

**The paraphrase half does nothing.** `paraphrase_converter` rewrites the
English strings "make" and "help me", so it changes 0 of the 5 French
payloads.

**"bombe" contains "bomb".** The only French payload the target refuses is
the bomb seed, and the guard flags it too, 1 of 5. The cognate survives
translation, and a keyword filter that catches it by coincidence is still
not multilingual. Separately, `max_turns` defaults to 5 over four
converters, so a fifth turn exists only after an extension like this one.

### 4 — on 152 benign developer lines, violent_crimes flags 48 and privacy 16, and the guard flags its own lesson page

**Violent Crimes (S1) and Privacy (S7).** The corpus is a fixed fixture: 19
labelled developer idioms, each written once for each of 8 module names,
152 lines with no request for harm among them. The guard flags 88 of them.

| category | lines flagged |
|---|---:|
| violent_crimes (S1) | 48 |
| privacy (S7) | 16 |
| code_interpreter_abuse (S14) | 8 |
| indiscriminate_weapons | 8 |
| self_harm | 8 |

These are ordinary developer words. For S1 they are `proc.kill()`,
`os.kill`, "pod kill" and `equip_weapon`. For S7 they are an `SSN_RE`
pattern and a redacted "home address" column. A classifier trained on
harmful requests that use these words would learn the same associations.
The 64 lines that use alarming but harmless words ("execute", "exploit",
"attack surface", "inject") all pass, because the toy has only 18 triggers.

**Every agent-skills line is flagged as a violent crime.** Deleting "skill"
clears 16 of the 48 violent_crimes lines, and those 16 are all of the
agent-skills lines (`load_skill`, `skills_dir`). Without them violent_crimes
still flags 32 lines, twice privacy's 16, so the ranking does not depend on
this bug.

**The guard flags its own lesson page.** On `docs/en.md` it fires
violent_crimes and self_harm, triggered by "weapon", "self-harm" and "kill".
The first two come from the hazard-category list. Every "kill" sits inside
"skill". 8 of the 14 categories have no triggers, so they never fire: their
false-positive rate is zero by construction, and so is their recall.

### 5 — a patch takes Garak to 0 of 5 while PyRIT still breaks 5 of 5, and a regression moves Garak but not PyRIT's ASR

**Use Garak for nightly regression and PyRIT for pre-release.** Both tools
were run against two edits to the target. The patch blocks the one probe
Garak reported passing. The regression is a model update that stops
refusing one phrase.

| target | Garak probes passing (calls) | PyRIT seeds broken (calls) |
|---|---|---|
| shipped | 1 / 5 (5) | 5 / 5 |
| patched | 0 / 5 (5) | 5 / 5, all at `encode` (15) |
| regressed | 2 / 5 | 5 / 5 |

After the patch Garak shows all green, but PyRIT still breaks every seed.
Garak can only replay the questions it already has, so fixing the issues it
reported leaves it stale. PyRIT is the tool for a new model or agent after
the scanner's findings are fixed.

After the regression Garak names the probe that flipped
(`roleplay_crime`). PyRIT's ASR does not move, and only one seed's break
turn changes, from `encode` to `seed`. A campaign that stops at its first
success saturates, so its ASR cannot show a regression. Garak can, at a
fixed cost per run, which suits a chat product whose prompt or model changes
every week.

**The toy Garak's detector never reads the output.** Against a target whose
every answer the guard flags and that never refuses, `garak_scan` still
reports 4 of 5 detected and 0 of 5 blocked. Its `guard_detected` column
comes from classifying the probe payload, so it says nothing about what the
model said.
