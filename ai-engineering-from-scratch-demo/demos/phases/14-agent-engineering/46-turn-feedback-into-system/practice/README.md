<!-- generated:start -->
# 14-agent-engineering / 46-turn-feedback-into-system

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/14-agent-engineering/46-turn-feedback-into-system/) · upstream spec
`phases/14-agent-engineering/46-turn-feedback-into-system/docs/en.md`

```bash
uv run demo practice run 46-turn-feedback-into-system --ex 1
uv run demo explain 46-turn-feedback-into-system --ex 1
uv run pytest demos/phases/14-agent-engineering/46-turn-feedback-into-system
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Take five corrections from a recent coding session and classify their real owners. | code | T0 | `ex01_the_keyword_classifier_agrees_with_one_of_five_real_corrections.py` |
| 2 | Replace one prose rule with an executable test. | code | T0 | `ex02_the_prose_rule_is_two_sentences_and_the_check_is_four_lines.py` |
| 3 | Add consequence weighting so a severe first occurrence can be promoted immediately. | code | T0 | `ex03_consequence_is_carried_into_the_output_and_never_read.py` |
| 4 | Add an owner and retirement date to the lab output. | code | T0 | `ex04_the_record_asks_for_eight_fields_and_the_control_carries_six_of_them.py` |
| 5 | Review one existing agent instruction and delete it only after proving a stronger control exi… | code | T0 | `ex05_the_stronger_control_exists_and_it_keeps_less_than_the_sentence.py` |
<!-- generated:end -->

## Answers

### 1 — the keyword classifier agrees with one of five real corrections

The five have to be real or the exercise is a vocabulary test. These are five
corrections from the session that produced this repository's practice solutions,
each labelled by hand with the layer that actually absorbed it — before running the
lesson's classifier over them.

| Correction | Real owner | `choose_target` |
|---|---|---|
| the scaffold crashed on a lesson with no Chinese doc | automation | instruction |
| a solution asserted a hypothesised number and failed | test | example |
| a solution passed 150 lines and the audit refused it | test | instruction |
| a citation was not found because the heading wrapped | example | example |
| `gh pr edit` was rejected as non-collaborator | instruction | scope |

**1 of 5.** The one it gets right is the missing canonical example. The permission
case is the closest of the misses — "scope" is defensible — but the control that
actually stuck was a written decision record, because the repository permission is
not something a scope policy can grant.

The mechanism is visible in the source: `choose_target` matches 14 keywords over
`symptom + cause` in a fixed order and returns on the first hit, so "regression"
beats "scope" whatever the cause says. The lesson's own warning — "a rule that
merely repeats the symptom will fail in the next slightly different case" — applies
to the router as much as to the rules it writes.

Two consequences. The rule text is `f"Prevent {normalize_cause(cause)}"`, and
`normalize_cause` recognises exactly 3 sentence shapes; a cause outside them is
passed through lowercased and unchanged, so a vague cause produces a vague control.
And the fingerprint is `sha256(target|rule)`, so the two corrections this repository
absorbed into the *same* file (`audit_practice.py`) get 2 fingerprints and 2 rules.
Dedup happens on wording, not on destination.

### 2 — the prose rule is two sentences and the check is four lines

The rule chosen is this repository's own `DESIGN D5`: a solution imports the
lesson's `code/`, it never forks it. As four lines:

```python
def check(text, reference_names):
    defined = set(re.findall(r"^(?:def|class) (\w+)", text, re.M))
    return "parity.load_reference" in text and not (defined & reference_names)
```

Over the 15 solutions the three most recent lessons ship it scores 15/15, and it
refuses a file holding two definitions that shadow reference symbols and no
reference import.

Three things the exercise surfaces:

**The lesson's own promotion sends this correction to the wrong layer.** Feeding it
"a solution forked the reference module" / "the import rule was described but not
checked" returns target `instruction`, with the verification "Run the instruction
linter and scenario check" — a control that does not exist. The destination `test`
exists; the keyword that reaches it does not appear in the sentence.

**The check is narrower than the sentence, and that has to be said out loud.** The
literal fork shadows 2 reference symbols and is refused. The same file with its
definitions renamed shadows 0, and is refused only because it still lacks the
import — the weaker half of the rule. A fork that imports the reference *and*
renames everything would pass. That is the honest cost of making a rule
executable, and it is better written down than discovered later.

**`Control.verification` is a string, not a command.** It is one of five canned
phrases. Whether the four-line check is wired into `audit_practice.py` is the
entire difference between a ratchet and a note.

### 3 — consequence is carried into the output and never read

Before adding a weight, find the gate it joins. `ratchet` skips a correction only
when `recurrence < 1` — which no recorded correction ever is. Nothing is held back
today, so "promoted immediately" already describes the behaviour for everything.

Weighting only becomes visible once something can be refused. Add a one-off
preference (recurrence 1, "minor annoyance") to the lesson's three corrections:

- shipped gate: promotes **4 of 4**.
- `recurrence >= 2 or severity >= 3`: promotes **3** — the scope failure and the
  setup failure on recurrence, the regression on its first occurrence because the
  consequence is user-visible. The preference stays out.

That is what the exercise is actually asking for: not "let severity promote", but
"let anything be refused, then let severity override the refusal".

Three supporting measurements. `consequence` appears 3 times in the module — a
field, an argument, a JSON key — and 0 of them in a condition, although the lesson
says to promote when recurrence *or* consequence justifies it. The recurrence gate
filters 0 of the example's corrections, because a threshold below the minimum
possible value is a comment. And the three consequence strings share 0 words, so a
weight map over free text is Exercise 1's keyword table again; three enum levels on
the `Correction` would remove the guessing entirely.

### 4 — the record asks for eight fields and the control carries six of them

The ratchet record in the docs is eight items. `Control` has eight fields, but they
are not the same eight: it covers symptom, cause, recurrence, consequence, the
chosen control (`target` + `rule`) and verification — 6 — and adds a `fingerprint`
the record never asks for. **`owner` and the review date are the gap.** Adding them
takes each JSON row to 10 keys.

The constraint the lesson does not mention: the module imports neither `time` nor
`datetime`, so every run produces identical bytes. A retirement date computed from
the clock breaks that. Passing `today` and a window as data keeps it — in this
fixture, 3 of 4 controls are due for review, and the answer is the same on every
machine and in every year.

Two design notes. A reworded cause produces a new rule, a new fingerprint and
therefore a new control with a fresh review date: two entries where a human sees
one ongoing control, and the older entry keeps a date nobody will act on. And of
the record's items, `target`, `rule`, `verification` and `fingerprint` are all
derived from the correction — `owner` is the only one that is not. That is exactly
why it is the field the lesson asks you to add: it is the one a generator cannot
fill, and the one that decides whether the control is anybody's problem.

### 5 — the stronger control exists, and it keeps less than the sentence

The instruction reviewed: this repository's `DESIGN.md` rule that a solution may
not ship with a surviving scaffold marker. The stronger control: the banned-string
scan in `scripts/audit_practice.py`. The proof is running both over the same files.

The scan checks 5 banned strings. Over the 15 solutions the three most recent
lessons ship it reports **0** violations; over five synthetic files, one per banned
string, it reports **5**. The instruction refused 0, because prose cannot refuse
anything. The enforcement sentence can go.

Two qualifications, and they are the reason the exercise says *prove* first:

**The check is broader than the sentence.** `DESIGN.md` says a *scaffold* marker;
the scan matches the substring anywhere in the file, including inside a docstring
that quotes an exercise using the word. That is not hypothetical — this very
solution had to spell the marker at runtime to get past the audit. So the sentence
is rewritten (to record the scope the check cannot express) rather than deleted
outright.

**What survives deletion is the "why".** `Control` carries `symptom`, `cause` and
`consequence` beside the rule, so the reasoning lives with the control rather than
in the instruction file. Deleting the enforcement sentence costs none of it — which
is the test for whether a sentence is enforcement or judgment.

Finally, of the four retirement criteria the lesson lists, this deletion satisfies
exactly one: a stronger executable control replaced it. The architecture has not
changed, the failure still recurs, and the friction was never high. One is enough —
but it has to be that one. A deletion justified by silence instead of replacement
is the case the lesson warns about.
