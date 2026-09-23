<!-- generated:start -->
# 16-multi-agent-and-swarms / 08-role-specialization

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/16-multi-agent-and-swarms/08-role-specialization/) · upstream spec
`phases/16-multi-agent-and-swarms/08-role-specialization/docs/en.md`

```bash
uv run demo practice run 08-role-specialization --ex 1
uv run demo explain 08-role-specialization --ex 1
uv run pytest demos/phases/16-multi-agent-and-swarms/08-role-specialization
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py` and observe how the verifier catches the bug the critic missed. Add a stat… | code | T0 | `ex01_the_static_check_catches_the_code_that_memorised_the_tests.py` |
| 2 | Add a 5th role: "requirements analyst" that translates user wish into planner-ready spec. Wha… | code | T0 | `ex02_the_questions_that_flow_up_are_the_fields_nobody_checks.py` |
| 3 | Read MetaGPT Section 3 ("Agents"). List the input/output schema of each of MetaGPT's 5 roles. | explain | T0 | prose, below |
| 4 | Read ChatDev's chat-chain diagram (arXiv:2307.07924 Figure 3). Identify where communicative d… | explain | T0 | prose, below |
| 5 | PwC's 7× accuracy gain came from verification loops. Hypothesize three tasks where adding a v… | code | T0 | `ex05_two_of_the_specs_four_fields_already_have_no_oracle.py` |
<!-- generated:end -->

## Answers

### 1 — the static check catches the code that memorised the tests

**What it catches: code that memorised the test cases.** Write the artifact
that defeats the runtime test first, or there is no way to tell whether the
new check is catching anything:

```python
def add_two(a, b):
    if (a, b) == (1, 2):   return 3
    if (a, b) == (10, 20): return 30
    if (a, b) == (-5, 5):  return 0
    return a * b
```

It passes the critic's **3** checks and all **3** of the verifier's tests.
It has **4** returns; the correct implementation has **1**.

The runtime test cannot catch it, and that is not a weakness in the suite. The
tests are the verifier's *only* oracle, so an artifact that satisfies all of
them is by construction indistinguishable from a correct one. The static check
works because it reads a property of the **text** rather than of the
behaviour — a different axis, which is the whole reason it adds anything.

Three other things fall out of running the pipeline.

**The verifier is not a sandbox.** Its docstring says "Run the code in a
sandbox namespace"; the call is `exec(art.code, ns, ns)` with a plain dict.
CPython injects `__builtins__` into any globals mapping lacking it, so
`import os` inside an artifact succeeds. The role the lesson describes as the
deterministic, trustworthy one is the only role that executes
attacker-controlled text, with nothing removed.

**The verdict hides the critic.** The branches are
`if approved and passed` / `elif not passed` / `elif not approved`. The second
catches every run where the verifier failed, so the third can only fire when
the verifier *passed*. A run where both roles object prints "verifier blocked
ship" and the critic's notes never reach the verdict, although they were
computed.

**The planner ignores the wish.** `task_name`, `signature` and `tests` are
literals; only `description` carries the argument through. Two entirely
different wishes produce the same specification.

### 2 — the questions that flow up are the fields nobody checks

**The requests that must flow up are `signature` and `description`.** Find them
by elimination — a request only needs to travel upward if nothing downstream
can settle it:

| Spec field | read by |
|---|---|
| `task_name` | critic, verifier |
| `tests` | verifier |
| `signature` | **nobody** |
| `description` | **nobody** |

So the two questions are: *what exactly are the parameter and return types*,
and *what does "the sum of two integers" mean at the boundaries* — non-integer
arguments, overflow, whether floats must be rejected. Everything else can be
settled without asking, because something downstream can check it.

Those fields are already shipping wrong answers. `def add_two(*args)`
contradicts the stated `add_two(a: int, b: int) -> int`, and is **approved by
the critic and passed by the verifier**, because nothing in the pipeline
compares the artifact against `signature`.

Two structural notes on adding the role. The executor cannot raise a request:
both executors return a constant `Artifact` and never touch the `spec`
parameter, so the role most likely to discover a gap never looks at the
requirements. And there is nowhere to put a question — `Artifact` has one
field, and the two reports carry booleans and note lists addressed to no one.
Adding the analyst means adding a channel, not just a function.

### 3 — five roles, and the schema that is a prompt

*Draws on "MetaGPT's SOP pattern".*

| role | input | output |
|---|---|---|
| Product Manager | user requirement (natural language) | PRD — goals, user stories, requirement pool |
| Architect | PRD | system design — file list, data structures, API/interface definitions |
| Project Manager | system design | task list — one unit of work per file, with dependencies |
| Engineer | one task + the interfaces it depends on | source file implementing it |
| QA Engineer | source files + the design | test cases, run results, bug reports back to Engineer |

The shape worth noticing is that each role's input is the previous role's
output — the schema is what makes `Code = SOP(Team)` mean anything, because it
is what lets a role be swapped without touching its neighbours.

And the shape worth being sceptical about is that the schema is enforced by a
*prompt*. "The role prompt says what the role is and what it must produce" —
which is a request, not a type. This module is the degenerate version:
`planner` returns a real `Spec` dataclass with four typed fields, and **two of
them are read by no one**, so even a genuine schema does not make the contract
binding. A schema that nothing validates is documentation.

### 4 — the loop it breaks is executor guessing, then QA testing the guess

*Draws on "ChatDev's communicative dehallucination".*

The infinite loop is between the implementer and the checker. The executor
needs a detail the plan does not contain — a field name, an error code, a
boundary rule. Without a way to ask, it invents something plausible and
proceeds. The reviewer or tester sees output that does not match the intent,
rejects it, and sends it back. The executor still does not have the detail, so
it invents a *different* plausible value. Nothing in that cycle adds
information, so it can run forever, and every turn costs a full generation on
both sides.

Communicative dehallucination breaks it by making the missing detail
addressable: "when you need specific information you were not given, ask the
relevant role by name before producing output." The request goes **up**, to
whoever owns the decision, rather than sideways to whoever checks the guess.
One message converts an unbounded rejection loop into a bounded question.

The mechanism is precisely what exercise 2 finds missing here. This pipeline
has no upward channel at all — `Artifact` carries only `code`, and the reports
carry booleans — so an executor that noticed a gap could not report it. It
also cannot notice one, since both executors ignore the spec. Put together,
the module can produce the failure ChatDev's move exists to prevent and has no
way to express the move.

### 5 — two of the spec's four fields already have no oracle

Three classes, with one worked example of each already in the module.

**Unbounded.** `description` claims "the sum of two integers" — a property over
an infinite domain, checked at **3** points. No finite suite decides it, and
the gap is not a sampling problem: a lookup table over those three points
passes all of them. Other tasks of this shape: "handles all valid UTF-8",
"never deadlocks", "terminates on every input".

**Erased.** `signature` says `add_two(a: int, b: int) -> int`. Python
annotations are not enforced at runtime, so `def add_two(*args)` satisfies the
verifier and contradicts the spec. The property is real; the artifact simply
does not carry it at the point where checking happens. Checking it needs a
different tool, not a longer test suite. Other tasks: "uses no floating point
internally", "holds no lock across an await", "does not log secrets".

**Judgements.** Readability, design fit, whether this is what the user meant.
`CriticReport` exists precisely because these cannot be a `VerifierReport`.

The measurement that makes it concrete: **2 of the Spec's 4 fields are read by
0 roles**, and the two that are read are exactly those with a mechanical
oracle — a name to compare and a list of cases to execute.

Two observations about the 7× claim in that light. The verifier's power here
is exactly the length of the test list: both the memorised artifact and the
widened one come back `(approved=True, passed=True)`. Adding verification
raises accuracy **on the sample**, which equals accuracy on the task only when
the sample determines the task. And `description` — the only field that varies
with the user's request — is the one nothing reads.

The last one is about the critic. Its three checks are a `def`, a `return`,
and a matching name: all decidable. The lesson's critic-versus-verifier
distinction is real, and the shipped critic sits on the wrong side of it —
**0** of its checks require judgement, which is why it is fooled by the buggy
code rather than sceptical of it.
