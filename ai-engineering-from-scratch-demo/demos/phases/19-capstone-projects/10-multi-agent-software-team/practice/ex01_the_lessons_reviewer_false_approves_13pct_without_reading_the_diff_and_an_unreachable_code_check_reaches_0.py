"""Exercise 1 — the lesson's reviewer false-approves 13% without reading the diff; an unreachable-code check reaches 0%.

    Inject an obvious bug into a diff mid-run (extra `return None` before the main body). Measure the reviewer's false-approve rate. Tune the reviewer prompt until false-approval is under 5%.

Reading of the exercise: the lesson's coders emit a dict of subtask name,
line count and a `has_bug` label, with no code. So the coders here are
wrapped to emit real source for each of the four planned subtasks (two
of them open with a legitimate `if ...: return None` guard). Mid-run, after
the coder and before the reviewer, one subtask per run gets an extra
`return None` right after its docstring. 200 seeded `run_team` runs rotate
the target over the four subtasks. A false approval is a run whose first
review says `approved`. The stub has no prompt, so a "prompt" here is a
review rule a reviewer applies to the diff text, and tuning means trying
rules until the rate is under 5% without rejecting clean runs.

**ANSWER: 26/200 (13.0%) for the lesson's reviewer; prompt v2 reaches 0/200
and rejects 0 of 200 clean runs.** v1, "reject any diff that adds
`return None`", also reaches 0% but rejects all 200 clean runs, because the
guard clauses match. v2, "reject a statement after a `return` in the same
block", flags only the unreachable body.

**FINDING: the lesson's reviewer and tester never read the diff.** They
branch on the `has_bug` label. With the `return None` in the text and the
label off, the reviewer approves 200/200 and the tests pass 193/200 (the 7
are the tester's coin-flip flakes). With the label on, the tester fails
every false approval: 168/200 pass.

**FINDING: every rejection is routed to coder-A.** `run_team` sends
feedback to the first coder whatever subtask was flagged, so 130 of the
lesson reviewer's 174 rejections, and 150 of v2's 200, reach a coder who
did not write the bug. The "revision" then relabels every diff clean.
"""

from __future__ import annotations

import ast
import contextlib
import random

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "10-multi-agent-software-team"
RUNS = 200
# one function per subtask of the lesson's plan; two open with a legitimate guard clause
SOURCE = {
    "parser": 'def parse(raw):\n    """Split a header line."""\n    if not raw:\n        return None\n'
              '    key, _, value = raw.partition(":")\n    return key.strip(), value.strip()\n',
    "cache": 'def get(cache, key):\n    """Read through the cache."""\n    hit = cache.get(key)\n'
             '    if hit is None:\n        return None\n    return hit.value\n',
    "api": 'def handler(request):\n    """Serve one widget."""\n    widget = load(request.id)\n'
           '    return {"id": widget.id, "name": widget.name}\n',
    "migration": 'def migrate(db):\n    """Add the widget index."""\n    db.execute("CREATE INDEX w ON widgets(id)")\n'
                 '    return db.version + 1\n',
}


def inject(src):
    """Insert `return None` after the docstring, before the main body."""
    lines = src.splitlines(keepends=True)
    return "".join(lines[:2] + ["    return None\n"] + lines[2:])


def unreachable(src):
    """True when a statement follows a `return` in the same block."""
    bodies = [getattr(n, "body", None) for n in ast.walk(ast.parse(src))]
    return any(isinstance(s, ast.Return) for b in bodies if isinstance(b, list) for s in b[:-1])


PROMPTS = {
    "v1: reject any diff that adds `return None`": lambda src: "return None" in src,
    "v2: reject a diff with a statement after a `return` in the same block": unreachable,
}


def content_reviewer(detect):
    """A reviewer that reads the patch text; same signature as the lesson's `reviewer_check`."""
    return lambda diffs, rng: next(((False, f"flagged {d['subtask']}") for d in diffs if detect(d["patch"])),
                                   (True, "lgtm"))


@contextlib.contextmanager
def patched(ref, target, reviewer=None, label=True):
    """Coders emit real source; `target` gets the injected line mid-run."""
    boards, real = [], (ref.coder_implement, ref.reviewer_check, ref.Board)

    class Recording(real[2]):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            boards.append(self)

    def coder(sub, rng):
        hit = sub.name == target
        return {**real[0](sub, rng), "patch": inject(SOURCE[sub.name]) if hit else SOURCE[sub.name], "has_bug": hit and label}

    ref.coder_implement, ref.reviewer_check, ref.Board = coder, reviewer or real[1], Recording
    try:
        yield boards
    finally:
        ref.coder_implement, ref.reviewer_check, ref.Board = real


def trial(ref, reviewer=None, inject_on=True, label=True):
    """Approvals, rejections routed to the wrong coder, and passes over RUNS seeded runs."""
    approved = misrouted = passed = 0
    for seed in range(RUNS):
        target = list(SOURCE)[seed % 4] if inject_on else None
        with patched(ref, target, reviewer, label) as boards:
            r = ref.run_team(f"issue-{seed}", rng=random.Random(seed))
        approved, passed = approved + r["approved"], passed + r["tested_passed"]
        fb = [m.to for m in boards[0].messages if m.kind == ref.MsgKind.REVIEW_FEEDBACK]
        misrouted += bool(target and fb) and fb[0] != f"coder-{chr(65 + list(SOURCE).index(target))}"
    return {"approved": approved, "misrouted": misrouted, "passed": passed}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    out = {"lesson": trial(ref), "lesson_text_only": trial(ref, label=False)}
    for name, detect in PROMPTS.items():
        out[name[:2]] = trial(ref, content_reviewer(detect))
        out[name[:2] + "_clean"] = trial(ref, content_reviewer(detect), inject_on=False)
    return out


def verify(result):
    r, (lesson, v1, v1c, v2, v2c) = result, (result[k] for k in ("lesson", "v1", "v1_clean", "v2", "v2_clean"))
    return [
        practice.Check(
            "ANSWER: the lesson's reviewer false-approves 13.0%; prompt v2 reaches 0% with 0 false rejections",
            (lesson["approved"], v1["approved"], v1c["approved"], v2["approved"], v2c["approved"])
            == (26, 0, 0, 0, RUNS) and v2["approved"] / RUNS < 0.05,
            f"false approvals on {RUNS} injected runs: lesson {lesson['approved']}, v1 {v1['approved']}, "
            f"v2 {v2['approved']}; clean runs approved: v1 {v1c['approved']}/{RUNS}, v2 {v2c['approved']}/{RUNS}",
        ),
        practice.Check(
            "FINDING: the lesson's reviewer and tester never read the diff; an unlabelled `return None` ships",
            (r["lesson_text_only"]["approved"], r["lesson_text_only"]["passed"], lesson["passed"]) == (RUNS, 193, 168),
            f"injected text with has_bug=False: approved {r['lesson_text_only']['approved']}/{RUNS}, tests pass "
            f"{r['lesson_text_only']['passed']}/{RUNS}; labelled: tests pass {lesson['passed']}/{RUNS}",
        ),
        practice.Check(
            "FINDING: every rejection goes to coder-A, so 3 of 4 reach the wrong coder",
            (lesson["misrouted"], RUNS - lesson["approved"], v2["misrouted"]) == (130, 174, 150),
            f"lesson reviewer: {lesson['misrouted']} of {RUNS - lesson['approved']} rejections misrouted; "
            f"v2: {v2['misrouted']} of {RUNS}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
