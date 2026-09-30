"""Exercise 5 — a documenter that reads the board costs as much as the run, for entries wrong 1 time in 4.

    Add a fifth role: documenter (Haiku 4.5). After review, it produces a changelog entry. Measure whether documentation quality justifies the extra token spend.

Reading of the exercise: the documenter runs at the `approved` message on
1,000 seeded runs and writes the entry from what the board holds: the issue,
subtask names and line counts, and any review note and revision. Quality is
whether the entry is true of what shipped: the change passed the tester,
and a revision is credited to the subtask that had the bug. Spend is Haiku
4.5 list price on the tokens it reads and writes (4 characters a token).

**ANSWER: no.** Reading the transcript costs 33,786 tokens ($0.0339) an entry,
0.97x the 34,973-token four-role run, and 744/1000 entries are true.

**FINDING: the wrong entries come from the lesson's routing and from
documenting before the tester.** 65 entries describe a change that then
failed tests. 198 of the 269 revised runs credit the revision to `parser`,
because the lesson's revised DIFF_READY always says `parser`.

**FINDING: the board holds names and line counts only.** Reading just those
payloads costs 103 tokens ($0.00021) and writes the same entry, which a
template writes for 0 tokens. `MsgKind` has no changelog kind among its 9.
"""

from __future__ import annotations

import contextlib
import random

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "10-multi-agent-software-team"
RUNS = 1000
HAIKU = (1.00, 5.00)  # $/MTok in/out, Claude Haiku 4.5, platform.claude.com/docs/en/about-claude/pricing, 2026-09-29
CHARS_PER_TOKEN = 4  # same page's FAQ: "1 token is approximately 4 characters"


@contextlib.contextmanager
def recording(ref):
    boards, real = [], ref.Board

    class Recording(real):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            boards.append(self)

    ref.Board = Recording
    try:
        yield boards
    finally:
        ref.Board = real


def document(ref, seen):
    """The changelog entry: everything the board holds when review approves."""
    k = ref.MsgKind
    plan = next(m.payload for m in seen if m.kind == k.PLAN_REQUEST)
    diffs = [m.payload for m in seen if m.kind == k.DIFF_READY]
    first = {d["subtask"]: d["lines"] for d in diffs[: len(plan["subtasks"])]}
    revised = [d["subtask"] for d in diffs[len(plan["subtasks"]):]]
    entry = f"{plan['issue']}: " + ", ".join(f"{n} (+{c} lines)" for n, c in first.items())
    if revised:
        note = next(m.payload["comment"] for m in seen if m.kind == k.REVIEW_FEEDBACK)
        entry += f"; revised {', '.join(revised)} after review ({note})"
    return entry, revised


def spend(k, seen, entry):
    """Tokens read (whole transcript, or only the payloads the entry uses) and written."""
    used = (k.PLAN_REQUEST, k.DIFF_READY, k.REVIEW_FEEDBACK)
    lean = sum(len(str(m.payload)) for m in seen if m.kind in used) // CHARS_PER_TOKEN
    return sum(m.tokens for m in seen), lean, -(-len(entry) // CHARS_PER_TOKEN)


def grade(ref, board, result):
    """Is the entry true of what shipped? Lists the failure reasons, empty if it is."""
    msgs, k = board.messages, ref.MsgKind
    seen = msgs[: next(i for i, m in enumerate(msgs) if m.kind == k.APPROVED) + 1]
    entry, revised = document(ref, seen)
    bugged = [m.payload["subtask"] for m in seen if m.kind == k.DIFF_READY and m.payload.get("has_bug")]
    wrong = ["entry for a change that failed tests"] * (not result["tested_passed"])
    wrong += ["revision credited to the wrong subtask"] * bool(revised and revised != bugged)
    return entry, wrong, spend(k, seen, entry)


def usd(tokens_in, tokens_out):
    return round((tokens_in * HAIKU[0] + tokens_out * HAIKU[1]) / 1e6 / RUNS, 6)


def tally(rows):
    reasons = [w for _, ws, _ in rows for w in ws]
    return {"correct": sum(not ws for _, ws, _ in rows),
            "reasons": {w: reasons.count(w) for w in sorted(set(reasons))}}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    with recording(ref) as boards:
        rs = [ref.run_team(f"issue-{s}", rng=random.Random(s)) for s in range(RUNS)]
    rows = [grade(ref, b, r) for b, r in zip(boards, rs)]
    full, lean, out = (sum(t[i] for _, _, t in rows) for i in range(3))
    return {
        "entries": len(rows), **tally(rows),
        "tokens_per_entry": round((full + out) / RUNS), "usd_per_entry": usd(full, out),
        "lean_tokens_per_entry": round((lean + out) / RUNS), "lean_usd_per_entry": usd(lean, out),
        "team_tokens": round(sum(r["total_tokens"] for r in rs) / RUNS),
        "example": rows[0][0], "kinds": [m.value for m in ref.MsgKind],
    }


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: no -- an entry reading the board costs 0.97x a whole run and is right 744/1000 times",
            (r["entries"], r["correct"], r["tokens_per_entry"], r["team_tokens"], r["usd_per_entry"])
            == (1000, 744, 33786, 34973, 0.033893),
            f"{r['correct']}/{r['entries']} entries true of what shipped; the documenter reads the transcript: "
            f"{r['tokens_per_entry']:,} tokens (${r['usd_per_entry']:.4f} on Haiku 4.5) against "
            f"{r['team_tokens']:,} for the four-role run",
        ),
        practice.Check(
            "FINDING: the wrong entries come from the lesson's routing and from documenting before the tester",
            r["reasons"] == {"entry for a change that failed tests": 65, "revision credited to the wrong subtask": 198},
            f"{r['reasons']}; the revised DIFF_READY always says 'parser'",
        ),
        practice.Check(
            "FINDING: the board holds names and line counts only, so a 103-token read writes the same entry",
            (r["lean_tokens_per_entry"], r["lean_usd_per_entry"], "changelog" in r["kinds"], r["example"])
            == (103, 0.00021, False,
                "issue-0: parser (+20 lines), cache (+48 lines), api (+80 lines), migration (+77 lines)"),
            f"payloads only: {r['lean_tokens_per_entry']} tokens (${r['lean_usd_per_entry']:.5f}); no changelog "
            f"kind among {len(r['kinds'])} MsgKinds; e.g. {r['example']!r}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
