"""Exercise 4 — at 3x compaction a front-to-back summary keeps 3 of 12 plan items right; pinning the plan keeps 12.

    Implement `PreCompact` summarization with a smaller model (Haiku 4.5). Measure how much plan fidelity is lost at 3x compaction.

Reading of the exercise: with no model call at run time, the summarizer is
the part that can be measured without one: what each compaction policy keeps
when the transcript must shrink to a third. A 36-turn session goes through
the lesson's `run_agent` with a 12-item plan that advances one item every 3
turns and a note on every item; each turn's transcript is the plan as
`PlanState.summary()` prints it, plus the `read_file` observation. Three
policies compress it 3x: `oldest` (a single-pass summarizer that runs out of
budget, keeping the head), `newest` (a sliding window), and `pinned` (the
latest plan verbatim as a prior-state block, then the newest turns). Plan
fidelity is the share of the final plan's 12 (id, status) pairs recoverable
from the compacted text, reading the last mention of each id.

**ANSWER: at 3x, `oldest` keeps 3/12 plan items right (9 stale statuses),
`newest` and `pinned` keep 12/12.** Because the plan is rewritten whole
every turn, any policy that keeps the last turn keeps the whole plan. What
3x costs is history: `pinned` keeps 11 of 36 observations. `pinned` keeps
12/12 up to 109x, where the budget gets too small for the 12 plan lines
themselves. A
Haiku 4.5 pass at the lesson's 150k mark costs $0.40 (150k in, 50k out at
$1/$5 per MTok) against Sonnet 4.6's $1.20, and $0.25 of the $0.40 is output.

**FINDING: the notes are gone before any compaction.** `summary()` prints
id, status and description and drops `note`, so 0 of the 12 notes reach the
transcript at 1x.

**FINDING: the harness never compacts.** Over the run, 3 of the 8 hook
events never fire: `PreCompact`, `UserPromptSubmit` and `Notification`.
`main.py` has no 150k threshold; its only token ceiling is a 200,000
cumulative total that ends the session.
"""

from __future__ import annotations

import os
import re
import tempfile

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "01-terminal-native-coding-agent"
ITEMS, TURNS, RATIO = 12, 36, 3
HAIKU_45, SONNET_46 = (1.0, 5.0), (3.0, 15.0)  # $/MTok in/out, platform.claude.com, read 2026-09-29
LINE = re.compile(r"\[(.)\] (\d+)\. ")


def plan_at(ref, turn):
    status = lambda k: "done" if k < turn // 3 else "in_progress" if k == turn // 3 else "pending"
    return [ref.TodoItem(k + 1, f"step {k + 1} of the fix", status(k), note=f"why-{k + 1}")
            for k in range(ITEMS)]


def record(ref, sandbox):
    turns, fired = [], {}

    class CountingBus(ref.HookBus):
        def fire(self, event, payload):
            fired[event] = fired.get(event, 0) + 1
            return super().fire(event, payload)

    def model_step(plan, turn):
        items = plan_at(ref, min(turn, TURNS - 1))
        call = ("read_file", {"path": f"obs_{turn:02d}.txt"}) if turn < TURNS else None
        turns.append(ref.PlanState("fix", items).summary())
        return {"plan": items, "tool": call, "tokens": 100, "cost": 0.0}

    read, bus, step = ref.TOOLS["read_file"], ref.HookBus, ref.model_step
    ref.TOOLS["read_file"] = lambda sb, **kw: turns.append(read(sb, **kw)) or turns[-1]
    ref.HookBus, ref.model_step = CountingBus, model_step
    try:
        ref.run_agent("fix", sandbox)
    finally:
        ref.TOOLS["read_file"], ref.HookBus, ref.model_step = read, bus, step
    pairs = ["\n".join(turns[i:i + 2]) for i in range(0, len(turns) - 1, 2)]
    return pairs, turns[-1], fired


def compact(pairs, final_plan, budget, policy):
    text = "\n".join(pairs)
    if policy == "oldest":
        return text[:budget]
    if policy == "newest":
        return text[-budget:]
    kept, room = [], budget - len(final_plan) - len("PRIOR STATE\n") - 1
    for turn in reversed(pairs):
        if room < len(turn):
            break
        kept.insert(0, turn)
        room -= len(turn) + 1
    return ("PRIOR STATE\n" + final_plan + "\n" + "\n".join(kept))[:budget]


def fidelity(text, final_plan):
    last = {int(i): m for m, i in LINE.findall(text)}
    want = {int(i): m for m, i in LINE.findall(final_plan)}
    return sum(last.get(i) == m for i, m in want.items())


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    with tempfile.TemporaryDirectory() as tmp:
        for t in range(TURNS):
            with open(os.path.join(tmp, f"obs_{t:02d}.txt"), "w") as fh:
                fh.write(f"fact-{t:02d}: " + "observed output line. " * 30)
        pairs, final_plan, fired = record(ref, tmp)
    full = "\n".join(pairs)
    budget = len(full) // RATIO
    out = {p: compact(pairs, final_plan, budget, p) for p in ("oldest", "newest", "pinned")}
    breaks = next(r for r in range(1, 1000) if fidelity(compact(pairs, final_plan, len(full) // r,
                                                                "pinned"), final_plan) < ITEMS)
    source = (parity.lesson_dir(PHASE, LESSON) / "code" / "main.py").read_text()
    cost = lambda p: round((150_000 * p[0] + 50_000 * p[1]) / 1e6, 2)
    return {"fidelity": {p: fidelity(t, final_plan) for p, t in out.items()},
            "facts_pinned": len(re.findall(r"fact-\d\d", out["pinned"])), "turns": len(pairs),
            "pinned_ok_up_to": breaks - 1, "notes_in_full": len(re.findall(r"why-\d+", full)),
            "never_fired": sorted(set(ref.HookBus.EVENTS) - set(fired)),
            "has_150k": bool(re.search(r"150[_,]?000|150k", source)),
            "haiku": cost(HAIKU_45), "sonnet": cost(SONNET_46), "haiku_out": 50_000 * HAIKU_45[1] / 1e6}


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: 3x keeps 3/12 plan items front-to-back, 12/12 when the latest plan survives",
            (r["fidelity"], r["facts_pinned"], r["turns"], r["pinned_ok_up_to"], r["haiku"], r["sonnet"],
             r["haiku_out"]) == ({"oldest": 3, "newest": 12, "pinned": 12}, 11, 36, 109, 0.4, 1.2, 0.25),
            f"plan fidelity {r['fidelity']} of 12; pinned keeps {r['facts_pinned']}/{r['turns']} "
            f"observations and stays 12/12 up to {r['pinned_ok_up_to']}x; one compaction "
            f"${r['haiku']} Haiku 4.5 vs ${r['sonnet']} Sonnet 4.6",
        ),
        practice.Check(
            "FINDING: PlanState.summary() drops the note field, so notes are lost at 1x",
            r["notes_in_full"] == 0,
            f"{r['notes_in_full']} of {ITEMS} notes appear in the uncompacted transcript",
        ),
        practice.Check(
            "FINDING: PreCompact never fires and main.py has no 150k compaction mark",
            r["never_fired"] == ["Notification", "PreCompact", "UserPromptSubmit"] and not r["has_150k"],
            f"events never fired: {r['never_fired']}; 150k threshold in main.py: {r['has_150k']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
