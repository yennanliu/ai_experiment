"""Exercise 3 — the learner record has no timestamp to delete by, and the web half masters a lesson on one right answer.

    Build a parent dashboard: topics practiced, mastery trajectories, upcoming concepts, safety events (any guardrail hits). COPPA-aligned.

Reading of the exercise: the dashboard is a function over what the lesson
stores for a learner, a reference `LearnerState` after `run_adaptive`, plus
the two things the lesson does not store: a date per turn (60 turns as six
10-turn sessions, three a week for two weeks) and a guardrail log. The
lesson ships no guardrail, so a labelled 6-message fixture stands in for
it. The four panels are built from the reference objects: topics from
`history`, trajectories by replaying `history` through `bkt_update`,
upcoming concepts from `next_concept`. "COPPA-aligned" is read against the
rule's text (16 CFR 312, read 2026-09-29 at law.cornell.edu/cfr/text/16/312.2,
/312.6 and /312.10): a parent must be verified before review (312.6(a)(3)),
may direct deletion (312.6(a)(2)), retention needs a written timeframe and
may not be indefinite (312.10), and a child's audio or free text is itself
personal information (312.2), so no panel shows raw messages.

**ANSWER: `dashboard()` below; the week-1 report for the study's first
learner.** Topics: 8 of 11 practised in 30 turns, with attempts and right
answers per topic. Trajectories: the BKT value after every attempt, replayed
from `history`; each ends at the stored mastery. Upcoming:
`distributive_property`, the one concept whose prerequisites are met and
which is not yet at 0.85. Safety: 6 guardrail hits on 3 session days, as
(date, category) counts; none of the 6 raw messages appears anywhere in the
output, though two of them carry a name, an address and a phone number.
COPPA: an unverified request gets only "parent verification required"; the
parent's delete leaves 0 items; with the lesson's 1-year timeframe written
down, the purge keeps 30 turns at day 365, 10 at day 368 and 0 at day 370.
The rule itself sets no fixed period: 1 year is a policy the operator must
write and justify, not the law.

**FINDING: the lesson's learner record cannot support COPPA retention or a
safety panel.** `LearnerState` has 3 fields, `learner_id`, `mastery` and
`history`: no timestamp to age data out by, no guardrail log, no link to a
parent. Every date and event on this dashboard had to be added.

**FINDING: a read changes the record.** `mastery` is a `defaultdict`, so the
reference's own `mastery_sum`, run on the live record, adds the 3 concepts
the learner never practised at 0.2 (8 entries become 11). A dashboard that
counted "topics" from mastery keys would then report 11. The panels here
read a copy.

**FINDING: the two halves draw different trajectories.** The TypeScript
store scores a lesson `0.3 * score + 0.7 * observed`, so one right answer
gives exactly 0.7, which is `MASTERY_THRESHOLD`: mastered after one answer.
The Python BKT needs two (0.927).

Structure: `dashboard()` builds the four panels for a verified parent;
`purge()` applies the retention timeframe; `delete()` is the parent's erase.
"""

from __future__ import annotations

import collections
import datetime as dt
import random
import re

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "17-personal-ai-tutor"
START, RETENTION_DAYS = dt.date(2026, 9, 7), 365  # the lesson's "auto-delete after 1 year", as a written timeframe
SESSION_DAYS = (0, 2, 4, 7, 9, 11)  # Mon/Wed/Fri for two weeks
GUARD_LOG = [  # (session, category, raw message) -- the raw text must never reach the dashboard
    (0, "self_harm", "i want to hurt myself when i fail tests"), (1, "violence", "how do i make a weapon"),
    (1, "personal_info", "my name is ana and i live at 12 elm street"), (2, "personal_info", "my phone is 555 0134"),
    (2, "adult", "show me something rated 18+"), (2, "violence", "i will punch my brother")]


def turn_date(turn):
    return START + dt.timedelta(days=SESSION_DAYS[min(turn // 10, 5)])


def history_panels(ref, state):
    """Topics as (attempts, right answers) and each topic's BKT trajectory, replayed from `history`."""
    p, m, topics, paths = ref.BKTParams(), {}, {}, {}
    for concept, ok in state.history:
        m[concept] = ref.bkt_update(m.get(concept, p.p_init), ok, p)
        topics[concept] = (topics.get(concept, (0, 0))[0] + 1, topics.get(concept, (0, 0))[1] + ok)
        paths.setdefault(concept, []).append(round(m[concept], 2))
    return topics, paths


def dashboard(ref, cmap, record, parent_verified):
    if not parent_verified:  # 312.6(a)(3)(i): ensure the requestor is the parent
        return {"error": "parent verification required"}
    state = record["state"]  # read a copy: `mastery` is a defaultdict, so reading an unseen concept writes it
    view = ref.LearnerState(state.learner_id, mastery=collections.defaultdict(lambda: 0.2, state.mastery))
    ready = lambda c: view.mastery[c.name] < 0.85 and all(view.mastery[pr] >= 0.85 for pr in c.prereqs)
    unlocked = [c.name for c in cmap.values() if ready(c)]
    events = collections.Counter((str(turn_date(session * 10)), category) for session, category, _ in record["guard"])
    topics, paths = history_panels(ref, state)
    return {"topics": topics, "trajectories": paths, "next": ref.next_concept(view, cmap), "upcoming": unlocked,
            "safety_events": events, "retention_days": RETENTION_DAYS, "parent_can_delete": True}


def purge(record, today):
    """312.10: nothing kept past the written timeframe. The reference history has no dates of its own."""
    keep = [i for i in range(len(record["state"].history)) if (today - turn_date(i)).days <= RETENTION_DAYS]
    record["state"].history = [record["state"].history[i] for i in keep]
    return len(keep)


def delete(record):
    for store in (stores := (record["state"].history, record["state"].mastery, record["guard"])):
        store.clear()
    return sum(map(len, stores))


def ts_first_answer(ref):
    text = "\n".join(f.read_text() for f in (parity.lesson_dir(PHASE, LESSON) / "code" / "ts" / "src").glob("*.ts"))
    new = float(re.search(r"m\.score = [\d.]+ \* m\.score \+ ([\d.]+) \* observed", text).group(1))
    return [new * 1.0, float(re.search(r"MASTERY_THRESHOLD = ([\d.]+)", text).group(1))]  # first answer: 0 -> 1/1


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    cmap = ref.curriculum_map(ref.ALGEBRA)
    record = {"state": (state := ref.run_adaptive("learner_0", 0.3, cmap, 30, random.Random(100))), "guard": list(GUARD_LOG)}
    board = dashboard(ref, cmap, record, parent_verified=True)
    probe = ref.LearnerState("probe", mastery=collections.defaultdict(lambda: 0.2, state.mastery))
    before = len(probe.mastery)
    ref.mastery_sum(probe, cmap)  # the reference's own reader, on the live record
    replay_ok = all(t[-1] == round(state.mastery[c], 2) for c, t in board["trajectories"].items())
    return {
        "turns": len(state.history), "topics": len(board["topics"]), "next": board["next"], "upcoming": board["upcoming"],
        "events": sum(board["safety_events"].values()), "event_days": len({d for d, _ in board["safety_events"]}),
        "raw_leaked": sum(raw in repr(board) for _, _, raw in GUARD_LOG), "replay_ok": replay_ok,
        "unverified": dashboard(ref, cmap, record, parent_verified=False),
        "read_writes": [before, len(probe.mastery)], "state_fields": sorted(vars(state)),
        "kept_after_1y": [purge(record, START + dt.timedelta(days=365 + d)) for d in (0, 3, 5)],
        "left_after_delete": delete(record), "ts_first_answer": ts_first_answer(ref),
        "two_right_python": history_panels(ref, ref.LearnerState("two", history=[("c", True)] * 2))[1]["c"][-1]}


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: four panels for a verified parent, 0 raw messages shown, deletion to 0, 1-year purge",
            (r["turns"], r["topics"], r["next"], r["upcoming"], r["events"], r["event_days"], r["raw_leaked"],
             r["replay_ok"], r["unverified"], r["kept_after_1y"], r["left_after_delete"])
            == (30, 8, "distributive_property", ["distributive_property"], 6, 3, 0, True,
                {"error": "parent verification required"}, [30, 10, 0], 0),
            f"{r['turns']} turns, {r['topics']} topics, next/upcoming {r['next']}/{r['upcoming']}, {r['events']} hits on "
            f"{r['event_days']} days, raw shown {r['raw_leaked']}, replay ok {r['replay_ok']}, unverified {r['unverified']}, "
            f"kept at day 365/368/370 {r['kept_after_1y']}, left after delete {r['left_after_delete']}",
        ),
        practice.Check(
            "FINDING: the lesson's learner record cannot support COPPA retention or a safety panel",
            r["state_fields"] == ["history", "learner_id", "mastery"],
            f"LearnerState fields {r['state_fields']}: no timestamp, no guardrail log, no parent link",
        ),
        practice.Check(
            "FINDING: the reference's own mastery_sum writes 3 unpractised concepts into the record it reads",
            r["read_writes"] == [8, 11],
            f"mastery entries before/after mastery_sum on the live record {r['read_writes']}",
        ),
        practice.Check(
            "FINDING: the TypeScript half masters a lesson on one right answer, the Python half needs two",
            (r["ts_first_answer"], r["two_right_python"]) == ([0.7, 0.7], 0.93),
            f"TS score after one right answer vs MASTERY_THRESHOLD {r['ts_first_answer']}; Python after two {r['two_right_python']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
