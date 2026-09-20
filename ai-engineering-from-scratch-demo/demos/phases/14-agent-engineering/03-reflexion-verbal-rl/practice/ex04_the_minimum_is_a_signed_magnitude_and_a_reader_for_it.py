"""Exercise 4 — the minimum is a signed magnitude, and a reader for it.

    Run Reflexion with an adversarial Actor that ignores reflections. What is
    the minimum reflection prompt engineering that forces the Actor to notice
    them?

Reading of the exercise: the adversarial Actor is already shipped.
`Actor.act` branches on `len(memory.items)` and never touches
`Reflection.text`, so it ignores reflections by construction. That makes the
question two questions -- what the reflection must contain, and what the
Actor must be -- and only the first is prompt engineering. Three reflection
styles are run against an Actor that does read text, and all three against
the shipped one.

**ANSWER: a signed magnitude is the minimum, and an instruction adds
nothing.** A generic `be more careful next time` never converges in **6**
trials. The lesson's own `sum 6 is 14 short; pick larger values` converges on
trial **2**. Adding an explicit constraint -- `next attempt must sum to 20` --
also converges on trial **2**. The extra sentence buys **0** trials.

**FINDING: no prompt can reach an Actor whose input is a count.** Across all
**3** styles the shipped Actor produces **1** distinct trajectory, identical
to the one it produces with no reflector at all. "Forcing the Actor to
notice" is not reachable from the reflection side; the parameter has to
change.

**FINDING: the counting Actor is not a weak policy, it is a lookup.** Move
the target to **24** and it fails **6** times out of **6**, because
`[6, 7, 7]` was the answer to one question. The reading Actor finds
`[9, 9, 6]` on trial **2** of the same run. What looked like learning was a
table.

**FINDING: the lesson's Self-Reflector already writes the minimum.** Its
output carries one magnitude and one direction word, which is exactly what
the reading Actor consumes -- **1** regex, **0** changes to the reflector.
The gap was never in the prompt.

Structure: `ReadingActor` is the policy the exercise implies; `StyledReflector`
varies only the wording.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "03-reflexion-verbal-rl"
STYLES = ("generic", "diagnostic", "prescriptive")
TRIALS = 6


def need_from(text):
    """The whole reader: one magnitude, one direction."""
    short = re.search(r"(\d+) short", text)
    over = re.search(r"overshoots by (\d+)", text)
    return int(short.group(1)) if short else -int(over.group(1)) if over else 0


def adjust(values, need):
    """Push `need` into the triple one slot at a time, staying inside 1..9."""
    out = list(values)
    for index, value in enumerate(out):
        room = 9 - value if need > 0 else 1 - value
        step = min(need, room) if need > 0 else max(need, room)
        out[index], need = value + step, need - step
    return out


class ReadingActor:
    """Conditioned on the newest reflection's text, not on how many there are."""

    def __init__(self):
        self.values = [1, 2, 3]

    def act(self, memory):
        if memory.items:
            self.values = adjust(self.values, need_from(memory.items[-1].text))
        return list(self.values)


class StyledReflector:
    """The lesson's reflector, reworded three ways."""

    def __init__(self, ref, style):
        self.inner, self.style = ref.SelfReflector(), style

    def reflect(self, attempt, delta):
        if self.style == "generic":
            return "be more careful next time"
        base = self.inner.reflect(attempt, delta)
        if self.style == "prescriptive":
            return f"{base}; next attempt must sum to {sum(attempt) - delta}"
        return base


def run(ref, actor, reflector, target=20, trials=TRIALS):
    memory, rows = ref.EpisodicMemory(), []
    for number in range(1, trials + 1):
        attempt = actor.act(memory)
        success, delta = ref.binary_evaluator(attempt, target)
        rows.append({"trial": number, "attempt": attempt, "success": success})
        if success:
            break
        memory.add(ref.Reflection(number, reflector.reflect(attempt, delta)))
    return rows


def won_at(rows):
    return rows[-1]["trial"] if rows[-1]["success"] else 0


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    reading = {style: run(ref, ReadingActor(), StyledReflector(ref, style))
               for style in STYLES}
    counting = {style: run(ref, ref.Actor(), StyledReflector(ref, style))
                for style in STYLES}
    moved_count = run(ref, ref.Actor(), StyledReflector(ref, "diagnostic"), target=24)
    moved_read = run(ref, ReadingActor(), StyledReflector(ref, "diagnostic"), target=24)
    return {
        "reading_won": {style: won_at(rows) for style, rows in reading.items()},
        "counting_won": {style: won_at(rows) for style, rows in counting.items()},
        "counting_shapes": len({tuple(tuple(r["attempt"]) for r in rows)
                                for rows in counting.values()}),
        "generic_attempts": len({tuple(r["attempt"]) for r in reading["generic"]}),
        "moved_count": won_at(moved_count), "moved_read": won_at(moved_read),
        "moved_attempt": moved_read[-1]["attempt"],
        "counting_attempt": moved_count[-1]["attempt"],
        "first_reflection": ref.SelfReflector().reflect([1, 2, 3], -14),
        "extracted": need_from(ref.SelfReflector().reflect([1, 2, 3], -14)),
    }


def verify(result):
    reading = result["reading_won"]
    return [
        practice.Check(
            "ANSWER: a signed magnitude is the minimum; an instruction adds nothing",
            all([reading["generic"] == 0, reading["diagnostic"] == 2,
                 reading["prescriptive"] == 2, result["generic_attempts"] == 1]),
            f"against an Actor that reads text, 'be more careful next time' never "
            f"converges in {TRIALS} trials and leaves {result['generic_attempts']} "
            f"distinct attempt; the lesson's own wording converges at trial "
            f"{reading['diagnostic']}, and adding an explicit constraint also converges "
            f"at trial {reading['prescriptive']} -- {0} trials bought",
        ),
        practice.Check(
            "FINDING: no prompt can reach an Actor whose input is a count",
            all([result["counting_shapes"] == 1,
                 set(result["counting_won"].values()) == {3}]),
            f"across all 3 styles the shipped Actor produces "
            f"{result['counting_shapes']} distinct trajectory and converges at trial "
            f"{sorted(set(result['counting_won'].values()))[0]} every time. Forcing it to "
            "notice is not reachable from the reflection side at all",
        ),
        practice.Check(
            "FINDING: the counting Actor is a lookup, not a weak policy",
            all([result["moved_count"] == 0, result["moved_read"] == 2,
                 result["moved_attempt"] == [9, 9, 6],
                 result["counting_attempt"] == [6, 7, 7]]),
            f"move the target to 24 and the shipped Actor never converges, settling on "
            f"{result['counting_attempt']} -- the answer to a different question -- while "
            f"the reading Actor finds {result['moved_attempt']} at trial "
            f"{result['moved_read']}. What looked like learning was a table",
        ),
        practice.Check(
            "FINDING: the Self-Reflector already writes the minimum",
            all([result["first_reflection"] == "sum 6 is 14 short; pick larger values",
                 result["extracted"] == 14]),
            f"the shipped reflection reads {result['first_reflection']!r} and the reader "
            f"extracts {result['extracted']} from it with one regex. The reflector needed "
            "0 changes; the gap was in the Actor's signature, which is where prompt "
            "engineering cannot go",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
