"""Exercise 2 — the TTL never fires because the count bound is tighter.

    Add a TTL of 10 trials to reflections. Do older reflections hurt or help
    after that point?

Reading of the exercise: `EpisodicMemory` is bounded by `max_len = 6`, which
is a count, while a TTL is an age. The two units only coincide while exactly
one reflection is written per trial, so the first thing to measure is whether
a TTL of 10 ever evicts anything the shipped cap has not already dropped.
"After that point" needs a run longer than 10 trials, so the target is moved
to **21**, which the scripted Actor can never reach.

**ANSWER: a TTL of 10 trials, and it never fires.** At trial **15** the
buffer holds **6** reflections with the TTL in place and **6** without it --
`max_len = 6` evicts trial 8 long before age 10 could. Lift the count bound
and the TTL holds **10**. Neither number changes a single action: trials **3**
through **15** are all `[6, 7, 7]`.

**FINDING: hurt or help is decided by the count crossing 2, not by age.**
`Actor.act` branches on `len(memory.items)` at 0, 1 and "2 or more", so every
buffer with at least **2** entries is the same buffer. A TTL of **10** and a
TTL of **3** produce identical runs; a TTL of **2** holds the Actor at
`[5, 6, 7]` for **14** straight trials. Ageing reflections out can only hurt,
and only by starving a counter.

**FINDING: the buffer is **6** slots holding **1** sentence.** From trial 3
the attempt is constant, so `SelfReflector` writes the same line every time:
at trial 15 the **6** stored reflections have **1** distinct text, and
`as_prompt` renders **6** lines of which **5** are duplicates. That is memory
rot with a number on it -- the prompt grows and the information does not.

**FINDING: the two bounds are different units.** With one reflection per
trial they agree; skip two writes and at trial 15 the
count bound holds six trials back to 9 while the age bound holds nine back to
6 -- the same rule, two units.

Structure: `TTLMemory` is the lesson's buffer plus an age bound; `run()` is
`run_reflexion` with the memory and target as parameters.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "03-reflexion-verbal-rl"
TARGET, TRIALS = 21, 15


class TTLMemory:
    """`EpisodicMemory` with both bounds: a count cap and an age cap."""

    def __init__(self, ttl=10, max_len=6):
        self.ttl, self.max_len, self.items, self.now = ttl, max_len, [], 0

    def add(self, reflection):
        self.items.append(reflection)
        self.expire()

    def expire(self):
        self.items = [r for r in self.items if self.now - r.trial < self.ttl]
        while len(self.items) > self.max_len:
            self.items.pop(0)

    def as_prompt(self):
        if not self.items:
            return "(no prior reflections)"
        return "\n".join(f"- trial {r.trial}: {r.text}" for r in self.items)


def run(ref, memory, target=TARGET, trials=TRIALS):
    actor, reflector, rows = ref.Actor(), ref.SelfReflector(), []
    for number in range(1, trials + 1):
        memory.now = number
        memory.expire()
        attempt = actor.act(memory)
        success, delta = ref.binary_evaluator(attempt, target)
        text = reflector.reflect(attempt, delta)
        rows.append(attempt)
        if success:
            break
        memory.add(ref.Reflection(trial=number, text=text))
    return rows, memory


def capped(ref):
    """The shipped buffer, which has a count bound and no age bound."""
    memory = ref.EpisodicMemory()
    memory.now, memory.expire = 0, lambda: None
    return memory


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    ttl_rows, ttl_mem = run(ref, TTLMemory(ttl=10, max_len=6))
    base_rows, base_mem = run(ref, capped(ref))
    _, wide_mem = run(ref, TTLMemory(ttl=10, max_len=99))
    short_rows, _ = run(ref, TTLMemory(ttl=2, max_len=99))
    three_rows, _ = run(ref, TTLMemory(ttl=3, max_len=99))
    aged, counted = TTLMemory(ttl=10, max_len=99), ref.EpisodicMemory()
    for number in [n for n in range(1, 16) if n not in (5, 10)]:
        aged.now = number
        aged.add(ref.Reflection(trial=number, text="x"))
        counted.add(ref.Reflection(trial=number, text="x"))
    aged.now = 15
    aged.expire()
    return {
        "ttl_size": len(ttl_mem.items), "base_size": len(base_mem.items),
        "wide_size": len(wide_mem.items),
        "steady": sorted({tuple(a) for a in ttl_rows[2:]}), "trials": len(ttl_rows),
        "ttl_equals_base": ttl_rows == base_rows,
        "ttl3_equals_ttl10": three_rows == ttl_rows,
        "short_rows": sorted({tuple(a) for a in short_rows}),
        "short_repeats": short_rows.count([5, 6, 7]),
        "texts": len({r.text for r in ttl_mem.items}),
        "lines": len(ttl_mem.as_prompt().splitlines()),
        "age_window": [r.trial for r in aged.items],
        "count_window": [r.trial for r in counted.items],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: a TTL of 10 trials, and the count bound gets there first",
            all([result["ttl_size"] == 6, result["base_size"] == 6,
                 result["wide_size"] == 10, result["trials"] == 15,
                 result["steady"] == [(6, 7, 7)], result["ttl_equals_base"] is True]),
            f"at trial {result['trials']} the buffer holds {result['ttl_size']} with the "
            f"TTL and {result['base_size']} without it -- max_len=6 evicts before age 10 "
            f"can. Lifting the count bound leaves {result['wide_size']}. Trials 3 to 15 "
            f"are all {result['steady'][0]} either way",
        ),
        practice.Check(
            "FINDING: hurt or help is decided by the count crossing 2, not by age",
            all([result["ttl3_equals_ttl10"] is True, result["short_repeats"] == 14,
                 result["short_rows"] == [(1, 2, 3), (5, 6, 7)]]),
            f"Actor.act branches at 0, 1 and 2-or-more, so a TTL of 3 and a TTL of 10 "
            f"give identical runs ({result['ttl3_equals_ttl10']}). A TTL of 2 starves the "
            f"counter and holds the Actor at [5, 6, 7] for {result['short_repeats']} "
            "straight trials. Ageing reflections out can only hurt, and only that way",
        ),
        practice.Check(
            "FINDING: six slots holding one sentence",
            all([result["texts"] == 1, result["lines"] == 6,
                 result["lines"] - result["texts"] == 5]),
            f"from trial 3 the attempt is constant, so the reflector writes the same line "
            f"every time: {result['ttl_size']} stored reflections carry "
            f"{result['texts']} distinct text and as_prompt renders {result['lines']} "
            "lines. The prompt grows and the information does not",
        ),
        practice.Check(
            "FINDING: a count bound and an age bound are different units",
            all([result["count_window"] == [9, 11, 12, 13, 14, 15],
                 result["age_window"] == [6, 7, 8, 9, 11, 12, 13, 14, 15],
                 len(result["age_window"]) == 9]),
            f"with one write per trial the two agree; skip a write and at trial 15 the "
            f"count bound holds {result['count_window']} while the age bound holds "
            f"{result['age_window']} -- {len(result['count_window'])} against "
            f"{len(result['age_window'])} under the same rule",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
