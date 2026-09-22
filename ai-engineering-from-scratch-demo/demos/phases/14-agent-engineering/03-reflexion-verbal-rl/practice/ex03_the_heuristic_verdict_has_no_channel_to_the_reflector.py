"""Exercise 3 — the heuristic verdict has no channel to the reflector.

    Implement heuristic evaluator: mark the trial as stuck if the same action
    repeats. How does this interact with Self-Reflector?

Reading of the exercise: "how does this interact" is a question about
plumbing, so it is answered by looking at what the Self-Reflector can be told.
`SelfReflector.reflect(attempt, delta)` takes the action and the distance and
nothing else, so a heuristic verdict has no parameter to arrive in. The
target is moved to **21**, which the scripted Actor can never reach, so the
repetition the heuristic looks for actually happens.

**ANSWER: `stuck_evaluator` fires on trials 4 through 8.** The Actor settles
on `[6, 7, 7]` at trial 3, so **5** of **8** trials repeat the previous
action. On the reachable target the same heuristic fires **0** times, because
the run ends before anything can repeat.

**FINDING: the verdict cannot reach the reflector.** `reflect` has **2**
parameters besides `self`, neither of them a verdict, so the reflection
written on the fifth stuck trial is byte-identical to the one written on the
trial that first produced that action. The heuristic can stop the loop; it
cannot change a word of what the loop remembers.

**FINDING: the only interaction available is arithmetic.** A heuristic
evaluator that writes its verdict into the same buffer changes
`len(memory.items)`, which is the one thing `Actor.act` reads. Seeding the
buffer with a single entry before trial 1 -- true or nonsense, it makes no
difference -- converges on trial **2** instead of trial **3**. A 33% faster
run, with nothing read.

**FINDING: stuck is a narrower predicate than failing.** An Actor that cycles
three distinct wrong answers fails **8** times out of **8** and is marked
stuck **0** times. Repetition catches the loop that has given up, not the one
that is exploring uselessly, so as a safety rail it is one signature rather
than a class of them.

Structure: `stuck_evaluator` is the heuristic; `run()` is the lesson's trial
loop with the evaluator, actor and seed as parameters.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "03-reflexion-verbal-rl"
HARD, EASY, TRIALS = 21, 20, 8


class CyclingActor:
    """Fails every trial, repeats no action -- invisible to a repetition rail."""

    def __init__(self):
        self.turn = 0

    def act(self, memory):
        self.turn += 1
        return [[1, 2, 3], [4, 4, 4], [9, 9, 1]][self.turn % 3]


def stuck_evaluator(attempt, previous):
    """The heuristic the exercise asks for: the same action twice means stuck."""
    return previous is not None and attempt == previous


def run(ref, target, actor=None, seed=None, trials=TRIALS):
    actor = actor or ref.Actor()
    memory, reflector = ref.EpisodicMemory(), ref.SelfReflector()
    if seed is not None:
        memory.add(ref.Reflection(trial=0, text=seed))
    rows, previous = [], None
    for number in range(1, trials + 1):
        attempt = actor.act(memory)
        success, delta = ref.binary_evaluator(attempt, target)
        text = reflector.reflect(attempt, delta)
        rows.append({"trial": number, "attempt": attempt, "success": success,
                     "stuck": stuck_evaluator(attempt, previous), "text": text})
        previous = attempt
        if success:
            break
        memory.add(ref.Reflection(trial=number, text=text))
    return rows


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    hard, easy = run(ref, HARD), run(ref, EASY)
    cycling = run(ref, HARD, actor=CyclingActor())
    reflect_params = [p for p in inspect.signature(
        ref.SelfReflector.reflect).parameters if p != "self"]
    return {
        "stuck_trials": [row["trial"] for row in hard if row["stuck"]],
        "hard_len": len(hard), "easy_stuck": sum(1 for row in easy if row["stuck"]),
        "easy_len": len(easy),
        "reflect_params": reflect_params,
        "same_text": hard[2]["text"] == hard[7]["text"], "settled": hard[7]["attempt"],
        "seeded_true": len(run(ref, EASY, seed="sum 6 is 14 short; pick larger values")),
        "seeded_noise": len(run(ref, EASY, seed="the moon is made of cheese")),
        "cycling_failures": sum(1 for row in cycling if not row["success"]),
        "cycling_stuck": sum(1 for row in cycling if row["stuck"]),
        "cycling_distinct": len({tuple(row["attempt"]) for row in cycling}),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the heuristic fires on 5 of 8 trials, and 0 of the easy run",
            all([result["stuck_trials"] == [4, 5, 6, 7, 8], result["hard_len"] == 8,
                 result["easy_stuck"] == 0, result["easy_len"] == 3,
                 result["settled"] == [6, 7, 7]]),
            f"the Actor settles on {result['settled']} at trial 3, so trials "
            f"{result['stuck_trials']} repeat the previous action -- "
            f"{len(result['stuck_trials'])} of {result['hard_len']}. On the reachable "
            f"target the run ends at trial {result['easy_len']} with "
            f"{result['easy_stuck']} stuck trials, before anything can repeat",
        ),
        practice.Check(
            "FINDING: the verdict has no parameter to arrive in",
            all([result["reflect_params"] == ["attempt", "delta"],
                 result["same_text"] is True]),
            f"reflect takes {result['reflect_params']} and nothing else, so the reflection "
            f"written on the fifth stuck trial is identical to the one written when that "
            f"action was new ({result['same_text']}). The heuristic can stop the loop; it "
            "cannot change a word of what the loop remembers",
        ),
        practice.Check(
            "FINDING: the only interaction available is arithmetic",
            all([result["seeded_true"] == 2, result["seeded_noise"] == 2,
                 result["easy_len"] == 3,
                 result["seeded_true"] == result["seeded_noise"]]),
            f"seeding the buffer with one entry before trial 1 converges at trial "
            f"{result['seeded_true']} instead of {result['easy_len']}, and a true "
            f"reflection and 'the moon is made of cheese' do it equally well. An "
            "evaluator that writes into the buffer moves the policy by counting",
        ),
        practice.Check(
            "FINDING: stuck is a narrower predicate than failing",
            all([result["cycling_failures"] == 8, result["cycling_stuck"] == 0,
                 result["cycling_distinct"] == 3]),
            f"an Actor cycling {result['cycling_distinct']} distinct wrong answers fails "
            f"{result['cycling_failures']} times and is marked stuck "
            f"{result['cycling_stuck']} times. Repetition catches the loop that has given "
            "up, not the one exploring uselessly -- one signature, not a class",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
