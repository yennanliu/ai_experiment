"""Exercise 1 — the binary evaluator already returns the scalar.

    Switch from binary to scalar evaluator that returns a distance metric (how
    far from target). Does it converge faster?

Reading of the exercise: `binary_evaluator` returns `(total == target, total
- target)` -- the distance is already the second element, and
`SelfReflector.reflect` already consumes it. So "switch to a scalar
evaluator" is a rename unless something downstream changes behaviour on the
number, and the only thing downstream is `Actor.act`. The exercise is
therefore answered by asking what the Actor is a function of.

**ANSWER: no, it does not converge faster -- **3** trials either way.** The
scalar evaluator produces the same attempts `[1, 2, 3]`, `[5, 6, 7]`,
`[6, 7, 7]` and the same success on trial **3** as the shipped run. The
deltas it reports, `[-14, -2, 0]`, are the ones `binary_evaluator` was
already returning.

**FINDING: the Actor is a function of `len(memory.items)` and nothing else.**
Its body names `memory.items` once, inside `len()`, and never touches
`Reflection.text`. Replacing the Self-Reflector with one that writes
`doing great, keep going` on every trial still converges on trial **3** with
identical attempts. The reflections are counted, not read.

**FINDING: so the evaluator cannot matter either.** An evaluator that always
reports success-distance **0** and one that reports the true distance drive
the same trajectory, because both add exactly one `Reflection` per failed
trial. The only channel from evaluator to policy in this toy is the *length*
of the buffer.

**FINDING: what the scalar would buy, if it were read.** Across the two
failures the binary verdict is `False` twice -- **1** distinct value, no
gradient -- while the distance is `14` then `2`, **2** distinct values that
order the failures. The scalar's advantage is real and entirely unrealised.

Structure: `trial_loop()` is `run_reflexion` with the evaluator, reflector
and actor as parameters; everything else is the lesson's.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "03-reflexion-verbal-rl"


class FlatReflector:
    """A reflector that carries no information at all."""

    def reflect(self, attempt, delta):
        return "doing great, keep going"


def scalar_evaluator(attempt, target):
    """The exercise's evaluator: a distance, with success read off distance 0."""
    distance = abs(sum(attempt) - target)
    return distance == 0, sum(attempt) - target


def blind_evaluator(attempt, target):
    """Always reports a perfect score -- and still drives the same trajectory."""
    return sum(attempt) == target, 0


def trial_loop(ref, evaluate, reflector, target=20, max_trials=6):
    actor, memory, rows = ref.Actor(), ref.EpisodicMemory(), []
    for number in range(1, max_trials + 1):
        attempt = actor.act(memory)
        success, delta = evaluate(attempt, target)
        text = reflector.reflect(attempt, delta)
        rows.append(ref.TrialResult(number, attempt, success, delta, text))
        if success:
            break
        memory.add(ref.Reflection(trial=number, text=text))
    return rows


def shape(rows):
    return {"trials": len(rows), "attempts": [row.attempt for row in rows],
            "deltas": [row.delta for row in rows], "won": rows[-1].success}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    shipped = shape(ref.run_reflexion(max_trials=6, use_memory=True))
    scalar = shape(trial_loop(ref, scalar_evaluator, ref.SelfReflector()))
    flat = shape(trial_loop(ref, scalar_evaluator, FlatReflector()))
    blind = shape(trial_loop(ref, blind_evaluator, ref.SelfReflector()))
    body = inspect.getsource(ref.Actor.act)
    return {
        "shipped": shipped, "scalar": scalar, "flat": flat, "blind": blind,
        "binary_out": ref.binary_evaluator([1, 2, 3], 20),
        "reads_text": body.count(".text"), "reads_len": body.count("len(memory.items)"),
        "binary_values": len({row.success for row in
                              ref.run_reflexion(max_trials=6, use_memory=True)[:-1]}),
        "scalar_values": len({abs(d) for d in scalar["deltas"][:-1]}),
    }


def verify(result):
    shipped, scalar = result["shipped"], result["scalar"]
    return [
        practice.Check(
            "ANSWER: no -- 3 trials either way, on identical attempts",
            all([shipped["trials"] == 3, scalar["trials"] == 3,
                 scalar["attempts"] == shipped["attempts"] == [[1, 2, 3], [5, 6, 7],
                                                               [6, 7, 7]],
                 scalar["deltas"] == [-14, -2, 0], scalar["won"] is True,
                 result["binary_out"] == (False, -14)]),
            f"the scalar evaluator converges in {scalar['trials']} trials on "
            f"{scalar['attempts']} with deltas {scalar['deltas']} -- the same as the "
            f"shipped run's {shipped['trials']}. binary_evaluator already returns "
            f"{result['binary_out']}, so the distance was never missing",
        ),
        practice.Check(
            "FINDING: the Actor is a function of len(memory.items) and nothing else",
            all([result["reads_text"] == 0, result["reads_len"] == 1,
                 result["flat"]["trials"] == 3,
                 result["flat"]["attempts"] == shipped["attempts"]]),
            f"Actor.act names len(memory.items) {result['reads_len']} time and "
            f"Reflection.text {result['reads_text']} times. A reflector that writes "
            f"'doing great, keep going' on every trial converges in "
            f"{result['flat']['trials']} on the same attempts -- reflections are counted",
        ),
        practice.Check(
            "FINDING: so the evaluator cannot matter either",
            all([result["blind"]["trials"] == 3,
                 result["blind"]["attempts"] == shipped["attempts"],
                 result["blind"]["deltas"] == [0, 0, 0]]),
            f"an evaluator that reports distance {result['blind']['deltas'][0]} on every "
            f"trial drives the identical trajectory in {result['blind']['trials']} "
            "trials, because both evaluators add exactly one Reflection per failure. The "
            "only channel from evaluator to policy is the length of the buffer",
        ),
        practice.Check(
            "FINDING: what the scalar would buy, if anything read it",
            all([result["binary_values"] == 1, result["scalar_values"] == 2,
                 scalar["deltas"][:2] == [-14, -2]]),
            f"across the two failures the binary verdict takes "
            f"{result['binary_values']} distinct value and the distance takes "
            f"{result['scalar_values']} -- 14 then 2, which orders the failures. That "
            "ordering is the scalar's whole advantage, and nothing here consumes it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
