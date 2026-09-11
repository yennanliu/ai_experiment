"""Exercise 3 — a validity test cannot see a tight constraint.

    **Hard.** Implement a regex-constrained decoder from scratch for phone numbers
    (`\\d{3}-\\d{3}-\\d{4}`). Verify 0 invalid outputs on 1000 samples.

Reading of the exercise: `code/main.py` already ships the decoder, and it scores
1000 of 1000. The interesting question is what that run establishes, and the
answer is almost nothing -- the check cannot fail on any decoder whose mask is
the pattern, and it cannot fail on decoders whose mask is *narrower* than the
pattern either.

Three mutations of `PhoneFSM`, each run through the shipped `generate_constrained`
for 1000 samples: an accept state one position early scores 0/1000, an extra `-`
allowed at the separator states scores 6/1000, and an FSM offering only the
digits 0-4 scores **1000/1000**. The third is the one that matters. It passes the
exercise's acceptance criterion unchanged while being able to emit only 9,765,625
of the 10,000,000,000 phone numbers -- 0.098% of the space. Validity is
invariant to how much of the legal space the constraint removes, so the test is
blind in exactly the direction constrained decoding fails in production: an FSM
that excludes the right answer.

Sample size is not the fix. 1000 draws touch 1000 of 10^10 numbers, one part in
ten million, and the shipped decoder produces 1000 distinct outputs -- so the run
is neither near-exhaustive nor repetitive enough to notice. Coverage has to be
computed from the FSM, not sampled from it: the reachable count is the product of
the alphabet sizes `valid_next` returns along the accepting path.

One thing the 1000 samples also cannot see: `sample()` ends with
`return len(probs) - 1`, taken whenever the accumulated probability never reaches
the drawn number. The last index of this alphabet is `-`, which the mask sets to
zero at every digit state, so the fallback returns a token the mask forbade. It
is guarded only by floating-point summation reaching 1.0, not by the mask -- and
a stub sampler with probabilities summing to 0.95 returns it on demand.

Structure: `variant` builds a mutated FSM from a per-state override of
`valid_next`; `run` samples 1000 outputs and counts regex matches; `reachable`
multiplies the branching factor along the accepting path.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "20-structured-outputs-constrained-decoding"

ALPHABET = list("0123456789-")
SAMPLES = 1000
DIGIT_STATES = (0, 1, 2, 4, 5, 6, 8, 9, 10, 11)
NARROW = "01234"


def variant(ref, *, accept=12, digits=None, separators=None):
    """A `PhoneFSM` with one part of the constraint changed."""
    class Mutated(ref.PhoneFSM):
        def __init__(self):
            super().__init__()
            self.accept_state = accept

        def valid_next(self, state):
            if digits is not None and state in DIGIT_STATES:
                return list(digits)
            if separators is not None and state in (3, 7):
                return list(separators)
            return super().valid_next(state)

    return Mutated()


def run(ref, fsm):
    """Outputs and match count for `SAMPLES` seeds through the shipped decoder."""
    outs = [ref.generate_constrained(ALPHABET, fsm, seed) for seed in range(SAMPLES)]
    return outs, sum(1 for out in outs if re.fullmatch(ref.PHONE_REGEX, out))


def reachable(fsm):
    """How many distinct strings the constraint admits, from the FSM rather than a sample."""
    total, state = 1, 0
    while not fsm.is_accept(state) and fsm.valid_next(state):
        total *= len(fsm.valid_next(state))
        state += 1
    return total


class Stub:
    """A sampler whose draw exceeds the probability mass, to reach the fallback branch."""

    def __init__(self, draw):
        self.draw = draw

    def random(self):
        return self.draw


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    shipped = ref.PhoneFSM()
    arms = {
        "shipped": shipped,
        "early accept": variant(ref, accept=11),
        "loose separator": variant(ref, separators="0123456789-"),
        "digits 0-4 only": variant(ref, digits=NARROW),
    }
    rows = {name: run(ref, fsm) for name, fsm in arms.items()}
    space = reachable(shipped)
    return {
        "samples": SAMPLES,
        "valid": {name: hits for name, (_, hits) in rows.items()},
        "space": space,
        "narrow_space": reachable(arms["digits 0-4 only"]),
        "narrow_share": round(reachable(arms["digits 0-4 only"]) / space * 100, 3),
        "distinct": len(set(rows["shipped"][0])),
        "explored": SAMPLES / space,
        "fallback": ref.sample([0.5, 0.45, 0.0], Stub(0.99)),
        "masked_index": len(ALPHABET) - 1,
        "masked_char": ALPHABET[-1],
    }


def verify(result):
    valid, samples = result["valid"], result["samples"]
    return [
        practice.Check(
            "ANSWER: the acceptance criterion passes on a decoder that is badly wrong",
            valid["digits 0-4 only"] == samples,
            f"the shipped FSM scores {valid['shipped']}/{samples}, and an FSM offering only the "
            f"digits {NARROW} scores {valid['digits 0-4 only']}/{samples} -- the same perfect "
            f"result while able to emit {result['narrow_space']:,} of {result['space']:,} phone "
            f"numbers, {result['narrow_share']}% of the space",
        ),
        practice.Check(
            "MECHANISM: the test is one-sided, and it catches the side that does not matter",
            valid["early accept"] < samples and valid["loose separator"] < samples,
            f"an accept state one position early scores {valid['early accept']}/{samples} and an "
            f"extra `-` at the separators scores {valid['loose separator']}/{samples}, so a mask "
            "that is too loose is caught immediately. A mask that is too tight is invisible, and "
            "too tight is the failure mode -- an FSM that excludes the right answer",
        ),
        practice.Check(
            "FINDING: 1000 samples is not a coverage test in either direction",
            result["explored"] < 1e-6,
            f"{samples} draws touch {result['explored']:.0e} of the space, and the shipped decoder "
            f"returns {result['distinct']} distinct outputs from {samples} seeds -- neither near "
            "exhaustive nor repetitive enough to notice the gap",
        ),
        practice.Check(
            "MECHANISM: coverage has to be computed from the FSM, not sampled from it",
            result["space"] == 10 ** 10,
            f"the reachable count is the product of the branching factors along the accepting "
            f"path: {result['space']:,} for the shipped FSM against {result['narrow_space']:,} "
            "for the narrowed one. That comparison is available before a single sample is drawn",
        ),
        practice.Check(
            "FINDING: the sampler has a fallback that ignores the mask",
            result["fallback"] == 2,
            f"`sample` ends with `return len(probs) - 1`, taken whenever the accumulated "
            f"probability never reaches the draw. Index {result['masked_index']} of this alphabet "
            f"is {result['masked_char']!r}, which the mask zeroes at every digit state -- a stub "
            "drawing 0.99 against 0.95 of mass returns it. The guarantee rests on floating-point "
            "summation reaching 1.0, not on the mask",
        ),
        practice.Check(
            "CONTROL: the shipped FSM is exactly the pattern, so the exercise's own answer is right",
            valid["shipped"] == samples and result["space"] == 10 ** 10,
            "three digits, a separator, three digits, a separator, four digits, accepting at 12 -- "
            f"the full {result['space']:,} numbers `\\d{{3}}-\\d{{3}}-\\d{{4}}` admits. The "
            "decoder is correct; the verification the exercise asks for is what does not "
            "establish it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
