"""Exercise 1 — one step and eight agree on all sixteen tokens.

    Masked discrete diffusion samples in ~16 steps. Why not 1? What breaks if
    you unmask everything at step 0?

Reading of the exercise: the answer is argued from the factorisation and then
*tested* against the lesson's own sampler by running the same prompt at T=8 and
at a single `step_unmask` with `keep_ratio=1.0`. The test is the point: the two
runs agree on every token, which says something about the toy rather than about
masked diffusion, and the difference between those two statements is the
exercise.

**ANSWER: because one step samples every position from its own marginal.**
Masked diffusion models the joint as a product of per-position conditionals
given whatever is already visible. At step 0 nothing is visible, so unmasking
everything draws all positions **independently** -- every patch individually
plausible, the picture jointly incoherent. Multiple steps exist so that each
round of decisions becomes context for the next.

**FINDING: the lesson's sampler cannot show this, because its model has no
cross-position dependence.** `mock_logits` reads `tokens[i]` and the index `i`,
and nothing else -- no neighbour, no window, no attention. Its logits for
position i are the same whatever the other fifteen positions hold.

**FINDING: so one step and eight produce identical output.** Both return
**16 of 16** matching tokens, and both return the deterministic cycle
`(prompt_seed + i) % 8`. Eight forward passes buy exactly nothing here, which is
the strongest possible demonstration that the sampling loop and the model are
separate concerns.

**FINDING: the +3.0 self-bias is a lock, not a condition.** The one place
`mock_logits` reads a decided token is to add 3.0 to *that same position's*
already-chosen value -- so the mechanism that looks like conditioning is a
commitment device that prevents a revisit. Show-o's real gain comes from
positions informing each other, and this toy implements only the half that keeps
a position from changing its mind.

Structure: `single_step` runs one unmasking at full keep ratio, `full_run` runs
the lesson's own `sample`, and `reads_other_positions` inspects the mock model's
source for any index other than its own.
"""

from __future__ import annotations

import inspect
import random

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "14-show-o-discrete-diffusion-unified"
PROMPT_SEED, STEPS, SEED = 3, 8, 2


def single_step(ref, prompt_seed=PROMPT_SEED, seed=SEED):
    random.seed(seed)
    return ref.step_unmask([ref.MASK] * ref.SEQ_LEN, prompt_seed, 1.0)


def full_run(ref, prompt_seed=PROMPT_SEED, steps=STEPS, seed=SEED):
    random.seed(seed)
    traces = ref.sample(prompt_seed, steps)
    return traces[-1], len(traces) - 1


def reads_other_positions(ref):
    """Does mock_logits index the sequence anywhere but at its own position?"""
    source = inspect.getsource(ref.mock_logits)
    return "tokens[" in source


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    one = single_step(ref)
    many, passes = full_run(ref)
    expected = [(PROMPT_SEED + i) % ref.VOCAB for i in range(ref.SEQ_LEN)]
    return {
        "one_step": one, "multi_step": many, "passes": passes,
        "agree": sum(a == b for a, b in zip(one, many)),
        "length": ref.SEQ_LEN,
        "is_cycle": many == expected,
        "cycle": expected,
        "cross_position": reads_other_positions(ref),
        "self_bias": "base[t] += 3.0" in inspect.getsource(ref.mock_logits),
        "index_bias": "(prompt_seed + i) % VOCAB" in inspect.getsource(ref.mock_logits),
        "no_masked_left": sum(1 for token in many if token == ref.MASK),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: one step samples every position from its own marginal",
            all([result["length"] == 16, result["passes"] == STEPS,
                 result["no_masked_left"] == 0]),
            f"masked diffusion models the joint as a product of per-position conditionals "
            f"given what is already visible. At step 0 nothing is visible, so unmasking all "
            f"{result['length']} draws them independently -- each plausible alone, the whole "
            f"incoherent. The lesson's run takes {result['passes']} passes to finish",
        ),
        practice.Check(
            "FINDING: the lesson's model has no cross-position dependence",
            all([not result["cross_position"], result["self_bias"],
                 result["index_bias"]]),
            "mock_logits reads the index i and the token at that same index, and nothing "
            "else -- no neighbour, no window, no attention. Its logits for position i are "
            "identical whatever the other fifteen positions hold",
        ),
        practice.Check(
            "FINDING: so one step and eight produce identical output",
            all([result["agree"] == result["length"] == 16,
                 result["one_step"] == result["multi_step"],
                 result["is_cycle"]]),
            f"both return {result['agree']} of {result['length']} matching tokens, and both "
            f"return the deterministic cycle (prompt_seed + i) % 8: {result['cycle']}. "
            f"{result['passes']} forward passes buy exactly nothing here, which is the "
            "strongest available demonstration that the sampling loop and the model are "
            "separate concerns",
        ),
        practice.Check(
            "FINDING: the +3.0 self-bias is a lock, not a condition",
            all([result["self_bias"], not result["cross_position"]]),
            "the one place mock_logits reads a decided token is to add 3.0 to that same "
            "position's already-chosen value, so the mechanism that looks like conditioning "
            "is a commitment device preventing a revisit. Show-o's real gain comes from "
            "positions informing each other; this toy implements only the half that stops a "
            "position changing its mind",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
