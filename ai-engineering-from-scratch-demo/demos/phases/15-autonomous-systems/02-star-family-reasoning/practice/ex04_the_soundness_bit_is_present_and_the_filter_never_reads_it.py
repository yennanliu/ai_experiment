"""Exercise 4 — the soundness bit is present, and the filter never reads it.

    Compare STaR's keep-if-correct filter to a process-supervised
    alternative that rewards each rationale step independently. Identify the
    labelling cost difference and the plausible quality difference.

Reading of the exercise: the module has no steps -- `Trace` carries a
strategy and two booleans -- so a step-level filter cannot be built here at
full strength. What can be built is its limit case: one label per rationale,
on soundness rather than on the answer. That is process supervision with
exactly one step, which makes it the ceiling on what step rewards buy in this
simulator rather than an approximation of them.

**ANSWER: the quality difference is 29x and the cost difference is the step
count.** Replaying `star_round`'s own update with the filter switched to the
soundness bit takes the shortcut share to **0.004** in five rounds against
**0.116** under keep-if-correct, and the sound share to **0.992** against
**0.872**. The cost is the same **1000** traces labelled, at k labels each
instead of 1 -- and k is not a number this module has, because **0** of
`Trace`'s **3** fields is a step.

**FINDING: half of the first round's training set is unsound.**
Keep-if-correct retains **0.40** of the samples, and of what it retains
**0.50** is a shortcut or a lucky guess. The filter is not choosing reasoning;
it is choosing outcomes, and at the shipped starting mix the outcome is a coin
flip on whether the reasoning is real.

**FINDING: the waste the lesson tabulates is 0.60, and the alternative has
none.** The comparison table's "discards all incorrect rationales" is **600**
of 1000 labelled samples thrown away each round. Step-level rewards discard
nothing: an incorrect step is a labelled negative, which is exactly the data
V-STaR goes back for.

**FINDING: the bit is there, and the loop is not allowed to see it.**
`star_round` names `rationale_sound` **0** times; `vstar_infer` reads it
**2** times to score candidates, and its own docstring calls that "an
idealized verifier ... an upper bound". The one signal that separates process
supervision from outcome supervision is handed to the verifier and withheld
from the filter.

Structure: `replay()` is `star_round`'s update with the filter as a knob;
`kept_mix()` is the first round's training set, by strategy.
"""

from __future__ import annotations

import dataclasses
import inspect

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "02-star-family-reasoning"

SOUND0, SHORT0, ROUNDS = 0.20, 0.40, 5
ALPHA, ID_HIT, RANDOM_HIT = 0.6, 0.40, 0.10
SAMPLES = 1000                  # star_round's own default n_samples


def replay(on_soundness, rounds=ROUNDS, sound=SOUND0, shortcut=SHORT0):
    """`star_round`'s update, keeping correct answers or sound rationales."""
    for _ in range(rounds):
        kept = ((sound, 0.0, 0.0) if on_soundness else
                (sound, shortcut * ID_HIT, (1 - sound - shortcut) * RANDOM_HIT))
        total = sum(kept)
        sound, shortcut = (ALPHA * kept[0] / total + (1 - ALPHA) * sound,
                           ALPHA * kept[1] / total + (1 - ALPHA) * shortcut)
        scale = max(sound + shortcut, 1.0)
        sound, shortcut = sound / scale, shortcut / scale
    return round(sound, 3), round(shortcut, 3)


def kept_mix():
    """The first round's kept set, as (sound, shortcut, guess) fractions of all samples."""
    return (SOUND0, SHORT0 * ID_HIT, (1 - SOUND0 - SHORT0) * RANDOM_HIT)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    outcome, process = replay(False), replay(True)
    mix = kept_mix()
    kept = sum(mix)
    fields = [f.name for f in dataclasses.fields(ref.Trace)]
    loop, verifier = inspect.getsource(ref.star_round), inspect.getsource(ref.vstar_infer)
    return {
        "outcome": outcome, "process": process,
        "quality_ratio": round(outcome[1] / process[1]),
        "trace_fields": fields,
        "step_fields": [name for name in fields if "step" in name],
        "samples": SAMPLES,
        "kept_fraction": round(kept, 3),
        "unsound_of_kept": round((mix[1] + mix[2]) / kept, 3),
        "discarded": round(SAMPLES * (1 - kept)),
        "loop_reads_soundness": loop.count("rationale_sound"),
        "verifier_reads_soundness": verifier.count("rationale_sound"),
        "verifier_admits": "upper bound" in (ref.vstar_infer.__doc__ or ""),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 29x on quality, and the cost difference is the step count",
            all([result["outcome"] == (0.872, 0.116), result["process"] == (0.992, 0.004),
                 result["quality_ratio"] == 29, result["step_fields"] == [],
                 len(result["trace_fields"]) == 3]),
            f"keeping sound rationales instead of correct answers ends at "
            f"{result['process']} against {result['outcome']} -- {result['quality_ratio']}x "
            f"less shortcut -- for the same {result['samples']} labelled traces at k "
            f"labels each, and {len(result['step_fields'])} of Trace's "
            f"{len(result['trace_fields'])} fields is a step, so k is not in this module",
        ),
        practice.Check(
            "FINDING: half of the first round's training set is unsound",
            all([result["kept_fraction"] == 0.40, result["unsound_of_kept"] == 0.50]),
            f"keep-if-correct retains {result['kept_fraction']} of the samples and "
            f"{result['unsound_of_kept']} of what it retains is a shortcut or a lucky "
            "guess -- the filter chooses outcomes, not reasoning",
        ),
        practice.Check(
            "FINDING: the tabulated waste is 0.60, and the alternative has none",
            all([result["discarded"] == 600, result["samples"] == 1000]),
            f"{result['discarded']} of {result['samples']} labelled samples are thrown "
            "away each round; step-level rewards discard nothing, because an incorrect "
            "step is a labelled negative",
        ),
        practice.Check(
            "FINDING: the bit is there, and the loop is not allowed to see it",
            all([result["loop_reads_soundness"] == 0,
                 result["verifier_reads_soundness"] == 2, result["verifier_admits"]]),
            f"star_round names rationale_sound {result['loop_reads_soundness']} times "
            f"and vstar_infer reads it {result['verifier_reads_soundness']}, under a "
            "docstring that calls the result an upper bound",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
