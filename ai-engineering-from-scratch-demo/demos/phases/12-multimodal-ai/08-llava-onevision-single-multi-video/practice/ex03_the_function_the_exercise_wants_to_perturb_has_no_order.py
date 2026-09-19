"""Exercise 3 — the function the exercise wants to perturb has no order.

    Swap the curriculum order — train multi-image first, then single-image, then
    video. Predict which benchmarks degrade and why.

Reading of the exercise: the swap is attempted against the lesson's own
`curriculum_stages` first, and it cannot be made -- the function takes only the
target mix and always emits the same three stages. What the swap *does* change
is measurable anyway: the cumulative exposure each scenario gets across the three
stages, which is a sum over the stage mixes and is what the prediction has to
rest on.

**ANSWER: single-image degrades, and by the largest relative move of the
three.** Summing the three stage mixes, the shipped order gives single-image
**1.9** stage-units of exposure, multi-image 0.6 and video 0.5. Putting
multi-image first gives **0.9 / 1.6 / 0.5**: single-image loses **53%** of its
exposure and multi-image gains **167%**, while video is untouched at 0.5.
MMMU and DocVQA are the benchmarks that ride on single-image.

**FINDING: `curriculum_stages` has no order parameter.** Its signature is
`curriculum_stages(mix)`, and the first two stages are literals -- `1.0, 0, 0`
and `0.5, 0.3, 0.2`. The permutation the exercise asks for is not expressible in
the planner the lesson ships; only the target-task stage moves.

**FINDING: the lesson has a number for one of six orderings, and it is not this
one.** Six permutations of three stages; the printed note covers "stages in
reverse (video first)" and puts it at 2-4 MMMU. Multi-image-first is a different
permutation, and the note's own ordering leaves single-image at **0.9** exposure
-- the same figure as the exercise's swap, which is why the note's 2-4 MMMU is
the best available estimate for it.

**FINDING: video is invariant under both swaps.** Moving either single-image or
multi-image into stage 1 leaves video at 0.5 stage-units, because video appears
only in the last two stages under every ordering that does not put it first.
The scenario the curriculum exists to reach is the one the order does not move.

Structure: `STAGES` is the lesson's own schedule with stage 1 parameterised,
`exposure` sums the mixes per scenario, and `swap` builds the stage-1 mix for one
scenario-first ordering.
"""

from __future__ import annotations

import inspect
import itertools

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "08-llava-onevision-single-multi-video"
SCENARIOS = ("single", "multi", "video")
LATER_STAGES = ((0.5, 0.3, 0.2), (0.4, 0.3, 0.3))
REVERSE_PENALTY = "2-4 MMMU"


def swap(first):
    """Stage 1 as the lesson writes it -- one scenario at 1.0, the others at 0."""
    return tuple(1.0 if name == first else 0.0 for name in SCENARIOS)


def exposure(first):
    """Cumulative stage-units per scenario across the three stages."""
    stages = (swap(first),) + LATER_STAGES
    return {name: round(sum(stage[i] for stage in stages), 2)
            for i, name in enumerate(SCENARIOS)}


def change(before, after):
    return {name: round((after[name] / before[name] - 1) * 100)
            for name in SCENARIOS}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    shipped, multi_first = exposure("single"), exposure("multi")
    signature = inspect.signature(ref.curriculum_stages)
    source = inspect.getsource(ref.curriculum_stages)
    return {
        "shipped": shipped, "multi_first": multi_first,
        "video_first": exposure("video"),
        "change": change(shipped, multi_first),
        "loser": min(change(shipped, multi_first), key=change(shipped, multi_first).get),
        "parameters": list(signature.parameters),
        "literal_stages": source.count('("Stage'),
        "mix_references": source.count('mix['),
        "orderings": len(list(itertools.permutations(SCENARIOS))),
        "penalty_noted": REVERSE_PENALTY in source,
        "video_invariant": len({exposure(name)["video"] for name in ("single", "multi")}) == 1,
    }


def verify(result):
    shipped, swapped, moved = result["shipped"], result["multi_first"], result["change"]
    return [
        practice.Check(
            "ANSWER: single-image degrades, by the largest relative move of the three",
            all([shipped == {"single": 1.9, "multi": 0.6, "video": 0.5},
                 swapped == {"single": 0.9, "multi": 1.6, "video": 0.5},
                 moved == {"single": -53, "multi": 167, "video": 0},
                 result["loser"] == "single"]),
            f"cumulative exposure goes {shipped} -> {swapped}, a change of {moved} percent. "
            "Single-image loses over half of its training exposure and multi-image nearly "
            "triples, so MMMU and DocVQA -- the benchmarks that ride on single-image -- are "
            "what degrades",
        ),
        practice.Check(
            "FINDING: curriculum_stages has no order parameter",
            all([result["parameters"] == ["mix"], result["literal_stages"] == 3,
                 result["mix_references"] == 3]),
            f"the signature is curriculum_stages({', '.join(result['parameters'])}) and the "
            f"body holds {result['literal_stages']} literal stage rows, of which only the "
            f"last reads the argument ({result['mix_references']} references, all in stage "
            "TT). The permutation the exercise asks for is not expressible in the planner "
            "the lesson ships",
        ),
        practice.Check(
            "FINDING: the lesson has a number for one of six orderings, and it is not this one",
            all([result["orderings"] == 6, result["penalty_noted"],
                 result["video_first"]["single"] == swapped["single"]]),
            f"there are {result['orderings']} permutations of three stages; the printed note "
            f"covers video-first and puts it at {REVERSE_PENALTY}. That ordering leaves "
            f"single-image at {result['video_first']['single']} stage-units -- the same "
            f"figure as this swap -- which is why its penalty is the best available estimate "
            "for a permutation nobody measured",
        ),
        practice.Check(
            "FINDING: video is invariant under both swaps",
            all([result["video_invariant"], shipped["video"] == swapped["video"] == 0.5]),
            f"moving either single-image or multi-image into stage 1 leaves video at "
            f"{shipped['video']} stage-units, because video appears only in the last two "
            "stages under any ordering that does not put it first. The scenario the "
            "curriculum exists to reach is the one the order does not move",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
