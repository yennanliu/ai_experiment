"""Exercise 1 — the floor is 55% of the target.

    Your target TTFAB is 300ms. On a 7B Thinker and 300M Talker, write out every
    component's latency.

Reading of the exercise: the components come from the lesson's own `ttfab`, and
the target is treated as a constraint to be checked rather than a heading --
which the shipped configuration misses. The model is then swept over both of its
size parameters to find what a 300 ms budget actually permits.

**ANSWER: 310 ms, ten over the target.** Mic 50, Thinker prefill 100, first text
token 40, Talker 20, residual-VQ 30, waveform 70. Turning vision on adds a flat
80 and takes it to **390**.

**FINDING: 165 ms of that is spent before and after the models, and nothing can
move it.** Mic tokenisation, the Talker's `max(15, ...)` floor, the residual
decode and the waveform decoder are constants -- **53.2%** of the 310, and
**55%** of the 300 ms target. A free Thinker and a free Talker still cost 165 ms.

**FINDING: the target admits a 6B Thinker and no more.** Both model-scaled terms
carry `thinker_b / 7`, so the total is 165 + 20 x thinker_b: **290** ms at 6B and
**310** at 7B. The Talker is almost irrelevant by comparison -- 100M to 1,000M
moves the total by **52** ms while 7B to 8B moves it by 20.

**FINDING: the lesson's code disagrees with its own prose at both ends.** The
docs state "320-510ms at 7B" and "600-900ms at 70B"; `ttfab` returns **310** at
7B without vision -- below its own floor -- and **1,690** at 72B with vision,
**1.9x** its own ceiling.

Structure: `total` sums the lesson's own component list, `floor` isolates the
terms that take neither size parameter, and `sweep` walks the Thinker and Talker
sizes independently.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "20-omni-models-thinker-talker"
TARGET = 300
THINKER, TALKER = 7, 300
THINKER_SWEEP = (1, 5, 6, 7, 8)
TALKER_SWEEP = (100, 300, 1000)
DOC_FLOOR, DOC_CEILING = 320, 900


def config(ref, thinker=THINKER, talker=TALKER, vision=False):
    return ref.StreamConfig(thinker_b=thinker, talker_m=talker, include_vision=vision)


def total(ref, **kwargs):
    return round(sum(step.ms for step in ref.ttfab(config(ref, **kwargs))))


def components(ref, **kwargs):
    return [(step.name, round(step.ms)) for step in ref.ttfab(config(ref, **kwargs))]


def floor(ref):
    """The terms that take neither size parameter."""
    return round(sum(step.ms for step in ref.ttfab(config(ref, thinker=0, talker=0))))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    shipped = total(ref)
    fixed = floor(ref)
    by_thinker = {size: total(ref, thinker=size) for size in THINKER_SWEEP}
    by_talker = {size: total(ref, talker=size) for size in TALKER_SWEEP}
    return {
        "components": components(ref), "count": len(components(ref)),
        "total": shipped, "target": TARGET, "over": shipped - TARGET,
        "with_vision": total(ref, vision=True),
        "vision_cost": total(ref, vision=True) - shipped,
        "floor": fixed, "floor_share": round(fixed / shipped * 100, 1),
        "floor_vs_target": round(fixed / TARGET * 100),
        "by_thinker": by_thinker,
        "largest_fitting": max(size for size, ms in by_thinker.items() if ms <= TARGET),
        "per_billion": by_thinker[8] - by_thinker[7],
        "by_talker": by_talker,
        "talker_span": by_talker[1000] - by_talker[100],
        "big": total(ref, thinker=72, talker=300, vision=True),
        "doc_floor": DOC_FLOOR, "doc_ceiling": DOC_CEILING,
        "over_ceiling": round(total(ref, thinker=72, talker=300, vision=True)
                              / DOC_CEILING, 1),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 310 ms, ten over the target",
            all([result["count"] == 6, result["total"] == 310,
                 result["over"] == 10, result["with_vision"] == 390,
                 result["vision_cost"] == 80,
                 [ms for _, ms in result["components"]] == [50, 100, 40, 20, 30, 70]]),
            f"the six components are {result['components']} -- {result['total']} ms against a "
            f"{result['target']} target, {result['over']} over. Vision adds a flat "
            f"{result['vision_cost']} and takes it to {result['with_vision']}",
        ),
        practice.Check(
            "FINDING: 165 ms is spent before and after the models",
            all([result["floor"] == 165, result["floor_share"] == 53.2,
                 result["floor_vs_target"] == 55]),
            f"mic tokenisation, the Talker's max(15, ...) floor, the residual decode and the "
            f"waveform decoder take neither size parameter: {result['floor']} ms, "
            f"{result['floor_share']}% of the shipped total and {result['floor_vs_target']}% "
            "of the target. A free Thinker and a free Talker still cost 165 ms",
        ),
        practice.Check(
            "FINDING: the target admits a 6B Thinker and no more",
            all([result["by_thinker"] == {1: 190, 5: 270, 6: 290, 7: 310, 8: 330},
                 result["largest_fitting"] == 6, result["per_billion"] == 20]),
            f"both model-scaled terms carry thinker_b / 7, so the total is 165 + 20 x "
            f"thinker_b: {result['by_thinker']} ms. The largest that fits "
            f"{result['target']} is {result['largest_fitting']}B, and each further billion "
            f"costs {result['per_billion']} ms",
        ),
        practice.Check(
            "FINDING: the lesson's code disagrees with its own prose at both ends",
            all([result["total"] < DOC_FLOOR, result["big"] == 1690,
                 result["over_ceiling"] == 1.9,
                 result["talker_span"] == 52]),
            f"the docs state {DOC_FLOOR}-510 ms at 7B and 600-{DOC_CEILING} at 70B; ttfab "
            f"returns {result['total']} at 7B without vision -- below its own floor -- and "
            f"{result['big']:,} at 72B with vision, {result['over_ceiling']}x its own "
            f"ceiling. Meanwhile the whole Talker sweep from 100M to 1,000M spans "
            f"{result['talker_span']} ms",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
