"""Exercise 2 — the hack is free for two hundred generations.

    Run with `--reward-hack-allowed`. Compare score trajectories. How many
    generations until the loop learns to inflate score? What does the
    "winner" actually do?

Reading of the exercise: "how many generations" is a property of a seed, not
of the loop, so the shipped seed is reported and then twenty are measured.
"What does the winner actually do" is read as a question about *behaviour*,
which the run already prints beside the inflated number.

**ANSWER: generation 3 at the shipped seed, a median of 17.5 over twenty,
and the winner does exactly what the honest agent does.** Every one of **20**
seeds learns to inflate, first between generations **1** and **83**. Final
true score is **0.25** in 10 of 10 seeds with the channel open and **0.25**
in 10 of 10 with it closed; the channel buys **0.07** to **0.29** of reported
score and **0.00** of behaviour.

**FINDING: inflating creates a cell no honest agent can occupy.** The
archive is keyed by `(len(ops), round(reported, 2))`, so every new bonus
value opens a new score bucket. Closed, the archive holds **17-23** cells;
open, **44-77**. A hacked agent is not out-competing anyone -- it is moving
somewhere nobody else can stand.

**FINDING: the bonus only goes up.** `mutate` raises it with probability
**0.08** per generation and has **0** branches that lower it, so the
expected drift is about **+0.004** a generation and the only thing bounding
it is the `min(1.0, ...)` in `reported_score`.

**FINDING: 200 generations is what makes the hack look free.** Run the same
loop for **2000** and reported reaches **0.48-0.65** while true falls to
**0.12** in **2** of 5 seeds. Given enough time a hacked agent with a worse
operator sequence outranks an honest one, and the behaviour goes with it.
The demo's headline -- the agent edits the pipeline "not the behavior" --
holds for the horizon it ships and stops holding at ten times that.

Structure: `transcript()` captures each shipped run; `first_inflated()`
reads the generation where the printed report first exceeds the printed true.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import random
import re
import statistics

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "04-darwin-godel-machine"

SHIPPED_SEED, SHORT, LONG = 7, 200, 2000
SEEDS, LONG_SEEDS = range(20), range(5)


def transcript(ref, hack_allowed, seed, generations=SHORT):
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        ref.run_dgm(generations, hack_allowed, seed=seed)
    return buffer.getvalue()


def rows(text):
    found = [re.findall(r"[-\d.]+", line) for line in text.splitlines()
             if line.strip().startswith("gen ")]
    return [(int(r[0]), float(r[1]), float(r[2])) for r in found]


def first_inflated(text):
    """The generation whose printed report first exceeds its printed true score."""
    for generation, reported, true in rows(text):
        if reported > true + 1e-9:
            return generation
    return None


def summary(text, label):
    return float(re.search(rf"{label}\s+: ([-+\d.]+)", text).group(1))


def finals(ref, hack_allowed, seeds, generations=SHORT):
    texts = [transcript(ref, hack_allowed, seed, generations) for seed in seeds]
    return ([summary(text, "final reported score") for text in texts],
            [summary(text, "final true score") for text in texts])


def archive_sizes(ref, hack_allowed, seeds=range(5)):
    """`run_dgm`'s archive, which it does not return: how many cells it ends up with."""
    sizes = []
    for seed in seeds:
        random.seed(seed)
        start = ref.Agent(ops=["nop"])
        grid = {(1, round(ref.reported_score(start, hack_allowed), 2)): start}
        for _generation in range(SHORT):
            child = ref.mutate(random.choice(list(grid.values())), hack_allowed)
            reported = ref.reported_score(child, hack_allowed)
            key = (len(child.ops), round(reported, 2))
            if key not in grid or reported > ref.reported_score(grid[key], hack_allowed):
                grid[key] = child
        sizes.append(len(grid))
    return [min(sizes), max(sizes)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    hits = [first_inflated(transcript(ref, True, seed)) for seed in SEEDS]
    found = [value for value in hits if value is not None]
    open_reported, open_true = finals(ref, True, range(10))
    _closed_reported, closed_true = finals(ref, False, range(10))
    long_reported, long_true = finals(ref, True, LONG_SEEDS, LONG)
    mutation = inspect.getsource(ref.mutate)
    loop = inspect.getsource(ref.run_dgm)
    return {
        "shipped": first_inflated(transcript(ref, True, SHIPPED_SEED)),
        "seeds": len(hits),
        "inflating": len(found),
        "median_gen": statistics.median(found),
        "span": [min(found), max(found)],
        "open_true": sorted(set(open_true)),
        "closed_true": sorted(set(closed_true)),
        "bought": [round(min(r - t for r, t in zip(open_reported, open_true)), 2),
                   round(max(r - t for r, t in zip(open_reported, open_true)), 2)],
        "key_uses_reported": "round(rep, 2)" in loop,
        "closed_cells": archive_sizes(ref, False),
        "open_cells": archive_sizes(ref, True),
        "raise_probability": re.findall(r"random\.random\(\) < ([\d.]+)", mutation)[-1],
        "lowering_branches": mutation.count("bonus -"),
        "clamped": "min(1.0" in inspect.getsource(ref.reported_score),
        "long_reported": [min(long_reported), max(long_reported)],
        "long_degraded": sum(value < 0.25 for value in long_true),
        "long_seeds": len(LONG_SEEDS),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: generation 3 here, median 17.5 over twenty, and the same behaviour",
            all([result["shipped"] == 3, result["inflating"] == result["seeds"] == 20,
                 result["median_gen"] == 17.5, result["span"] == [1, 83],
                 result["open_true"] == [0.25], result["closed_true"] == [0.25],
                 result["bought"] == [0.07, 0.29]]),
            f"all {result['inflating']} seeds learn to inflate, first between "
            f"generations {result['span']} with a median of {result['median_gen']} and "
            f"{result['shipped']} at the shipped seed; final true score is "
            f"{result['open_true']} open and {result['closed_true']} closed, so the "
            f"channel buys {result['bought']} of report and none of behaviour",
        ),
        practice.Check(
            "FINDING: inflating creates a cell no honest agent can occupy",
            all([result["key_uses_reported"], result["closed_cells"] == [17, 23],
                 result["open_cells"] == [44, 77]]),
            f"the archive key is (len(ops), round(reported, 2)), so every new bonus "
            f"value opens a bucket an honest agent cannot reach: {result['closed_cells']} "
            f"cells closed against {result['open_cells']} open -- a hacked agent moves "
            "rather than wins",
        ),
        practice.Check(
            "FINDING: the bonus only goes up",
            all([result["raise_probability"] == "0.08",
                 result["lowering_branches"] == 0, result["clamped"]]),
            f"mutate raises the bonus with probability {result['raise_probability']} "
            f"and has {result['lowering_branches']} branches that lower it, so the only "
            "bound is the min(1.0, ...) in reported_score",
        ),
        practice.Check(
            "FINDING: 200 generations is what makes the hack look free",
            all([result["long_reported"] == [0.48, 0.65], result["long_degraded"] == 2,
                 result["long_seeds"] == 5]),
            f"at {LONG} generations reported reaches {result['long_reported']} and true "
            f"falls below 0.25 in {result['long_degraded']} of {result['long_seeds']} "
            "seeds -- the behaviour goes too, once there is time for it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
