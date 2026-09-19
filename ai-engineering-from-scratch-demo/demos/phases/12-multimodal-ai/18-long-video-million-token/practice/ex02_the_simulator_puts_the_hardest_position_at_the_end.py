"""Exercise 2 — the simulator puts the hardest position at the end.

    Design a needle-in-a-haystack test: at what minute do you inject the marker,
    and what is the exact query format?

Reading of the exercise: the design is written against the lesson's own
`nih_trial` and then checked by sweeping it over insertion depth, because "at
what minute" has a standard answer -- the middle, where the lost-in-the-middle
effect is strongest -- and the lesson's recall model turns out to put its minimum
somewhere else.

**ANSWER: sweep eleven depths from 0% to 100% and run N trials at each**, with
the middle as the position that separates models. The marker must be lexically
unique and semantically out of place, and the query must admit one string:

```text
At what time does the red umbrella appear?
Answer with only the timestamp in seconds, e.g. 123.4
```

**FINDING: the lesson's recall model is monotone, so it has no middle.** Swept
over eleven depths its curve is 0.95, 0.95, 0.85 x4, 0.75 x5 -- its minimum is at
depth **0.6 and beyond**, and the last value is the lowest. A lost-in-the-middle
curve has its minimum at **0.5** and recovers afterwards; the two disagree by
**0.23** at the very end, which is where the lesson is most confident and the
literature is least.

**FINDING: `nih_trial` returns a probability and never samples it.** There is no
draw against `recall_prob`, so a "trial" produces no hit and no miss and N of
them produce no recall. The function is named for an experiment and implements a
lookup.

**FINDING: and the demo runs one trial per model at a random depth.** The needle
position is `random.uniform(0, duration)`, so the four printed probabilities are
four different depths on four different curves -- not comparable to each other,
and not a measurement of any of them.

**ANSWER: so the design is three things the simulator lacks.** A depth *sweep*
rather than a random draw; a *sampled* outcome so recall is a frequency; and a
query whose answer is checkable by string equality, because a grader that accepts
"early in the video" cannot distinguish a model that found the needle from one
that guessed the prior.

Structure: `recall_at` reproduces the lesson's threshold lookup, `DEPTHS` is the
sweep, `u_shaped` is the lost-in-the-middle alternative, and `argmin` locates
each curve's hardest position.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "18-long-video-million-token"
CURVE = ((0.1, 0.95), (0.5, 0.85), (1.0, 0.75))
DEPTHS = tuple(index / 10 for index in range(11))
PEAK, DIP = 0.98, 0.25
QUERY = ("At what time does the red umbrella appear?\n"
         "Answer with only the timestamp in seconds, e.g. 123.4")


def recall_at(depth, curve=CURVE):
    for threshold, recall in curve:
        if depth <= threshold:
            return recall
    return curve[-1][1]


def u_shaped(depth, peak=PEAK, dip=DIP):
    """Lost in the middle: best at both ends, worst at the midpoint."""
    return round(peak - dip * (1 - abs(2 * depth - 1)), 3)


def argmin(values, depths=DEPTHS):
    return depths[values.index(min(values))]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    lesson = [recall_at(depth) for depth in DEPTHS]
    alternative = [u_shaped(depth) for depth in DEPTHS]
    gaps = [abs(a - b) for a, b in zip(lesson, alternative)]
    source = inspect.getsource(ref.nih_trial)
    demo = inspect.getsource(ref.nih_simulation)
    return {
        "depths": list(DEPTHS), "lesson": lesson, "alternative": alternative,
        "lesson_argmin": argmin(lesson), "alternative_argmin": argmin(alternative),
        "monotone": all(a >= b for a, b in zip(lesson, lesson[1:])),
        "recovers": alternative[-1] > alternative[5],
        "worst_gap": round(max(gaps), 2),
        "worst_gap_at": DEPTHS[gaps.index(max(gaps))],
        "samples_outcome": "random.random()" in source,
        "returns_probability": "recall_prob" in source,
        "random_depth": "random.uniform" in source,
        "trials_per_model": demo.count("nih_trial("),
        "models": demo.count("(0.1,"),
        "query_lines": len(QUERY.splitlines()),
        "query_constrains_format": "only the timestamp" in QUERY,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: sweep eleven depths, N trials each, with the middle as the separator",
            all([len(result["depths"]) == 11, result["depths"][5] == 0.5,
                 result["query_lines"] == 2, result["query_constrains_format"]]),
            f"eleven depths from {result['depths'][0]} to {result['depths'][-1]} with the "
            f"middle at {result['depths'][5]}, and a query that admits one string: "
            f"{QUERY.splitlines()[0]!r} followed by a format constraint. The marker must be "
            "lexically unique and semantically out of place, so it cannot be guessed from "
            "context",
        ),
        practice.Check(
            "FINDING: the lesson's recall model is monotone, so it has no middle",
            all([result["lesson"] == [0.95, 0.95, 0.85, 0.85, 0.85, 0.85,
                                      0.75, 0.75, 0.75, 0.75, 0.75],
                 result["monotone"], result["lesson_argmin"] == 0.6,
                 result["alternative_argmin"] == 0.5, result["recovers"],
                 result["worst_gap"] == 0.23, result["worst_gap_at"] == 1.0]),
            f"swept over the depths its curve is {result['lesson']} -- monotone, with its "
            f"minimum first reached at depth {result['lesson_argmin']} and never recovering. "
            f"A lost-in-the-middle curve bottoms at {result['alternative_argmin']} and "
            f"recovers; the two disagree most at depth {result['worst_gap_at']}, by "
            f"{result['worst_gap']}, which is where the lesson is most confident",
        ),
        practice.Check(
            "FINDING: nih_trial returns a probability and never samples it",
            all([result["returns_probability"], not result["samples_outcome"]]),
            "there is no draw against recall_prob anywhere in the function, so a trial "
            "produces no hit and no miss and N of them produce no recall. It is named for an "
            "experiment and implements a lookup",
        ),
        practice.Check(
            "FINDING: the demo runs one trial per model at a random depth",
            all([result["random_depth"], result["trials_per_model"] == 1,
                 result["models"] == 4]),
            f"the needle position is random.uniform(0, duration) and nih_simulation calls "
            f"nih_trial {result['trials_per_model']} time inside a loop over "
            f"{result['models']} models, so the four printed probabilities are four "
            "different depths on four different curves -- not comparable, and not a "
            "measurement of any of them",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
