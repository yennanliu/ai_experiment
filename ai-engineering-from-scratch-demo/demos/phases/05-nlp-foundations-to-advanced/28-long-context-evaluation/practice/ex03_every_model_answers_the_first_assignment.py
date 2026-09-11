"""Exercise 3 — every model answers the first assignment.

    **Hard.** Construct a variable-tracing task (X1 → X2 → X3, with 3 hops)
    embedded in 64k of filler. Measure accuracy across 3 frontier models. Report
    effective reasoning length per model.

Reading of the exercise: no frontier-model client is installed -- `openai`,
`anthropic` and `transformers` are all absent -- so the three models are three
copies of the lesson's `mock_retrieval_model` at different effective capacities,
5,000 / 20,000 / 80,000 words, which is the one axis the mock actually has. The
task is the lesson's own: `X1 = 42`, `X2 = X1 + 10`, `X3 = X2 * 2`, planted at
depths 0.15 / 0.45 / 0.75, swept over six lengths around the specified 64k.
Three-hop accuracy is **0 of 18 cells**. Every model's effective reasoning
length is **0**, and the number the exercise asks you to report does not exist.

The mock returns the first regex match inside its capacity and never reads the
question. `X1 = 42` is the shallowest plant, so at 64k of filler the answer to
"What is X1?", "What is X2?" and "What is X3?" is **`42` in all three cases**.
The exercise's third hop is answered with the first hop's right-hand side.

Reordering so `X3 = X2 * 2` is the shallowest plant does not help: the pattern
`x3\\s*= [A-Z0-9a-z_]+` captures `X3 = X2` and the model returns **`X2`** -- a
variable name, not a value. There is no arithmetic anywhere in
`mock_retrieval_model`, so no hop is representable at any capacity, and the
degradation curve the lesson wants plotted is flat on the floor.

The retrieval axis does work. Asking for X1 instead of X3 separates the three
models cleanly: effective retrieval lengths of **32,000 / 128,000 / 256,000**
words against effective reasoning lengths of **0 / 0 / 0**. The lesson's own
prescription -- report retrieval-effective and reasoning-effective -- produces
one informative number and one that is identical for every model, so it can rank
models on retrieval and cannot rank them at all on reasoning.

At the single length the exercise specifies, 64k, the three models score 0, 1
and 1 on the retrieval question. One length is one point; the boundaries at
32,000 and 128,000 only appear once the sweep is widened.

And a model that reads nothing beats all three. Answering the constant `42` to
every question scores **18 of 18** on the one-hop task, against the mocks'
**14 of 18** -- they return "no answer" whenever the plant is out of capacity --
and ties them at 0 of 18 on the three-hop task. Both halves of the spec sheet
are matched or beaten by a system with no context at all.

Fixtures: the three assignments, the depths and the two questions are invented
here (the lesson's Step 3 sketches the task in prose but ships no fixture); the
haystack layout reuses the lesson's own placement so capacity is the only
variable.

Structure: `place` plants the assignments at fixed depths; `ask` calls the mock;
`accuracy` sweeps one question at one capacity across the length grid;
`effective` reports the longest length clearing a threshold; `effective_lengths`
and `total` aggregate across models; `constant_score` is the read-nothing
baseline; `absent` probes for the model clients without importing them.
"""

from __future__ import annotations

import importlib.util

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "28-long-context-evaluation"

HOPS = ("X1 = 42", "X2 = X1 + 10", "X3 = X2 * 2")
DEPTHS = (0.15, 0.45, 0.75)
Q1, Q2, Q3 = "What is X1?", "What is X2?", "What is X3?"
ANSWERS = {Q1: "42", Q2: "52", Q3: "104"}
LENGTHS = (1000, 8000, 32000, 64000, 128000, 256000)
SPECIFIED = 64000
MODELS = {"mock-small": 5000, "mock-mid": 20000, "mock-large": 80000}
PROBED = ("openai", "anthropic", "transformers")
THRESHOLD = 0.7


def absent(names):
    """The probed clients that are not installed — probed by spec, never imported."""
    return tuple(n for n in names if importlib.util.find_spec(n) is None)


def place(ref, length, hops=HOPS, seed=7):
    """The three assignments planted at DEPTHS in `length` words of the lesson's filler."""
    words = ref.make_filler(length, seed=seed).split()
    for depth, hop in sorted(zip(DEPTHS, hops), reverse=True):
        pos = int(len(words) * depth)
        words = words[:pos] + [hop] + words[pos:]
    return " ".join(words)


def ask(ref, haystack, question, capacity):
    """One call to the lesson's mock, stripped."""
    return ref.mock_retrieval_model(haystack, question, capacity).strip()


def accuracy(ref, hays, question, capacity):
    """Per-length correctness of one model on one question."""
    want = ANSWERS[question]
    return [1 if ask(ref, hays[n], question, capacity) == want else 0 for n in LENGTHS]


def effective(scores, threshold=THRESHOLD):
    """Longest swept length whose score clears the threshold, or 0 when none does."""
    return max([n for n, s in zip(LENGTHS, scores) if s >= threshold], default=0)


def effective_lengths(rows):
    """{model: longest length clearing the threshold}."""
    return {m: effective(s) for m, s in rows.items()}


def total(rows):
    """Correct cells and total cells across every model and length."""
    return sum(sum(s) for s in rows.values()), sum(len(s) for s in rows.values())


def constant_score(question, guess="42"):
    """A model that answers `guess` without reading anything, over every cell."""
    return (1 if ANSWERS[question] == guess else 0) * len(LENGTHS) * len(MODELS)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    hays = {n: place(ref, n) for n in LENGTHS}
    one = {m: accuracy(ref, hays, Q1, c) for m, c in MODELS.items()}
    three = {m: accuracy(ref, hays, Q3, c) for m, c in MODELS.items()}
    spoken = {q: ask(ref, hays[SPECIFIED], q, MODELS["mock-mid"]) for q in ANSWERS}
    reordered = place(ref, 8000, tuple(reversed(HOPS)))
    return {
        "one": one, "three": three, "spoken": spoken,
        "one_total": total(one), "three_total": total(three),
        "retrieval": effective_lengths(one), "reasoning": effective_lengths(three),
        "at_specified": [row[LENGTHS.index(SPECIFIED)] for row in one.values()],
        "reordered": ask(ref, reordered, Q3, MODELS["mock-mid"]),
        "const_one": constant_score(Q1), "const_three": constant_score(Q3),
        "absent": absent(PROBED), "n_probed": len(PROBED),
    }


def verify(result):
    three_hits, cells = result["three_total"]
    one_hits, _ = result["one_total"]
    spoken = result["spoken"]
    return [
        practice.Check(
            "ANSWER: no frontier client is installed, and the reasoning length is 0 for all 3",
            len(result["absent"]) == result["n_probed"] and three_hits == 0,
            f"{list(result['absent'])} are all absent, so the three models are the lesson's mock "
            f"at capacities {list(MODELS.values())}. Three-hop accuracy is {three_hits}/{cells} "
            f"cells, so effective reasoning length at the {THRESHOLD} threshold is "
            f"{list(result['reasoning'].values())} — the number asked for does not exist",
        ),
        practice.Check(
            "MECHANISM: the mock ignores the question and returns the first plant in capacity",
            spoken[Q1] == spoken[Q2] == spoken[Q3],
            f"at the specified {SPECIFIED} words of filler the answers to '{Q1}', '{Q2}' and "
            f"'{Q3}' are '{spoken[Q1]}', '{spoken[Q2]}' and '{spoken[Q3]}'. The third hop is "
            "answered with the first hop's right-hand side",
        ),
        practice.Check(
            "MECHANISM: the regex returns a variable name, so no hop is representable",
            result["reordered"] not in ANSWERS.values(),
            f"reordering so '{HOPS[2]}' is the shallowest plant returns "
            f"'{result['reordered']}' — the pattern captures the right-hand side token, not a "
            "value, and mock_retrieval_model contains no arithmetic at any capacity",
        ),
        practice.Check(
            "FINDING: the retrieval axis ranks the models, the reasoning axis cannot",
            len(set(result["retrieval"].values())) == len(MODELS)
            and len(set(result["reasoning"].values())) == 1,
            f"effective retrieval lengths {result['retrieval']} against effective reasoning "
            f"lengths {result['reasoning']}. The lesson asks for both numbers; one separates "
            "three models and the other is identical for every model on the grid",
        ),
        practice.Check(
            "FINDING: the single specified length is one point, not a curve",
            len(set(result["at_specified"])) > 1
            and min(result["retrieval"].values()) < SPECIFIED,
            f"at {SPECIFIED} the three models score {result['at_specified']} on the retrieval "
            f"question. The boundaries at {min(result['retrieval'].values())} and "
            f"{sorted(result['retrieval'].values())[1]} only appear once the sweep is widened "
            f"to {list(LENGTHS)}",
        ),
        practice.Check(
            "CONTROL: a model that reads nothing beats all three on retrieval and ties on hops",
            result["const_one"] > one_hits and result["const_three"] == three_hits,
            f"answering the constant '42' to every question scores {result['const_one']}/{cells} "
            f"on the one-hop task against the mocks' {one_hits}/{cells} — they return 'no "
            f"answer' whenever the plant is out of capacity — and ties them at "
            f"{result['const_three']}/{cells} on the three-hop task",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
