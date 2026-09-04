"""Exercise 3 — the two most similar of five 50-dimensional vectors.

    Given 5 random word-like vectors (dimension 50), find the two most similar
    using cosine similarity

Reading of the exercise: "random" cannot mean random *here* — a test that
regenerates its own input asserts nothing repeatable. The five vectors are a
committed fixture (`fixtures/word_vectors.json`) built from a seeded draw, with
'bank' planted at cosine 0.90 from 'finance' so the expected answer is known
before the code runs, not read off its output.
"""

from __future__ import annotations

import itertools
import json
import pathlib

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "01-linear-algebra-intuition"
FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "word_vectors.json"


def load_fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "vectors")
    data = load_fixture()
    vectors = {name: ref.Vector(components) for name, components in data["vectors"].items()}
    scored = [
        (vectors[a].cosine_similarity(vectors[b]), a, b)
        for a, b in itertools.combinations(sorted(vectors), 2)
    ]
    scored.sort(reverse=True)
    return {"scored": scored, "meta": data["_meta"], "dim": data["dim"],
            "vectors": data["vectors"]}


def verify(result):
    scored = result["scored"]
    top_score, top_a, top_b = scored[0]
    runner_up = scored[1][0]
    expected = set(result["meta"]["expected_top_pair"])
    target = result["meta"]["expected_top_cosine"]
    checks = [
        practice.Check("all 10 pairs scored", len(scored) == 10, f"{len(scored)} pairs"),
        practice.Check("every vector is 50-dimensional",
                       all(len(v) == 50 for v in result["vectors"].values())
                       and result["dim"] == 50, "dim 50 x 5"),
        practice.Check(f"most similar pair is {sorted(expected)}",
                       {top_a, top_b} == expected, f"got ({top_a}, {top_b}) at {top_score:.4f}"),
        practice.Check(f"planted cosine is {target}",
                       abs(top_score - target) < 1e-9, f"measured {top_score:.9f}"),
        practice.Check("the answer is unambiguous (margin >= 0.15)",
                       top_score - runner_up >= 0.15,
                       f"margin {top_score - runner_up:.4f} over runner-up {runner_up:.4f}"),
    ]
    numpy = parity.try_numpy()
    if numpy is not None:
        matrix = numpy.array([result["vectors"][n] for n in sorted(result["vectors"])])
        norms = numpy.linalg.norm(matrix, axis=1, keepdims=True)
        grid = (matrix / norms) @ (matrix / norms).T
        names = sorted(result["vectors"])
        theirs = max(((grid[i][j], names[i], names[j])
                      for i in range(len(names)) for j in range(i + 1, len(names))))
        checks.append(practice.Check(
            "matches numpy's cosine matrix",
            {theirs[1], theirs[2]} == expected and abs(theirs[0] - top_score) < 1e-12,
            f"numpy -> ({theirs[1]}, {theirs[2]}) at {theirs[0]:.9f}"))
    return checks


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
