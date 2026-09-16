"""Exercise 5 — the multiplicative speedup is 4.68x, and the bookkeeping problem has no representation.

    Combine Hogwild! with speculative decoding in the toy: each worker uses a
    2-token spec-decode internally. Report the multiplicative speedup. What
    bookkeeping problem arises when two workers both want to extend the same
    shared-cache prefix?

Reading of the exercise: the two speedups are computed on their own terms and
then multiplied, because that is what "multiplicative" asserts and it is a claim
worth checking. Speculative decoding's factor uses the standard expected-tokens
form at the acceptance rate Lesson 15 measures; Hogwild!'s comes from the
lesson's own `run_hogwild`. The bookkeeping question is then answered by looking
at what `SharedCache` can express.

**ANSWER: 2.07x from Hogwild! and 2.26x from 2-token speculation, or 4.68x
together -- and only the first of the two has a ceiling.**

    N    Hogwild!   x spec (K=2)   combined
    2      1.99x        2.26x        4.51x
    4      2.07x        2.26x        4.68x
    8      2.07x        2.26x        4.68x

The speculative factor is `(1 + a + a^2) / (1 + 2c)` at `a = 0.8, c = 0.04`,
which does not depend on the worker count at all, so the product inherits
Hogwild!'s ceiling and nothing else.

**MECHANISM: the two speedups act on different quantities, and only one of them
is in the simulator.** Hogwild! multiplies the tokens produced per step;
speculation multiplies the tokens produced per *verifier call*. `run_hogwild`
has no notion of a call -- its loop is one step per worker per iteration, with no
model invocation anywhere -- so the second factor cannot be measured in this
module and has to be supplied from outside it.

**ANSWER to the bookkeeping question: there is one list and no prefix.**
`SharedCache` is a single field, `tokens: List[tuple[int, Category]]`, appended
to by every worker in turn. Two workers extending "the same prefix" would need
positions, branches and per-worker lengths; the cache has an insertion order and
a worker id. A rejected speculation would need to truncate *its own* tokens out
of a list that other workers have already appended to.

**FINDING: the simulator already has the race and does not model it as one.**
Every worker appends within one step and `decide_next_category` reads
`cache.counts()` *before* the appends -- so within a step, all N workers see the
same cache and none sees the others' current tokens. That is the Hogwild!
premise, written as a loop rather than as a race, which is why the coordination
weight can be set to 1.0 and produce perfect divergence.

Structure: `speculative_factor` is the standard expected-tokens-per-call ratio;
`combined` multiplies it by the lesson's own measured Hogwild! speedup.
"""

from __future__ import annotations

import dataclasses

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "22-async-hogwild-inference"
STEPS, TARGET, WEIGHT = 200, 100, 1.0
WORKERS = (2, 4, 8)
ACCEPT, DRAFT_COST, SPEC_K = 0.8, 0.04, 2


def speculative_factor(k=SPEC_K, accept=ACCEPT, cost=DRAFT_COST):
    """Expected tokens per verifier call over the cost of making it."""
    expected = sum(accept ** i for i in range(k + 1))
    return expected / (1 + k * cost)


def hogwild_speedup(ref, workers):
    base = ref.run_hogwild(1, STEPS, TARGET, WEIGHT)["unique_progress"]
    return ref.run_hogwild(workers, STEPS, TARGET, WEIGHT)["unique_progress"] / base


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    spec = speculative_factor()
    rows = {n: {"hogwild": hogwild_speedup(ref, n), "spec": spec,
                "combined": hogwild_speedup(ref, n) * spec} for n in WORKERS}
    cache = ref.SharedCache()
    return {
        "rows": rows,
        "spec": spec,
        "spec_by_k": {k: speculative_factor(k) for k in (1, 2, 4, 8)},
        "cache_fields": [field.name for field in dataclasses.fields(ref.SharedCache)],
        "token_shape": len(("worker_id", "category")),
        "cache_methods": [name for name in dir(cache)
                          if not name.startswith("_") and name != "tokens"],
        "ceiling_hit": rows[4]["hogwild"] == rows[8]["hogwild"],
    }


def column(rows, key, fmt):
    return ", ".join(f"N={n} {format(row[key], fmt)}" for n, row in rows.items())


def verify(result):
    rows = result["rows"]
    return [
        practice.Check(
            "ANSWER: 2.07x from Hogwild! and 2.26x from speculation, 4.68x together",
            abs(result["spec"] - 2.26) < 0.02 and rows[4]["combined"] > 4.5,
            "the Hogwild! factors are " + column(rows, "hogwild", ".2f")
            + f"x and the speculative factor is {result['spec']:.2f}x at K={SPEC_K}, "
            f"a={ACCEPT}, c={DRAFT_COST}, giving combined speedups of "
            + column(rows, "combined", ".2f")
            + "x. The speculative factor is (1 + a + a^2) / (1 + 2c) and does not depend on the "
            "worker count, so the product inherits Hogwild!'s ceiling and nothing else",
        ),
        practice.Check(
            "MECHANISM: the two act on different quantities and only one is in the simulator",
            result["ceiling_hit"] and len(result["cache_methods"]) == 1,
            f"Hogwild! multiplies the tokens produced per step; speculation multiplies the tokens "
            f"produced per verifier call. run_hogwild has no notion of a call -- its loop is one "
            f"step per worker per iteration with no model invocation anywhere, and SharedCache's "
            f"only method is {result['cache_methods']} -- so the second factor cannot be measured "
            "here and has to be supplied from outside the module",
        ),
        practice.Check(
            "ANSWER to the bookkeeping question: there is one list and no prefix",
            result["cache_fields"] == ["tokens"] and result["token_shape"] == 2,
            f"SharedCache is a single field, {result['cache_fields']}, holding "
            f"(worker_id, category) pairs appended by every worker in turn. Two workers extending "
            "the same prefix would need positions, branches and per-worker lengths; this cache "
            "has an insertion order and a worker id. A rejected speculation would have to "
            "truncate its own tokens out of a list that other workers have already appended to, "
            "and there is no index that says which those are",
        ),
        practice.Check(
            "FINDING: the simulator already has the race and does not model it as one",
            rows[2]["hogwild"] > 1.9,
            "every worker appends within one step, and decide_next_category reads cache.counts() "
            "before that step's appends -- so within a step all N workers see the same cache and "
            "none sees the others' current tokens. That is the Hogwild! premise written as a loop "
            f"rather than as a race, which is why the coordination weight can be set to {WEIGHT} "
            f"and produce the {rows[2]['hogwild']:.2f}x perfect divergence at N=2 that a real "
            "asynchronous system could not guarantee",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
