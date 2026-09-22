"""Exercise 3 — twenty tests accept a network that does not sort.

    Swap the evolutionary search evaluator to a real test suite. Evolve a sort
    function that passes 20 test cases; report generations to convergence.

Reading of the exercise: the shipped demo evolves two integers against a sum
of squared errors. Swapping in a test suite changes the fitness from a
continuous surface to a count of failures, and changes the *program* from two
numbers to a variable-length sorting network -- a list of compare-exchange
pairs. Both changes are made here, and the independent oracle for "did it
actually sort" is the 0-1 principle: a comparator network sorts every input
if and only if it sorts all 2^n zero-one vectors.

**ANSWER: convergence in 7 generations, on a 9-comparator network.** Over 20
seeded permutations of five distinct integers, the search reaches **0**
failures at generation **7** with a length penalty of 0.01 per comparator.

**FINDING: the winner is not a sorting network.** That 9-comparator result
passes **20/20** tests and **31** of the **32** zero-one vectors. The
evaluator accepted a program that is wrong on inputs it was never shown, and
reported it as converged -- which is the whole of AlphaEvolve's evaluator
argument in one number.

**FINDING: the length penalty bought the wrong answer faster.** Dropping it
converges in **5** generations on a **12**-comparator network that passes
**32/32**. The penalty that made the result prettier is what made it
incorrect: shorter networks fit 20 tests before they sort.

**FINDING: a test count is a much flatter landscape than a squared error.**
Of 500 single mutations from a fixed network, **124** leave the failure count
unchanged; of 500 mutations to the demo's two integers, **18** leave the
squared error unchanged. Swapping the evaluator for a test suite removes most
of the gradient the shipped demo was climbing.

Structure: `evolve()` is the lesson's loop with a genome and an evaluator
swapped in; `zero_one()` is the oracle the evaluator is scored against.
"""

from __future__ import annotations

import itertools
import random

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "11-planning-htn-and-evolutionary"
N, CASES, SEED = 5, 20, 7
PAIRS = tuple((i, j) for i in range(N) for j in range(i + 1, N))
_RNG = random.Random(SEED)
TESTS = tuple(tuple(_RNG.sample(range(1, 50), N)) for _ in range(CASES))
ZERO_ONE = tuple(itertools.product((0, 1), repeat=N))


def run_network(network, values):
    out = list(values)
    for i, j in network:
        if out[i] > out[j]:
            out[i], out[j] = out[j], out[i]
    return out


def failures(network, cases=TESTS):
    return sum(run_network(network, case) != sorted(case) for case in cases)


def zero_one(network):
    return sum(run_network(network, case) == sorted(case) for case in ZERO_ONE)


def mutate(network, rng):
    kind, genome = rng.choice(("add", "drop", "swap")), list(network)
    if kind == "add" or not genome:
        genome.insert(rng.randrange(len(genome) + 1), rng.choice(PAIRS))
    elif kind == "drop":
        genome.pop(rng.randrange(len(genome)))
    else:
        genome[rng.randrange(len(genome))] = rng.choice(PAIRS)
    return tuple(genome)


def seed_population(rng, size):
    return [tuple(rng.choice(PAIRS) for _ in range(rng.randint(3, 9)))
            for _ in range(size)]


def evolve(seed=0, penalty=0.01, generations=200, size=12, keep=4):
    rng = random.Random(seed)
    score = lambda net: failures(net) + penalty * len(net)  # noqa: E731
    ranked = sorted((score(net), net) for net in seed_population(rng, size))
    for generation in range(1, generations + 1):
        survivors = ranked[:keep]
        children = [(score(child), child) for _, net in survivors
                    for child in (mutate(net, rng) for _ in range(3))]
        ranked = sorted(survivors + children)[:size]
        if failures(ranked[0][1]) == 0:
            return generation, ranked[0][1]
    return None, ranked[0][1]


def flatness(mutator, seed=3, trials=500):
    rng = random.Random(seed)
    base = tuple(rng.choice(PAIRS) for _ in range(6))
    return sum(mutator(base, rng) for _ in range(trials))


def sort_flat(base, rng):
    return failures(mutate(base, rng)) == failures(base)


def linear_flat(_base, rng, a=0, b=0):
    def error(x, y):
        return sum((3 * k + 7 - (x * k + y)) ** 2 for k in range(-5, 6))
    return error(a + rng.choice((-2, -1, 0, 1, 2)),
                 b + rng.choice((-2, -1, 0, 1, 2))) == error(a, b)


def solve():
    parity.load_reference(PHASE, LESSON, "main")
    generation, network = evolve()
    loose_gen, loose_net = evolve(penalty=0.0)
    return {
        "tests": len(TESTS), "generations": generation,
        "comparators": len(network), "passed": len(TESTS) - failures(network),
        "zero_one": zero_one(network), "space": len(ZERO_ONE),
        "loose_generations": loose_gen, "loose_comparators": len(loose_net),
        "loose_zero_one": zero_one(loose_net),
        "loose_passed": len(TESTS) - failures(loose_net),
        "sort_flat": flatness(sort_flat), "linear_flat": flatness(linear_flat),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 7 generations to 20/20 on a 9-comparator network",
            all([result["generations"] == 7, result["comparators"] == 9,
                 result["passed"] == 20, result["tests"] == 20]),
            f"over {result['tests']} seeded permutations of five distinct integers the "
            f"search reaches 0 failures at generation {result['generations']}, on a "
            f"{result['comparators']}-comparator network passing "
            f"{result['passed']}/{result['tests']}",
        ),
        practice.Check(
            "FINDING: the winner is not a sorting network",
            all([result["zero_one"] == 31, result["space"] == 32,
                 result["passed"] == result["tests"]]),
            f"the converged network passes {result['passed']}/{result['tests']} tests and "
            f"{result['zero_one']} of the {result['space']} zero-one vectors. By the 0-1 "
            "principle that is not a sorting network -- the evaluator accepted a program "
            "wrong on inputs it was never shown and reported it as converged",
        ),
        practice.Check(
            "FINDING: the length penalty bought the wrong answer faster",
            all([result["loose_generations"] == 5, result["loose_comparators"] == 12,
                 result["loose_zero_one"] == 32, result["loose_passed"] == 20]),
            f"dropping the penalty converges in {result['loose_generations']} generations "
            f"on a {result['loose_comparators']}-comparator network that passes "
            f"{result['loose_zero_one']}/{result['space']} zero-one vectors. The term "
            "that made the result shorter is what made it incorrect",
        ),
        practice.Check(
            "FINDING: a test count is a much flatter landscape",
            all([result["sort_flat"] == 124, result["linear_flat"] == 18,
                 result["sort_flat"] > 6 * result["linear_flat"]]),
            f"of 500 single mutations from a fixed network, {result['sort_flat']} leave "
            f"the failure count unchanged; of 500 mutations to the demo's two integers, "
            f"{result['linear_flat']} leave the squared error unchanged. A test suite "
            "removes most of the gradient the shipped demo was climbing",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
