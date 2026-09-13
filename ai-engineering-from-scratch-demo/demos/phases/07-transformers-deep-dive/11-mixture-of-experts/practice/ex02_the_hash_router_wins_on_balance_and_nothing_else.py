"""Exercise 2 — the hash router wins on balance and loses everything else.

    **Medium.** Replace the learned router with a hash-based router
    (deterministic, no learning). Compare quality and balance. Why is the learned
    router better?

Reading of the exercise: "quality" has no loss to measure it with here, so it is
measured as the property quality depends on -- whether an expert receives a
coherent slice of the input distribution. Both routers run over the same 1,000
clustered tokens, and balance is read from the lesson's own `entropy`.

**ANSWER: the hash router is better balanced, by a lot.** Entropy **2.0788**
against the learned router's **1.5492**, out of `ln 8 = 2.0794` -- 99.97% of the
ceiling -- and max/min expert usage **1.13** against **657**. It needs no bias
correction, no auxiliary loss and no warm-up, because hashing is uniform by
construction.

**FINDING: and it routes at chance under any perturbation.** Nudge each token by
Gaussian noise and ask whether it keeps its top-1 expert:

| noise | learned | hash | chance |
|---|---:|---:|---:|
| 0.001 | **99.8%** | 12.7% | 12.5% |
| 0.01 | 99.1% | 12.9% | 12.5% |
| 0.1 | 94.2% | 13.9% | 12.5% |

The hash router is at chance at every scale. It is deterministic in the *token*,
which is not the same as continuous in the token, and only the second is useful.

**FINDING: that is the whole answer to "why is the learned router better".**
Tokens with cosine similarity above 0.9 share the learned router's top-1 expert
**88.2%** of the time, against 46.2% for random pairs and 12.5% by chance. Under
the hash router the same measurement reads **12.4%** against 12.3% -- no locality
at all. An expert can only specialise if the tokens it sees have something in
common; hashing guarantees they do not, so every expert converges toward the
average of the whole distribution and the mixture buys nothing.

**CONTROL: the hash router has no gradient either.** `argmax` over learned scores
is at least differentiable in the gate weights the lesson's `route` returns;
`hash(x) % E` connects the routing decision to the loss through nothing.

Structure: `hash_route` is the replacement; `spread` measures balance; `keeps`
measures perturbation stability; `locality` measures whether neighbours share.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "11-mixture-of-experts"
WIDTH, EXPERTS, TOP_K, TOKENS, CLUSTERS = 16, 8, 2, 1_000, 3
NOISE, PAIRS, NEAR = (0.001, 0.01, 0.1), 20_000, 0.9


def hash_route(token, top_k=TOP_K, experts=EXPERTS):
    """A deterministic router: no weights, no learning, no bias."""
    digest = hash(tuple(round(v, 6) for v in token))
    return [(digest + i * 2654435761) % experts for i in range(top_k)], [1.0 / top_k] * top_k


def spread(ref, router, tokens):
    """(entropy, max/min usage) for one router over the corpus."""
    counts = [0] * EXPERTS
    for token in tokens:
        for expert in router(token)[0]:
            counts[expert] += 1
    return ref.entropy(counts), max(counts) / max(1, min(counts))


def keeps(router, tokens, noise, seed=7):
    """Share of tokens whose top-1 expert survives Gaussian noise of this scale."""
    rng = random.Random(seed)
    return sum(router([v + noise * rng.gauss(0, 1) for v in token])[0][0] == router(token)[0][0]
               for token in tokens) / len(tokens)


def cosine(a, b):
    """Cosine similarity, for deciding which token pairs count as neighbours."""
    return (sum(p * q for p, q in zip(a, b))
            / math.sqrt(sum(v * v for v in a) * sum(v * v for v in b)))


def locality(router, tokens, pairs):
    """(share of near pairs on one expert, share of all pairs) -- 1/EXPERTS is chance."""
    top = [router(token)[0][0] for token in tokens]
    near = [(i, j) for i, j in pairs if cosine(tokens[i], tokens[j]) > NEAR]
    return (sum(top[i] == top[j] for i, j in near) / len(near),
            sum(top[i] == top[j] for i, j in pairs) / len(pairs), len(near))


def corpus(rng):
    """1,000 tokens from 3 Gaussian clusters, and a router to score them with."""
    weights = [[rng.gauss(0, 0.3) for _ in range(WIDTH)] for _ in range(EXPERTS)]
    centres = [[rng.gauss(0, 1) for _ in range(WIDTH)] for _ in range(CLUSTERS)]
    return weights, [[c + 0.15 * rng.gauss(0, 1) for c in rng.choice(centres)]
                     for _ in range(TOKENS)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    weights, tokens = corpus(random.Random(42))
    learned = lambda token: ref.route(token, weights, TOP_K, [0.0] * EXPERTS)
    picker = random.Random(3)
    pairs = [(picker.randrange(TOKENS), picker.randrange(TOKENS)) for _ in range(PAIRS)]
    return {
        "balance": {"learned": spread(ref, learned, tokens),
                    "hash": spread(ref, hash_route, tokens)},
        "stability": {noise: (keeps(learned, tokens, noise), keeps(hash_route, tokens, noise))
                      for noise in NOISE},
        "locality": {"learned": locality(learned, tokens, pairs),
                     "hash": locality(hash_route, tokens, pairs)},
        "ceiling": math.log(EXPERTS), "chance": 1 / EXPERTS,
        "gates": (learned(tokens[0])[1], hash_route(tokens[0])[1]),
        "at_chance": all(abs(keeps(hash_route, tokens, n) - 1 / EXPERTS) < 0.02 for n in NOISE),
        "steady": min(keeps(learned, tokens, n) for n in NOISE),
    }


def verify(result):
    learned, hashed = result["balance"]["learned"], result["balance"]["hash"]
    stable, local = result["stability"], result["locality"]
    shown = str({n: (round(l, 3), round(h, 3)) for n, (l, h) in stable.items()})
    return [
        practice.Check(
            "ANSWER: the hash router is better balanced, by a lot",
            hashed[0] > learned[0] and hashed[1] < 1.5 and learned[1] > 100,
            f"entropy {hashed[0]:.4f} against the learned router's {learned[0]:.4f}, out of "
            f"ln {EXPERTS} = {result['ceiling']:.4f} -- {hashed[0] / result['ceiling']:.2%} of "
            f"the ceiling -- and max/min usage {hashed[1]:.2f} against {learned[1]:.0f}. No bias "
            "correction, no auxiliary loss, no warm-up: hashing is uniform by construction",
        ),
        practice.Check(
            "FINDING: and it routes at chance under any perturbation",
            result["at_chance"] and result["steady"] > 0.9,
            f"noise -> (learned, hash) share keeping their top-1 expert: {shown}, against "
            f"{result['chance']:.3f} by chance. Deterministic in the token is not the same as "
            "continuous in the token, and only the second is useful",
        ),
        practice.Check(
            "FINDING: the learned router puts neighbours on the same expert 88% of the time",
            local["learned"][0] > 0.8 and abs(local["hash"][0] - result["chance"]) < 0.02,
            f"tokens with cosine similarity above {NEAR} share the learned router's top-1 expert "
            f"{local['learned'][0]:.1%} of the time against {local['learned'][1]:.1%} for random "
            f"pairs; under the hash router the same measurement reads {local['hash'][0]:.1%} "
            f"against {local['hash'][1]:.1%} -- chance. {local['learned'][2]:,} near pairs",
        ),
        practice.Check(
            "FINDING: that is the whole answer to why the learned router is better",
            local["learned"][0] > 6 * result["chance"],
            "an expert can only specialise if the tokens it sees have something in common. "
            "Hashing guarantees they do not, so every expert converges toward the average of the "
            f"whole distribution and the mixture buys nothing -- {local['hash'][0]:.1%} shared "
            f"against {local['learned'][0]:.1%} is the difference between eight experts and one",
        ),
        practice.Check(
            "CONTROL: the hash router has no gradient either",
            result["gates"][1] == [1 / TOP_K] * TOP_K,
            f"the lesson's route returns gate weights from the scores, "
            f"{[round(g, 3) for g in result['gates'][0]]} here, so the mixture is differentiable "
            f"in them. hash(x) % E returns {result['gates'][1]} by definition -- uniform, "
            "constant, and connected to the loss through nothing at all",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
