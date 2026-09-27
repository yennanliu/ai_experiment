"""Exercise 4 — a 10-expert PMixED answers 617 queries at epsilon 8, less accurately than DP-SGD answers any number.

    Design a deployment using PMixED (arXiv:2403.15638) that operates entirely at inference time. What is the threat model that PMixED addresses that DP-SGD does not?

Reading of the exercise: the design is built and measured on the lesson's
own trainer rather than drawn. The shipped seed-59 private set is split into
10 disjoint shards of 50. Each shard trains one expert, with no DP (sigma =
0). A public model is trained on 20 separate public records. Per query, each
expert's probability is pulled toward the public model's until the two are
within Renyi divergence beta (order alpha = 4, both directions). The
deployment returns one label sampled from the average -- never the
probability. The per-query privacy loss is computed exactly: changing one
record changes one expert anywhere inside its ball, and Renyi divergence is
quasi-convex, so the worst case sits at the ball's endpoints. Queries are
counted against a total epsilon = 8 at delta = 1e-5.

**ANSWER: at beta = 0.1 the deployment answers 617 queries at epsilon 8,
with 0.798 expected accuracy.** Loosening beta buys accuracy and burns the
budget: beta = 0.5 gives 0.832 for 25 queries, and beta = 1 gives 0.843 for
4. Without a public model (uniform prior) beta = 0.1 answers 958 queries at
0.604. A DP-SGD model at the same epsilon 8 (sigma = 6.17, Renyi accounting
as in exercise 1) has mean accuracy 0.858 over 20 runs on the shipped
data, and answers any number of queries. On this task DP-SGD dominates.
PMixED pays twice: it must sample instead of taking the argmax, and each
expert sees a tenth of the data. The public model alone costs no privacy: its sampled accuracy is
0.749 and its argmax accuracy 0.930.

**FINDING: the threat model is a query-only adversary, and the budget is
per deployment, not per user.** DP-SGD's guarantee covers the released
weights, and every later query is post-processing. PMixED's experts are not
private at all: if one leaks, the log-loss test finds a mislabeled canary in
its shard in 100% of 60 runs at a 5% false-alarm rate. What PMixED adds is
a deployment for data that cannot be DP-trained. A shard can be retrained
or dropped alone, and no training run pays the DP utility cost. In exchange,
every answer to every user spends from one shared budget, so collusion is
free.

**FINDING: the privacy comes from sampling, and the lesson says otherwise.**
The lesson describes PMixED as "aggregation adds noise for DP". The model
here adds no noise and still has a finite per-query epsilon, because the
output is one draw from a distribution pinned near public. Return the
probability instead and the guarantee is gone: swapping in the expert
trained with the canary moves the returned probability by up to 0.006 on
the test queries. The shift is small, but it is deterministic, so an
attacker who sees the probability can tell the two datasets apart.

Structure: `ball()` finds the beta-ball around p0 by bisection; `answer()`
mixes and averages; `query_eps()` is the worst Renyi loss over the ball's
vertices; `deploy()` scores one (beta, prior) setting.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "22-differential-privacy-for-llms"
ALPHA, BUDGET, DELTA, N, CANARY = 4.0, 8.0, 1e-5, 10, ((2.0, -2.0), 0)


def renyi(p, q, a=ALPHA):
    terms = [(u, v) for u, v in ((p, q), (1 - p, 1 - q)) if u > 0]
    if any(v <= 0 for _, v in terms):
        return math.inf                # the output the neighbour never produces
    return math.log(sum(u**a * v ** (1 - a) for u, v in terms)) / (a - 1)


def ball(p0, beta):
    def edge(target, lo=0.0, hi=1.0):
        for _ in range(40):
            mid, m = (lo + hi) / 2, p0 + (lo + hi) / 2 * (target - p0)
            lo, hi = (mid, hi) if max(renyi(m, p0), renyi(p0, m)) <= beta else (lo, mid)
        return p0 + lo * (target - p0)
    return edge(0.0), edge(1.0)


def prob(model, x):
    z = model[2] + model[0] * x[0] + model[1] * x[1]
    return min(max(1 / (1 + math.exp(-max(-30.0, min(30.0, z)))), 1e-6), 1 - 1e-6)


def answer(experts, p0, x, beta):
    lo, hi = ball(p0, beta)
    return sum(min(max(prob(e, x), lo), hi) for e in experts) / len(experts)


def query_eps(p0, beta):
    ends = ball(p0, beta)
    return max(renyi(((N - 1) * s + a) / N, ((N - 1) * s + b) / N)
               for s in ends for a in ends for b in ends)


def deploy(experts, prior, test, beta):
    acc = sum(abs(1 - y - answer(experts, prior(x), x, beta)) for x, y in test) / len(test)
    eps = max(query_eps(prior(x), beta) for x, _ in test)
    return round(acc, 3), int((BUDGET - math.log(1 / DELTA) / (ALPHA - 1)) / eps)


def seeded(ref, seed, fn, *args):
    saved, ref.random = ref.random, random.Random(seed)
    try:
        return fn(*args)
    finally:
        ref.random = saved


def fit(ref, data, sigma=0.0, seed=1):
    return seeded(ref, seed, ref.dp_sgd, list(data), 10, 0.05, sigma, 1.0)


def leak_test(ref, shard, runs=60):          # log-loss canary test on leaked weights
    loss = lambda m: math.log(1 / (1 - prob(m, CANARY[0])))  # noqa: E731
    member = [loss(fit(ref, [CANARY] + shard[1:], seed=s)) for s in range(runs)]
    out = sorted(loss(fit(ref, shard, seed=1000 + s)) for s in range(runs))
    return sum(v < out[int(0.05 * runs)] for v in member) / runs


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    private, test = seeded(ref, 59, lambda: (ref.gen(500), ref.gen(200)))
    experts = [fit(ref, private[i::N]) for i in range(N)]
    public = fit(ref, seeded(ref, 7, ref.gen, 20))
    by_public = lambda x: prob(public, x)  # noqa: E731
    sigma8 = 80 / (math.sqrt(160 * (math.log(1 / DELTA) + BUDGET)) - math.sqrt(160 * math.log(1 / DELTA)))
    dpsgd = [ref.accuracy(fit(ref, private, sigma8, s), test) for s in range(20)]
    swapped = [fit(ref, [CANARY] + private[0::N][1:])] + experts[1:]
    return {
        "table": {b: deploy(experts, by_public, test, b) for b in (0.1, 0.5, 1.0)},
        "uniform": deploy(experts, lambda x: 0.5, test, 0.1),
        "dpsgd": (round(sigma8, 2), round(sum(dpsgd) / len(dpsgd), 3)),
        "public": (round(sum(abs(1 - y - prob(public, x)) for x, y in test) / len(test), 3),
                   ref.accuracy(public, test)),
        "leak": leak_test(ref, private[0::N]),
        "shift": round(max(abs(answer(experts, by_public(x), x, 0.1)
                               - answer(swapped, by_public(x), x, 0.1)) for x, _ in test), 3),
        "doc_noise": "aggregation adds noise" in parity.doc_text(PHASE, LESSON, "en"),
    }


def verify(r):
    return [
        practice.Check(
            "ANSWER: beta 0.1 answers 617 queries at epsilon 8 with 0.798 accuracy, below DP-SGD's 0.858",
            r["table"] == {0.1: (0.798, 617), 0.5: (0.832, 25), 1.0: (0.843, 4)}
            and r["uniform"] == (0.604, 958) and r["dpsgd"] == (6.17, 0.858)
            and r["public"] == (0.749, 0.93)
            and all(acc < r["dpsgd"][1] for acc, _ in r["table"].values()),
            f"beta -> (accuracy, queries) {r['table']}; uniform prior {r['uniform']}; "
            f"DP-SGD (sigma, acc) {r['dpsgd']}; public (sampled, argmax) {r['public']}",
        ),
        practice.Check(
            "FINDING: the threat model is a query-only adversary with one shared budget",
            r["leak"] == 1.0,
            f"a leaked expert reveals the canary in its shard in {r['leak']:.0%} of runs",
        ),
        practice.Check(
            "FINDING: the privacy comes from sampling, and the lesson says otherwise",
            r["doc_noise"] and r["shift"] == 0.006,
            f"lesson says aggregation adds noise: {r['doc_noise']}; returning the probability "
            f"shifts it by up to {r['shift']} when one shard changes",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
