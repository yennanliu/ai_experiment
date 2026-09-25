"""Exercise 5 — the fast path saves the judge, not the pheromones, and pays in staleness.

    Read AMRO-S (arXiv:2603.12933). Implement the decoupled "inference
    fast-path" with asynchronous pheromone update. How does this change system
    latency under sustained load?

Reading of the exercise: the paper's online update needs a quality signal
from "a lightweight LLM-Judge" and batches recorded requests in a FIFO buffer
of size B before updating. So the fast path is modelled as two costs -- one
inference (1.0 time unit) and one judge call (0.3) -- on a single server fed
by seeded Poisson arrivals, 20,000 requests per load level, with the judge
and deposit either in the request path (sync) or moved to a background
updater (async). Routing quality under the buffer is measured on the
reference router, with a port of `run_amro_s` that equals it exactly at B=1.

**ANSWER: it removes the judge from the service time, and the gain grows
without bound as load approaches sync capacity.** Mean latency sync / async:
2.46 / 1.49 at 0.5 requests per unit, 8.07 / 2.11 at 0.7, 34.4 / 2.41 at
0.75, and at 0.8 the sync server is past capacity (0.8 x 1.3 = 1.04) --
mean 530 and still rising -- while async sits at 2.89. p99 at 0.7 goes from
28.3 to 6.8.

**FINDING: in the lesson's simulation the fast path would change nothing.**
`simulate_task` hands back the quality score with the output, and `deposit`
is 3 multiplications and 1 addition. With the judge free, sync latency equals
async latency at every load, to the digit: what AMRO-S decouples is the
*evaluation*, and the reference has none to decouple.

**FINDING: the buffer costs routing quality.** Routing on pheromones up to B
requests stale, over 40 seeds of 200 tasks: B=1 scores 0.670, B=8 0.667,
B=32 0.643, B=64 0.614, B=100 0.578, and never flushing 0.502 -- the random
baseline's 0.501. B=64 gives up a third of what the router learns over
random; the latency win is paid for in staleness, a knob the paper leaves
unquantified.

Structure: `latency()` is Lindley's recursion for a single FIFO server;
`amro()` is `run_amro_s` with deposits held in a buffer of size B.
"""

from __future__ import annotations

import inspect
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "19-swarm-optimization-pso-aco"
INFER, JUDGE, N = 1.0, 0.3, 20000
LOADS, BATCHES, SEEDS = (0.5, 0.7, 0.75, 0.8), (1, 8, 32, 64, 100, 201), range(40)


def latency(rate, service, seed=0):
    """(mean, p99) response time: wait + service, arrivals Poisson at `rate`."""
    rng, wait, out = random.Random(seed), 0.0, []
    for _ in range(N):
        out.append(wait + service)
        wait = max(0.0, wait + service - rng.expovariate(rate))
    out.sort()
    return statistics.mean(out), out[int(0.99 * N)]


def amro(ref, seed, batch, n=200):
    """run_amro_s, same draw order, with deposits applied B at a time."""
    rng, types = random.Random(seed), ["code", "math", "writing", "planning"]
    agents = list(ref.AGENT_TASK_AFFINITY)
    router, rand_q, aco_q, buffer = ref.PheromoneRouter(types, agents), 0.0, 0.0, []
    for i in range(n):
        task = types[i % 4]
        rand_q += ref.simulate_task(rng.choice(agents), task, rng)
        agent = router.choose(task, rng)
        quality = ref.simulate_task(agent, task, rng)
        aco_q, buffer = aco_q + quality, buffer + [(task, agent, quality)]
        if len(buffer) >= batch:
            for entry in buffer:
                router.deposit(*entry)
            buffer = []
    return rand_q / n, aco_q / n, router


def deposit_ops(ref):
    source = inspect.getsource(ref.PheromoneRouter.deposit)
    return [source.count("*="), source.count("+=")]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    mine, theirs = amro(ref, 0, 1), ref.run_amro_s(200, 0)
    return {
        "parity": mine[:2] == theirs[:2] and mine[2].pheromones == theirs[2].pheromones,
        "sync": {r: latency(r, INFER + JUDGE) for r in LOADS},
        "async": {r: latency(r, INFER) for r in LOADS},
        "free_judge": {r: latency(r, INFER + 0.0) for r in LOADS},
        "quality": {b: statistics.mean(amro(ref, s, b)[1] for s in SEEDS) for b in BATCHES},
        "random": statistics.mean(amro(ref, s, 1)[0] for s in SEEDS),
        "deposit_ops": deposit_ops(ref),
        "agents": len(ref.AGENT_TASK_AFFINITY),
        "scored_inline": "return max(0.0, min(1.0, base" in inspect.getsource(ref.simulate_task),
    }


def verify(result):
    sync, fast, q = result["sync"], result["async"], result["quality"]
    lost = (q[1] - q[64]) / (q[1] - result["random"])
    return [
        practice.Check(
            "ANSWER: it removes the judge from the service time; the gain grows with load",
            all([sync[r][0] > fast[r][0] for r in LOADS]
                + [sync[0.8][0] > 100 * fast[0.8][0], sync[0.7][1] > 4 * fast[0.7][1]]),
            f"mean latency sync/async {[(round(sync[r][0], 2), round(fast[r][0], 2)) for r in LOADS]} "
            f"at loads {list(LOADS)}; p99 at 0.7 {sync[0.7][1]:.1f} -> {fast[0.7][1]:.1f}; "
            "at 0.8 the sync server is past capacity",
        ),
        practice.Check(
            "FINDING: in the lesson's simulation the fast path would change nothing",
            all([result["free_judge"] == fast, result["deposit_ops"] == [1, 1],
                 result["agents"] == 3, result["scored_inline"]]),
            "simulate_task returns the quality with the output and deposit is 3 "
            "multiplications and 1 addition; with the judge free, sync equals async "
            "latency exactly at every load",
        ),
        practice.Check(
            "FINDING: the buffer costs routing quality",
            all([result["parity"], q[1] > q[8] > q[32] > q[64] > q[100] > q[201],
                 abs(q[201] - result["random"]) < 0.01, 0.25 < lost < 0.4]),
            f"quality by buffer size {({b: round(v, 3) for b, v in q.items()})} against "
            f"random {result['random']:.3f}; B=64 gives up {lost:.0%} of the router's gain",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
