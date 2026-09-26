"""Exercise 1 — the flag moves the bill 17%, not 5x, and the whole penalty is paid in the first 106 requests.

    Run `code/main.py`. Toggle the parallelization flag. How much does the bill change?

Reading of the exercise: "the parallelization flag" is `Config.parallel_penalty`;
toggling it is the shipped pair of rows "parallel penalty active" / "parallel
fixed". The bill is read in total and on input alone (output is a flat 200
tokens a request that no cache touches), then set against the 5-10x the
lesson says the anti-pattern costs.

**ANSWER: $6.84 -> $5.85, a 17% (1.17x) change.** Toggling the flag removes
65 cache writes (77 -> 12) and turns them into reads. On input alone the bill
goes $2.93 -> $1.94, 1.51x; the $3.91 of output is the same in both rows.

**FINDING: every penalised write lands in the first 106 requests.** The
simulated cache never expires, and single (non-wave) requests do populate it,
so once each of the 12 prefixes has been seen by a single request no wave can
miss again. The penalty is a cold-start cost and dilutes with traffic:
1.17x at the shipped 500 iterations, 1.04x at 2000, 1.02x at 5000 -- always
the same 77 writes.

**FINDING: the waves do not share a prompt, and the header's "500 requests" is
1304.** `make_workload(500)` loops 500 times and emits 5 requests per wave:
1304 requests, 1005 of them in waves. Each wave member draws its own prefix,
so 0 of 201 waves share one (4.2 distinct prefixes per wave) -- the lesson's
"10 tool calls with the same 4K-token system prompt" is not what is simulated.

**FINDING: 5x needs 8 cold parallel calls on one prefix; 10x needs 46.**
For N simultaneous misses on one prefix against 1 write + N-1 reads, input
costs N*1.25 / (1.25 + 0.1(N-1)): 3.79x for the simulator's wave of 5, 5.13x
at 8, 5.81x for the lesson's 10, and 10x only at N = 46. The ceiling is 12.5x (1.25 / 0.1).

Structure: `bill()` runs the reference `simulate` with only `parallel_penalty`
varied; `penalised()` replays the cache to find where the extra writes fall.
"""

from __future__ import annotations

import collections

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "14-prompt-semantic-caching"
SIZES = (500, 2000, 5000)


def config(ref, penalty):
    return ref.Config(l1_enabled=False, l2_enabled=True, parallel_penalty=penalty,
                      l1_threshold=0.95, l1_hit_prob=0.0, ttl="5min")


def bill(ref, reqs):
    return {flag: ref.simulate(reqs, config(ref, flag)) for flag in (True, False)}


def penalised(reqs):
    """Indices of wave requests that miss a prefix no single request has cached yet."""
    cache, out = set(), []
    for i, r in enumerate(reqs):
        if r.prefix_hash in cache:
            continue
        if r.is_parallel_wave:
            out.append(i)
        else:
            cache.add(r.prefix_hash)
    return out


def inflation(n, write=1.25, read=0.1):
    return n * write / (write + read * (n - 1))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    reqs = ref.make_workload()
    waves = collections.defaultdict(list)
    for r in reqs:
        if r.is_parallel_wave:
            waves[r.arrived_at].append(r.prefix_hash)
    scaled = {}
    for n in SIZES:
        b = bill(ref, ref.make_workload(n))
        scaled[n] = (round(b[True]["cost"] / b[False]["cost"], 2), b[True]["l2_writes"])
    return {
        "bill": bill(ref, reqs), "n": len(reqs),
        "output": len(reqs) * 200 / 1e6 * ref.BASE_OUTPUT,
        "in_waves": sum(r.is_parallel_wave for r in reqs), "waves": len(waves),
        "shared": sum(len(set(v)) == 1 for v in waves.values()),
        "distinct": sum(len(set(v)) for v in waves.values()) / len(waves),
        "penalised": penalised(reqs), "scaled": scaled,
        "inflation": {n: round(inflation(n), 2) for n in (5, 8, 10, 46)},
    }


def verify(result):
    on, off = result["bill"][True], result["bill"][False]
    out, pen = result["output"], result["penalised"]
    return [
        practice.Check(
            "ANSWER: $6.84 -> $5.85, a 17% (1.17x) change",
            all([round(on["cost"], 2) == 6.84, round(off["cost"], 2) == 5.85,
                 (on["l2_writes"], off["l2_writes"]) == (77, 12)]),
            f"bill ${on['cost']:.2f} -> ${off['cost']:.2f} ({on['cost'] / off['cost']:.2f}x), "
            f"writes {on['l2_writes']} -> {off['l2_writes']}; input alone "
            f"${on['cost'] - out:.2f} -> ${off['cost'] - out:.2f}, output ${out:.2f} in both",
        ),
        practice.Check(
            "FINDING: every penalised write lands in the first 106 requests",
            len(pen) == 65 and max(pen) == 105
            and result["scaled"] == {500: (1.17, 77), 2000: (1.04, 77), 5000: (1.02, 77)},
            f"{len(pen)} penalised writes, the last at request index {max(pen)}; "
            f"(ratio, writes) by workload size {result['scaled']}",
        ),
        practice.Check(
            "FINDING: the waves do not share a prompt, and '500 requests' is 1304",
            all([result["n"] == 1304, result["in_waves"] == 1005, result["waves"] == 201,
                 result["shared"] == 0, round(result["distinct"], 1) == 4.2]),
            f"{result['n']} requests, {result['in_waves']} in {result['waves']} waves; "
            f"{result['shared']} waves share one prefix, {result['distinct']:.1f} distinct each",
        ),
        practice.Check(
            "FINDING: 5x needs 8 cold parallel calls on one prefix; 10x needs 46",
            result["inflation"] == {5: 3.79, 8: 5.13, 10: 5.81, 46: 10.0},
            f"N-way cold miss vs 1 write + N-1 reads, input multiple {result['inflation']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
