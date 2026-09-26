"""Exercise 3 — fallback to native makes an outage cheap, and at the lesson's rates an outage makes the cluster faster.

    The LMCache server is a single point of failure. Design the HA strategy
    (replicas, fallback to native).

Reading of the exercise: the strategy is designed and then run, on the
lesson's 200-request workload with 4 engines of 4 prefix slots and the server
down for requests 50-149. It is priced at a paying load cost (0.2 ms/block,
half of exercise 1's break-even) and at the shipped 3.0. Four designs:
`single`, where a lookup against a dead server errors the request; `fallback`,
where the connector treats the server as best-effort and a dead lookup is a
miss, so the engine re-prefills; `cold_standby`, which fails over to an empty
standby; and `warm_replica`, where every write goes to both copies. A
restarted primary comes back empty, since DRAM does not survive a restart.
The warm replica resyncs it.

**ANSWER: make the connector fall back to native prefill, then add a
write-through replica.** At 0.2 ms/block the unprotected server fails 37 of
200 requests. Fallback serves all 200 for +2475 ms (373208 against 370733)
and stays under native's 374883. A cold standby costs +225 ms and the warm
replica +0. Fallback is what removes the single point of failure; replicas
only buy back hit rate.

**FINDING: at the lesson's 3.0 ms/block every outage makes the cluster
faster.** Without an outage LMCache takes 428833 ms. With the server down
the fallback run takes 396658 ms and native 374883. The warm replica keeps
the slowest configuration running, at 428833. `single` looks faster still
(329685) only because it drops 37 requests.

**FINDING: in the shipped configuration the server is only needed for
warm-up.** With the lesson's unlimited engine caches the last LMCache hit is
request 89. An outage from request 90 on changes nothing. One from request 88
fails 5 requests without fallback, because a failed request never populates
the engine's cache and the same prefix fails again there.

Structure: `cluster()` is exercise 1's loop with a primary and a standby
shared cache; `view()` and `write()` are the strategy.
"""

from __future__ import annotations

import pathlib
import random

from harness import parity, practice

HERE = pathlib.Path(__file__).resolve().parent
EX01 = practice.load_module(next(HERE.glob("ex01_*.py")))
STRATEGIES = ("single", "fallback", "cold_standby", "warm_replica")
OUTAGE, SLOTS = (50, 150), 4


def view(strategy, down, primary, standby):
    """The shared cache a request can read, or None when a lookup errors."""
    if not down:
        return primary
    return {"single": None, "fallback": set(), "cold_standby": standby,
            "warm_replica": standby}[strategy]


def write(strategy, down, primary, standby, prefix):
    if not down:
        primary.add(prefix)
    if strategy == "warm_replica" or (down and strategy == "cold_standby"):
        standby.add(prefix)


def cluster(ref, reqs, strategy, load_ms, outage=OUTAGE, slots=SLOTS):
    """(total ms, failed requests, LMCache hits) for one outage window."""
    local, primary, standby = [{} for _ in range(4)], set(), set()
    rng, total, failed, hits = random.Random(11), 0.0, 0, 0
    for i, r in enumerate(reqs):
        eng, down = local[rng.randrange(4)], outage[0] <= i < outage[1]
        if i == outage[1]:
            primary = set(standby) if strategy == "warm_replica" else set()
        shared = view(strategy, down, primary, standby)
        if shared is None and r.prefix_id not in eng:
            failed += 1
            continue
        ms, _, hit = EX01.cost(ref, "LMCACHE", r, eng, shared or set(), load_ms)
        write(strategy, down, primary, standby, r.prefix_id)
        EX01.admit(eng, r.prefix_id, slots)
        total, hits = total + ms + r.output_tokens / ref.DECODE_TOK_PER_MS, hits + hit
    return round(total), failed, hits



def solve():
    ref = parity.load_reference(EX01.PHASE, EX01.LESSON, "main")
    reqs, never = ref.make_workload(), (len(ref.make_workload()),) * 2
    runs = {c: {s: cluster(ref, reqs, s, c) for s in STRATEGIES} for c in (0.2, 3.0)}
    return {
        "runs": runs, "calm": {c: cluster(ref, reqs, "single", c, never) for c in runs},
        "native": round(EX01.run(ref, reqs, "NATIVE_ONLY", SLOTS)["total_ms"]),
        "last_hit": EX01.run(ref, reqs, "LMCACHE")["hits"][-1],
        "late": cluster(ref, reqs, "single", 3.0, (90, 200), None),
        "shipped": cluster(ref, reqs, "single", 3.0, never, None),
        "at88": cluster(ref, reqs, "single", 3.0, (88, 200), None),
    }


def verify(result):
    paying, shipped, calm = result["runs"][0.2], result["runs"][3.0], result["calm"]
    return [
        practice.Check(
            "ANSWER: fall back to native prefill, then add a write-through replica",
            all([paying["single"][1] == 37, paying["fallback"][1] == 0,
                 paying["fallback"][0] - calm[0.2][0] == 2475,
                 paying["fallback"][0] < result["native"],
                 paying["cold_standby"][0] - calm[0.2][0] == 225,
                 paying["warm_replica"] == calm[0.2]]),
            f"outage 50-149 at 0.2 ms/block (total ms, failed, hits): {paying}; no outage "
            f"{calm[0.2]}, native {result['native']}",
        ),
        practice.Check(
            "FINDING: at the lesson's 3.0 ms/block every outage makes the cluster faster",
            all(shipped[s][0] < calm[3.0][0] for s in ("fallback", "cold_standby"))
            and shipped["warm_replica"] == calm[3.0] and shipped["single"][1] == 37,
            f"no outage {calm[3.0]}; with the outage {shipped}; native {result['native']}",
        ),
        practice.Check(
            "FINDING: in the shipped configuration the server is only needed for warm-up",
            result["last_hit"] == 89 and result["late"] == result["shipped"]
            and result["at88"][1] == 5,
            f"last LMCache hit at request {result['last_hit']}; outage from 90 "
            f"{result['late']} = no outage {result['shipped']}; from 88 {result['at88']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
