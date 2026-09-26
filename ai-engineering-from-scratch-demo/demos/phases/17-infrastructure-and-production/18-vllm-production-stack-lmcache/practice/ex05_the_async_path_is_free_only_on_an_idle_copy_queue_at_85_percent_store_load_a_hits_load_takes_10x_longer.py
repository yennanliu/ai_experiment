"""Exercise 5 — the async path is free only on an idle copy queue: at 85% store load a hit's load takes 10x longer.

    Argue whether the vLLM 0.11.0 asynchronous path is "free" — where does the
    overhead hide?

Reading of the exercise: "free" means the offload costs nothing anywhere,
not just nothing on the request path, so the argument is made by following
the transfers. A copy queue moves 500 MB KVs (exercise 4's size) at the
83.4 GB/s DMA rate the vLLM offloading blog reports, 6.0 ms each. Loads for
cache hits arrive at 10% of its capacity and async stores at 0-85%, as
seeded Poisson streams over 20 simulated minutes. It is run two ways: with
stores and loads sharing one queue, and with each on its own queue, as
full-duplex PCIe with separate copy engines allows. The lesson's own
CPU_OFFLOAD config is measured beside it.

**ANSWER: no. Async takes the store off the request path and puts its cost
in the copy queue.** A hit still needs its KV on the GPU before it can
decode. Sharing the queue with stores, its load takes 6.35 ms mean when no
stores are running, 8.01 ms at 30% store load, 13.14 at 60% and 62.35 at 85%,
with p99 11.75 -> 280 ms. That is ordinary M/D/1 queueing at 95% total load
(the formula gives 63 ms). With a queue of its own the load stays at 6.35 ms
at every store load.

**FINDING: a separate queue moves the cost to HBM residency.** A block
being offloaded cannot be freed until its copy completes. With stores on
their own queue that is 6.0 ms uncontended, a mean 7.31 ms at 30% store load
and 23.29 ms (p99 94.4) at 85%; in the shared queue it is 62.17 ms (p99 275)
at 85%. For that long HBM holds
KV the scheduler has already given up, and in a preemption-heavy workload
that HBM is exactly what is short.

**FINDING: the lesson's own offload is pure overhead with the "async" as a
0.1 factor.** CPU_OFFLOAD stores the 6125 blocks of its 24 misses at
0.15 ms/block (10% of 1.5), 918.75 ms charged on the request path, and never
reads one back. The blog measures the real-world counterpart: its custom
copy kernel costs 6% throughput at a 0% hit rate, because it competes with
the model for GPU cores.

**FINDING: the lesson dates and credits the async path wrongly.** It says
"vLLM 0.11.0 (January 2026) adds an asynchronous offload path" and "vLLM
0.9.0 introduced a Connector API". GitHub dates v0.11.0 to 2025-10-02.
January 8, 2026 is the date of the blog post. The blog says 0.9.0 "extended
the connector API to support asynchronous loading and storing" and that the
offloading feature arrived in 0.11.0. Its speedups, up to 4x lower TTFT and
5x throughput, came with the 0.12.0 memory layout. The lesson's claim about
speculative-decoding interactions is not in the blog and was not verified.

Structure: `arrivals()` and `fifo()` are the queue; `queues()` runs it
shared and split; exercise 1's `run()` prices the lesson's CPU_OFFLOAD.
"""

from __future__ import annotations

import pathlib
import random

from harness import parity, practice

HERE = pathlib.Path(__file__).resolve().parent
EX01 = practice.load_module(next(HERE.glob("ex01_*.py")))
LINK_BPS, KV_BYTES = 83.4e9, 500e6        # blog's DMA figure; exercise 4's 4K-token KV
SERVICE_MS = KV_BYTES / LINK_BPS * 1000
LOAD_RHO, STORE_RHOS, HORIZON_MS = 0.1, (0.0, 0.3, 0.6, 0.85), 1_200_000


def arrivals(rng, rho, kind):
    """Poisson arrivals at utilisation rho, each one KV transfer long."""
    t, out = rng.expovariate(rho / SERVICE_MS) if rho else HORIZON_MS, []
    while t < HORIZON_MS:
        out.append((t, kind))
        t += rng.expovariate(rho / SERVICE_MS)
    return out


def fifo(jobs):
    """Sojourn times (wait + transfer) per kind through one FIFO copy queue."""
    free, out = 0.0, {"load": [], "store": []}
    for t, kind in sorted(jobs):
        start = max(t, free)
        free = start + SERVICE_MS
        out[kind].append(free - t)
    return out


def summary(times):
    times = sorted(times)
    return round(sum(times) / len(times), 2), round(times[int(0.99 * len(times))], 2)


def queues(store_rho, seed=0):
    """(mean, p99) ms: loads and stores through one shared queue, then each on its own."""
    rng = random.Random(seed)
    loads, stores = arrivals(rng, LOAD_RHO, "load"), arrivals(rng, store_rho, "store")
    shared, own = fifo(loads + stores), fifo(stores)
    none = (0.0, 0.0)
    return {"load_shared": summary(shared["load"]), "load_own": summary(fifo(loads)["load"]),
            "store_shared": summary(shared["store"]) if stores else none,
            "store_own": summary(own["store"]) if stores else none}


def solve():
    ref = parity.load_reference(EX01.PHASE, EX01.LESSON, "main")
    reqs = ref.make_workload()
    cpu = EX01.run(ref, reqs, "CPU_OFFLOAD")
    misses = [r for r, ms in zip(reqs, cpu["costs"]) if ms]
    doc = parity.doc_text(EX01.PHASE, EX01.LESSON)
    return {
        "service": round(SERVICE_MS, 2),
        "queues": {rho: queues(rho) for rho in STORE_RHOS},
        "misses": len(misses),
        "offload_ms": round(cpu["prefill_ms"] - sum(r.prompt_tokens for r in misses)
                            / ref.PREFILL_TOK_PER_MS, 2),
        "stored_blocks": sum(-(-r.prompt_tokens // ref.KV_BLOCK_TOKENS) for r in misses),
        "claims": ["0.11.0 (January 2026) adds an asynchronous offload path" in doc,
                   "vLLM 0.9.0 introduced a Connector API" in doc],
    }



def verify(result):
    q = result["queues"]
    load = [q[r]["load_shared"][0] for r in STORE_RHOS]
    return [
        practice.Check(
            "ANSWER: no -- async takes the store off the request path and puts it in the copy queue",
            all([result["service"] == 6.0, load == [6.35, 8.01, 13.14, 62.35],
                 all(q[r]["load_own"] == q[0.0]["load_own"] for r in STORE_RHOS),
                 q[0.85]["load_shared"][1] > 20 * q[0.0]["load_shared"][1]]),
            f"hit load (mean, p99) ms by store load, shared queue "
            f"{ {r: q[r]['load_shared'] for r in STORE_RHOS} }; own queue "
            f"{q[0.0]['load_own']} at every store load",
        ),
        practice.Check(
            "FINDING: a separate queue moves the cost to HBM residency",
            q[0.85]["store_own"][0] > 3 * result["service"]
            and q[0.85]["store_shared"][0] > 10 * result["service"],
            f"store sojourn (mean, p99) ms on its own queue "
            f"{ {r: q[r]['store_own'] for r in STORE_RHOS} }; shared "
            f"{q[0.85]['store_shared']} at 85%",
        ),
        practice.Check(
            "FINDING: the lesson's own offload is pure overhead with the async as a 0.1 factor",
            (result["misses"], result["stored_blocks"], result["offload_ms"]) == (24, 6125, 918.75),
            f"CPU_OFFLOAD stores {result['stored_blocks']} blocks from {result['misses']} misses, "
            f"{result['offload_ms']} ms on the request path, and reads none back",
        ),
        practice.Check(
            "FINDING: the lesson dates and credits the async path wrongly",
            all(result["claims"]),
            "the lesson dates 0.11.0 to January 2026 and credits it with async offload; GitHub "
            "dates v0.11.0 2025-10-02, and the vLLM blog credits async load/store to 0.9.0",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
