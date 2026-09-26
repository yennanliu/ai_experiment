"""Exercise 1 — LMCache pays at no HBM utilization, because a load costs 7.5x the prefill it replaces.

    Run `code/main.py`. At what HBM utilization does LMCache start paying?

Reading of the exercise: the module has no HBM-utilization input, so
utilization is made one: each engine holds `slots` prefixes (FIFO, the
shipped NATIVE_ONLY rule) and utilization is the 6-prefix working set over
that capacity, 100% at 6 slots to 600% at 1. The same capacity is applied to
all three configs, and the LMCache load cost is swept beside it. `run()`
reproduces `simulate()` exactly for CPU_OFFLOAD and LMCACHE before anything is
varied. NATIVE_ONLY's shipped eviction has no defined order (third finding),
so `run()` evicts the oldest prefix.

**ANSWER: at none -- in this model the break-even is a per-block cost, not a
utilization.** A re-prefill costs 16 / 40 = 0.4 ms per block and an LMCache
load 3.0 ms, 7.5x the work it replaces. So every LMCache hit adds time: the
shipped run is 0.99x, and LMCache loses at every utilization from 100% (0.9676x)
to 600% (0.7579x). Drop the load cost below 0.4 ms/block and it wins at every
utilization: 1.0026x to 1.0252x at 0.2 ms/block, 1.0052x to 1.0517x when
free. At 0.4 it is exactly 1.0 at all six. Utilization only sets how many hits there are, 18 to 164.

**FINDING: the simulator has no HBM.** `hbm_capacity_blocks_per_engine = 900`
is assigned and never read. Only NATIVE_ONLY evicts, at 4 prefixes;
CPU_OFFLOAD and LMCACHE never evict at all. CPU_OFFLOAD's printed 1.01-1.02x is
that unlimited capacity. It never reads an offloaded block back. Its only offload
code is a 0.15 ms/block charge on a miss.

**FINDING: the baseline depends on PYTHONHASHSEED.** NATIVE_ONLY evicts with
`set.pop()`, which removes whichever prefix the string hash puts first, and
may remove the one it just added. Over hash seeds 0..9 its total runs from
373983 to 376133 ms and its avoided re-prefills 111 to 128, so the printed
LMCache speedup is 0.98x or 0.99x run to run.

**FINDING: decode is 97.6% of the time, so no cache can move throughput much.**
Decode is 366033 of 374883 ms (FIFO baseline); a free cache tops out at
1.024x over it. The best cell in the sweep, 1.0517x, is measured against a
1-slot baseline.

Structure: `run()` is `simulate()` with two knobs, `slots` and `load_ms`;
`hash_seeds()` runs the shipped module under ten fixed hash seeds.
"""

from __future__ import annotations

import inspect
import os
import pathlib
import random
import subprocess
import sys

import harness
from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "18-vllm-production-stack-lmcache"
LOADS, SLOTS = (3.0, 0.4, 0.2, 0.0), (6, 5, 4, 3, 2, 1)
KEYS = ("total_ms", "prefill_ms", "re_prefills_avoided")
PROBE = (f"from harness import parity; r = parity.load_reference({PHASE!r}, {LESSON!r}, 'main'); "
         "o = r.simulate('NATIVE_ONLY', r.make_workload()); print(o['total_ms'], o['re_prefills_avoided'])")


def cost(ref, config, request, local, shared, load_ms):
    """(prefill ms, avoided re-prefill, LMCache hit) for one request."""
    blocks = -(-request.prompt_tokens // ref.KV_BLOCK_TOKENS)
    if request.prefix_id in local:
        return 0.0, True, False
    if config == "LMCACHE" and request.prefix_id in shared:
        return blocks * load_ms, True, True
    extra = blocks * ref.CPU_OFFLOAD_TIME_MS_PER_BLOCK * 0.1 if config == "CPU_OFFLOAD" else 0
    return request.prompt_tokens / ref.PREFILL_TOK_PER_MS + extra, False, False


def admit(eng, prefix, slots):
    """Cache a prefix on an engine, evicting the oldest past `slots` (None = unlimited)."""
    eng[prefix] = None
    if slots and len(eng) > slots:
        del eng[next(iter(eng))]


def run(ref, reqs, config, slots=None, load_ms=None, engines=4):
    """simulate() with an engine capacity in prefixes and a load cost."""
    load_ms = ref.LMCACHE_TIME_MS_PER_BLOCK if load_ms is None else load_ms
    local, shared, rng = [{} for _ in range(engines)], set(), random.Random(11)
    out = dict.fromkeys(KEYS, 0.0) | {"hits": [], "costs": []}
    for i, r in enumerate(reqs):
        eng = local[rng.randrange(engines)]
        ms, avoided, hit = cost(ref, config, r, eng, shared, load_ms)
        shared.update([r.prefix_id] if config == "LMCACHE" else [])
        admit(eng, r.prefix_id, slots)
        out["total_ms"] += ms + r.output_tokens / ref.DECODE_TOK_PER_MS
        out["prefill_ms"] += ms
        out["re_prefills_avoided"] += avoided
        out["hits"] += [i] if hit else []
        out["costs"].append(ms)
    return out


def hash_seeds(seeds=range(10)):
    root, runs = pathlib.Path(harness.__file__).resolve().parent.parent, set()
    for seed in seeds:
        env = {**os.environ, "PYTHONHASHSEED": str(seed), "PYTHONPATH": str(root)}
        out = subprocess.run([sys.executable, "-c", PROBE], env=env, cwd=root,
                             capture_output=True, text=True, check=True).stdout.split()
        runs.add((round(float(out[0])), int(out[1])))
    return sorted(runs)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    reqs = ref.make_workload()
    shipped = {c: ref.simulate(c, reqs) for c in ("CPU_OFFLOAD", "LMCACHE")}
    return {
        "parity": all(abs(v - run(ref, reqs, c)[k]) < 1e-6
                      for c in shipped for k, v in shipped[c].items() if k in KEYS),
        "native": run(ref, reqs, "NATIVE_ONLY", 4)["total_ms"],
        "lmcache": shipped["LMCACHE"]["total_ms"], "cpu": shipped["CPU_OFFLOAD"]["total_ms"],
        "cols": {c: [round(run(ref, reqs, "NATIVE_ONLY", k)["total_ms"]
                           / run(ref, reqs, "LMCACHE", k, c)["total_ms"], 4) for k in SLOTS]
                 for c in LOADS},
        "hits": {k: len(run(ref, reqs, "LMCACHE", k)["hits"]) for k in SLOTS},
        "ratio": ref.LMCACHE_TIME_MS_PER_BLOCK / (ref.KV_BLOCK_TOKENS / ref.PREFILL_TOK_PER_MS),
        "capacity_reads": inspect.getsource(ref.simulate).count("hbm_capacity_blocks_per_engine"),
        "decode": sum(r.output_tokens / ref.DECODE_TOK_PER_MS for r in reqs),
        "seeds": hash_seeds(),
    }


def verify(result):
    col, native, seeds, cpu = result["cols"], result["native"], result["seeds"], result["cpu"]
    speedups, best = sorted({round(t / result["lmcache"], 2) for t, _ in seeds}), max(map(max, col.values()))
    return [
        practice.Check(
            "ANSWER: at none -- in this model the break-even is a per-block cost, not a utilization",
            all([result["parity"], result["ratio"] == 7.5, max(col[3.0]) < 1,
                 set(col[0.4]) == {1.0}, min(col[0.2]) > 1, col[0.0] == sorted(col[0.0])]),
            f"a load is {result['ratio']}x the prefill it replaces; LMCache/native by load cost "
            f"at utilization 100%..600%: {col}; LMCache hits by slots {result['hits']}",
        ),
        practice.Check(
            "FINDING: the simulator has no HBM",
            result["capacity_reads"] == 1 and round(native / cpu, 2) == 1.01,
            f"hbm_capacity_blocks_per_engine is only assigned; CPU_OFFLOAD ({cpu:.0f} ms) never evicts",
        ),
        practice.Check(
            "FINDING: the baseline depends on PYTHONHASHSEED",
            len(seeds) > 1 and len(speedups) > 1,
            f"NATIVE_ONLY (ms, avoided) over hash seeds 0..9 {seeds}; LMCache speedup {speedups}",
        ),
        practice.Check(
            "FINDING: decode is 97.6% of the time, so no cache can move throughput much",
            round(result["decode"] / native, 3) == 0.976 and best < 1.06,
            f"decode {result['decode']:.0f} of {native:.0f} ms caps a free cache at "
            f"{native / result['decode']:.3f}x; the sweep's best is {best}x",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
