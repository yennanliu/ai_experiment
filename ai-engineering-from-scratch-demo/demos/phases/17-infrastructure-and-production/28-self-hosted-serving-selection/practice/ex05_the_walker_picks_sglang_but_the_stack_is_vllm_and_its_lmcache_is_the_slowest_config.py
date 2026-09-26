"""Exercise 5 — the walker picks SGLang, but the stack is vLLM's, and its LMCache is the slowest config.

    RAG product with P99 prefix length 8K and high reuse across tenants. Pick
    an engine and stack it with Phase 17 · 11 + 18.

Reading of the exercise: the pick is asked of this lesson's walker, and the
stack is asked of lessons 11 (cache-aware multi-region router) and 18 (vLLM
production-stack + LMCache) by running their own simulators. "P99 prefix 8K"
becomes lesson 18's workload with every prompt at 8000 tokens. For lesson 11,
the 800 ms miss is scaled to 3200 ms, on the assumption that prefill cost is
linear in length (its figures are for a 2K prompt). Both simulators evict with
`set.pop()`, whose order depends on the per-process hash seed. They therefore
run in fresh interpreters under PYTHONHASHSEED 0-4, which makes the run
repeatable.

**ANSWER: vLLM with prefix caching, lesson 11's router and lesson 18's
LMCache.** The walker, given the exercise's words, says SGLang on Hopper and
AMD because the string contains "prefix". Phrased "RAG with high reuse across
tenants" it says vLLM, and on Blackwell it says TRT-LLM either way. But the
stack the exercise names is vLLM's. Lesson 18 is the vLLM production-stack
and its Connector API, and lesson 11 calls SGLang's RadixAttention "the
intra-replica equivalent", with cross-replica routing "strictly upstream".
So the stack decides the engine: vLLM.

**FINDING: in lesson 18's own simulator LMCache is the slowest config.** On
its shipped workload it takes 380,833 ms against native's 375,033-376,133 and
CPU offload's 369,402. It avoids the most re-prefills, 194, and spends the most
prefill time doing it, 14,800 ms. On the all-8K workload it is 394,233 ms,
with prefill 28,200. A hit costs 500 blocks x 3 ms = 1500 ms, and
re-prefilling 8000 tokens costs 200 ms. LMCache would break even at 0.4
ms/block, 7.5x cheaper than coded. `hbm_capacity_blocks_per_engine` is set and
never read, so the "KV exceeds HBM" case that lesson 18 says LMCache is for
cannot occur. Decode is 366,033 ms of every run, 97.3-97.6% of native's
total, so no prefill cache can move the total by more than 2.7%.

**FINDING: cache-aware routing lifts the hit rate but not the P99.** GLOBAL
routing takes hits from 0.30-0.31 (REGIONAL) to 0.86-0.89 across hash seeds,
and moves 565-603 of 1000 requests across regions. Tenant data-residency rules
forbid that. Misses remain over 1% under every strategy, so P99 TTFT is the
miss itself: 800 ms, or 3200 ms at 8K.

Structure: `seeded()` runs the two hash-dependent simulators in subprocesses;
`lmcache()` runs lesson 18's deterministic configs in-process.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "28-self-hosted-serving-selection"
L11, L18 = "11-multi-region-kv-locality", "18-vllm-production-stack-lmcache"
SEEDS, MISS_8K, PROMPT = range(5), 3200, 8000
PHRASES = ("RAG with P99 prefix length 8K and high reuse across tenants",
           "RAG with high reuse across tenants")
SNIPPET = """import json, sys; sys.path.insert(0, %r)
from harness import parity
a = parity.load_reference(%r, %r, 'main'); b = parity.load_reference(%r, %r, 'main')
out = {}
for miss in (800, %d):
    a.CACHE_MISS_MS = miss
    for s in ('REGIONAL', 'GLOBAL'):
        r = a.simulate(s, [a.Request(q.origin_region, q.prefix_hash) for q in a.make_workload()])
        out[f'{s}{miss}'] = [round(r['hit_rate'], 3), r['p99_ttft'], r['crossregion']]
w = b.make_workload()
for tag, n in (('mixed', None), ('8k', %d)):
    reqs = [b.Request(n or q.prompt_tokens, q.output_tokens, q.prefix_id) for q in w]
    out['native_' + tag] = b.simulate('NATIVE_ONLY', reqs)['total_ms']
print(json.dumps(out))"""


def seeded():
    root = str(pathlib.Path(parity.__file__).resolve().parents[1])
    code = SNIPPET % (root, PHASE, L11, PHASE, L18, MISS_8K, PROMPT)
    runs = [subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={**os.environ, "PYTHONHASHSEED": str(s)}, check=True)
            for s in SEEDS]
    return [json.loads(r.stdout) for r in runs]


def lmcache(l18):
    base = l18.make_workload()
    run = lambda cfg, n=None: l18.simulate(  # noqa: E731
        cfg, [l18.Request(n or r.prompt_tokens, r.output_tokens, r.prefix_id) for r in base])
    decode = sum(r.output_tokens / l18.DECODE_TOK_PER_MS for r in base)
    return {"mixed": {c: run(c) for c in ("CPU_OFFLOAD", "LMCACHE")},
            "8k": {c: run(c, PROMPT)["total_ms"] for c in ("CPU_OFFLOAD", "LMCACHE")},
            "decode": decode, "hit_ms": -(-PROMPT // l18.KV_BLOCK_TOKENS) * l18.LMCACHE_TIME_MS_PER_BLOCK,
            "prefill_ms": PROMPT / l18.PREFILL_TOK_PER_MS,
            "break_even": l18.KV_BLOCK_TOKENS / l18.PREFILL_TOK_PER_MS,
            "hbm_reads": parity.lesson_dir(PHASE, L18).joinpath("code", "main.py")
            .read_text(encoding="utf-8").count("hbm_capacity_blocks_per_engine")}


def routing(runs):
    """Lesson 11 and lesson 18 native results, gathered across hash seeds."""
    col = lambda key, i: [r[key][i] for r in runs]  # noqa: E731
    return {"global": col("GLOBAL800", 0), "regional": col("REGIONAL800", 0),
            "cross": col("GLOBAL800", 2), "regional_cross": set(col("REGIONAL800", 2)),
            "p99": set(col("REGIONAL800", 1) + col("GLOBAL800", 1)),
            "p99_8k": set(col("REGIONAL3200", 1) + col("GLOBAL3200", 1)),
            "native": [round(r["native_mixed"]) for r in runs],
            "native_8k": max(r["native_8k"] for r in runs)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    picks = {(hw, p): ref.pick_engine(hw, "production", p)["engine"] for p in PHRASES
             for hw in ("NVIDIA Hopper", "AMD", "NVIDIA Blackwell")}
    doc11, doc18 = parity.doc_text(PHASE, L11), parity.doc_text(PHASE, L18)
    return {"picks": picks, "seeded": routing(seeded()),
            "lm": lmcache(parity.load_reference(PHASE, L18, "main")),
            "vllm_stack": "Connector API" in doc18 and "vLLM production-stack" in doc18,
            "radix_intra": "SGLang RadixAttention** (Phase 17 · 06) is the intra-replica" in doc11}


def verify(result):
    picks, lm, sd = result["picks"], result["lm"], result["seeded"]
    native, lmc, cpu = sd["native"], lm["mixed"]["LMCACHE"], lm["mixed"]["CPU_OFFLOAD"]
    return [
        practice.Check(
            "ANSWER: vLLM with prefix caching, lesson 11's router and lesson 18's LMCache",
            all([picks[("NVIDIA Hopper", PHRASES[0])] == picks[("AMD", PHRASES[0])] == "SGLang",
                 picks[("NVIDIA Hopper", PHRASES[1])] == "vLLM",
                 {picks[("NVIDIA Blackwell", p)] for p in PHRASES} == {"TRT-LLM"},
                 result["vllm_stack"], result["radix_intra"]]),
            f"walker picks {picks}; lesson 18 is vLLM's production-stack and Connector API, and "
            "lesson 11 puts SGLang's RadixAttention inside one replica",
        ),
        practice.Check(
            "FINDING: in lesson 18's own simulator LMCache is the slowest config",
            all([lmc["total_ms"] > max(native) > cpu["total_ms"], lmc["re_prefills_avoided"] == 194,
                 lm["8k"]["LMCACHE"] > sd["native_8k"], lm["hbm_reads"] == 1,
                 (lm["hit_ms"], lm["prefill_ms"], lm["break_even"]) == (1500, 200, 0.4),
                 round(lm["decode"] / min(native), 3) == 0.976]),
            f"totals LMCACHE {lmc['total_ms']:.0f} vs native {min(native)}-{max(native)} vs CPU "
            f"offload {cpu['total_ms']:.0f} ms; an 8K hit costs {lm['hit_ms']:.0f} ms vs "
            f"{lm['prefill_ms']:.0f} ms re-prefill; HBM capacity is written once and never "
            f"read; decode is {lm['decode']:.0f} ms",
        ),
        practice.Check(
            "FINDING: cache-aware routing lifts the hit rate but not the P99",
            all([min(sd["global"]) > 2.5 * max(sd["regional"]), sd["p99"] == {800},
                 sd["p99_8k"] == {MISS_8K}, sd["regional_cross"] == {0},
                 min(sd["cross"]) > 500, len(set(native)) > 1]),
            f"GLOBAL hit {sd['global']} vs REGIONAL {sd['regional']} over hash seeds "
            f"{list(SEEDS)}; cross-region {sd['cross']}; P99 is the miss, 800 / {MISS_8K} ms; "
            f"native totals differ by seed: {sorted(set(native))}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
