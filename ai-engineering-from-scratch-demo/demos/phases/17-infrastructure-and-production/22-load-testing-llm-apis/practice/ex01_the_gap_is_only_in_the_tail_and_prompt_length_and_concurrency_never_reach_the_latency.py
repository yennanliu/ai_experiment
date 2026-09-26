"""Exercise 1 — the gap is only in the tail, and prompt length and concurrency never reach the latency.

    Run `code/main.py`. Compare uniform vs realistic distribution — where is the
    gap?

Reading of the exercise: "where" is read literally -- which statistic, and
which input of the simulator, carries the difference -- so the shipped run is
compared column by column, and then each knob the module exposes (concurrency,
prompt length, the unused constants) is varied to see whether the gap moves.

**ANSWER: the gap is in the tail and the mean, not the median.** Both
workloads have TTFT P50 80 ms. P99 is 80 ms uniform against 800 ms
realistic, the mean 81.44 against 193.76 ms, cache hits 499 against 421 of
500. TTFT takes only two values -- the hit and miss constants -- so the whole
gap is the miss count, 1 against 79: the realistic generator draws from 80
prefixes and meets 79 of them in 500 requests.

**FINDING: concurrency is a dead knob.** The printout repeats the same two
rows at 10, 50 and 200, because a request's cache insert is visible to the
rest of its own batch; `unique_prefixes` is computed and never read. If
concurrent requests cannot see each other's inserts, the uniform workload
misses once per request in the first batch and the gap closes as concurrency
rises: mean 94.4 vs 199.52 ms at 10, 152 vs 235.52 at 50, 368 vs 389.6 at 200.

**FINDING: prompt length never reaches the latency.** `prompt_tokens` is
never read by `simulate`: the uniform prompts are 2000 tokens against a
realistic mean of 503.9, 4x the prefill, and still report faster; setting
every realistic prompt to 2000 tokens changes no number. The generator also
uses stddev 180 where the lesson's LLMPerf example uses 150.

**FINDING: nothing measures TPOT.** "Use It" says the code "measures
effective TPOT"; `TPOT_MS` and `BATCH_EFFICIENCY_SHARED_PREFIX` are defined
and never read, and the result dict has only TTFT fields.

Structure: `shipped()` runs the reference `simulate`; `unshared()` is the same
loop with the cache updated after each batch; `loads()` reads the module's
source with `ast` to list which names are ever read.
"""

from __future__ import annotations

import ast
import dataclasses
import statistics

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "22-load-testing-llm-apis"
CONCURRENCY = (10, 50, 200)


def shipped(ref, reqs):
    return {c: ref.simulate(reqs, c) for c in CONCURRENCY}


def unshared(ref, reqs, concurrency):
    """Mean TTFT when a batch cannot see its own cache inserts."""
    cache, ttft = set(), []
    for i in range(0, len(reqs), concurrency):
        batch = reqs[i : i + concurrency]
        ttft += [ref.PREFIX_CACHE_HIT_TTFT_MS if r.prefix_hash in cache
                 else ref.PREFIX_CACHE_MISS_TTFT_MS for r in batch]
        cache |= {r.prefix_hash for r in batch}
    return round(statistics.mean(ttft), 2)


def loads():
    """Every bare name and attribute the module ever reads."""
    tree = ast.parse((parity.lesson_dir(PHASE, LESSON) / "code" / "main.py").read_text())
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return names | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    uniform, real = ref.make_uniform_workload(500), ref.make_realistic_workload(500)
    long_real = [dataclasses.replace(r, prompt_tokens=2000) for r in real]
    used = loads()
    return {
        "uniform": shipped(ref, uniform), "real": shipped(ref, real),
        "long_real": ref.simulate(long_real, 10),
        "prefixes": len({r.prefix_hash for r in real}),
        "mean_tokens": round(statistics.mean(r.prompt_tokens for r in real), 1),
        "unshared": {c: (unshared(ref, uniform, c), unshared(ref, real, c)) for c in CONCURRENCY},
        "unused": sorted(n for n in ("unique_prefixes", "prompt_tokens", "TPOT_MS",
                                     "BATCH_EFFICIENCY_SHARED_PREFIX") if n not in used),
        "stddev": ("gauss(500, 180)" in ref.__loader__.get_source(ref.__name__),
                   "--stddev-input-tokens 150" in parity.doc_text(PHASE, LESSON)),
        "claims_tpot": "measures effective TPOT" in parity.doc_text(PHASE, LESSON),
    }


def verify(result):
    u, r = result["uniform"][10], result["real"][10]
    same = all(result["uniform"][c] == u and result["real"][c] == r for c in CONCURRENCY)
    gaps = result["unshared"]
    return [
        practice.Check(
            "ANSWER: the gap is in the tail and the mean, not the median",
            all([u["p50"] == r["p50"] == 80, (u["p99"], r["p99"]) == (80, 800),
                 (u["mean"], r["mean"]) == (81.44, 193.76),
                 (u["cache_hits"], r["cache_hits"]) == (499, 421), result["prefixes"] == 79]),
            f"P50 {u['p50']}/{r['p50']} ms, P99 {u['p99']}/{r['p99']} ms, mean "
            f"{u['mean']}/{r['mean']} ms, hits {u['cache_hits']}/{r['cache_hits']}; the "
            f"realistic run meets {result['prefixes']} of its 80 prefixes",
        ),
        practice.Check(
            "FINDING: concurrency is a dead knob",
            same and "unique_prefixes" in result["unused"]
            and gaps == {10: (94.4, 199.52), 50: (152, 235.52), 200: (368, 389.6)},
            f"identical rows at concurrency {CONCURRENCY}; with inserts visible only "
            f"after the batch, mean uniform/realistic TTFT is {gaps}",
        ),
        practice.Check(
            "FINDING: prompt length never reaches the latency",
            all(["prompt_tokens" in result["unused"], result["long_real"] == r,
                 result["mean_tokens"] == 503.9, result["stddev"] == (True, True)]),
            f"prompt_tokens is never read; uniform 2000 tokens vs realistic mean "
            f"{result['mean_tokens']}, and 2000-token realistic prompts give the same "
            "row; the generator's stddev is 180 against the lesson's 150",
        ),
        practice.Check(
            "FINDING: nothing measures TPOT",
            result["claims_tpot"] and {"TPOT_MS", "BATCH_EFFICIENCY_SHARED_PREFIX"}
            <= set(result["unused"]),
            f"the lesson says the code 'measures effective TPOT'; never read: {result['unused']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
