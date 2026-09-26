"""Exercise 3 — all three leaks draw the same 50 MB line, and the lesson's own workload hides the cache leak.

    Your soak test shows memory growing 50 MB/hour. Name three causes and the
    instrumentation to pick between them.

Reading of the exercise: the three causes are the three the lesson says a
soak catches -- memory leaks (here an unbounded prefix/response cache),
connection-pool drift and observability overflow. Each is modelled as a
counter driven by a request stream, calibrated so that alone it produces the
observed 50 MB/hour at 10 RPS on real traffic (unique prompts, no keep-alive,
a `request_id` metric label). "Pick between them" is then tested: which
experiment changes which slope, and which gauge reads each counter directly.

**ANSWER: an unbounded cache, connection-pool drift, metric-label
cardinality -- and one A/B soak per cause.** Instrument the process with
three gauges: cache entry count (plus a `tracemalloc` snapshot diff), open
connections / `process_open_fds`, and the metric registry's series count
(`prometheus_tsdb_head_series` on the server side). Then rerun the soak once
per cause, each run removing one driver. Replaying a fixed prompt set zeroes
the cache slope, keep-alive zeroes the pool slope, and dropping the
per-request label zeroes the series slope. The other two stay at 50 MB/h
each time.

**FINDING: the memory graph cannot tell them apart, and neither can load.**
All three give 50.0 MB/h at baseline, and doubling RPS doubles all three to
100.0 MB/h. The one knob a soak test turns moves every candidate together.

**FINDING: the lesson's realistic workload hides an unbounded cache.**
`make_realistic_workload` draws from 80 prefixes and has met all 80 by
request 833 -- 83 s into a 10 RPS soak -- so over hour two it adds 0 entries
and the cache slope is 0 MB/h; the uniform workload's single prefix is the
same. A soak must replay real, mostly-unique prompts to see this leak at all.

Structure: `streams()` builds one hour-two request stream per scenario from the
reference `Request` and generators; `slopes()` counts each cause's growth
events in that hour and scales them by its calibration.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "22-load-testing-llm-apis"
RPS, MB_PER_HOUR, POOL_LEAK = 10, 50.0, 100  # one pooled connection in 100 is never returned
PER_REQUEST_MB = MB_PER_HOUR / (RPS * 3600)  # calibration: each cause alone gives 50 MB/h
MB_PER_EVENT = {"cache": PER_REQUEST_MB, "pool": PER_REQUEST_MB * POOL_LEAK, "series": PER_REQUEST_MB}
INSTRUMENT = {
    "cache": "cache entry gauge + tracemalloc snapshot diff",
    "pool": "open connections / process_open_fds",
    "series": "registry series count / prometheus_tsdb_head_series",
}
BASE = {"rps": RPS, "prompts": "unique", "keepalive": False, "request_id_label": True}
SCENARIOS = {
    "baseline": {}, "2x rps": {"rps": 2 * RPS}, "replay realistic": {"prompts": "realistic"},
    "keep-alive": {"keepalive": True}, "no request_id label": {"request_id_label": False},
}


def hour_two(ref, prompts, n):
    """Requests of the soak's first two hours; hour one warms the cache."""
    if prompts == "realistic":
        return ref.make_realistic_workload(2 * n)
    return [ref.Request(500, f"prompt_{i}") for i in range(2 * n)]


def slopes(ref, knobs):
    n = knobs["rps"] * 3600
    reqs = hour_two(ref, knobs["prompts"], n)
    seen = {r.prefix_hash for r in reqs[:n]}
    new_prefixes = len({r.prefix_hash for r in reqs[n:]} - seen)
    opened = 0 if knobs["keepalive"] else n  # keep-alive holds the VUs' connections open
    new_series = n if knobs["request_id_label"] else 0
    counts = {"cache": new_prefixes, "pool": opened // POOL_LEAK, "series": new_series}
    return {k: round(counts[k] * MB_PER_EVENT[k], 2) for k in counts}


def first_full(reqs, size=80):
    seen = set()
    for i, r in enumerate(reqs, 1):
        seen.add(r.prefix_hash)
        if len(seen) == size:
            return i
    return None


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return {
        "slopes": {name: slopes(ref, {**BASE, **over}) for name, over in SCENARIOS.items()},
        "full_at": first_full(ref.make_realistic_workload(5000)),
        "uniform_prefixes": len({r.prefix_hash for r in ref.make_uniform_workload(36000)}),
        "lesson_names": all(s in parity.doc_text(PHASE, LESSON) for s in (
            "memory leaks", "connection-pool drift", "observability overflow")),
    }


def verify(result):
    s = result["slopes"]
    flat = {"cache": 50.0, "pool": 50.0, "series": 50.0}
    isolated = {"replay realistic": "cache", "keep-alive": "pool", "no request_id label": "series"}
    isolates = all(s[name] == {**flat, cause: 0.0} for name, cause in isolated.items())
    return [
        practice.Check(
            "ANSWER: unbounded cache, pool drift, label cardinality -- one A/B soak each",
            result["lesson_names"] and isolates and len(set(INSTRUMENT.values())) == 3,
            f"MB/h by cause per scenario {s}; gauges {INSTRUMENT}",
        ),
        practice.Check(
            "FINDING: the memory graph cannot tell them apart, and neither can load",
            s["baseline"] == flat and s["2x rps"] == {k: 100.0 for k in flat},
            f"baseline {s['baseline']}, doubled RPS {s['2x rps']}",
        ),
        practice.Check(
            "FINDING: the lesson's realistic workload hides an unbounded cache",
            result["full_at"] == 833 and s["replay realistic"]["cache"] == 0.0
            and result["uniform_prefixes"] == 1,
            f"all 80 prefixes met by request {result['full_at']} "
            f"({result['full_at'] / RPS:.0f} s at {RPS} RPS); hour-two cache slope "
            f"{s['replay realistic']['cache']} MB/h; uniform has "
            f"{result['uniform_prefixes']} prefix",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
