"""Exercise 2 — a request ID in the prefix takes hit rate from 99% to 0% and bills more than no cache.

    Your system prompt has a date. Move it out. Show before/after hit rate math.

Reading of the exercise: the simulator has no prompt text, only a
`prefix_hash`, so "the date in the prefix" is modelled by appending a stamp
to each request's hash, derived from its `arrived_at` (read as seconds). The
stamps are the ones The Problem names -- the time to the second and to the
minute, a request ID -- plus the day. "Moved out" is the unstamped hash. Each
variant runs the reference `simulate` (L2 5-min, parallel fixed), and hit rate
is `l2_reads / requests`.

**ANSWER: hit rate = 1 - distinct prefixes / requests; moving the stamp out
takes it back to 1 - 12/1304 = 99.1%.** On the lesson's 1304 requests over
519 s:

| stamp in prefix | distinct prefixes | hit rate | bill |
|---|---:|---:|---:|
| request ID | 1304 | 0.0% | $26.06 |
| time to the second | 1091 | 16.3% | $22.58 |
| time to the minute | 108 | 91.7% | $7.23 |
| day, or moved out | 12 | 99.1% | $5.85 |

The minute row follows 1 - 1/(r*g): each prefix sees r = 1304/12/519 = 0.209
requests a second, g = 60 s, so 12.6 requests share a stamp and the model
gives 92.0% against 91.7% measured.

**FINDING: with a request ID in the prefix, caching costs 20% more than no
caching.** Every request misses and pays the 1.25x write premium: $26.06
against $21.63 uncached. A dynamic prefix is not neutral -- it is worse than
switching caching off.

**FINDING: the same timestamp costs 7 points or 83 depending on traffic.** At
this simulator's rate a minute stamp still hits 91.7%; a stamp to the second
drops it to 16.3%. What matters is requests per stamp value, not whether a
stamp exists -- the day stamp costs nothing within one day.

**FINDING: ProjectDiscovery's post is dated 2026-04-10, not 2025-11, and ends
at 84%.** Fetched 2026-09-26: "Cache hit rate jumped from under 8% to 74%
overnight" after relocating dynamic context to the conversation tail, and
"The data: from 7% to 84%" after later optimisations. The lesson cites it as
"(project blog, 2025-11)".

Structure: `stamped()` rewrites `prefix_hash` on copies of the reference's
Requests; `run()` bills each variant.
"""

from __future__ import annotations

import dataclasses

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "14-prompt-semantic-caching"
STAMPS = {
    "request_id": lambda r, i: str(i),
    "second": lambda r, i: str(int(r.arrived_at)),
    "minute": lambda r, i: str(int(r.arrived_at // 60)),
    "day": lambda r, i: "day0",
    "moved_out": None,
}


def stamped(reqs, stamp):
    if stamp is None:
        return list(reqs)
    return [dataclasses.replace(r, prefix_hash=f"{r.prefix_hash}|{stamp(r, i)}")
            for i, r in enumerate(reqs)]


def config(ref, l2=True):
    return ref.Config(l1_enabled=False, l2_enabled=l2, parallel_penalty=False,
                      l1_threshold=0.95, l1_hit_prob=0.0, ttl="5min")


def run(ref, reqs):
    res = ref.simulate(reqs, config(ref))
    keys = len({r.prefix_hash for r in reqs})
    return {"keys": keys, "hit": res["l2_reads"] / len(reqs),
            "math": 1 - keys / len(reqs), "cost": round(res["cost"], 2)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    reqs = ref.make_workload()
    span = reqs[-1].arrived_at
    rate = len(reqs) / 12 / span
    return {
        "n": len(reqs), "span": span, "rate": rate, "model_minute": 1 - 1 / (rate * 60),
        "rows": {name: run(ref, stamped(reqs, f)) for name, f in STAMPS.items()},
        "uncached": round(ref.simulate(reqs, config(ref, l2=False))["cost"], 2),
        "doc_date": "(project blog, 2025-11)" in parity.doc_text(PHASE, LESSON),
    }


def verify(result):
    rows = result["rows"]
    hits = {k: round(v["hit"], 3) for k, v in rows.items()}
    return [
        practice.Check(
            "ANSWER: hit rate = 1 - distinct prefixes / requests; moving the stamp out gives 99.1%",
            all([hits == {"request_id": 0.0, "second": 0.163, "minute": 0.917,
                          "day": 0.991, "moved_out": 0.991},
                 all(abs(v["hit"] - v["math"]) < 1e-12 for v in rows.values()),
                 round(result["model_minute"], 3) == 0.920]),
            f"hit rate by stamp {hits}, each equal to 1 - keys/{result['n']} with keys "
            f"{ {k: v['keys'] for k, v in rows.items()} }; 1 - 1/(r*60) predicts "
            f"{result['model_minute']:.3f} for the minute stamp",
        ),
        practice.Check(
            "FINDING: with a request ID in the prefix, caching costs 20% more than no caching",
            rows["request_id"]["cost"] == 26.06 and result["uncached"] == 21.63,
            f"${rows['request_id']['cost']} with every request a 1.25x write, against "
            f"${result['uncached']} with caching off "
            f"({rows['request_id']['cost'] / result['uncached']:.2f}x)",
        ),
        practice.Check(
            "FINDING: the same timestamp costs 7 points or 83 depending on traffic",
            rows["minute"]["cost"] == 7.23 and rows["second"]["cost"] == 22.58
            and round(result["rate"], 3) == 0.209,
            f"at {result['rate']:.3f} requests/s per prefix the minute stamp bills "
            f"${rows['minute']['cost']} and the second stamp ${rows['second']['cost']}, "
            f"against ${rows['moved_out']['cost']} moved out",
        ),
        practice.Check(
            "FINDING: ProjectDiscovery's post is dated 2026-04-10, not 2025-11, and ends at 84%",
            result["doc_date"],
            "the lesson dates it '(project blog, 2025-11)'; the post, fetched 2026-09-26, is "
            "dated Apr 10, 2026: 'under 8% to 74%' after the relocation, 84% after later work",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
