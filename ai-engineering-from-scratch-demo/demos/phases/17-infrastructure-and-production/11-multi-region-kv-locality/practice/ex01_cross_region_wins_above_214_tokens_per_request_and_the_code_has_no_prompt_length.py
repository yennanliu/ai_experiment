"""Exercise 1 — cross-region wins above 214 tokens per request, and the code has no prompt length.

    Run `code/main.py`. At what prompt length does cross-region routing beat
    local-only routing, given 75 ms RTT?

Reading of the exercise: the shipped simulator has no prompt length, so one
is added the only way the lesson's numbers allow. TTFT is linear in prompt
length and calibrated at the lesson's 2K-token point: a miss is 800 ms x
L/2048. Two readings of the hit are run. The *scaled* hit keeps the lesson's
10x gap, 80 ms x L/2048. The *fixed* hit is a flat 80 ms. "Cross-region" is
the reference GLOBAL strategy and "local-only" is REGIONAL, with every RTT set
to 75 ms. Both are run on the reference's own `simulate()` over its
1000-request workload. The shipped cache evicts with `set.pop()`, whose order
follows string hashes and changes between processes (see exercise 2). Here it
is replaced by a FIFO cache of the same 12 slots, so the numbers repeat.

**ANSWER: per request, above 214 tokens (scaled hit) or 397 tokens (fixed
hit); on the fleet, above 234 or 416.** A remote hit beats a local miss when
miss(L) > hit(L) + 75 ms, which gives L > 213.3 scaled and L > 396.8 fixed.
The fleet crossover sits higher. GLOBAL's rule is exactly that per-request
test, so at 214 tokens it sends 588 requests across regions and loses: mean
TTFT 63.1 ms against REGIONAL's 59.4 ms. The reason is that a remote hit
never warms the local cache, so the next local request misses again. GLOBAL
first wins at 234 tokens (scaled) and 416 tokens (fixed), and it keeps
winning above that.

**FINDING: the shipped code cannot ask the question.** `Request` has no
length field and TTFT is the constants 80 and 800 ms. Every remote hit costs
at most 80 + 130 = 210 ms against an 800 ms miss, so at the shipped constants
cross-region "wins" for every prompt and every region pair.

**FINDING: the lesson's own numbers contradict its APAC example.** It says a
hot prefix 220 ms away is "dwarfed by 440 ms round-trip" against the saved
800 -> 80 ms. But 720 ms saved beats 440 ms. The 440 also double-counts a
number that is already a round trip. On the scaled model the break-even is
626 tokens at 220 ms and 1252 at 440 ms, both under the 2K prompt the example
uses. The skill file's rule, "long prompts (>8K tokens) benefit", puts the
threshold 38x above 214. And at fleet level the question mostly disappears
once the local tie-breaker works (exercise 2): cross-region then buys 0.09 ms
at 2K tokens.

Structure: `fleet()` runs the reference `simulate()` with FIFO caches and
reports the replicas afterwards. `patched()` swaps module constants and
restores them. `solve()` scans prompt lengths upward from the per-request
break-even for the first one where GLOBAL wins.
"""

from __future__ import annotations

import contextlib
import math

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "11-multi-region-kv-locality"
RTT, CALIBRATION = 75, 2048


class Cache(dict):
    """FIFO prefix cache; reports len <= 12 so the reference's own pop never fires."""

    def __init__(self, cap=12):
        super().__init__()
        self.cap, self.evictions = cap, 0

    def __len__(self):
        return min(super().__len__(), 12)

    def add(self, key):
        self[key] = None
        if super().__len__() > self.cap:
            self.pop(next(iter(self)))
            self.evictions += 1


@contextlib.contextmanager
def patched(ref, **values):
    saved = {name: getattr(ref, name) for name in values}
    try:
        yield ref.__dict__.update(values)
    finally:
        ref.__dict__.update(saved)


def fleet(ref, strategy, reqs, cap=12, regions=None, per_region=None):
    """Reference simulate() on fresh requests; returns (stats, replicas, requests)."""
    replicas = [ref.Replica(r, i, prefix_cache=Cache(cap)) for r in regions or ref.REGIONS
                for i in range(per_region or ref.REPLICAS_PER_REGION)]
    fresh = [ref.Request(r.origin_region, r.prefix_hash) for r in reqs]
    with patched(ref, make_replicas=lambda: replicas):
        return ref.simulate(strategy, fresh), replicas, fresh


def at_length(ref, length, scaled=True):
    ratio, table = length / CALIBRATION, {pair: RTT for pair in ref.CROSSREGION_RTT}
    with patched(ref, CACHE_MISS_MS=800 * ratio, CACHE_HIT_MS=80 * ratio if scaled else 80,
                 CROSSREGION_RTT=table):
        glob, local = (fleet(ref, s, ref.make_workload())[0] for s in ("GLOBAL", "REGIONAL"))
    return glob["mean_ttft"], local["mean_ttft"], glob["crossregion"]


def wins(ref, length, scaled):
    glob, local, _ = at_length(ref, length, scaled)
    return glob < local


def breakeven(rtt, scaled=True):
    """Smallest integer L with miss(L) > hit(L) + rtt."""
    per_token = (720 if scaled else 800) / CALIBRATION
    return math.floor((rtt + (0 if scaled else 80)) / per_token) + 1


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    skill = (parity.lesson_dir(PHASE, LESSON) / "outputs" / "skill-multi-region-router.md")
    per = {s: breakeven(RTT, s) for s in (True, False)}
    cross = {s: next(L for L in range(per[s], 4 * per[s]) if wins(ref, L, s)) for s in per}
    return {
        "per": per, "fleet": cross, "at_per": at_length(ref, per[True]),
        "before": any(wins(ref, cross[s] - 1, s) for s in per),
        "above": all(wins(ref, L, s) for s in per for L in (512, 1024, 2048, 8192)),
        "fields": sorted(ref.Request.__dataclass_fields__),
        "worst_remote": ref.CACHE_HIT_MS + max(ref.CROSSREGION_RTT.values()),
        "apac": (breakeven(220), breakeven(440)),
        "doc": ("dwarfed by 440 ms round-trip" in parity.doc_text(PHASE, LESSON),
                "(>8K tokens) benefit" in skill.read_text(encoding="utf-8")),
    }


def verify(result):
    at_per, fleet_ = result["at_per"], result["fleet"]
    return [
        practice.Check(
            "ANSWER: above 214 tokens per request (397 fixed-hit); on the fleet above 234 (416)",
            all([result["per"] == {True: 214, False: 397}, fleet_ == {True: 234, False: 416},
                 at_per[2] == 588, at_per[0] > at_per[1], not result["before"],
                 result["above"]]),
            f"per-request break-even {result['per']}; at 214 GLOBAL sends {at_per[2]} across "
            f"at {at_per[0]:.1f} ms against REGIONAL {at_per[1]:.1f}; GLOBAL first wins on "
            f"the fleet at {fleet_} and still wins at 512, 1K, 2K and 8K",
        ),
        practice.Check(
            "FINDING: the shipped code cannot ask the question",
            "prompt_len" not in result["fields"] and result["worst_remote"] < 800,
            f"Request fields {result['fields']}; the worst remote hit is "
            f"{result['worst_remote']} ms against an 800 ms miss at any length",
        ),
        practice.Check(
            "FINDING: the lesson's own numbers contradict its APAC example",
            all(result["doc"]) and max(result["apac"]) < 2048,
            f"720 ms saved beats the doc's 440 ms; break-even is {result['apac'][0]} tokens "
            f"at 220 ms and {result['apac'][1]} at 440; the skill file says 8K",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
