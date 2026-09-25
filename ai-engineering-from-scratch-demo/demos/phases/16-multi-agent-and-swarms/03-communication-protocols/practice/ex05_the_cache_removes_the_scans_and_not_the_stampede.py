"""Exercise 5 — the cache removes the scans and not the stampede.

    **Rate-limited discovery.** Add a `RateLimitedRegistry` wrapper that caches
    Agent Card lookups with a configurable TTL and limits discovery queries
    per agent per second. Simulate a thundering herd of 100 agents discovering
    each other on startup and measure the difference.

Reading of the exercise: build it, run the herd, and then check what the
measurement is a measurement of -- because the cache fixes the registry's
work and leaves the thing a thundering herd actually breaks completely
untouched.

**ANSWER: the cache removes 99% of the lookup work and none of the
concentration.** A herd of **100** agents each calling `discoverBySkillTag`
once costs **10000** tag comparisons through the shipped registry, because
`[...this.cards.values()].filter(...)` rebuilds and rescans the whole map on
every call. A TTL cache turns that into **100** -- one miss and **99** hits,
a **100.0x** reduction. Every one of the 100 agents still receives the same
list, and still delegates to the same agent.

**FINDING: the stampede is downstream of the thing being cached.**
`discoverAndDelegate` selects `candidates[0]` -- **1** call site, no ranking,
Map insertion order. With **10** agents carrying the sought tag, all **100**
callers pick the same one: **1** distinct target. The rate limiter protects
the registry, which was never at risk, while the elected agent takes **100**
delegations either way. Caching a fan-out does not spread it.

**FINDING: the TTL buys the reduction with a blind window.** An agent that
registers one tick after the first lookup is invisible to every cached query
until the entry expires. Over a **30**-tick TTL the herd sees **10**
candidates while the registry holds **11**, and the newcomer receives **0**
queries -- so the startup case the exercise simulates is exactly the case a
startup cache gets wrong.

**FINDING: the two discovery methods disagree about what they scan.**
`discoverBySkillTag` reads only `skill.tags`. `discoverByInputMode` reads
`card.defaultInputModes` **or** `skill.inputModes`, so a card can be
discoverable by mode through a field its skills never declare. **2** methods,
**2** different notions of what a card offers, and the registry exposes both
without saying which one `discoverAndDelegate` uses.

Structure: `Registry` ports the shipped scan with a comparison counter;
`Cached` is the wrapper the exercise asks for.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "03-communication-protocols"
HERD, CARRIERS, TTL = 100, 10, 30
TAG = "search"


def cards(count, carriers):
    """`count` agent cards, of which `carriers` declare the sought tag."""
    return [{"name": f"agent-{index:03d}",
             "skills": [{"tags": [TAG if index < carriers else "other", "common"]}]}
            for index in range(count)]


class Registry:
    """AgentRegistry.discoverBySkillTag, ported, counting `tags.includes` calls."""

    def __init__(self, entries):
        self.cards = {card["name"]: card for card in entries}
        self.comparisons = 0

    def register(self, card):
        self.cards[card["name"]] = card

    def discover(self, tag):
        found = []
        for card in list(self.cards.values()):
            for skill in card["skills"]:
                self.comparisons += 1
                if tag in skill["tags"]:
                    found.append(card)
                    break
        return found


class Cached:
    """The wrapper: a TTL cache over lookups, plus a per-agent query budget."""

    def __init__(self, registry, ttl=TTL, per_tick=1):
        self.registry, self.ttl, self.per_tick = registry, ttl, per_tick
        self.entries, self.spent = {}, {}

    def discover(self, agent, tag, now=0):
        budget = self.spent.setdefault((agent, now), 0)
        if budget >= self.per_tick:
            return None
        self.spent[(agent, now)] = budget + 1
        cached = self.entries.get(tag)
        if cached and now - cached[0] < self.ttl:
            return cached[1]
        found = self.registry.discover(tag)
        self.entries[tag] = (now, found)
        return found


def herd(wrapper=None):
    """A hundred agents discovering on startup, with and without the wrapper."""
    registry = Registry(cards(HERD, CARRIERS))
    cache = Cached(registry) if wrapper else None
    results = [cache.discover(f"caller-{n}", TAG, now=1) if cache else registry.discover(TAG)
               for n in range(HERD)]
    return registry, cache, results


def staleness():
    """Register a newcomer after the first cached lookup and count its queries."""
    registry = Registry(cards(HERD, CARRIERS))
    cache = Cached(registry)
    cache.discover("caller-0", TAG, now=1)
    registry.register({"name": "agent-late", "skills": [{"tags": [TAG]}]})
    during = cache.discover("caller-1", TAG, now=TTL - 1)
    after = cache.discover("caller-2", TAG, now=TTL + 2)
    return len(during), len(after), sum(c["name"] == "agent-late" for c in during)


def solve():
    src = (parity.lesson_dir(PHASE, LESSON) / "code" / "main.ts").read_text("utf-8")
    start = src.index("class AgentRegistry")
    registry_src = src[start:src.index("type TaskState", start)]
    plain, _, _ = herd()
    cached_registry, _, results = herd(wrapper=True)
    during, after, late_seen = staleness()
    return {
        "herd": HERD, "carriers": CARRIERS, "ttl": TTL, "during": during,
        "plain": plain.comparisons, "cached": cached_registry.comparisons,
        "hits": len(results) - 1, "delegations": len(results),
        "targets": len({found[0]["name"] for found in results}),
        "after": after, "late_seen": late_seen,
        "candidates_zero": src.count("candidates[0]"),
        "tag_reads": registry_src.count("skill.tags.includes"),
        "mode_reads": (registry_src.count("defaultInputModes.includes")
                       + registry_src.count("skill.inputModes.includes")),
        "methods": len(re.findall(r"(?m)^  discover\w+\(", registry_src)),
    }


def verify(result):
    ratio = result["plain"] / result["cached"]
    return [
        practice.Check(
            "ANSWER: the cache removes 99% of the lookup work and none of the concentration",
            all([result["plain"] == 10000, result["cached"] == 100,
                 result["hits"] == 99, round(ratio, 1) == 100.0]),
            f"a herd of {result['herd']} agents costs {result['plain']} tag comparisons "
            f"through the shipped registry, which rescans the map per call; the TTL cache "
            f"makes it {result['cached']} -- one miss and {result['hits']} hits, "
            f"{ratio:.1f}x less -- and every caller still gets the same list",
        ),
        practice.Check(
            "FINDING: the stampede is downstream of the thing being cached",
            all([result["candidates_zero"] == 1, result["targets"] == 1,
                 result["delegations"] == result["herd"]]),
            f"discoverAndDelegate selects candidates[0] in {result['candidates_zero']} "
            f"place, by Map insertion order; with {result['carriers']} agents carrying the "
            f"tag all {result['delegations']} callers pick {result['targets']}, so the "
            "elected agent takes the whole herd either way",
        ),
        practice.Check(
            "FINDING: the TTL buys the reduction with a blind window",
            all([result["during"] == result["carriers"],
                 result["after"] == result["carriers"] + 1, result["late_seen"] == 0]),
            f"an agent registering after the first lookup is invisible until the entry "
            f"expires: over a {result['ttl']}-tick TTL the herd sees {result['during']} "
            f"candidates while the registry holds {result['after']}, the newcomer in "
            f"{result['late_seen']} cached results",
        ),
        practice.Check(
            "FINDING: the two discovery methods disagree about what they scan",
            all([result["methods"] == 2, result["tag_reads"] == 1,
                 result["mode_reads"] == 2]),
            f"{result['methods']} discovery methods: discoverBySkillTag reads "
            f"skill.tags in {result['tag_reads']} place, while discoverByInputMode reads "
            f"{result['mode_reads']} different fields, so a card can be discoverable by "
            "mode through a field its skills never declare",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
