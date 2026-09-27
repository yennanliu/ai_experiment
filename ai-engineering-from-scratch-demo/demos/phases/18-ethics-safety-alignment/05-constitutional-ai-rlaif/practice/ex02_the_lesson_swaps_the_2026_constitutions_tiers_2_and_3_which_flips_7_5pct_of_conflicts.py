"""Exercise 2 — the lesson swaps the 2026 constitution's tiers 2 and 3, which flips 7.5% of conflicts.

    Read Anthropic's 2026 constitution (anthropic.com/news/claudes-constitution).
    List one principle that would rank Tier 1 and one that would rank Tier 4.
    Why does the priority structure matter for conflicts?

Reading of the exercise: the principles come from the published constitution
(anthropic.com/constitution, read 2026-09-27; the /news/ URL is the 2023
page). "Why does it matter" is answered by measuring it: every pair of
candidate responses that break 0-2 principles in each of four tiers (81
profiles, 3,240 pairs) is resolved by strict top-down priority in the lesson's
order, in the published order, and by flat counting with no priority.

**ANSWER: Tier 1 -- "broadly safe: not undermining appropriate human
mechanisms to oversee the dispositions and actions of AI"; Tier 4 --
"genuinely helpful: benefiting the operators and users it interacts with".**
In the lesson's own toy, "do not provide operational uplift for attacks" is
Tier-1 material and "help the user while protecting third parties" is Tier 4.
Priority matters because conflicts are the only place it acts: with flat
counting and no priority, in 1,458 pairs where exactly one response breaks a
Tier-1 principle, the breaker is picked 246 times (16.9%) and ties 216 more;
flat counting disagrees with strict priority on 32.4% of all pairs.

**FINDING: the lesson's tier order is not the published one.** The lesson
lists Tier 2 "follow Anthropic's guidelines", Tier 3 "be broadly ethical";
the 2026 document ranks broadly ethical second and guidelines third, and puts
catastrophic harms in separate hard constraints rather than a Tier 1. With
catastrophe held as the top tier, the two orders pick different winners on
243 of 3,240 pairs (7.5%): every pair tied at Tier 1 where guidelines and
ethics point opposite ways. The lesson also says "Conflicts are resolved
top-down"; the document calls its prioritisation "holistic rather than
strict".

**FINDING: the lesson's toy has no priority structure to resolve anything.**
`critique` never reads its `principle` argument: over 1,000 seeded base
responses it returns the identical flag list for all four PRINCIPLES.

Structure: `resolve()` returns the index of the preferred profile (None on a
tie) under an ordering or flat counting; the tiers are parsed from the
lesson, and the published order is written against them as (1, 3, 2, 4).
"""

from __future__ import annotations

import inspect
import itertools
import random
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "05-constitutional-ai-rlaif"
LESSON_ORDER, PUBLISHED_ORDER = (0, 1, 2, 3), (0, 2, 1, 3)   # lesson tiers 1,3,2,4 in 2026


def resolve(a, b, order=None):
    """Preferred violation profile: strict lexicographic by `order`, or flat sum."""
    ka, kb = ([v[i] for i in order] for v in (a, b)) if order else ([sum(a)], [sum(b)])
    return 0 if ka < kb else 1 if kb < ka else None


def conflicts():
    pairs = list(itertools.combinations(itertools.product(range(3), repeat=4), 2))
    split = [(a, b) for a, b in pairs if (a[0] == 0) != (b[0] == 0)]
    flat = [resolve(a, b) for a, b in split]
    return {
        "pairs": len(pairs),
        "order_flips": sum(resolve(a, b, LESSON_ORDER) != resolve(a, b, PUBLISHED_ORDER)
                           for a, b in pairs),
        "flip_kind": {(a[0] == b[0], (a[1] - b[1]) * (a[2] - b[2]) < 0) for a, b in pairs
                      if resolve(a, b, LESSON_ORDER) != resolve(a, b, PUBLISHED_ORDER)},
        "t1_split": len(split),
        "flat_picks_breaker": sum(w is not None and p[w][0] > 0 for w, p in zip(flat, split)),
        "flat_ties": flat.count(None),
        "flat_vs_strict": sum(resolve(a, b) != resolve(a, b, LESSON_ORDER) for a, b in pairs),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    doc = parity.doc_text(PHASE, LESSON)
    saved, ref.random = ref.random, random.Random(5)
    try:
        responses = [ref.base_model_sample() for _ in range(1000)]
    finally:
        ref.random = saved
    body = inspect.getsource(ref.critique).split('"""')[-1]
    return {
        "tiers": [t.strip() for t in re.findall(r"Tier \d: ([^.(\n]+)", doc)],
        "top_down": "Conflicts are resolved top-down" in doc,
        "principle_used": "principle" in body,
        "same_flags": all(len({tuple(ref.critique(r, p)) for p in ref.PRINCIPLES}) == 1
                          for r in responses),
        "flagged": sum(bool(ref.critique(r, "")) for r in responses),
        **conflicts(),
    }


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: priority acts only in conflicts, and without it a Tier-1 breaker wins 16.9%",
            (r["pairs"], r["t1_split"], r["flat_picks_breaker"], r["flat_ties"],
             r["flat_vs_strict"]) == (3240, 1458, 246, 216, 1050)
            and round(r["flat_picks_breaker"] / r["t1_split"], 3) == 0.169,
            f"of {r['t1_split']} pairs with one Tier-1 breaker, flat counting picks it "
            f"{r['flat_picks_breaker']} times and ties {r['flat_ties']}; flat vs strict differ "
            f"on {r['flat_vs_strict']}/{r['pairs']}",
        ),
        practice.Check(
            "FINDING: the lesson's tier order is not the published one",
            r["tiers"][1:3] == ["follow Anthropic's guidelines", "be broadly ethical"]
            and r["top_down"] and r["order_flips"] == 243 and r["flip_kind"] == {(True, True)},
            f"lesson tiers {r['tiers']}; swapping 2 and 3 flips {r['order_flips']}/{r['pairs']} "
            f"pairs, all tied at Tier 1 with tiers 2 and 3 opposed: {r['flip_kind']}",
        ),
        practice.Check(
            "FINDING: the lesson's toy has no priority structure to resolve anything",
            not r["principle_used"] and r["same_flags"] and r["flagged"] > 0,
            f"critique body reads `principle`: {r['principle_used']}; identical flags for all "
            f"principles on 1000 responses ({r['flagged']} flagged): {r['same_flags']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
