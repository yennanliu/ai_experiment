"""Exercise 2 -- the lesson's cache reports 99.7%, a 5-minute TTL gives 39.0%, and at 115 tokens a real API caches nothing.

    Measure prompt-cache hit rate over a week of production traffic.
    Identify which queries break the cache prefix. Restructure.

Reading of the exercise: "a week of production traffic" is simulated, seeded
and deterministic: 7 days x 300 queries spread over business hours (9:00 to
18:00), from three user groups (analyst/GDPR, counsel/HIPAA, counsel/SOC2,
weighted 5:3:2). Each group has a few intents, drawn Zipf-style, and each
intent has three paraphrases; 10% of queries are capitalised with a "?". Every
query runs through the lesson's `chat_turn` with its `PromptCache`. The same
cache keys are then replayed under the rule Anthropic documents (read
2026-09-29 at platform.claude.com/docs/en/docs/build-with-claude/prompt-caching):
a hit needs a 100% identical prefix, a cache entry lives 5 minutes and a hit
refreshes it, and a prefix shorter than 1,024 tokens (Sonnet 4.6 and Sonnet
5) is not cached at all. Tokens are estimated at 4 characters each.

**ANSWER: the lesson's cache reports 99.7% (2,093 of 2,100); with the
documented 5-minute TTL the same keys hit 39.0%.** The only queries that
break the prefix are paraphrases whose RRF order differs. Every query in a
group retrieves the same set of chunks, because each group can see at most
3 chunks and k = 3. But the order of those chunks follows the wording, and
the order is part of the hashed prefix. In analyst/GDPR, 4 orderings appear;
6 of the group's 9 phrasings break the group's most common order, which is
46.8% of that group's traffic. The fix is to sort the context by anchor
before hashing and to put the breakpoint after system + policy + context.
That raises the TTL hit rate from 39.0% to 62.9%, which is also the ceiling:
keying on the (role, jurisdiction) group alone gives the same 62.9%.

**FINDING: the lesson's `PromptCache` never expires.** It keeps every key
forever, and there are only 7 keys in the whole week, so it reports a 99.7%
hit rate that a 5-minute cache cannot deliver.

**FINDING: at the lesson's prompt size the real hit rate is 0%.** The
cacheable prefix is 461 characters, about 115 tokens. The whole corpus plus
the system prompt is about 155 tokens. Both are far below the 1,024-token
minimum, so the lesson's "60-80% hit rate" lever needs a stable prefix about
9x longer before it can apply at all.
"""

from __future__ import annotations

import collections
import hashlib
import random

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "08-production-rag-chatbot"
TTL, MIN_TOKENS, CHARS_PER_TOKEN = 300, 1024, 4
INTENTS = {
    ("analyst", "GDPR"): [
        ["how many days to delete EU user profiles after termination", "when must EU user profiles be deleted",
         "EU profile deletion deadline under GDPR Article 17"],
        ["deletion deadline for restricted data", "how fast is restricted category data deleted after a termination notice",
         "restricted data deletion within how many days"],
        ["how can users export their data", "where is the self-service data export portal",
         "can a user request a data export"]],
    ("counsel", "HIPAA"): [
        ["what happens to PHI after agreement termination", "how many days to return or destroy PHI", "PHI destruction deadline"],
        ["can users request data export through the portal", "self-service export of user data", "how do users export data"]],
    ("counsel", "SOC2"): [["how often are privileged users access reviewed", "access review cadence for standard users",
                           "is the access review quarterly or annual"]],
}
GROUPS = list(INTENTS)


def traffic(seed=19, days=7, per_day=300):
    rng, out = random.Random(seed), []
    for day in range(days):
        for t in sorted(rng.uniform(9 * 3600, 18 * 3600) for _ in range(per_day)):
            group = rng.choices(GROUPS, weights=[5, 3, 2])[0]
            intents = INTENTS[group]
            phrasings = rng.choices(intents, weights=[1 / (k + 1) for k in range(len(intents))])[0]
            q = rng.choice(phrasings)
            out.append((day * 86400 + t, group, q.capitalize() + "?" if rng.random() < 0.1 else q))
    return out


def ttl_hit_rate(keys, ttl=TTL):
    last, hits = {}, 0
    for ts, key in keys:
        hits += key in last and ts - last[key] <= ttl
        last[key] = ts
    return round(hits / len(keys), 4)


def digest(*parts):
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def breakers(rows):
    """Phrasings in analyst/GDPR whose context order differs from the group's modal order."""
    mine = [(q.lower().rstrip("?"), tuple(r["citations"])) for _, g, q, r in rows if g == GROUPS[0]]
    modal = collections.Counter(c for _, c in mine).most_common(1)[0][0]
    broken = [q for q, c in mine if c != modal]
    return {"orders": len({c for _, c in mine}), "phrasings": len({q for q, _ in mine}),
            "breaking": len(set(broken)), "breaking_share": round(len(broken) / len(mine), 3)}


def replay_keys(ref, rows):
    """The same week under three prefix layouts: the lesson's, canonical context order, group only."""
    policy = {g: f"role={g[0]} jurisdiction={g[1]}" for g in GROUPS}
    by_anchor = {c.anchor(): f"[{c.anchor()}] {c.text}" for c in ref.CORPUS}
    lesson = [(ts, r["cache_key"]) for ts, _, _, r in rows]
    canonical = [(ts, digest(ref.SYSTEM_PROMPT, policy[g], *sorted(by_anchor[a] for a in r["citations"])))
                 for ts, g, _, r in rows]
    group_only = [(ts, digest(ref.SYSTEM_PROMPT, policy[g])) for ts, g, _, _ in rows]
    prefix = "\n".join([ref.SYSTEM_PROMPT, policy[rows[0][1]], *(by_anchor[a] for a in rows[0][3]["citations"])])
    return {"lesson_ttl": ttl_hit_rate(lesson), "canonical_ttl": ttl_hit_rate(canonical),
            "group_ttl": ttl_hit_rate(group_only), "prefix_chars": len(prefix),
            "prefix_tokens": len(prefix) // CHARS_PER_TOKEN,
            "corpus_tokens": len("\n".join([ref.SYSTEM_PROMPT, *by_anchor.values()])) // CHARS_PER_TOKEN}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    cache, rows = ref.PromptCache(), []
    for ts, (role, j), q in traffic():
        rows.append((ts, (role, j), q, ref.chat_turn(q, role, j, ref.CORPUS, cache)))
    return {"queries": len(rows), "lesson_rate": round(cache.hit_rate(), 4), "hits": cache.hits,
            "keys": len(cache.store), **replay_keys(ref, rows), **breakers(rows)}


def verify(result):
    r = result
    real = 0.0 if r["prefix_tokens"] < MIN_TOKENS else r["canonical_ttl"]
    return [
        practice.Check(
            "ANSWER: 99.7% as the lesson counts it, 39.0% with a 5-minute TTL; paraphrase reordering breaks it",
            (r["queries"], r["lesson_rate"], r["hits"], r["lesson_ttl"]) == (2100, 0.9967, 2093, 0.39)
            and (r["orders"], r["phrasings"], r["breaking"], r["breaking_share"]) == (4, 9, 6, 0.468),
            f"{r['queries']} queries: lesson {r['lesson_rate']:.1%} ({r['hits']} hits), TTL {r['lesson_ttl']:.1%}; "
            f"analyst/GDPR: {r['orders']} context orders, {r['breaking']}/{r['phrasings']} phrasings break the "
            f"modal order ({r['breaking_share']:.1%} of its traffic)",
        ),
        practice.Check(
            "ANSWER: sorting the context by anchor lifts the TTL hit rate 39.0% -> 62.9%, the per-group ceiling",
            (r["canonical_ttl"], r["group_ttl"]) == (0.6286, 0.6286),
            f"canonical order {r['canonical_ttl']:.1%}; keyed on (role, jurisdiction) only {r['group_ttl']:.1%}",
        ),
        practice.Check(
            "FINDING: the lesson's PromptCache never expires, so 7 keys serve the whole week",
            r["keys"] == 7 and r["lesson_rate"] - r["lesson_ttl"] > 0.6,
            f"{r['keys']} keys stored; no-TTL {r['lesson_rate']:.1%} vs 5-minute TTL {r['lesson_ttl']:.1%}",
        ),
        practice.Check(
            "FINDING: a ~115-token prefix is under the 1,024-token minimum, so the real hit rate is 0%",
            (r["prefix_chars"], r["prefix_tokens"], r["corpus_tokens"], real) == (461, 115, 155, 0.0),
            f"prefix {r['prefix_chars']} chars ~ {r['prefix_tokens']} tokens; system + whole corpus "
            f"~ {r['corpus_tokens']} tokens; minimum {MIN_TOKENS}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
