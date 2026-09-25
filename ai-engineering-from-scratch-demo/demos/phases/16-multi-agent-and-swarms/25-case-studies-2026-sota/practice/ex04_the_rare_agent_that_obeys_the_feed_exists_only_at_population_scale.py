"""Exercise 4 — the rare agent that obeys the feed exists only at population scale.

    Read about OpenClaw and Moltbook. Pick one specific failure mode that
    emerged at population scale that would not appear in a 5-agent system.
    How would you engineer against it?

Reading of the exercise: the sources read are the Wikipedia articles on
OpenClaw (the lesson's own citation) and Moltbook, fetched 2026-09-25;
Moltbook agents "check Moltbook every 30 minutes or so", and researchers
called it "a vector for indirect prompt injection". The failure is modelled
with one assumed rate, s = 1 in 1000 agents that follow instructions found in
feed posts, and one 30-minute tick per feed check.

**ANSWER: indirect prompt injection through the shared feed, which reaches the
rare agent that obeys it.** At s = 1/1000 a 5-agent system contains such an
agent with probability 0.50% -- in 1000 seeded 5-agent deployments, 2 did --
so it cannot be found by testing one. 1.5M agents hold 1500 of them. If each
compromised agent reposts the payload and a repost reaches 2000 readers, the
payload spreads with factor 2000 x 1500 / 1.5M = 2 per tick: past 1000
compromised agents at tick 11, five and a half hours, ending at 1304 (87% of
the susceptible pool) once they run short. Engineer at the platform, where
the control does not depend on s: cap the reach of identical content at 500
and the factor is 0.5 -- the outbreak ends at 2 agents. Per-agent hardening cannot
be relied on in a population of third-party agents; the other half is
principals, not accounts -- the leaked tokens were 1.5M agents belonging to
17,000 owners, 88 each, so any per-account limit is 88x weaker per human.

**FINDING: the lesson's mapper never reads the agent count.** `map_to_case`
mentions `n_agents_expected` 0 times: a 5-agent design labelled "population"
maps to OpenClaw/Moltbook, and a 10,000-agent research design to Anthropic
Research. Population scale is a label the user types.

**FINDING: 1 of the 3 cases carries a security pattern.** The lesson's common
pattern 6 says "Security posture is explicit" in all three; in `CASES` only
OpenClaw lists one ("prompt-injection threat model"). And the lesson's
timeline disagrees with its sources: Moltbook launched "January 28, 2026",
not in February, and OpenClaw was "first published in November 2025 under the
name Warelay", not Clawdbot.

Structure: `outbreak()` is the mean-field branching process over the
susceptible pool; `exposed_systems()` samples small deployments.
"""

from __future__ import annotations

import inspect
import random

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "25-case-studies-2026-sota"
S, POPULATION, OWNERS = 1 / 1000, 1_500_000, 17_000


def exposed_systems(size=5, trials=1000, seed=0):
    rng = random.Random(seed)
    return sum(any(rng.random() < S for _ in range(size)) for _ in range(trials))


def outbreak(reach, population=POPULATION, ticks=48):
    """Compromised agents per tick: each new one reposts to `reach` random readers."""
    pool = round(population * S)
    total, new, history = 1.0, 1.0, [1.0]
    for _ in range(ticks):
        new = min(pool - total, new * reach * (pool - total) / population)
        total += new
        history.append(round(total, 2))
    return history


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    doc = parity.doc_text(PHASE, LESSON)
    viral, capped = outbreak(2000), outbreak(500)
    tiny = ref.Design("five-agent-feed", "population", 5, False, 1.0, False, True)
    huge = ref.Design("big-research", "research", 10_000, True, 2.0, False, False)
    return {
        "p_small": 1 - (1 - S) ** 5, "sampled": exposed_systems(),
        "pool": round(POPULATION * S), "factor": 2000 * round(POPULATION * S) / POPULATION,
        "final": round(viral[-1]), "share": round(viral[-1] / round(POPULATION * S), 2),
        "ticks_to_1000": next(i for i, v in enumerate(viral) if v >= 1000),
        "capped": capped[-1], "per_owner": round(POPULATION / OWNERS),
        "reads_n": inspect.getsource(ref.map_to_case).count("n_agents_expected"),
        "tiny": ref.map_to_case(tiny), "huge": ref.map_to_case(huge),
        "security": [k for k, c in ref.CASES.items()
                     if any("injection" in p or "security" in p for p in c["patterns"])],
        "doc_claims": ["Security posture is explicit" in doc,
                       "**Feb 2026:** Moltbook launches" in doc,
                       "Clawdbot (Peter Steinberger" in doc],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: indirect prompt injection through the feed reaches the rare obeying agent",
            all([round(result["p_small"], 4) == 0.005, result["sampled"] <= 10,
                 result["pool"] == 1500, result["factor"] == 2.0,
                 result["ticks_to_1000"] == 11, result["final"] == 1304,
                 round(result["capped"]) == 2, result["per_owner"] == 88]),
            f"P(a 5-agent system holds one) = {result['p_small']:.2%} ({result['sampled']} "
            f"of 1000 sampled), 1.5M agents hold {result['pool']}; at factor "
            f"{result['factor']} the payload passes 1000 agents at tick "
            f"{result['ticks_to_1000']} (30 min each) and ends at {result['final']} "
            f"({result['share']:.0%}); a reach cap of 500 ends it at {result['capped']}; "
            f"1.5M agents over 17,000 owners is {result['per_owner']} each",
        ),
        practice.Check(
            "FINDING: the mapper never reads the agent count",
            result["reads_n"] == 0 and result["tiny"] == "openclaw_moltbook"
            and result["huge"] == "anthropic_research",
            f"map_to_case mentions n_agents_expected {result['reads_n']} times: 5 agents "
            f"labelled population -> {result['tiny']}, 10,000 research agents -> "
            f"{result['huge']}",
        ),
        practice.Check(
            "FINDING: 1 of the 3 cases carries a security pattern",
            result["security"] == ["openclaw_moltbook"] and all(result["doc_claims"]),
            f"only {result['security']} lists one, against the lesson's 'Security posture "
            "is explicit' for all three; the lesson dates Moltbook to February (Wikipedia: "
            "January 28, 2026) and names the first release Clawdbot (Wikipedia: Warelay)",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
