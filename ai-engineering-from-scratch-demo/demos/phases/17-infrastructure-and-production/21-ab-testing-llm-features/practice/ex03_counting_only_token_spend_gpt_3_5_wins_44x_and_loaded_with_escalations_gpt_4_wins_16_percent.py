"""Exercise 3 — counting only token spend GPT-3.5 wins 44x, and loaded with escalations GPT-4 wins 16%.

    Design an A/B that tests GPT-4 vs GPT-3.5 on cost-per-resolved-ticket.
    What's the primary metric, guardrail metric, secondary?

Reading of the exercise: a support bot either resolves a ticket or escalates
it to a human; the design is run as a seeded simulation of 10,000 tickets per
arm, randomized by ticket. Prices are OpenAI's legacy list prices (GPT-4 8k
$30/$60 per M tokens, gpt-3.5-turbo $0.50/$1.50); resolution rates 60% vs
50%, 2,000 input / 400 output tokens a ticket and $4 per human-handled ticket
are ASSUMPTIONS, stated so the arithmetic can be redone with real ones.

**ANSWER: primary -- fully loaded cost per resolved ticket, (LLM spend +
human cost of escalations) / tickets closed; guardrail -- 7-day reopen rate
(and CSAT, P99 latency), which must not be significantly worse; secondary --
bot resolution rate and tokens per ticket.** Every ticket is eventually
closed, so the primary is a per-ticket mean and a plain two-sample z-test
applies, randomized by ticket. Simulated: GPT-4 $1.67 against GPT-3.5 $1.98
a ticket, z = -11.0 -- GPT-4 wins; resolution rate z = 13.9; reopens per
resolved ticket are flat (z = -0.17).

**FINDING: the metric's definition decides the winner before the test runs.**
LLM spend per bot-resolved ticket is $0.139 vs $0.0032 -- GPT-3.5 is 44x
cheaper and no A/B is needed to learn that. Loaded with escalations GPT-4 is
16% cheaper. The lesson's "accuracy + cost/request + latency" for model
selection is the first definition.

**FINDING: the test must be powered on the break-even resolution gap, 2.06
points.** GPT-4 pays for itself when its resolution rate beats GPT-3.5's by
($0.084 - $0.0016) / $4 = 2.06 points; the reference `fixed_sample_size`
gives 9,232 tickets per arm to detect that at 50% -- and the reference's only
test, `z_statistic`, takes success counts, so it can size the resolution
rate but cannot test the cost metric itself.

**FINDING: a reopen guardrail counted per ticket penalizes the model that
resolves more.** Both arms reopen 5% of what they resolve, but GPT-4 resolves
more, so per ticket it reopens 1.18x as often, z = 1.96 -- at the edge of
tripping the guardrail on a model that is no worse. Divide by resolved tickets.

Structure: `arm()` simulates one arm; `welch_z()` tests the loaded cost; the
reference's `z_statistic` tests the two proportions.
"""

from __future__ import annotations

import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "21-ab-testing-llm-features"
PRICE = {"gpt-4": (30.0, 60.0), "gpt-3.5": (0.5, 1.5)}  # $ per M tokens, in / out
RESOLVE = {"gpt-4": 0.60, "gpt-3.5": 0.50}  # assumed
# assumed $ per escalation and reopen rate; tickets per arm
HUMAN, REOPEN, N = 4.0, 0.05, 10_000


def llm_cost(model, tokens_in=2000, tokens_out=400):
    p_in, p_out = PRICE[model]
    return (tokens_in * p_in + tokens_out * p_out) / 1e6


def arm(model, seed):
    rng = random.Random(seed)
    rows = []
    for _ in range(N):
        cost = llm_cost(model, rng.randint(1000, 3000), rng.randint(200, 600))
        resolved = rng.random() < RESOLVE[model]
        reopened = resolved and rng.random() < REOPEN
        rows.append((cost, resolved, reopened, cost + (0 if resolved else HUMAN)))
    return rows


def welch_z(a, b):
    se = math.sqrt(statistics.variance(a) / len(a) + statistics.variance(b) / len(b))
    return (statistics.fmean(a) - statistics.fmean(b)) / se


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    g4, g35 = arm("gpt-4", 1), arm("gpt-3.5", 2)

    def count(rows, i):
        return sum(r[i] for r in rows)

    def per_bot_resolved(rows):
        return sum(r[0] for r in rows) / count(rows, 1)

    gap = (llm_cost("gpt-4") - llm_cost("gpt-3.5")) / HUMAN
    return {
        "loaded": (
            statistics.fmean(r[3] for r in g4),
            statistics.fmean(r[3] for r in g35),
        ),
        "z_loaded": welch_z([r[3] for r in g4], [r[3] for r in g35]),
        "reopen_ratio": count(g4, 2) / count(g35, 2),
        "z_reopen_ticket": ref.z_statistic(count(g35, 2), N, count(g4, 2), N),
        "z_reopen": ref.z_statistic(
            count(g35, 2), count(g35, 1), count(g4, 2), count(g4, 1)
        ),
        "z_resolve": ref.z_statistic(count(g35, 1), N, count(g4, 1), N),
        "bot_only": (per_bot_resolved(g4), per_bot_resolved(g35)),
        "gap": gap,
        "n_gap": ref.fixed_sample_size(0.5, gap / 0.5),
    }


def verify(result):
    g4, g35 = result["loaded"]
    b4, b35 = result["bot_only"]
    return [
        practice.Check(
            "ANSWER: primary loaded cost per resolved ticket, guardrail reopen rate, "
            "secondary resolution rate",
            round(g4, 2) == 1.67
            and round(g35, 2) == 1.98
            and result["z_loaded"] < -1.96
            and abs(result["z_reopen"]) < 1.96,
            f"loaded ${g4:.2f} vs ${g35:.2f} a ticket, z = {result['z_loaded']:.1f}; reopen "
            f"z = {result['z_reopen']:.2f}; resolution z = {result['z_resolve']:.1f}",
        ),
        practice.Check(
            "FINDING: the metric's definition decides the winner before the test runs",
            round(b4 / b35) == 44 and round(1 - g4 / g35, 2) == 0.16,
            f"LLM spend per bot-resolved ticket ${b4:.3f} vs ${b35:.4f} ({b4 / b35:.0f}x "
            f"for GPT-3.5); loaded, GPT-4 is {1 - g4 / g35:.0%} cheaper",
        ),
        practice.Check(
            "FINDING: the test must be powered on the break-even resolution gap, 2.06 points",
            round(result["gap"] * 100, 2) == 2.06 and result["n_gap"] == 9232,
            f"break-even gap {result['gap'] * 100:.2f} points; fixed_sample_size(0.5, "
            f"{result['gap'] / 0.5:.4f}) = {result['n_gap']} tickets per arm",
        ),
        practice.Check(
            "FINDING: a reopen guardrail counted per ticket penalizes the model that resolves more",
            round(result["reopen_ratio"], 1) == 1.2
            and result["z_reopen_ticket"] > 1.9
            and abs(result["z_reopen"]) < 1.96,
            f"both arms reopen {REOPEN:.0%} of what they resolve; per ticket GPT-4 reopens "
            f"{result['reopen_ratio']:.2f}x as often, z = {result['z_reopen_ticket']:.2f} at "
            f"the edge of flagging; per resolved ticket z = "
            f"{result['z_reopen']:.2f}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
