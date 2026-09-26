"""Exercise 5 — retroactive attribution is exact per key and 12% wrong once one tenant runs agents.

    Argue whether retroactive tagging can ever work. When is it acceptable?

Reading of the exercise: "retroactive" is attribution done after the fact
from what the provider's bill records -- API key, tokens, cost -- plus
whatever logs already exist, with no tenant stamped on the call. The argument
is made by measurement: a seeded 10-day call log in the reference's own
traffic model (its three tenants, request counts, N(600, 150) tokens, flat
$10/M), attributed three ways, scored by the share of spend landing on the
wrong tenant.

**ANSWER: it works exactly when the dimension is already on the billing
record, and approximately when requests are alike.** A key per tenant:
0.0% misattributed over 29,457 calls. A shared key split by gateway request
counts: 1.5% with daily buckets, 1.1% over the whole period. Make
tenant_A_normal an agent workload -- 4x tokens per request -- and the same
split misattributes 12.2% daily and 12.0% over the period. Acceptable, then,
when the tenant has its own key or project, or when the numbers feed
showback and a percent or two does not change a decision. Not acceptable for
invoicing, kill switches, or anything where one tenant's requests differ in
size -- which is the agent case this lesson's layer table is about.

**FINDING: finer time buckets do not fix it.** Daily buckets are no better
than one bucket for the whole period (12.2% vs 12.0%): the error comes from
request size differing by tenant, which is persistent, not from traffic
moving over time.

**FINDING: allocation misattributes spend; it does not lose it.** Every
method above conserves the bill to the cent. The lesson's skill file says
"retroactive tagging loses ~10-30% of spend", but what it loses is
correctness, and the measured error runs from 0% to 12% depending only on
how alike the tenants' requests are -- not a constant.

**FINDING: the reference simulator cannot show retroactive attribution
failing.** All three of its tenants draw tokens per request from one
distribution, so splitting by request count is within 1.5%; the failure
needs heterogeneity the simulator does not model.

Structure: `call_log()` generates billing rows; `group(rows, "key")` and
`by_request_share()` are the retroactive methods; `misattributed()` is half
the L1 distance to the true per-tenant spend, over the total.
"""

from __future__ import annotations

import collections
import random
import re

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "27-finops-llms"
DAYS, PRICE = 10, 10.0                  # the reference's horizon and flat $/1M


def call_log(ref, agent=None, seed=0):
    """Per-call billing rows in the reference's traffic model.

    Mirrors simulate_day: requests = int(100 * mult * U(0.8, 1.3)) and one token
    draw per tenant-day from N(600, 150). `agent` = (tenant, factor) makes one
    tenant's requests `factor` times longer.
    """
    rng, rows = random.Random(seed), []
    for day in range(DAYS):
        for name, (_, _, mult) in ref.TENANTS.items():
            requests = int(100 * mult * rng.uniform(0.8, 1.3))
            tokens = int(rng.gauss(600, 150)) * (agent[1] if agent and agent[0] == name else 1)
            rows += [{"day": day, "tenant": name, "key": f"key_{name}", "cost":
                      tokens / 1e6 * PRICE}] * requests
    return rows


def group(rows, field):
    """Spend by one column: "tenant" is the truth, "key" the provider's usage export."""
    out = collections.defaultdict(float)
    for r in rows:
        out[r[field].removeprefix("key_")] += r["cost"]
    return out


def by_request_share(rows, bucket):
    """Retroactive on a shared key: split each bucket's bill by gateway request counts."""
    bill, counts = collections.defaultdict(float), collections.defaultdict(collections.Counter)
    for r in rows:
        b = bucket(r)
        bill[b] += r["cost"]
        counts[b][r["tenant"]] += 1
    out = collections.defaultdict(float)
    for b, usd in bill.items():
        total = sum(counts[b].values())
        for tenant, n in counts[b].items():
            out[tenant] += usd * n / total
    return out


def misattributed(rows, method, *args):
    """(share of spend on the wrong tenant, whether the bill is conserved)."""
    real, est = group(rows, "tenant"), method(rows, *args)
    total = sum(real.values())
    wrong = sum(abs(real[t] - est.get(t, 0.0)) for t in real) / 2 / total
    return round(wrong, 4), abs(sum(est.values()) - total) < 1e-6


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    shipped = call_log(ref)
    agent = call_log(ref, agent=("tenant_A_normal", 4))
    daily, whole = (lambda r: r["day"]), (lambda r: 0)
    skill = (parity.lesson_dir(PHASE, LESSON) / "outputs" / "skill-finops-plan.md").read_text()
    return {
        "key": misattributed(shipped, group, "key"),
        "daily": misattributed(shipped, by_request_share, daily),
        "whole": misattributed(shipped, by_request_share, whole),
        "agent_daily": misattributed(agent, by_request_share, daily),
        "agent_whole": misattributed(agent, by_request_share, whole),
        "calls": len(shipped),
        "skill_claim": re.search(r"retroactive tagging loses ([^.]*?) of spend", skill).group(1),
    }


def verify(result):
    methods = ("key", "daily", "whole", "agent_daily", "agent_whole")
    err = {m: result[m][0] for m in methods}
    return [
        practice.Check(
            "ANSWER: exact when the dimension is on the billing record, approximate when "
            "requests are alike",
            err == {"key": 0.0, "daily": 0.0147, "whole": 0.0112, "agent_daily": 0.1217,
                    "agent_whole": 0.1203} and result["calls"] == 29457,
            f"share of spend misattributed over {result['calls']} calls: {err}",
        ),
        practice.Check(
            "FINDING: finer time buckets do not fix it",
            abs(err["agent_daily"] - err["agent_whole"]) < 0.005,
            f"agent tenant, daily buckets {err['agent_daily']:.1%} vs one bucket "
            f"{err['agent_whole']:.1%}",
        ),
        practice.Check(
            "FINDING: allocation misattributes spend; it does not lose it",
            all(result[m][1] for m in methods) and result["skill_claim"] == "~10-30%",
            f"every method conserves the bill; the skill file claims {result['skill_claim']} "
            f"lost, measured misattribution runs {min(err.values()):.1%} to "
            f"{max(err.values()):.1%}",
        ),
        practice.Check(
            "FINDING: the reference simulator cannot show retroactive attribution failing",
            max(err["daily"], err["whole"]) < 0.02,
            f"its tenants share one token distribution, so request-count allocation is "
            f"off by {err['daily']:.1%} daily and {err['whole']:.1%} overall",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
