"""Exercise 2 — the lesson's pair gives each model one host, so its failover swaps the model.

    Your product needs Claude 3.7 Sonnet and GPT-4o. Design a two-provider
    deployment — which goes to which hyperscaler, what gateway sits in front,
    what is the failover policy?

Reading of the exercise: the design is written as a routing table a gateway
executes -- an ordered list of (provider, model) targets per requested
model -- and graded by enumerating every up/down state of Bedrock, Azure
OpenAI and Vertex. The lesson gives no availability figure, so each provider
is *assumed* independently up 99.9% of the time; the comparison between
tables does not depend on that number. Costs come from the reference's own
rates and its `lock_in_cost()`.

**ANSWER: Claude on Bedrock, GPT-4o on Azure OpenAI, one self-hosted gateway
in front, same-model failover first and cross-model last.** GPT-4o is only
in Azure's catalog and Claude is not in it, so the split is forced. The
gateway (LiteLLM-style, one key and one budget per team) routes Claude to
Bedrock, then to Claude on Vertex's Model Garden, then to GPT-4o on Azure;
GPT-4o goes to Azure, then to Claude on Bedrock. Every request is then
answered by some model 99.9999% of the time and Claude requests by Claude
99.9999% of the time. Falling Claude traffic over to GPT-4o costs less, not
more: $4.375/M against $6.00/M blended at 3:1 in the reference's rates.

**FINDING: the lesson's pair gives each model one host, so its failover
swaps the model.** "Claude from one, GPT from the other, failover between
them" means a Bedrock outage serves Claude users GPT-4o. Same-model
availability stays 99.9% -- 43.2 minutes a month -- with or without that
failover; only a second Claude host takes it to 0.04 minutes. GPT-4o has no
second hyperscaler in the lesson's catalogs, so its prompts must run on
Claude too, or the failover is to a different product.

**FINDING: `lock_in_cost()` charges 10% for headroom that on-demand does not
bill and that covers a tenth of a failover.** It prices redundancy at 13% of
spend, $195/month on $1,500. An on-demand secondary costs nothing while
idle, so the uplift is the 3% gateway, $45/month; and reserved headroom of
10% absorbs 10% of a failed-over load the size of the primary's. The real
constraint is the secondary's rate-limit quota, which the code does not model.

Structure: `availability()` enumerates the 8 provider states for a routing
table; `uplift()` parses the reference's printout.
"""

from __future__ import annotations

import contextlib
import io
import itertools

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "01-managed-llm-platforms"
UP, MINUTES = 0.999, 30 * 24 * 60  # assumed per-provider availability; minutes a month
PROVIDERS = ("bedrock", "azure", "vertex")
TABLES = {
    "pinned pair": {"claude": [("bedrock", "claude")], "gpt-4o": [("azure", "gpt-4o")]},
    "lesson pair": {
        "claude": [("bedrock", "claude"), ("azure", "gpt-4o")],
        "gpt-4o": [("azure", "gpt-4o"), ("bedrock", "claude")],
    },
    "design": {
        "claude": [("bedrock", "claude"), ("vertex", "claude"), ("azure", "gpt-4o")],
        "gpt-4o": [("azure", "gpt-4o"), ("bedrock", "claude")],
    },
}


def route(targets, up):
    return next((model for provider, model in targets if provider in up), None)


def availability(table):
    """{model: (served by the same model, served by any model)} over all 8 states."""
    out = {m: [0.0, 0.0] for m in table}
    for state in itertools.product((True, False), repeat=len(PROVIDERS)):
        up = {p for p, ok in zip(PROVIDERS, state) if ok}
        weight = 1.0
        for ok in state:
            weight *= UP if ok else 1 - UP
        for model, targets in table.items():
            served = route(targets, up)
            out[model][0] += weight * (served == model)
            out[model][1] += weight * (served is not None)
    return {m: (round(same, 9), round(anyone, 9)) for m, (same, anyone) in out.items()}


def uplift(ref):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ref.lock_in_cost()
    money = {
        line.split(":")[0].split(" (")[0]: float(line.split("$")[1].split("/")[0])
        for line in out.getvalue().splitlines()
        if "$" in line
    }
    keys = (
        "Primary daily spend",
        "Gateway overhead",
        "Idle secondary headroom",
        "Monthly uplift",
    )
    return tuple(money[k] for k in keys)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    bedrock, azure = ref.PLATFORMS[0], ref.PLATFORMS[1]
    daily, gateway, headroom, monthly = uplift(ref)
    return {
        "avail": {name: availability(table) for name, table in TABLES.items()},
        "blend": tuple(
            (3 * p.per_mtok_input + p.per_mtok_output) / 4 for p in (bedrock, azure)
        ),
        "uplift": (monthly, round(monthly / (daily * 30), 4), round(gateway * 30, 2)),
        # spare capacity as a share of an equal failed-over load
        "headroom_covers": headroom / daily,
        "claim": "failover between them" in parity.doc_text(PHASE, LESSON),
    }


def verify(result):
    a = result["avail"]
    down = {name: round((1 - a[name]["claude"][0]) * MINUTES, 2) for name in a}
    return [
        practice.Check(
            "ANSWER: Claude on Bedrock, GPT-4o on Azure, one gateway, same-model failover first",
            a["design"]["claude"][0] >= 0.999999
            and a["design"]["gpt-4o"][1] >= 0.999999
            and result["blend"] == (6.0, 4.375),
            f"design: Claude served by Claude {a['design']['claude'][0]}, GPT-4o requests "
            f"answered {a['design']['gpt-4o'][1]}; failover to GPT-4o costs "
            f"${result['blend'][1]}/M against ${result['blend'][0]}/M",
        ),
        practice.Check(
            "FINDING: the lesson's pair gives each model one host, so its failover swaps the model",
            result["claim"]
            and a["lesson pair"]["claude"][0] == a["pinned pair"]["claude"][0] == UP
            and a["lesson pair"]["claude"][1] > UP
            and down["design"] < 0.1,
            f"Claude-by-Claude availability {a['lesson pair']['claude'][0]} with or without "
            f"the lesson's failover ({down['lesson pair']} min/month down); a second Claude "
            f"host takes it to {down['design']} min",
        ),
        practice.Check(
            "FINDING: lock_in_cost() charges 10% for headroom that on-demand does not bill",
            result["uplift"] == (195.0, 0.13, 45.0),
            f"${result['uplift'][0]:.0f}/month, {result['uplift'][1]:.0%} of spend; on-demand "
            f"idles free, leaving the gateway's ${result['uplift'][2]:.0f}; 10% reserved "
            f"headroom absorbs {result['headroom_covers']:.0%} of an equal failed-over load",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
