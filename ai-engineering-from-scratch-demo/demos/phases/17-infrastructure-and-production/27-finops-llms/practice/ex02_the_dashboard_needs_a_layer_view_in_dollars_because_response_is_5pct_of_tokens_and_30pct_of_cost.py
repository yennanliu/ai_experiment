"""Exercise 2 — the dashboard needs a layer view in dollars, because response is 5% of tokens and 30% of cost.

    Design a per-tenant, per-task cost dashboard. What are the 5 views you
    build first?

Reading of the exercise: a dashboard design is only testable once it runs, so
the five views are built over a seeded log of 3000 traces in the lesson's own
trace shape (`user_id`, `tenant_id`, `task_id`, `route`, four layers,
`cached_input`), stamped at the call site, priced at GPT-5 and GPT-5 mini list
prices (OpenAI model pages, read 2026-09-26: $1.25 / $0.125 cached / $10 and
$0.25 / $0.025 / $2 per 1M). The lesson's tenants come from the reference.

**ANSWER: tenant spend vs contract, cost per resolved by task, layer dollars
by task, per-user spend by tenant, and route x cache split.** View 1 carries
the enforcement ladder's state, view 2 is the lesson's unit metric, view 3 is
where levers are chosen, view 4 finds power users and abuse inside a tenant,
view 5 shows whether caching and routing are engaged. The design rule that
makes them trustworthy: every view is a partition of the same bill, and all
five sum to the log's total to within float rounding.

**FINDING: the layer view has to be in dollars, not tokens.** The lesson's
sample trace (1800 / 600 / 400 / 150) is 61.0 / 20.3 / 13.6 / 5.1% of
tokens -- outside the lesson's own "typical %" table on prompt (40-60%) and
response (10-30%). Priced at GPT-5, response is 30% of the dollars and prompt
45%: output costs 8x input, so a token-share chart puts the output-length
lever at a sixth of its real size.

**FINDING: the reference records enough for view 1 only.** `TenantState`
holds spend_today_usd, minute_count, daily_history and paused -- no user,
task, route or layer -- and the lesson's skill file lists five views with no
layer view, although the same file hard-rejects single-bucket billing.

Structure: `trace_log()` stamps the traces; `views()` builds the five as
rollups of one log; `lesson_trace()` reads the sample trace out of the docs.
"""

from __future__ import annotations

import collections
import dataclasses
import random
import re

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "27-finops-llms"
# $ per 1M tokens (input, cached input, output), OpenAI model pages, read 2026-09-26
PRICES = {"gpt-5": (1.25, 0.125, 10.0), "gpt-5-mini": (0.25, 0.025, 2.0)}
LAYERS, TRACES = ("prompt", "tool", "memory", "response"), 3000
TASKS = {"task_classify_doc": 1.0, "task_draft_reply": 1.5, "task_agent_run": 4.0}


def lesson_trace():
    """The four layer counts and cost_usd of the lesson's sample trace."""
    doc = parity.doc_text(PHASE, LESSON)
    counts = {k: int(re.search(rf"{k}_tokens: (\d+)", doc).group(1)) for k in LAYERS}
    return counts, float(re.search(r"cost_usd: ([\d.]+)", doc).group(1))


def layer_dollars(layers, route="gpt-5", cached=False):
    rate, out = PRICES[route][1 if cached else 0], PRICES[route][2]
    return {k: layers[k] * (out if k == "response" else rate) / 1e6 for k in LAYERS}


def trace_log(ref, seed=0):
    """Traces in the lesson's shape, stamped with every dimension at the call site."""
    rng, base, log = random.Random(seed), lesson_trace()[0], []
    for i in range(TRACES):
        tenant, task = rng.choice(list(ref.TENANTS)), rng.choice(list(TASKS))
        layers = {k: int(v * TASKS[task] * rng.uniform(0.5, 1.5)) for k, v in base.items()}
        route, cached = "gpt-5-mini" if task == "task_classify_doc" else "gpt-5", rng.random() < 0.5
        log.append(dict(trace_id=i, tenant_id=tenant, task_id=task, route=route, cached_input=cached,
                        user_id=f"{tenant[:8]}_u{min(int(rng.paretovariate(1.2)), 20)}",
                        layers=layer_dollars(layers, route, cached), resolved=rng.random() < 0.8))
    return log


def rollup(log, *fields, by_layer=False):
    out = collections.defaultdict(float)
    for t in log:
        for layer, usd in t["layers"].items():
            out[tuple(t[f] for f in fields) + ((layer,) if by_layer else ())] += usd
    return dict(out)


def views(log, contracts):
    """The five views; every one is a partition of the same bill."""
    resolved = collections.Counter(t["task_id"] for t in log if t["resolved"])
    return {
        "1 tenant spend vs contract": {k: (usd, usd / contracts[k[0]])
                                       for k, usd in rollup(log, "tenant_id").items()},
        "2 cost per resolved, by task": {k: (usd, usd / resolved[k[0]])
                                         for k, usd in rollup(log, "task_id").items()},
        "3 layer dollars, by task": rollup(log, "task_id", by_layer=True),
        "4 per-user spend, by tenant": rollup(log, "tenant_id", "user_id"),
        "5 route x cache split": rollup(log, "route", "cached_input"),
    }


def reconcile(board):
    """Each view's dollar total; the first element when a cell also carries a ratio."""
    return {name: sum(v[0] if isinstance(v, tuple) else v for v in view.values())
            for name, view in board.items()}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    log = trace_log(ref)
    board = views(log, {n: p.contracted_daily_usd for n, (p, _, _) in ref.TENANTS.items()})
    counts = lesson_trace()[0]
    tokens, dollars = sum(counts.values()), layer_dollars(counts)
    skill = (parity.lesson_dir(PHASE, LESSON) / "outputs" / "skill-finops-plan.md").read_text()
    top5 = re.search(r"Top 5 views: (.*)\.", skill).group(1).split(", ")
    return {
        "total": sum(sum(t["layers"].values()) for t in log),
        "sums": reconcile(board), "views": len(board),
        "token_share": {k: round(v / tokens, 3) for k, v in counts.items()},
        "dollar_share": {k: round(v / sum(dollars.values()), 3) for k, v in dollars.items()},
        "state_fields": [f.name for f in dataclasses.fields(ref.TenantState)],
        "skill_views": top5, "skill_layer_view": any("layer" in v for v in top5),
    }


def verify(result):
    total, share, dollar = result["total"], result["token_share"], result["dollar_share"]
    sums = {k: round(v, 6) for k, v in result["sums"].items()}
    fields = ["spend_today_usd", "minute_count", "daily_history", "paused"]
    return [
        practice.Check(
            "ANSWER: tenant vs contract, cost per resolved, layer dollars, per-user, route x cache",
            result["views"] == 5 and all(abs(v - total) < 1e-9 for v in result["sums"].values()),
            f"five views over one log, each summing to the ${total:.2f} bill: {sums}",
        ),
        practice.Check(
            "FINDING: the layer view has to be in dollars, not tokens",
            share["prompt"] > 0.6 and share["response"] < 0.1 and dollar["response"] == 0.3,
            f"sample trace token shares {share}, outside the lesson's 40-60% prompt and "
            f"10-30% response rows; at GPT-5 prices the dollar shares are {dollar}",
        ),
        practice.Check(
            "FINDING: the reference records enough for view 1 only",
            result["state_fields"] == fields and not result["skill_layer_view"],
            f"TenantState fields {result['state_fields']}; the skill file's five views "
            f"{result['skill_views']} include no layer view",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
