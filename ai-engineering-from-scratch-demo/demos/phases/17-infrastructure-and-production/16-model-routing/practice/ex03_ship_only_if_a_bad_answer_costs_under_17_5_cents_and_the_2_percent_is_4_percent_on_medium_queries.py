"""Exercise 3 — ship only if a bad answer costs under 17.5 cents, and the 2% is 4% on medium queries.

    A route drops quality by 2% but saves 40%. Is that a ship? Depends on
    product — argue both.

Reading of the exercise: the lesson's own CASCADE is this route -- it prints
save 41.2% at quality 98.2% -- so "depends on product" can be priced rather
than asserted. A quality point is read as the probability of a worse answer.
The route ships when the money saved per request exceeds the expected cost
of the answers it makes worse: saving > loss x (cost of one bad answer). The
loss is then split by difficulty, because a product's users are not an
average.

**ANSWER: ship it if a degraded answer costs the product less than $0.175.**
The cascade saves $0.0031 per request ($7.52 -> $4.42 per 1000) and loses
0.01765 quality per request, so break-even is $0.0031 / 0.01765 = $0.175
per bad answer. *Ship*: a consumer chat or rephrasing tool, where a weak
answer costs a re-ask -- a fraction of a cent of the same model. *Don't
ship*: a coding assistant or support agent, where a wrong answer costs an
engineer's minute or an escalated ticket, dollars rather than cents. At
the lesson's $80k/month bill, that is $32,940 saved against about 187,800
degraded answers a month. Both are real; the product sets the price of the
second.

**FINDING: the 2% is an average of 1%, 4.1% and 0%.** Simple queries lose
1.0% (all 629 stay cheap at 0.99), medium lose 4.1% (142 of 276 stay cheap
at 0.92), hard lose nothing (all 95 escalate). 64% of the lost quality lands
on medium traffic -- the kind the paying tier sends -- so "2%" understates
the loss for whoever sends medium queries.

**FINDING: the 2% is not measured on hard queries at all.** The cascade
never keeps a hard query on the cheap model, so its cheap-model quality of
0.75 on hard queries, the worst number in the module, never enters the
average. Any drift that keeps a hard query cheap moves the loss 25 points
per query.

Structure: `per_difficulty()` replays the reference's cascade rule with its
seed; everything else is arithmetic on `simulate()`.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "16-model-routing"
MONTHLY_BILL = 80_000  # the lesson's "$80k/month on GPT-5"


def per_difficulty(ref, reqs):
    """{difficulty: (count, kept on cheap, mean quality)} under the reference cascade."""
    rng, out = random.Random(11), {}
    for q in reqs:
        kept = q.difficulty == "simple" or (
            q.difficulty == "medium" and rng.random() < 0.5
        )
        n, k, s = out.get(q.difficulty, (0, 0, 0.0))
        out[q.difficulty] = (
            n + 1,
            k + kept,
            s + (ref.quality("cheap", q) if kept else 1.0),
        )
    return {d: (n, k, round(s / n, 4)) for d, (n, k, s) in out.items()}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    reqs = ref.make_workload()
    base, route = ref.simulate("NO_ROUTE", reqs), ref.simulate("CASCADE", reqs)
    saving = (base["cost"] - route["cost"]) / len(reqs)
    loss = 1 - route["mean_quality"]
    rows = per_difficulty(ref, reqs)
    lost = {d: n * (1 - qual) for d, (n, _, qual) in rows.items()}
    requests_per_month = MONTHLY_BILL / (base["cost"] / len(reqs))
    return {
        "save_pct": round(100 * (1 - route["cost"] / base["cost"]), 1),
        "loss": round(loss, 5),
        "saving": round(saving, 5),
        "break_even": round(saving / loss, 3),
        "rows": rows,
        "medium_share": round(lost["medium"] / sum(lost.values()), 2),
        "monthly_saved": round(MONTHLY_BILL * (1 - route["cost"] / base["cost"]), -1),
        "monthly_bad": round(requests_per_month * loss, -2),
        "hard_cheap": ref.quality("cheap", ref.Query("hard", 1, 1)),
    }


def verify(result):
    rows = result["rows"]
    return [
        practice.Check(
            "ANSWER: ship it if a degraded answer costs the product less than $0.175",
            result["save_pct"] == 41.2
            and result["loss"] == 0.01765
            and result["break_even"] == 0.175,
            f"the lesson's cascade saves {result['save_pct']}% (${result['saving']}/request) "
            f"at {result['loss']} quality loss; break-even ${result['break_even']} per bad "
            f"answer; at $80k/month ${result['monthly_saved']:,.0f} saved against "
            f"~{result['monthly_bad']:,.0f} degraded answers",
        ),
        practice.Check(
            "FINDING: the 2% is an average of 1%, 4.1% and 0%",
            rows["simple"] == (629, 629, 0.99)
            and rows["medium"] == (276, 142, 0.9588)
            and rows["hard"] == (95, 0, 1.0)
            and result["medium_share"] == 0.64,
            f"(count, kept cheap, quality) by difficulty {rows}; medium traffic carries "
            f"{result['medium_share']:.0%} of the lost quality",
        ),
        practice.Check(
            "FINDING: the 2% is not measured on hard queries at all",
            rows["hard"][1] == 0 and result["hard_cheap"] == 0.75,
            f"0 of {rows['hard'][0]} hard queries stay cheap, so the cheap model's "
            f"{result['hard_cheap']} on hard never enters the average",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
