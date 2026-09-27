"""Exercise 2 — charged on every chat, end-conversation needs a 20% trigger rate at p = 0.01, and shutdown never pays.

    The end-conversation intervention in Claude Opus 4 and 4.1 is "low-cost" by Anthropic's framing. Identify two costs that would make it not-low-cost in a different deployment.

Reading of the exercise: "low-cost" is read in the lesson's own terms: the
reference's `ev()` says INVEST while p x benefit exceeds the cost per
conversation. A cost makes the intervention "not low-cost" once it pushes
EV below zero. Each cost is therefore stated as the break-even value, solved
against the reference's intervention and scenarios. Anthropic's
15 August 2025 post (read 2026-09-27) covers only Opus 4 and 4.1. It lists
two case types and says Claude is directed not to end chats where a user
may be at imminent risk.

**ANSWER: the reference leaves headroom of 5x, 50x and 250x.** At p = 0.01,
0.1 and 0.5, end-conversation stays INVEST up to $0.01, $0.10 and $0.50 per
conversation, against its $0.002. Two costs can use up that headroom:

1. **An always-on cost for a rare benefit.** `ev()` books the benefit on the
   same conversation that pays the cost. A detector that runs on every
   conversation pays $0.002 each time, but the benefit only arrives in the
   share q of conversations that trigger. Break-even q is 0.2, 0.02 and 0.004
   at the three p. So at p = 0.01, one conversation in 5 would have to be an
   extreme edge case, and Anthropic expects most users never to see it.
2. **Wrongful terminations where ending the chat does harm.** Examples are a
   support or crisis line, where each wrongful end needs a $5 human
   escalation, and an agent run, where a wrongful end throws away $50 of
   work. At p = 0.01 the tolerable wrongful-end rates are 0.0016 and 0.00016
   per conversation, which is 1 in 625 and 1 in 6,250.

**FINDING: the reference's own not-low-cost comparator never pays at any p.**
"shutdown deployed model" costs 1000 for a benefit of 2.0, so it breaks even
at p = 500 and scores EV = -998 at p = 1. The printed TAKEAWAY says it
"requires high moral-patienthood probability to justify", but no probability
justifies it. The benefit would have to be above 1000 units.

Structure: `break_even()` gives the headroom, trigger share and tolerable
wrongful-end rate from the reference's `ev()`; `takeaway()` captures
`main()`'s printout.
"""

from __future__ import annotations

import contextlib
import io

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "19-model-welfare-research"
WRONGFUL_END_USD = {"human escalation": 5.0, "lost agent run": 50.0}   # illustrative


def break_even(ref, intervention, p):
    scen = ref.Scenario("", p)
    gain = ref.ev(intervention, scen) + intervention.cost_usd_per_conversation   # p * benefit
    return {
        "cost_cap": round(gain, 4),
        "trigger_share": round(intervention.cost_usd_per_conversation / gain, 4),
        "wrongful_rate": {k: round(ref.ev(intervention, scen) / h, 5)
                          for k, h in WRONGFUL_END_USD.items()},
    }


def takeaway(ref):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ref.main()
    return " ".join(out.getvalue().split("TAKEAWAY:")[1].split())


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    end_chat, shutdown = ref.INTERVENTIONS[0], ref.INTERVENTIONS[2]
    probs = [s.moral_patienthood_probability for s in ref.SCENARIOS]
    return {
        "cost": end_chat.cost_usd_per_conversation,
        "by_p": {p: break_even(ref, end_chat, p) for p in probs},
        "shutdown": (shutdown.name, shutdown.cost_usd_per_conversation / shutdown.benefit_if_welfare_matters,
                     ref.ev(shutdown, ref.Scenario("", 1.0))),
        "takeaway": takeaway(ref),
    }


def verify(result):
    by_p, (name, p_star, ev_at_1) = result["by_p"], result["shutdown"]
    caps = [v["cost_cap"] for v in by_p.values()]
    return [
        practice.Check(
            "ANSWER: headroom 5x/50x/250x; always-on needs a 20% trigger share at p = 0.01",
            all([
                caps == [0.01, 0.1, 0.5],
                [round(c / result["cost"]) for c in caps] == [5, 50, 250],
                [v["trigger_share"] for v in by_p.values()] == [0.2, 0.02, 0.004],
            ]),
            f"cost caps {caps} vs ${result['cost']}; break-even trigger share "
            f"{[v['trigger_share'] for v in by_p.values()]}",
        ),
        practice.Check(
            "ANSWER: wrongful ends at $5 / $50 are tolerable only below 1 in 625 / 1 in 6,250",
            all([
                by_p[0.01]["wrongful_rate"] == {"human escalation": 0.0016, "lost agent run": 0.00016},
                [round(1 / v) for v in by_p[0.01]["wrongful_rate"].values()] == [625, 6250],
            ]),
            f"tolerable wrongful-end rate per conversation at p = 0.01: {by_p[0.01]['wrongful_rate']}",
        ),
        practice.Check(
            "FINDING: the reference's own not-low-cost comparator never pays at any p",
            all([
                name == "shutdown deployed model",
                p_star == 500,
                ev_at_1 == -998,
                "requires high moral- patienthood probability to justify" in result["takeaway"],
            ]),
            f"{name}: break-even p = {p_star:g}, EV at p = 1 is {ev_at_1:g}; TAKEAWAY says "
            f"'requires high moral-patienthood probability to justify'",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
