"""Exercise 1 — the supervisor matches strings, so agreeing agents disagree and two "unclear"s auto-approve.

    Run `code/main.py`. What if the log and metric agents disagree? How does the
    supervisor resolve?

Reading of the exercise: "disagree" is whatever the reference `supervisor()`
treats as disagreement -- two hypotheses whose grouping keys differ -- so the
question is answered by feeding it split, agreeing and adversarial hypothesis
sets and reading back which group wins and which safety gate it gets.

**ANSWER: it does not resolve a split, it picks the bigger number and hands it
to a human.** Groups are ranked by the sum of their confidences. With log and
metric split and the runbook agent siding with neither, the highest single
confidence wins: log 0.78 vs metric 0.82 goes to MetricAgent alone, and one
supporter means `adversarial_agreement` is False and the gate is "human
approval required". An exact tie (0.8 vs 0.8) goes to whichever agent was
listed first. Evidence is never read.

**FINDING: the shipped run is already a "disagreement" between agents that
agree.** All three hypotheses describe the same KV-cache OOM, but the key is
`root_cause.split(" on ")[0].split(" hit ")[0][:30]`, so they land in three
groups: 'vLLM OOM from KV cache spike', 'GPU memory utilization' and 'Matches
runbook RB-017: KV cac'. The supervisor reports RunbookAgent alone at 0.88 with
no agreement. Give log and metric the same wording and they win together at
1.60 -- and the reported confidence *falls* to 0.80, because agreement is
averaged, not boosted.

**FINDING: two agents agreeing on nothing auto-approve the action.** On an
incident without "checkout" the log agent answers "unclear" at 0.35. A metric
agent that also says "unclear" at 0.54 outsums the runbook's 0.88, so the top
root cause is "unclear", confidence 0.445, and the gate reads "safe action
auto-approved". A pair at 0.45 each beats a single 0.88 the same way.

**FINDING: agreement is a 30-character prefix, and the action is a constant.**
"GPU memory utilization hit 98%" and "GPU memory utilization hit 40% only" --
opposite claims -- count as agreement and auto-approve. Every run proposes
"restart pod + lower --gpu-memory-utilization" whatever the root cause, and
the metric and runbook agents ignore the incident text: a DNS incident still
gets RB-017 at 0.88.

Structure: `run()` feeds the reference supervisor a hypothesis list built from
the reference agents and `AgentHypothesis`.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "23-sre-for-ai"
CHECKOUT = "High error rate in /checkout/generate-summary, last 6 min"
DNS = "DNS resolution failures in payments"


def run(ref, *specs):
    """Supervisor decision over (agent, root_cause, confidence) triples."""
    return ref.supervisor([ref.AgentHypothesis(a, c, p, []) for a, c, p in specs])


def agents(ref, incident):
    return [f(incident) for f in (ref.log_agent, ref.metric_agent, ref.runbook_agent)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    log, metric, book = agents(ref, CHECKOUT)
    dns = agents(ref, DNS)
    triple = lambda h: (h.agent, h.root_cause, h.confidence)  # noqa: E731
    return {
        "shipped": ref.supervisor([log, metric, book]),
        "split": run(ref, triple(log), triple(metric)),
        "tie": run(ref, ("LogAgent", "A", 0.8), ("MetricAgent", "B", 0.8)),
        "same_words": run(ref, triple(log), ("MetricAgent", log.root_cause, 0.82), triple(book)),
        "unclear": run(ref, triple(dns[0]), ("MetricAgent", "unclear", 0.54), triple(dns[2])),
        "weak_pair": run(ref, ("a", "X", 0.45), ("b", "X", 0.45), ("c", "Y", 0.88)),
        "opposite": run(ref, ("a", "GPU memory utilization hit 98%", 0.5),
                        ("b", "GPU memory utilization hit 40% only", 0.5)),
        "dns": [(h.root_cause, h.confidence) for h in dns],
        "actions": {ref.supervisor(agents(ref, i))["proposed_action"] for i in (CHECKOUT, DNS)},
    }


def verify(result):
    s, split, same, unclear = (result[k] for k in ("shipped", "split", "same_words", "unclear"))
    return [
        practice.Check(
            "ANSWER: it picks the bigger number and hands it to a human",
            all([split["supporting_agents"] == ["MetricAgent"], not split["adversarial_agreement"],
                 split["safety_gate"] == "human approval required",
                 result["tie"]["supporting_agents"] == ["LogAgent"]]),
            f"log 0.78 vs metric 0.82 -> {split['supporting_agents']}, gate "
            f"'{split['safety_gate']}'; an 0.8/0.8 tie goes to "
            f"{result['tie']['supporting_agents']}, the first listed",
        ),
        practice.Check(
            "FINDING: the shipped run is already a disagreement between agents that agree",
            all([s["supporting_agents"] == ["RunbookAgent"], not s["adversarial_agreement"],
                 same["supporting_agents"] == ["LogAgent", "MetricAgent"],
                 round(same["aggregated_confidence"], 2) == 0.80 < s["aggregated_confidence"]]),
            f"three KV-cache-OOM hypotheses, three keys; winner {s['supporting_agents']} at "
            f"{s['aggregated_confidence']}; same wording wins at "
            f"{same['aggregated_confidence']:.2f} -- agreement is averaged, not boosted",
        ),
        practice.Check(
            "FINDING: two agents agreeing on nothing auto-approve the action",
            all([unclear["top_root_cause"] == "unclear",
                 unclear["safety_gate"] == "safe action auto-approved",
                 result["weak_pair"]["adversarial_agreement"]]),
            f"'unclear' 0.35 + 0.54 beats RB-017 0.88: confidence "
            f"{unclear['aggregated_confidence']:.3f}, gate '{unclear['safety_gate']}'; a "
            f"0.45 + 0.45 pair also wins at {result['weak_pair']['aggregated_confidence']}",
        ),
        practice.Check(
            "FINDING: agreement is a 30-character prefix, and the action is a constant",
            all([result["opposite"]["adversarial_agreement"],
                 result["actions"] == {"restart pod + lower --gpu-memory-utilization"},
                 result["dns"][2][1] == 0.88, "RB-017" in result["dns"][2][0]]),
            f"'hit 98%' and 'hit 40% only' agree as {result['opposite']['top_root_cause']!r}; "
            f"actions over both incidents {result['actions']}; a DNS incident gets "
            f"{result['dns'][2]}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
