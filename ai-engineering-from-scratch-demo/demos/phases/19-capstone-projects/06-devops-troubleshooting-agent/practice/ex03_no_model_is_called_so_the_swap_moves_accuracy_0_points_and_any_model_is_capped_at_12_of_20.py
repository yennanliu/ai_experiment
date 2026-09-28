"""Exercise 3 -- no model is called, so the swap moves accuracy 0 points; any model is capped at 12 of 20.

    Swap the hypothesis model from Claude Sonnet 4.7 to a self-hosted Llama 3.3
    70B. Measure RCA accuracy delta and dollar per incident.

Reading of the exercise: the lesson's `code/main.py` has no model call:
`root_cause` emits up to three template hypotheses and ranks them with a fixed
formula. So the swap is measured at the only seam a model could occupy, the
re-ranker over `root_cause`'s candidates, on a 20-incident suite built from
the lesson's list (OOMKill cascade, DNS flap, HPA thrash, PVC fill, noisy
neighbour, faulty sidecar, bad ConfigMap rollout, certificate rotation,
image-pull backoff, probe failure) plus ten more. Each incident is the lesson's
sample cluster with its own deploy age, alerted at the Deployment or a Pod,
labelled with the true cause: rollout, node, dns, or other. Accuracy = the
#1 hypothesis names the true cause. The ceiling for *any* model is an oracle
that picks the right candidate whenever one exists. Dollars per incident = the
re-rank prompt (graph neighbourhood + candidates as JSON, tokens estimated as
characters / 4) plus a 250-token ranked answer, at list prices read on
2026-09-29: Claude Sonnet 4.6 $3 / $15 per MTok, Sonnet 5 $2 / $10, and Llama 3.3
70B Instruct Turbo on Together AI $1.04 / $1.04, a per-token stand-in for
self-hosting.

**ANSWER: the delta is 0 points: 9/20 (45%) under either model, because
neither is called.** Plugged into the only seam, a model can reach at most
12/20 (60%): 8 of 20 true causes (OOM, HPA, PVC, cert, quota, NetworkPolicy,
DB outage, secret rotation) are not among the candidates. That is below the
lesson's 80% rubric whichever model sits there. The re-rank prompt is about
146 tokens, so 1,000 incidents cost $4.19 on Sonnet 4.6, $2.79 on Sonnet 5 and
$0.41 on Llama: Llama is 9.8% of Sonnet 4.6, and every option is under half a
cent per incident, so cost cannot decide the swap.

**FINDING: "Claude Sonnet 4.7" is not a model.** Anthropic's lineup (read
2026-09-29) has Opus 4.7 and Sonnet 4.6, then Sonnet 5; no Sonnet 4.7.

**FINDING: the heuristic's answer depends only on where the alert fires.**
All 12 Deployment alerts get "bad rollout" at #1 and all 8 Pod alerts get
"node-level pressure", whatever the incident.
"""

from __future__ import annotations

import json

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "06-devops-troubleshooting-agent"
PRICES = {"claude-sonnet-4-6": (3.00, 15.00), "claude-sonnet-5": (2.00, 10.00),
          "llama-3.3-70b (Together)": (1.04, 1.04)}
ANSWER_TOKENS = 250
# (incident, alerted at, deployed_at, true cause) -- labelled, deterministic
SUITE = [("OOMKill cascade", "Pod", "2d ago", "other"), ("DNS flap", "Deployment", "4d ago", "dns"),
         ("HPA thrash", "Deployment", "1d ago", "other"), ("PVC fill", "Pod", "9d ago", "other"),
         ("noisy neighbour", "Pod", "3d ago", "node"), ("faulty sidecar", "Deployment", "20m ago", "rollout"),
         ("bad ConfigMap rollout", "Deployment", "35m ago", "rollout"),
         ("certificate rotation", "Deployment", "6d ago", "other"),
         ("image-pull backoff", "Deployment", "5m ago", "rollout"),
         ("probe failure", "Deployment", "10m ago", "rollout"),
         ("bad image rollout", "Deployment", "14m ago", "rollout"),
         ("kernel regression", "Pod", "8d ago", "node"), ("disk-pressure eviction", "Pod", "2d ago", "node"),
         ("CoreDNS OOM", "Pod", "5d ago", "dns"), ("NetworkPolicy blocks egress", "Deployment", "2d ago", "other"),
         ("upstream DB outage", "Deployment", "30m ago", "other"), ("namespace quota hit", "Pod", "1d ago", "other"),
         ("node clock skew", "Pod", "7d ago", "node"), ("ndots search latency", "Deployment", "3d ago", "dns"),
         ("secret rotation", "Deployment", "45m ago", "other")]
CLASS = {"bad rollout": "rollout", "node-level": "node", "DNS flap": "dns"}


def cause_of(title):
    return next(c for prefix, c in CLASS.items() if title.startswith(prefix))


def prompt(g, alerted, hyps):
    hood = {k: g.nodes[k].attrs for _, k in g.neighbors(alerted) if k in g.nodes}
    return json.dumps({"alert": alerted, "attrs": g.nodes[alerted].attrs, "neighbours": hood,
                       "candidates": [{"title": h.title, "citations": h.citations} for h in hyps],
                       "task": "rank the candidates by likelihood of being the root cause"})


def run(ref, row):
    name, where, age, truth = row
    g = ref.build_sample_cluster()
    g.nodes["Deployment/checkout-api"].attrs["deployed_at"] = age
    alerted = "Deployment/checkout-api" if where == "Deployment" else "Pod/checkout-api-abc-0"
    hyps = ref.root_cause(g, alerted)
    causes = [cause_of(h.title) for h in hyps]
    return {"name": name, "where": where, "heuristic": causes[0] == truth, "oracle": truth in causes,
            "top": causes[0], "tokens": len(prompt(g, alerted, hyps)) // 4}


def costs(rows):
    tokens = sum(r["tokens"] for r in rows) / len(rows)
    cost = {m: round((tokens * i + ANSWER_TOKENS * o) / 1e3, 2) for m, (i, o) in PRICES.items()}  # $ per 1k
    share = round(cost["llama-3.3-70b (Together)"] / cost["claude-sonnet-4-6"], 3)
    return {"tokens": round(tokens), "cost": cost, "llama_share": share}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = [run(ref, row) for row in SUITE]
    return {
        "n": len(rows), "heuristic": sum(r["heuristic"] for r in rows), "oracle": sum(r["oracle"] for r in rows),
        "unreachable": [r["name"] for r in rows if not r["oracle"]], **costs(rows),
        "top_by_where": sorted({(r["where"], r["top"]) for r in rows}),
        "model_calls": [n for n in dir(ref) if "model" in n.lower() or "llm" in n.lower()],
    }


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: delta 0 points (9/20 both ways, no model is called); any model is capped at 12/20",
            (r["n"], r["heuristic"], r["oracle"], len(r["unreachable"]), r["model_calls"]) == (20, 9, 12, 8, [])
            and r["oracle"] / r["n"] < 0.80,
            f"heuristic {r['heuristic']}/20, oracle over candidates {r['oracle']}/20, "
            f"unreachable {r['unreachable']}",
        ),
        practice.Check(
            "ANSWER: dollars per 1,000 incidents on a ~146-token re-rank prompt",
            r["tokens"] == 146 and r["cost"] == {"claude-sonnet-4-6": 4.19, "claude-sonnet-5": 2.79,
                                                 "llama-3.3-70b (Together)": 0.41} and r["llama_share"] == 0.098,
            f"{r['tokens']} prompt tokens; $ per 1k incidents {r['cost']}; Llama/Sonnet 4.6 = {r['llama_share']}",
        ),
        practice.Check(
            "FINDING: 'Claude Sonnet 4.7' is not a model, and the lesson text names it",
            "Claude Sonnet 4.7" in parity.doc_text(PHASE, LESSON),
            "Anthropic's lineup on 2026-09-29: Opus 4.7, Sonnet 4.6, Sonnet 5 -- no Sonnet 4.7",
        ),
        practice.Check(
            "FINDING: the heuristic's #1 depends only on where the alert fires",
            r["top_by_where"] == [("Deployment", "rollout"), ("Pod", "node")],
            f"(alerted at, #1 cause) pairs seen: {r['top_by_where']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
