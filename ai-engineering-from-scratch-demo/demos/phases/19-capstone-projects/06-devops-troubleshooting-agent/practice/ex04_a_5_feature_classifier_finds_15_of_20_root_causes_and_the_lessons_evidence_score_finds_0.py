"""Exercise 4 -- a 5-feature classifier finds 15 of 20 root causes, and the lesson's evidence score finds 0.

    Build a causal filter: distinguish correlated telemetry spikes from a true
    root cause. Train a small classifier on the 20-scenario labels.

Reading of the exercise: each of the 20 lesson scenarios (the ten named ones
plus ten more, as in exercise 3) is a set of four telemetry spikes on cluster
objects: one on the root cause, two downstream symptoms, one coincidental
spike. Each spike has the five features an agent can read from telemetry and
the graph: onset lead before the alert (minutes), graph hops from the alerted
object, whether a change event (rollout, config edit) preceded it, how many
series spiked, and magnitude (z-score). A seeded `make_fixture()` draws them
under three cascade assumptions: symptoms start after their cause, sit nearer
the alert, and light up more series with bigger spikes. Labels: 1 for the root
cause. The classifier is a standardised logistic regression, scored
leave-one-scenario-out: train on 19 scenarios, pick the highest-probability
spike in the 20th. Baselines: the biggest spike, the earliest spike, and the
lesson's own `Hypothesis.score` (recency = minutes since onset, specificity =
magnitude / 8, citations = series, path length = hops).

**ANSWER: the classifier finds the root cause in 15 of 20 held-out
scenarios**, against 12 for "the earliest spike" and 0 for "the biggest
spike". Fit on all 80 spikes, it learns that more series (-1.52) and bigger
magnitude (-1.37) mean symptom, and that an earlier onset (+0.50) and a
preceding change (+0.32) mean cause. Hops barely matter (-0.09).

**FINDING: the lesson's evidence score is anti-causal.** Every term of
`Hypothesis.score` points at the symptom. Recency rewards the latest onset,
the citation count rewards spikes on more series, and path inverse rewards
spikes near the alert. It picks a downstream symptom in 19 of 20 scenarios and
the root in 0; the 20th pick is the coincidental spike.

**FINDING: the code is not the formula the doc states.** The lesson says
score = recency * specificity * path inverse * citation count. The code is a
weighted sum, so a hypothesis with no citations still scores 0.09, where a
product gives 0. Recency is clamped at 60 minutes: a cause that began 61
minutes or 999 minutes before the alert scores the same as one at 60 (0.265).
A PVC that has been filling for days gets no recency credit at all.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "06-devops-troubleshooting-agent"
SCENARIOS = ["OOMKill cascade", "DNS flap", "HPA thrash", "PVC fill", "noisy neighbour", "faulty sidecar",
             "bad ConfigMap rollout", "certificate rotation", "image-pull backoff", "probe failure",
             "bad image rollout", "kernel regression", "disk-pressure eviction", "CoreDNS OOM",
             "NetworkPolicy blocks egress", "upstream DB outage", "namespace quota hit", "node clock skew",
             "ndots search latency", "secret rotation"]
CHANGE_DRIVEN = {"faulty sidecar", "bad ConfigMap rollout", "image-pull backoff", "probe failure",
                 "bad image rollout", "certificate rotation", "secret rotation"}
FEATURES = ("lead_min", "hops", "change_before", "series", "magnitude")


def spike(rng, role, lead, change_driven):
    if role == "root":
        return [lead, rng.randint(1, 3), int(change_driven or rng.random() < 0.2), rng.randint(1, 2),
                round(rng.uniform(2, 4), 2)]
    if role == "symptom":
        return [round(lead - rng.uniform(2, 8), 1), rng.randint(0, 1), int(rng.random() < 0.1),
                rng.randint(2, 5), round(rng.uniform(3, 8), 2)]
    return [round(rng.uniform(0, 60), 1), rng.randint(2, 4), int(rng.random() < 0.3), rng.randint(1, 3),
            round(rng.uniform(2, 5), 2)]


def make_fixture(seed=19064):
    """20 labelled scenarios x 4 spikes: [(scenario, role, features, label)]."""
    rng, rows = random.Random(seed), []
    for name in SCENARIOS:
        lead = round(rng.uniform(10, 45), 1)
        for role in ("root", "symptom", "symptom", "coincidental"):
            rows.append((name, role, spike(rng, role, lead, name in CHANGE_DRIVEN), int(role == "root")))
    return rows


def fit(rows):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0))
    return clf.fit([r[2] for r in rows], [r[3] for r in rows])


def held_out_hit(rows, name):
    """Fit on the other 19 scenarios; 1 if the top-probability spike in `name` is the root."""
    test = [r for r in rows if r[0] == name]
    probs = fit([r for r in rows if r[0] != name]).predict_proba([r[2] for r in test])[:, 1]
    return test[int(probs.argmax())][3]


def loso(rows):
    """Leave-one-scenario-out hits, plus the coefficients of a fit on all 20."""
    hits = sum(held_out_hit(rows, name) for name in SCENARIOS)
    coefs = dict(zip(FEATURES, (round(float(c), 2) for c in fit(rows)[-1].coef_[0])))
    return hits, coefs


def pick_rate(rows, key):
    """Scenarios where the spike maximising `key` is the root cause."""
    return sum(max((r for r in rows if r[0] == n), key=key)[3] for n in SCENARIOS)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = make_fixture()

    def ref_score(r):
        lead, hops, _change, series, mag = r[2]
        return ref.Hypothesis("spike", ["s"] * series, int(lead), mag / 8, hops).score()

    hits, coefs = loso(rows)
    picked = [max((r for r in rows if r[0] == n), key=ref_score)[1] for n in SCENARIOS]
    return {
        "n": len(SCENARIOS), "spikes": len(rows), "clf": hits, "coefs": coefs,
        "biggest": pick_rate(rows, lambda r: r[2][4]), "earliest": pick_rate(rows, lambda r: r[2][0]),
        "ref": picked.count("root"), "ref_symptom": picked.count("symptom"),
        "stale": [round(ref.Hypothesis("h", ["s"], m, 0.5, 1).score(), 3) for m in (60, 61, 999)],
        "uncited": round(ref.Hypothesis("DNS", [], 60, 0.2, 4).score(), 3),
        "doc_product": "recency * specificity * graph-path length inverse * citation count"
        in parity.doc_text(PHASE, LESSON),
    }


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: the classifier finds the root cause in 15 of 20 held-out scenarios",
            (r["n"], r["spikes"], r["clf"], r["earliest"], r["biggest"]) == (20, 80, 15, 12, 0)
            and r["coefs"] == {"lead_min": 0.5, "hops": -0.09, "change_before": 0.32, "series": -1.52,
                               "magnitude": -1.37},
            f"logistic LOSO {r['clf']}/20 vs earliest spike {r['earliest']}/20, biggest {r['biggest']}/20; "
            f"coefficients {r['coefs']}",
        ),
        practice.Check(
            "FINDING: the lesson's evidence score picks a downstream symptom in 19 of 20 scenarios",
            (r["ref"], r["ref_symptom"]) == (0, 19),
            f"Hypothesis.score picks the root {r['ref']}/20, a symptom {r['ref_symptom']}/20",
        ),
        practice.Check(
            "FINDING: the score is a weighted sum, not the doc's product, and forgets anything past 60 min",
            r["doc_product"] and r["uncited"] == 0.09 and r["stale"] == [0.265, 0.265, 0.265],
            f"0-citation hypothesis scores {r['uncited']} (a product gives 0); recency 60/61/999 min "
            f"scores {r['stale']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
