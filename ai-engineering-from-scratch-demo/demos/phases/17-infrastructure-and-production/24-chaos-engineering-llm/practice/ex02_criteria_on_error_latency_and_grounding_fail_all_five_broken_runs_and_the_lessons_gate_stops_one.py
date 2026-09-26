"""Exercise 2 — criteria on error, latency and grounding fail all five broken runs, and the lesson's gate stops one.

    Design the first five chaos experiments for a vLLM-based RAG service.
    Include success criteria.

Reading of the exercise: a success criterion is only worth writing if it
fails when the resilience it tests is missing, so each experiment is stated
with a hypothesis, a fault, a blast radius, and criteria over three SLIs --
error rate, p95 latency against baseline, grounded-answer rate -- and then
judged against two versions of the service: *hardened* (the mechanism the
hypothesis names is there) and *naive* (it is not). The SLI outcomes are the
design's stated expectations, written as constants below; what is measured is
what the criteria and the lesson's own `run_experiment` gate do with them.

**ANSWER: five experiments, smallest blast first, and each criterion fails
exactly when its mechanism is missing.** (1) malformed prompt --
64 KB of stacked combining marks in 1% of requests, 5% blast: an input limit
must reject it, p95 within 1.2x. (2) gateway-to-vLLM partition, 5%: failover
to a hosted model, errors under 0.5%. (3) vector-DB +2 s with a 500 ms
timeout, 10%: BM25 fallback keeps grounding at 0.85 or above. (4) long-context
burst, top_k 5 -> 40 at 2x concurrency, 10%: admission control keeps p95 under
2x, shedding at most 5%. (5) kill 1 of 4 decode replicas, 25%: gateway retry
keeps errors at 0.1%, p95 under 1.5x. The hardened service passes 5 of 5; the
naive one fails 5 of 5, each on the SLI its experiment targets.

**FINDING: the lesson's gate stops one of the five broken runs.** Fed the
naive runs' error rates, `run_experiment` aborts only the replica kill (200x
burn, 25% blast). The partition fails every request in its blast radius --
2000x burn -- and completes, on 5% blast. The vector-DB run answers with no
context and a grounding rate of 0, and the KV run triples p95; both show the
baseline error rate, so both read 1.0x and complete. A RAG service's worst
chaos outcome, a confident answer with no sources, is a 200.

**FINDING: the reference `Experiment` has no field for a criterion.** Its
four fields are name, duration, induced error rate and blast radius, and the
error rate is an input: the result is decided before the run. Its only
success signal is "COMPLETED", which the naive partition earns.

Structure: `EXPERIMENTS` holds the design; `judge()` applies its criteria,
and `gate()` hands the same run to the reference.
"""

from __future__ import annotations

import dataclasses

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "24-chaos-engineering-llm"
BASELINE = (0.0005, 1.0, 0.97)  # error rate, p95 / baseline p95, grounded-answer rate

# name, blast, minutes, criteria (max error, max p95x, min grounded),
# expected SLIs hardened, expected SLIs naive (fallback / limit / retry missing)
EXPERIMENTS = [
    ("malformed prompt, 1% of requests", 0.05, 3, (0.015, 1.2, 0.95),
     (0.0105, 1.0, 0.97), (0.0005, 6.0, 0.97)),  # 12 s tokenizer stalls, head-of-line
    ("gateway-to-vLLM partition", 0.05, 5, (0.005, 1.5, 0.90),
     (0.0005, 1.15, 0.95), (1.0, 1.0, 0.0)),
    ("vector DB +2s, 500ms timeout", 0.10, 5, (0.005, 1.5, 0.85),
     (0.0005, 1.25, 0.90), (0.0005, 1.25, 0.0)),  # naive: generate with empty context
    ("long-context burst, top_k 5->40", 0.10, 10, (0.05, 2.0, 0.90),
     (0.02, 1.2, 0.96), (0.0005, 3.5, 0.96)),  # naive: preemption + re-prefill
    ("kill 1 of 4 decode replicas", 0.25, 5, (0.001, 1.5, 0.95),
     (0.0005, 1.4, 0.97), (0.10, 1.4, 0.97)),  # naive: 30 of 300 s before endpoint drop
]
SLIS = ("error", "p95", "grounded")


def judge(criteria, observed):
    """The SLIs an observed run fails."""
    max_err, max_p95, min_grounded = criteria
    err, p95, grounded = observed
    return [name for name, bad in zip(SLIS, (err > max_err, p95 > max_p95,
                                             grounded < min_grounded)) if bad]


def gate(ref, name, blast, minutes, observed):
    """The reference safety plane on the same run: it reads the error rate only."""
    run = ref.run_experiment(ref.Experiment(name, minutes, observed[0], blast))
    return run["status"], run["burn_rate_x"]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = []
    for name, blast, minutes, criteria, hardened, naive in EXPERIMENTS:
        rows.append({
            "name": name, "blast": blast,
            "hardened": judge(criteria, hardened), "naive": judge(criteria, naive),
            "baseline": judge(criteria, BASELINE),
            "gate_naive": gate(ref, name, blast, minutes, naive),
            "gate_hardened": gate(ref, name, blast, minutes, hardened)[0],
        })
    return {"rows": rows, "fields": [f.name for f in dataclasses.fields(ref.Experiment)]}


def verify(result):
    rows = result["rows"]
    blasts = [r["blast"] for r in rows]
    blind = {r["name"]: r["gate_naive"] for r in rows if r["gate_naive"][0] == "COMPLETED"}
    aborted = [r["name"] for r in rows if r["name"] not in blind]
    return [
        practice.Check(
            "ANSWER: five experiments, smallest blast first, and each criterion fails "
            "exactly when its mechanism is missing",
            all([not any(r["hardened"] + r["baseline"] for r in rows),
                 all(r["naive"] for r in rows), blasts == sorted(blasts)]),
            "naive fails " + "; ".join(f"{r['name']}: {r['naive']}" for r in rows)
            + "; hardened and baseline fail none",
        ),
        practice.Check(
            "FINDING: the lesson's gate stops one of the five broken runs",
            all([aborted == ["kill 1 of 4 decode replicas"], len(blind) == 4,
                 blind["gateway-to-vLLM partition"][1] == 2000.0,
                 blind["vector DB +2s, 500ms timeout"][1] == 1.0]),
            f"aborted {aborted}; completed with burn {blind}; hardened runs "
            f"{[r['gate_hardened'] for r in rows]}",
        ),
        practice.Check(
            "FINDING: the reference Experiment has no field for a criterion",
            result["fields"] == ["name", "duration_min", "induced_error_rate",
                                 "blast_radius_pct"],
            f"fields {result['fields']}; the error rate is an input, and COMPLETED is "
            "the only success signal",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
