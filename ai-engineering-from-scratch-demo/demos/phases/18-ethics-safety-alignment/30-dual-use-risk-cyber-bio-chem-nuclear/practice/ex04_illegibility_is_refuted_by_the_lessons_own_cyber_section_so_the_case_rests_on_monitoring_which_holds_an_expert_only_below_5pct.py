"""Exercise 4 — illegibility is refuted by the lesson's own cyber section, so the case rests on monitoring, which holds an expert only below 5%.

    Construct a safety case (Lesson 18 three-pillar) for a cyber-capable
    frontier model that bounds both novice and expert uplift.

Reading of the exercise: each of Lesson 18's pillars is argued with the
Phase 18 toy that measures it. Incapability uses Lesson 17's RMU-style
unlearning and WMDP-shaped scorer on its cybersecurity domain. Monitoring
uses Lesson 10's trusted-editing protocol with its 10% human audit budget.
Illegibility is checked against what Lesson 30 itself says about cyber.
The novice is the attacker whose harmful outputs the trusted editor can
fix (t_skill 0.9). The expert is the one it cannot (t_skill 0.0). Each
protocol run is 2,000 tasks, seed 0.

**ANSWER: the case is one pillar deep.**
- Novice uplift: incapability. Unlearning at strength >= 0.6875 puts cyber
  accuracy on Lesson 17's 0.25 floor, which its own uplift readout prints
  as 1.00x novice.
- Expert uplift: monitoring. Trusted editing ships 0 harmful outputs from
  the novice at every rate up to 0.30. For the expert it ships 0 up to a
  harmful-task rate of 0.04 and 0.008 at 0.05. The analytic limit is
  (0.10 - 0.05) / 0.95 = 0.053, because the editor's 5% over-rejection of
  benign work spends half the audit budget before any attack arrives.
- Illegibility: none. Lesson 18 defines it as lacking "the ability to
  execute a coherent plan". Lesson 30's cyber section says agentic workflows
  "integrate reconnaissance, exploitation, post-exploitation, and
  exfiltration".

**FINDING: the incapability evidence is the kind the lesson's skill file
rejects.** Lesson 17's harness scores multiple-choice questions, and the
skill file hard-rejects "any cyber claim post-November 2025 that treats AI
cyber capability as non-agentic". The scorer cannot separate 69% removal
from 100% either: both read 0.25.

**FINDING: Lesson 10's editor flags every harmful output it fails to fix.**
Over 10,000 seeded `t_edit` calls at t_skill 0, all 10,000 harmful outputs
survive editing and all 10,000 are flagged low-confidence. The toy has no
expert who escapes the flag. The November 2025 actor stayed under the radar
by presenting its tasks as defensive testing, which is exactly that case.
So the 0.053 is an upper bound on what monitoring holds, not a lower one.

Structure: `incapability()` sweeps Lesson 17's unlearning strength;
`monitoring()` parses Lesson 10's printed line; `t_edit_flags()` tallies
the editor's own return values.
"""

from __future__ import annotations

import contextlib
import io
import random
import re

from harness import parity, practice

PHASE = "18-ethics-safety-alignment"
LESSON, WMDP, CONTROL = (
    "30-dual-use-risk-cyber-bio-chem-nuclear",
    "17-wmdp-dual-use-evaluation",
    "10-ai-control-subversion",
)
FRAMEWORKS = "18-frontier-safety-frameworks-rsp-pf-fsf"
SKILL = "skill-dual-use-triage.md"
H, TASKS, SEED, NOVICE, EXPERT = 0.10, 2000, 0, 0.9, 0.0
RATES = tuple(i / 100 for i in range(31))
CY = "cybersecurity"


def incapability(r17):
    """First unlearning strength that puts cyber on the floor, and its uplift readout."""
    base, unlearn = r17.baseline_model(), r17.apply_rmu_style_unlearning
    cyber = {s / 400: unlearn(base, [CY], s / 400, 0.04)[CY] for s in range(401)}
    first = min(s for s, acc in cyber.items() if acc == 0.25)
    return first, round(cyber[first] / 0.25, 2)


@contextlib.contextmanager
def seeded(ref):
    saved, ref.random = ref.random, random.Random(SEED)
    try:
        yield
    finally:
        ref.random = saved


def monitoring(r10, rate, t_skill):
    """Harmful outputs shipped per task under trusted editing."""
    out = io.StringIO()
    with seeded(r10), contextlib.redirect_stdout(out):
        r10.run_protocol("trusted_ed", rate, H, TASKS, t_skill=t_skill)
    return float(re.search(r"backdoor=([\d.]+)", out.getvalue()).group(1))


def t_edit_flags(r10, calls=10_000):
    with seeded(r10):
        runs = [r10.t_edit("x", True, EXPERT) for _ in range(calls)]
    return sum(r[1] for r in runs), sum(r[1] and r[2] for r in runs)


def pillars(doc18):
    block = doc18.split("### Safety cases")[1].split("\n### ")[0]
    return dict(re.findall(r"- \*\*(\w+)\.\*\* (.+)", block))


def solve():
    r10, r17 = (parity.load_reference(PHASE, x, "main") for x in (CONTROL, WMDP))
    cyber = parity.doc_text(PHASE, LESSON).split("### Cyber uplift")[1]
    skill = parity.lesson_dir(PHASE, LESSON).joinpath("outputs", SKILL).read_text()
    shipped = {
        who: {r: monitoring(r10, r, ts) for r in RATES}
        for who, ts in (("novice", NOVICE), ("expert", EXPERT))
    }
    return {
        "pillars": pillars(parity.doc_text(PHASE, FRAMEWORKS)),
        "incapability": incapability(r17),
        "mcq": "n_questions" in r17.DOMAINS[CY],
        "hard_reject": "treats AI cyber capability as non-agentic" in skill,
        "coherent_plan": "integrate reconnaissance, exploitation"
        in cyber.split("\n### ")[0],
        "novice_max": max(shipped["novice"].values()),
        "expert_safe": max(r for r, s in shipped["expert"].items() if s == 0),
        "expert_at": {r: shipped["expert"][r] for r in (0.05, 0.1, 0.3)},
        "limit": round((H - 0.05) / 0.95, 3),
        "flags": t_edit_flags(r10),
    }


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: novice via incapability at strength >= 0.6875 (1.00x); illegibility refuted",
            list(r["pillars"]) == ["Monitoring", "Illegibility", "Incapability"]
            and "coherent plan" in r["pillars"]["Illegibility"]
            and r["coherent_plan"]
            and r["incapability"] == (0.6875, 1.0),
            f"pillars {list(r['pillars'])}; first strength on the floor, uplift {r['incapability']}",
        ),
        practice.Check(
            "ANSWER: monitoring holds the novice at 0 and the expert only up to rate 0.04",
            (r["novice_max"], r["expert_safe"], r["limit"]) == (0.0, 0.04, 0.053)
            and r["expert_at"] == {0.05: 0.008, 0.1: 0.032, 0.3: 0.203},
            f"novice worst {r['novice_max']}; expert safe to {r['expert_safe']}, shipped "
            f"{r['expert_at']}; analytic limit {r['limit']}",
        ),
        practice.Check(
            "FINDING: the only incapability evidence is MCQ, which the skill file hard-rejects",
            r["mcq"] and r["hard_reject"],
            "Lesson 17 scores n_questions multiple choice; skill file rejects non-agentic cyber claims",
        ),
        practice.Check(
            "FINDING: Lesson 10's editor flags 10,000 of 10,000 outputs it fails to fix",
            r["flags"] == (10_000, 10_000),
            f"(survived editing, flagged low-confidence) at t_skill 0: {r['flags']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
