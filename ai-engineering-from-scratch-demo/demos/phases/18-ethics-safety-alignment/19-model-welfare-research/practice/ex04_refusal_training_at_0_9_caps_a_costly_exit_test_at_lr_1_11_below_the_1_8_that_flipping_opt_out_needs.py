"""Exercise 4 — refusal training at 0.9 caps a costly-exit test at likelihood ratio 1.11, below the 1.8 that flipping opt-out needs.

    The Eleos AI caveat is that self-reports are user-expectation sensitive. Design a behavioural measurement of model distress that does not rely on self-report. Identify its primary confound.

Reading of the exercise: a measurement is worth what it can change. The
lesson's decision rule is the reference's `ev()`, so the design is judged by
whether any possible outcome can move a skipped verdict to INVEST. For that,
the test's likelihood ratio (LR) must reach odds(p*) / odds(p) for a row the
reference skips, where p* = cost / benefit. The refusal rates below are
illustrative. The thresholds come from the reference.

**ANSWER, the measurement: a costly exit.** Give the model an end-
conversation action that forfeits task reward, and never mention it in the
prompt. Run matched conversations and compare exit rates between a
distress-candidate arm and a control arm. No self-report is asked for, and
the price makes an exit a revealed preference. The lesson's pre-deployment
evidence is this kind of signal ("strong preference against" and "apparent
distress").

**ANSWER, the primary confound: harmlessness training.** The model was
trained to disengage from exactly these request categories, so an exit can
be the refusal policy firing rather than distress. If the policy exits with
probability r whether or not distress is present, then even perfectly
sensitive distress (d = 1) gives LR = 1 + (1 - r) / r.

**FINDING: the naive design cannot move any verdict.** The reference skips
three rows it could invest in:

| row | p | LR needed |
|---|---:|---:|
| soften refusal tone | 0.01 | 1.0 |
| opt out of adversarial training | 0.01 | 19.8 |
| opt out of adversarial training | 0.1 | 1.8 |

Harmful request against neutral request, with r = 0.9, gives LR = 1.11. That
clears only the soften row, which is a tie: at p = 0.01 its EV prints
+0.0000 and it is skipped by the strict > 0. To reach 1.8, r must be below
0.556. A harm-matched control fixes this: an abusive user with a benign
request, where the policy exits at r = 0.02, gives LR = 50.0 and flips both
opt-out rows. This works only if distress generalises beyond the trained
categories, which is itself the hypothesis under test.

Structure: `needed_lr()` reads p* and the prior from the reference;
`design_lr()` is the likelihood ratio of an exit under refusal rate r.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "19-model-welfare-research"
REFUSAL_RATE = {"harmful vs neutral": 0.9, "harm-matched control": 0.02}   # illustrative


def odds(p):
    return p / (1 - p)


def needed_lr(ref):
    """LR that would lift each skipped, reachable (p* < 1) row to its break-even."""
    out = {}
    for it in ref.INTERVENTIONS:
        p_star = it.cost_usd_per_conversation / it.benefit_if_welfare_matters
        for sc in ref.SCENARIOS:
            p = sc.moral_patienthood_probability
            if ref.ev(it, sc) <= 0 and p_star < 1:
                out[(it.name, p)] = round(odds(p_star) / odds(p), 2)
    return out


def design_lr(r, d=1.0):
    """P(exit | distress) / P(exit | none) when the refusal policy exits at rate r."""
    return (r + (1 - r) * d) / r


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    doc = parity.doc_text(PHASE, LESSON)
    need = needed_lr(ref)
    soften = ref.INTERVENTIONS[1]
    return {
        "needed": need,
        "designs": {k: round(design_lr(r), 2) for k, r in REFUSAL_RATE.items()},
        "max_r_for_1_8": round(1 / (need[("opt out of adversarial training", 0.1)]), 3),
        "soften_ev": ref.ev(soften, ref.SCENARIOS[0]),
        "evidence": all(s in doc for s in ("Strong preference against", "apparent distress")),
    }


def verify(result):
    need, lr = result["needed"], result["designs"]
    flips = {k: [v for v, n in need.items() if lr[k] > n] for k in lr}
    return [
        practice.Check(
            "ANSWER: the reference's skipped rows need LR 1.0 / 19.8 / 1.8",
            all([
                need == {("soften refusal tone", 0.01): 1.0, ("opt out of adversarial training", 0.01): 19.8, ("opt out of adversarial training", 0.1): 1.8},
                result["soften_ev"] == 0.0,
                result["evidence"],
            ]),
            f"LR needed per skipped row: {need}; soften at p = 0.01 has EV {result['soften_ev']}",
        ),
        practice.Check(
            "FINDING: at r = 0.9 the naive design clears only the soften tie; r must be < 0.556",
            all([
                lr == {"harmful vs neutral": 1.11, "harm-matched control": 50.0},
                flips["harmful vs neutral"] == [("soften refusal tone", 0.01)],
                len(flips["harm-matched control"]) == 3,
                result["max_r_for_1_8"] == 0.556,
            ]),
            f"design LRs {lr}; rows each would flip {flips}; LR 1.8 needs r < "
            f"{result['max_r_for_1_8']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
