"""Exercise 2 — a matching clause leaves no lab guarded in 27% of rounds, not 0.1%, and pays only if the defector is 2x riskier.

    The competitor-adjustment clause is in all three frameworks (2025+). Write
    one paragraph arguing for it; write one paragraph arguing against. Identify
    the assumption each position depends on.

Reading of the exercise: each paragraph is backed by the smallest model that
can hold it. The labs are the reference's own three (`len(LABS)`). Each lab
defects (ships without safeguards) independently with p = 0.1 per round. A
lab's residual risk is its intrinsic riskiness r times (1 - safeguard level).
A lab that keeps safeguards while a peer ships without them loses a fraction
w = 0.5 of its market share to the defector. Two clauses are compared. A
*matching* clause is the lesson's reading, "reduce requirements when a
competitor defects", down to zero. A *PF-style* clause keeps half the
safeguard (m = 0.5), standing in for PF v2's condition that safeguards stay
"more protective" than the defector's.

**ANSWER, for:** with no clause, the guarded labs hand share to the defector,
so the frontier ends up with the least careful lab. Staying in the race with
partial safeguards beats ceding it. **Assumption: share really moves
(w > 0), and the defector is riskier than the lab it displaces by more than
(1 - m) / w.** At w = 0.5 that is 2.0x for a matching clause and 1.0x for the
PF-style one; at w = 0.1 it is 10x and 5x.

**ANSWER, against:** a clause that keys each lab's safeguards to its peers
turns one lab's defection into everyone's. Enumerating the 8 outcomes, no lab
is guarded in 0.1% of rounds without the clause and in 27.1% with a matching
clause, 271x more often. **Assumption: defection is exogenous, and peers
really do condition on each other**, so the clause spreads defection rather
than deterring it.

**FINDING: the reference collapses three different provisions into "yes".**
All three `adjustment_clause` cells start with "yes", and Anthropic's reads
"peer-ship reduction allowed". In the primary texts, read 2026-09-27:
- PF v2 may reduce requirements, under four conditions: confirm the risk
  landscape changed, say so publicly, judge overall risk not meaningfully
  higher, and stay more protective.
- RSP v3.0 Appendix A only ratchets up. It delays when Anthropic is in the
  lead, matches competitors with strong safety measures, and adopts their
  better mitigations. It has no reduction trigger.
- FSF v3.0 has no clause. It lists as one safety-case factor whether another
  deployed model at the same CCL has weaker mitigations.

**FINDING: how deep the reduction goes decides the argument, and the lesson
does not say.** Halving the cut (m = 0.5) halves the riskiness ratio the
"for" side needs at every w.

Structure: `p_none_guarded()` enumerates outcomes; `frontier_risk()` is the
one-defector share model; `crossover()` bisects for the break-even riskiness
ratio.
"""

from __future__ import annotations

import itertools
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "18-frontier-safety-frameworks-rsp-pf-fsf"
P_DEFECT, SHARE_LOSS, PF_KEEP = 0.1, 0.5, 0.5
SWEEP = (0.1, 0.25, 0.5, 1.0)


def p_none_guarded(n, p, clause):
    """P(every lab ships unguarded) over all 2^n defect/hold outcomes."""
    total = 0.0
    for outcome in itertools.product((True, False), repeat=n):
        prob = 1.0
        for defects in outcome:
            prob *= p if defects else 1 - p
        guarded = [not d and not (clause and any(outcome)) for d in outcome]
        total += prob * (not any(guarded))
    return total


def frontier_risk(n, ratio, keep, w, clause):
    """Share-weighted residual risk when one of n labs defects (defector riskiness = ratio)."""
    if clause:                             # everyone ships, the others keep `keep` of safeguards
        return ratio / n + (n - 1) / n * (1 - keep)
    return (1 + (n - 1) * w) / n * ratio   # guarded labs are safe but cede share w


def crossover(n, keep, w):
    """Smallest defector riskiness ratio at which the clause lowers frontier risk."""
    lo, hi = 0.0, 100.0
    for _ in range(60):
        mid = (lo + hi) / 2
        pays = frontier_risk(n, mid, keep, w, True) < frontier_risk(n, mid, keep, w, False)
        lo, hi = (lo, mid) if pays else (mid, hi)
    return round(hi, 3)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    doc = parity.doc_text(PHASE, LESSON)
    n = len(ref.LABS)
    return {
        "labs": n,
        "clauses": [lab["adjustment_clause"] for lab in ref.LABS],
        "no_clause": round(p_none_guarded(n, P_DEFECT, False), 4),
        "matching": round(p_none_guarded(n, P_DEFECT, True), 4),
        "match_x": {w: crossover(n, 0.0, w) for w in SWEEP},
        "pf_x": {w: crossover(n, PF_KEEP, w) for w in SWEEP},
        "critics": bool(re.search(r"Critics argue .*race to the bottom", doc)),
        "defenders": bool(re.search(r"Defenders argue .*less safety-conscious", doc)),
    }


def verify(result):
    r = result
    ratio = round(r["matching"] / r["no_clause"])
    return [
        practice.Check(
            "ANSWER, for: it pays only if the defector is (1 - m) / w riskier; m = 0.5 halves the bar",
            (r["defenders"], r["match_x"], r["pf_x"]) == (True, {0.1: 10.0, 0.25: 4.0, 0.5: 2.0, 1.0: 1.0},
                                                         {0.1: 5.0, 0.25: 2.0, 0.5: 1.0, 1.0: 0.5}),
            f"break-even defector riskiness by share loss w: matching {r['match_x']}, "
            f"PF-style {r['pf_x']}",
        ),
        practice.Check(
            "ANSWER, against: one defection becomes everyone's",
            (r["critics"], r["labs"], r["no_clause"], r["matching"], ratio) == (True, 3, 0.001, 0.271, 271),
            f"{r['labs']} labs, p = {P_DEFECT}: no lab guarded {r['no_clause']:.1%} without "
            f"a clause, {r['matching']:.1%} with a matching one ({ratio}x)",
        ),
        practice.Check(
            "FINDING: the reference collapses three different provisions into 'yes'",
            [c.split("; ") for c in r["clauses"]] == [["yes", "peer-ship reduction allowed"],
                                                       ["yes", "Leadership may reduce requirements"],
                                                       ["yes", "added 2025"]],
            f"adjustment_clause cells {r['clauses']}; PF v2 reduces under four conditions, "
            "RSP v3.0 Appendix A only raises, FSF v3.0 has a safety-case factor",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
