"""Exercise 4 — calibration fixes overconfidence, and monoculture is not overconfidence.

    Read CP-WBFT (arXiv:2511.10400). Implement the confidence-probe
    calibration step (a separate calibration model checks each agent's
    self-reported confidence). Measure the accuracy gain on the monoculture
    scenario.

Reading of the exercise: one monoculture vote list has one outcome, so the
scenario is turned into a distribution. The three clones share a model, so
they are right together (probability a) or wrong together on 42%. The two
diverse agents are each right with probability 0.8 and wrong on distinct
answers. Self-reports are the scenario's own: 0.70/0.68/0.72 and 0.85/0.82.
The 8 outcomes are enumerated, so every accuracy is exact. The calibrator
fits P(correct | agent, reported confidence) on that labelled history and
replaces each self-report with it; aggregation is the reference `cp_wbft`.

**ANSWER: the gain is +30 points when the clones are 50% accurate, and
exactly 0 when they are as accurate as they say.**

| clone accuracy a | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---:|---:|---:|---:|
| CP-WBFT, self-reported | 0.300 | 0.500 | 0.700 | 0.900 |
| CP-WBFT, calibrated | 0.736 | 0.800 | 0.700 | 0.900 |
| clones counted as one voter | 0.736 | 0.800 | 0.864 | 0.928 |

Calibration moves the result only while 3a < 2 x 0.8, i.e. a < 0.53: it
removes overconfidence, and a monoculture is a *correlation*. At a = 0.7 each
clone is individually perfectly calibrated, three agreeing clones still carry
one draw's worth of evidence, and calibration adds nothing -- the correlated
row gains 16.4 points by fixing the right thing. On the single shipped vote
list (clones wrong, diverse right) calibration flips the verdict only for
a < 0.53.

**FINDING: uncalibrated CP-WBFT is plurality here.** 3 x 0.68 = 2.04 already
exceeds 0.85 + 0.82 = 1.67, so the self-reported row equals a at every a:
the confidences the lesson chose can never overturn the head count.

**On the paper: its probe is not the lesson's.** CP-WBFT's §3 has a
prompt-level probe -- verbalised "Confidence: [0.00-1.00]" -- and a hidden-level
one: a logistic regression on PCA-reduced hidden states. The lesson's `Vote`
carries only the first kind, and the paper aggregates by *average*
confidence, not the sum `cp_wbft` computes. The lesson's "+85.71% BFT
improvement" is the paper's fault *rate*: 6 Byzantine agents of 7.

Structure: `outcomes()` enumerates the 8 joint outcomes with their
probabilities; `calibrate()` fits per-agent accuracy from them -- the maximum-
likelihood fit of a probe whose only input is a constant self-report;
`accuracy()` scores a weighting over the distribution with reference votes.
"""

from __future__ import annotations

import itertools

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "14-consensus-and-bft"
CLONES = {"agent-a": 0.70, "agent-b": 0.68, "agent-c": 0.72}
DIVERSE = {"agent-d": (0.85, "5%"), "agent-e": (0.82, "3.9%")}
B, RIGHT, SHARED_WRONG = 0.8, "4.2%", "42%"
LEVELS = (0.3, 0.5, 0.7, 0.9)


def outcomes(a):
    """(probability, {agent: (answer, self-reported confidence)}) for all 8 cases."""
    for clones_ok, d_ok, e_ok in itertools.product((True, False), repeat=3):
        p = (a if clones_ok else 1 - a) * (B if d_ok else 1 - B) * (B if e_ok else 1 - B)
        shared = RIGHT if clones_ok else SHARED_WRONG
        votes = {n: (shared, c) for n, c in CLONES.items()}
        votes.update({name: ((RIGHT, wrong)[not ok], conf)
                      for (name, (conf, wrong)), ok in zip(DIVERSE.items(), (d_ok, e_ok))})
        yield p, votes


def calibrate(history):
    """Per-agent P(correct) on the labelled history (each agent reports one level)."""
    seen, right = {}, {}
    for p, votes in history:
        for agent, (answer, _) in votes.items():
            seen[agent] = seen.get(agent, 0.0) + p
            right[agent] = right.get(agent, 0.0) + p * (answer == RIGHT)
    return {agent: right[agent] / seen[agent] for agent in seen}


def accuracy(ref, a, weight):
    total = 0.0
    for p, votes in outcomes(a):
        ballot = [ref.Vote(n, ans, weight(n, conf)) for n, (ans, conf) in votes.items()]
        total += p * (ref.cp_wbft(ballot)[0] == RIGHT)
    return round(total, 4)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = {}
    for a in LEVELS:
        cal = calibrate(list(outcomes(a)))
        rows[a] = {
            "raw": accuracy(ref, a, lambda n, c: c),
            "calibrated": accuracy(ref, a, lambda n, c, cal=cal: cal[n]),
            "grouped": accuracy(ref, a, lambda n, c, cal=cal: cal[n] / (3 if n in CLONES else 1)),
        }
    shipped = {a: ref.cp_wbft([ref.Vote(n, SHARED_WRONG, a) for n in CLONES]
                              + [ref.Vote(n, RIGHT, B) for n in DIVERSE])[0]
               for a in (0.5, 0.53, 0.54, 0.7)}
    return {"rows": rows, "shipped": shipped,
            "clone_floor": round(3 * min(CLONES.values()), 2),
            "diverse_sum": round(sum(c for c, _ in DIVERSE.values()), 2)}


def verify(result):
    rows, shipped = result["rows"], result["shipped"]
    return [
        practice.Check(
            "ANSWER: +30 points at a = 0.5, exactly 0 when the clones are as accurate as they say",
            all([round(rows[0.5]["calibrated"] - rows[0.5]["raw"], 4) == 0.3,
                 rows[0.7]["calibrated"] == rows[0.7]["raw"] == 0.7,
                 rows[0.7]["grouped"] == 0.864,
                 shipped[0.53] == RIGHT and shipped[0.54] == SHARED_WRONG]),
            f"accuracy by clone accuracy {rows}; calibration only matters while 3a < 1.6, "
            f"and on the shipped vote list flips it for a <= 0.53 ({shipped}); counting the "
            "clones as one voter is the fix that survives a = 0.7",
        ),
        practice.Check(
            "FINDING: uncalibrated CP-WBFT is plurality here",
            all(row["raw"] == a for a, row in rows.items())
            and result["clone_floor"] > result["diverse_sum"],
            f"3 x 0.68 = {result['clone_floor']} already beats 0.85 + 0.82 = "
            f"{result['diverse_sum']}, so self-reported CP-WBFT scores exactly a at every a",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
