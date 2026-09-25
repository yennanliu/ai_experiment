"""Exercise 1 — CP-WBFT ships the monoculture answer until the clones drop below 0.56.

    Run `code/main.py`. Confirm plurality fails the monoculture attack but
    CPWBFT partially mitigates it when the monoculture confidence is below 0.7.

Reading of the exercise: "the monoculture confidence" is the confidence the
three clones report, so it is swept from 0.30 to 0.95 with the two honest
agents held at the scenario's 0.85 and 0.82, and each aggregator's answer is
read at every step -- the claim is about where CP-WBFT's answer flips.

**ANSWER: plurality fails, confirmed; CP-WBFT does not mitigate below 0.7.**
Plurality returns 42% on the monoculture scenario. So does CP-WBFT, at the
shipped confidences 0.70, 0.68 and 0.72 -- two of the three are already
below 0.7. The weights are summed, so the wrong cluster loses only when
3c < 0.85 + 0.82, that is c < 0.557: swept in steps of 0.01, CP-WBFT answers
4.2% at 0.55 and 42% at 0.56. "Below 0.7" is off by 0.14, and the mitigation
is not partial -- it is a step.

**FINDING: DecentLLMs never mitigates it, at any confidence.** Its score is
`len(cluster) * max(0, 1 - dist)`, where `dist` is the spread of the cluster's
confidences around their median. Three clones reporting the same confidence
have spread 0 and score 3.0 whether that confidence is 0.30 or 0.95; the
honest pair scores 1.94. The docstring says "scoring = confidence", but the
confidence *level* never enters -- and there is no geometric median of
anything, only a median of confidences.

**FINDING: the four scenarios cannot tell the three aggregators apart.** All
three return the same answer in all 4 scenarios -- 4.2%, 4.2%, 4.2%, 42% --
so the table the lesson presents as a comparison has three identical
columns. The claim "CPWBFT's confidence weighting mitigates sycophancy" is
true, and plurality gets sycophancy right too, 3 votes to 2.

**FINDING: CP-WBFT's threshold can never reject in any shipped scenario.**
Every scenario has exactly 2 clusters, and the larger of two weights is at
least half their sum, so `weights[winner] / total < 0.5` is unreachable. The
"[rejected below threshold]" branch of the table never prints.

**FINDING: the paper's rule gets monoculture right and the byzantine lie
wrong.** CP-WBFT (arXiv:2511.10400, §3) selects "the answer with the highest
average confidence, with supporter count as tie-breaker". On the lesson's
scenarios that rule answers 4.2% on monoculture (0.835 against 0.70) and 42%
on the byzantine lie (0.95 against 0.725). The lesson implements a sum, which
is plurality weighted by confidence -- a different aggregator.

Structure: `scenarios()` records the four vote lists `main()` builds by
replacing the module's `scenario` printer; every verdict comes from the
reference `plurality`, `cp_wbft` and `decentllms`.
"""

from __future__ import annotations

import contextlib
import io

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "14-consensus-and-bft"


def scenarios(ref):
    """{name: votes} exactly as main() passes them to scenario()."""
    seen, printer = {}, ref.scenario
    ref.scenario = lambda name, correct, votes: seen.setdefault(name, votes)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ref.main()
    finally:
        ref.scenario = printer
    return seen


def average_rule(votes):
    """arXiv:2511.10400 §3: highest average confidence, supporter count breaks ties."""
    clusters = {}
    for vote in votes:
        clusters.setdefault(vote.canonical(), []).append(vote)
    best = max(clusters.values(),
               key=lambda m: (sum(v.confidence for v in m) / len(m), len(m)))
    return best[0].answer


def sweep(ref, votes):
    """{clone confidence: (CP-WBFT answer, DecentLLMs 42% score)}."""
    out = {}
    for step in range(30, 96):
        c = step / 100
        swapped = [ref.Vote(v.agent, v.answer, c if v.answer == "42%" else v.confidence)
                   for v in votes]
        out[c] = (ref.cp_wbft(swapped)[0], round(ref.decentllms(swapped)[1]["42%"], 6))
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    runs = scenarios(ref)
    mono = runs["monoculture (correlated errors)"]
    table = {name: (ref.plurality(v)[0], ref.cp_wbft(v)[0], ref.decentllms(v)[0])
             for name, v in runs.items()}
    swept = sweep(ref, mono)
    return {
        "table": table, "swept": swept,
        "last_right": max(c for c, (ans, _) in swept.items() if ans == "4.2%"),
        "first_wrong": min(c for c, (ans, _) in swept.items() if ans == "42%"),
        "clone_conf": [v.confidence for v in mono if v.answer == "42%"],
        "decent_scores": sorted({score for _, score in swept.values()}),
        "clusters": {name: len(ref.cp_wbft(v)[1]) for name, v in runs.items()},
        "paper": {name: average_rule(v) for name, v in runs.items()},
    }


def verify(result):
    table, mono = result["table"], result["table"]["monoculture (correlated errors)"]
    paper = result["paper"]
    return [
        practice.Check(
            "ANSWER: plurality fails, confirmed; CP-WBFT does not mitigate below 0.7",
            all([mono[:2] == ("42%", "42%"), result["last_right"] == 0.55,
                 result["first_wrong"] == 0.56, min(result["clone_conf"]) < 0.7]),
            f"at the shipped clone confidences {result['clone_conf']} both return "
            f"{mono[1]}; CP-WBFT answers 4.2% up to {result['last_right']} and 42% from "
            f"{result['first_wrong']} -- the sum flips at 3c = 0.85 + 0.82, c = 0.557",
        ),
        practice.Check(
            "FINDING: DecentLLMs never mitigates it, at any confidence",
            result["decent_scores"] == [3.0] and mono[2] == "42%",
            f"with equal clone confidences the 42% cluster scores "
            f"{result['decent_scores']} at every level from 0.30 to 0.95 against the "
            "honest pair's 1.94 -- the score reads confidence spread, never level",
        ),
        practice.Check(
            "FINDING: the four scenarios cannot tell the three aggregators apart",
            all(len(set(row)) == 1 for row in table.values()),
            f"verdicts per scenario (plurality, CP-WBFT, DecentLLMs): "
            f"{ {k: v[0] for k, v in table.items()} } -- three identical columns",
        ),
        practice.Check(
            "FINDING: CP-WBFT's threshold can never reject in any shipped scenario",
            set(result["clusters"].values()) == {2} and None not in
            {row[1] for row in table.values()},
            f"clusters per scenario {list(result['clusters'].values())}; the larger of "
            "two weights is at least half their sum, so the 0.5 threshold is unreachable",
        ),
        practice.Check(
            "FINDING: the paper's rule gets monoculture right and the byzantine lie wrong",
            paper["monoculture (correlated errors)"] == "4.2%"
            and paper["byzantine lie"] == "42%",
            f"highest-average-confidence, per arXiv:2511.10400 §3, answers {paper} -- the "
            "lesson's sum is confidence-weighted plurality, a different aggregator",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
