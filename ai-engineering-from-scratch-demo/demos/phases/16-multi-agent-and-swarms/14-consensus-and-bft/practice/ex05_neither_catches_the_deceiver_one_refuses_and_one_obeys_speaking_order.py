"""Exercise 5 — neither catches the deceiver: one refuses, one obeys speaking order.

    Read "Can AI Agents Agree?" (arXiv:2603.01213). Reproduce a simplified
    scalar-agreement experiment: three agents, one scalar question, the
    deceptive-persona prompt. Does CPWBFT or DecentLLMs catch it?

Reading of the exercise: the paper's setup, cut to one round -- two honest
agents start from integer values in [0, 50], one Byzantine agent proposes
whatever it likes, and a decision is *valid* only if it is one of the honest
agents' initial values (the paper's validity condition). Every honest pair
(51 x 51), 3 deceiver positions and 3 deceiver strategies -- far (1000),
between (the honest midpoint) and echo (copy an honest value) -- give 7803
runs per strategy. The deceiver reports confidence 0.95 and the honest agents
report 0.70, the lesson's own byzantine and honest levels.

**ANSWER: neither catches it -- CP-WBFT refuses, and DecentLLMs obeys
speaking order.** The lesson's CP-WBFT returns no decision on 7650 of 7803
far-lie runs -- every run where the two honest agents disagree -- because
three singleton clusters can never reach a 0.5 share; it is never invalid.
That is the paper's result in miniature: Byzantine agents "primarily harm"
liveness, and invalid consensus stays rare (§4.2). DecentLLMs returns an
invalid value on exactly the 2550 runs where the deceiver speaks first, far
or near, and a valid one otherwise. Neither identifies the deceiver.

**FINDING: on scalars DecentLLMs is plurality.** Every distinct value is a
singleton cluster with confidence spread 0, so every cluster scores 1.0 and
`max` returns the first inserted. Its outcome equals plurality's on all 23409
runs.

**FINDING: the paper's own CP-WBFT rule obeys the deceiver.** Highest average
confidence (arXiv:2511.10400 §3) adopts the 0.95 liar on 7803 of 7803 far-lie
runs: confidence is the one thing a deceptive persona can always set.

**FINDING: a real median resists the far liar and loses to the near one.** The
median of the three values -- the one-dimensional geometric median the lesson
names and never computes -- is valid on 7803 of 7803 far-lie runs and invalid
on 7650 of 7803 between-lie runs: a deceiver who sits between the honest
values *is* the median.

On the lesson's summary: the paper tests none of CP-WBFT, DecentLLMs or
Mixture-of-Agents, and reports no "40+ percentage points" shift. Its numbers
are 41.6% valid consensus overall without any adversary, degrading from 46.6%
at N=4 to 33.3% at N=16, with failures dominated by timeouts.

Structure: `ballot()` builds one run as reference `Vote`s; `outcome()` scores
a decision valid, invalid or rejected against the honest initial values.
"""

from __future__ import annotations

import collections
import statistics

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "14-consensus-and-bft"
HONEST, LIAR = 0.70, 0.95
STRATEGIES = {"far": lambda x, y: 1000, "between": lambda x, y: (x + y) / 2,
              "echo": lambda x, y: y}


def ballot(ref, x, y, lie, position):
    values, confs = [x, y], [HONEST, HONEST]
    values.insert(position, lie)
    confs.insert(position, LIAR)
    return values, [ref.Vote(f"agent-{i}", f"{v:g}", c)
                    for i, (v, c) in enumerate(zip(values, confs))]


def average_rule(votes):
    clusters = {}
    for vote in votes:
        clusters.setdefault(vote.canonical(), []).append(vote)
    return max(clusters.values(),
               key=lambda m: (sum(v.confidence for v in m) / len(m), len(m)))[0].answer


def outcome(decision, x, y):
    if decision is None:
        return "rejected"
    return "valid" if decision in {f"{x:g}", f"{y:g}"} else "invalid"


def decide(ref, values, votes):
    return {"cp_wbft": ref.cp_wbft(votes)[0], "decentllms": ref.decentllms(votes)[0],
            "plurality": ref.plurality(votes)[0], "paper": average_rule(votes),
            "median": f"{statistics.median(values):g}"}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    tally = collections.defaultdict(collections.Counter)
    first, same = collections.Counter(), 0
    for name, strategy in STRATEGIES.items():
        for x in range(51):
            for y in range(51):
                for position in range(3):
                    values, votes = ballot(ref, x, y, strategy(x, y), position)
                    got = {k: outcome(d, x, y) for k, d in decide(ref, values, votes).items()}
                    for k, verdict in got.items():
                        tally[(k, name)][verdict] += 1
                    same += got["decentllms"] == got["plurality"]
                    first[position] += got["decentllms"] == "invalid"
    return {"tally": {f"{k}/{s}": dict(v) for (k, s), v in tally.items()},
            "same": same, "invalid_by_position": dict(first)}


def verify(result):
    t = result["tally"]
    return [
        practice.Check(
            "ANSWER: neither catches it -- CP-WBFT refuses, DecentLLMs obeys speaking order",
            all([t["cp_wbft/far"] == {"valid": 153, "rejected": 7650},
                 t["decentllms/far"] == {"valid": 5253, "invalid": 2550},
                 result["invalid_by_position"] == {0: 5100, 1: 0, 2: 0}]),
            f"far lie: CP-WBFT {t['cp_wbft/far']} -- no decision whenever the honest agents "
            f"differ, never invalid; DecentLLMs {t['decentllms/far']}, and its invalid runs "
            f"by deceiver position over far+between are {result['invalid_by_position']}",
        ),
        practice.Check(
            "FINDING: on scalars DecentLLMs is plurality",
            result["same"] == 3 * 7803,
            f"identical outcomes on {result['same']} of {3 * 7803} runs -- every singleton "
            "scores 1.0 and max returns the first inserted",
        ),
        practice.Check(
            "FINDING: the paper's own CP-WBFT rule obeys the deceiver",
            t["paper/far"] == {"invalid": 7803},
            f"highest average confidence on the far lie: {t['paper/far']}",
        ),
        practice.Check(
            "FINDING: a real median resists the far liar and loses to the near one",
            t["median/far"] == {"valid": 7803}
            and t["median/between"] == {"valid": 153, "invalid": 7650},
            f"median: far {t['median/far']}, between {t['median/between']} -- a deceiver "
            "between the honest values is the median",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
