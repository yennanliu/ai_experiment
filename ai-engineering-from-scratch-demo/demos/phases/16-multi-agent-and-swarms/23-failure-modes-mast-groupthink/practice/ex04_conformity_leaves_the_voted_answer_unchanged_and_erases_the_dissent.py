"""Exercise 4 — conformity leaves the voted answer unchanged and erases the dissent.

    Read the Groupthink paper (arXiv:2508.05687). Identify which of the five
    patterns is hardest to detect in production. Propose a proxy metric.

Reading of the exercise: "hardest to detect" is tested, not asserted -- a
pattern is hard to detect when the metrics a production system already logs
read the same with and without it. Three agents, each right with p = 0.7 and
otherwise on one of 4 wrong answers, answer first alone and then after seeing
their peers; every probability is enumerated exactly over the 125 joint
answers.

**ANSWER: conformity bias, and the proxy is the agreement gained from
exposure.** Under the conformity rule -- adopt the answer two peers share --
majority-vote accuracy is 0.784 before exposure and 0.784 after: the vote is
exactly what it was, so no accuracy dashboard can see it. What changes is the
dissent. Unanimity rises from 0.3447 to 0.8481, and the unanimous-and-wrong
rate -- answers nothing downstream would question -- from 0.0017 to 0.0641,
38x.
The proxy: log each agent's answer *before* it sees the others, and track
post-exposure minus pre-exposure agreement. It is +0.3356 under conformity
and exactly 0 for healthy agents, monoculture and easy questions alike.

**FINDING: the final agreement rate cannot separate the three.** Conformity
ends at 0.8481 agreement; monoculture drift at c = 0.6884 and independent
agents at p = 0.9201 end at the same value. The triple (pre-exposure
agreement, exposure gain, canary accuracy) separates them: conformity has a
normal pre-exposure rate and a gain, monoculture a high pre-exposure rate and
no gain, easy questions a high rate, no gain and high canary accuracy.

**FINDING: the paper names six failure modes, not five, and never says
"groupthink".** arXiv:2508.05687 is "Risk Analysis Techniques for Governed
LLM-based Multi-Agent Systems" (Reid et al.), and its sixth mode is
inter-agent communication failures -- which the lesson drops: its section
says "Five related failures" and the module's GROUPTHINK dict has 5 keys, none
of them communication. The paper does not rank detectability either.

Structure: `outcomes()` enumerates joint answers with their probabilities;
`expose()` applies the conformity rule; the monoculture and easy scenarios are
solved in closed form to match conformity's final agreement.
"""

from __future__ import annotations

import itertools

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "23-failure-modes-mast-groupthink"
P, WRONG = 0.7, 4


def prob(answer, p=P):
    return p if answer == 0 else (1 - p) / WRONG


def outcomes(p=P):
    for joint in itertools.product(range(WRONG + 1), repeat=3):
        yield joint, prob(joint[0], p) * prob(joint[1], p) * prob(joint[2], p)


def expose(joint):
    """Each agent adopts the answer its two peers share, if they share one."""
    out = list(joint)
    for i in range(3):
        a, b = (joint[j] for j in range(3) if j != i)
        out[i] = a if a == b else joint[i]
    return tuple(out)


def stats(transform=lambda j: j, p=P):
    agree = unanimous = wrong_unanimous = major = 0.0
    for joint, weight in outcomes(p):
        x = transform(joint)
        agree += weight * ((x[0] == x[1]) + (x[0] == x[2]) + (x[1] == x[2])) / 3
        unanimous += weight * (x[0] == x[1] == x[2])
        wrong_unanimous += weight * (x[0] == x[1] == x[2] != 0)
        major += weight * (sum(v == 0 for v in x) >= 2)
    return {k: round(v, 4) for k, v in
            dict(agree=agree, unanimous=unanimous, wrong=wrong_unanimous, major=major).items()}


def easy_p(target):
    """p at which independent agents agree `target`: p^2 + (1-p)^2/4 = target."""
    a, b, c = 1 + 1 / WRONG, -2 / WRONG, 1 / WRONG - target
    return (-b + (b * b - 4 * a * c) ** 0.5) / (2 * a)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    pre, post = stats(), stats(expose)
    base = pre["agree"]
    return {
        "pre": pre, "post": post, "gain": round(post["agree"] - base, 4),
        "mono_c": round((post["agree"] - base) / (1 - base), 4),
        "easy_p": round(easy_p(post["agree"]), 4),
        "easy_agree": stats(p=easy_p(post["agree"]))["agree"],
        "five": "Five related failures" in parity.doc_text(PHASE, LESSON),
        "keys": sorted(ref.GROUPTHINK),
    }


def verify(result):
    pre, post = result["pre"], result["post"]
    return [
        practice.Check(
            "ANSWER: conformity bias; the proxy is the agreement gained from exposure",
            all([pre["major"] == post["major"] == 0.784, post["wrong"] > 30 * pre["wrong"],
                 result["gain"] == 0.3356]),
            f"majority accuracy {pre['major']} before and {post['major']} after exposure; "
            f"unanimity {pre['unanimous']} -> {post['unanimous']}, unanimous-and-wrong "
            f"{pre['wrong']} -> {post['wrong']}; exposure gain +{result['gain']} under "
            "conformity and 0 for any pattern that does not revise",
        ),
        practice.Check(
            "FINDING: the final agreement rate cannot separate the three",
            result["easy_agree"] == post["agree"] and 0 < result["mono_c"] < 1,
            f"conformity ends at {post['agree']}; monoculture at c={result['mono_c']} and "
            f"independent agents at p={result['easy_p']} end at {result['easy_agree']} -- "
            "only pre-exposure agreement, gain and canary accuracy tell them apart",
        ),
        practice.Check(
            "FINDING: the lesson's five patterns drop the paper's sixth",
            result["five"] and len(result["keys"]) == 5
            and not any("comm" in k for k in result["keys"]),
            f"the lesson says 'Five related failures' and GROUPTHINK holds "
            f"{result['keys']}; arXiv:2508.05687 names six, the sixth being inter-agent "
            "communication failures",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
