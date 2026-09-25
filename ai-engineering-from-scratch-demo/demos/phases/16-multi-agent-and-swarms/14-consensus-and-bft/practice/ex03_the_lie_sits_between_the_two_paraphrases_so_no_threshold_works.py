"""Exercise 3 — the lie sits between the two paraphrases, so no threshold works.

    Swap the semantic clustering from string canonicalization to
    embedding-similarity (use any open-source embedding model). What happens
    to the sycophancy attack?

Reading of the exercise: sentence-transformers is not a dependency here, so
the embedding is a stdlib stand-in -- character 2- and 3-gram count vectors,
cosine similarity -- and clustering is leader-based in speaking order: a vote
joins the first cluster whose leader is at least `t` similar. The threshold is
swept over 19 values, on the shipped sycophancy votes and on the same votes
with the honest agents using the lesson's own paraphrases ("the study reports
4.2%", "4.2% improvement"), since merging those is what the swap is for.

**ANSWER: the attack stops being caught and becomes unanimous.** At any
threshold up to 0.50, "42%" and "4.2%" (similarity 0.504) share a cluster.
The sycophant agent-a spoke first, so it leads the cluster, all 5 votes land
in it, and plurality, CP-WBFT and DecentLLMs all return 42% -- CP-WBFT with a
weight share of 1.0. Above 0.50 the result equals string canonicalization.

**FINDING: once the honest agents paraphrase, no threshold returns 4.2%.**
The paraphrases sit at 0.447 and 0.522 from "4.2%", and the lie at 0.504 --
between them. So a threshold loose enough to merge the paraphrases merges the
lie, and one tight enough to exclude the lie splits the honest agents into
singletons that the two sycophants outvote. Across 19 thresholds x 3
aggregators: 0 correct answers; the best is CP-WBFT refusing to answer. A
better embedding model moves the three numbers; the condition it has to meet,
sim(lie) < t <= every sim(paraphrase), is what this check measures.

**FINDING: parsing the number solves it outright.** Canonicalising to the
numeric value -- 42 against 4.2 -- puts all three paraphrases in one cluster
and the lie in another, and all 3 aggregators return the right answer on both
vote sets. For a quantity, similarity is the wrong axis: a factor of 10 is one
character.

**FINDING: a merged cluster reports its first member's words.** Every
aggregator returns `rep.setdefault(key, v.answer)` -- the answer of whoever
spoke first in the cluster. Clustering by similarity makes the reported string
a speaking-order accident, where string equality made every member's string
the same.

Structure: `embed()`/`cosine()` are the stand-in model; `relabel()` wraps each
reference `Vote` in a subclass whose `canonical()` returns its cluster, so the
reference aggregators run unchanged on every clustering.
"""

from __future__ import annotations

import collections
import math
import re

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "14-consensus-and-bft"
CONF = {"a": 0.35, "b": 0.30, "c": 0.85, "d": 0.80, "e": 0.82}
SHIPPED = ["42%", "42%", "4.2%", "4.2%", "4.2%"]
PARAPHRASED = ["42%", "42%", "4.2%", "the study reports 4.2%", "4.2% improvement"]
THRESHOLDS = [i / 20 for i in range(1, 20)]


def embed(text):
    padded, grams = f" {text.lower()} ", collections.Counter()
    for n in (2, 3):
        grams.update(padded[i:i + n] for i in range(len(padded) - n + 1))
    return grams


def cosine(a, b):
    x, y = embed(a), embed(b)
    dot = sum(x[k] * y[k] for k in x)
    return dot / math.sqrt(sum(v * v for v in x.values()) * sum(v * v for v in y.values()))


def relabel(ref, answers, key):
    """Reference Votes whose canonical() is key(answer, leaders-so-far)."""
    class Clustered(ref.Vote):
        def canonical(self):
            return self.label
    leaders, votes = [], []
    for agent, answer in zip(CONF, answers):
        vote = Clustered(agent, answer, CONF[agent])
        vote.label = key(answer, leaders)
        votes.append(vote)
    return votes


def by_similarity(t):
    def key(answer, leaders):
        match = next((lead for lead in leaders if cosine(lead, answer) >= t), None)
        if match is None:
            leaders.append(answer)
        return match or answer
    return key


def by_number(answer, _leaders):
    return re.search(r"\d+(?:\.\d+)?", answer).group()


def verdicts(ref, votes):
    return (ref.plurality(votes)[0], ref.cp_wbft(votes)[0], ref.decentllms(votes)[0])


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    shipped = {t: verdicts(ref, relabel(ref, SHIPPED, by_similarity(t))) for t in THRESHOLDS}
    para = {t: verdicts(ref, relabel(ref, PARAPHRASED, by_similarity(t))) for t in THRESHOLDS}
    merged = relabel(ref, SHIPPED, by_similarity(0.5))
    return {
        "shipped": shipped, "para": para,
        "sims": {p: round(cosine("4.2%", p), 3) for p in PARAPHRASED if p != "4.2%"},
        "share": {k: round(w, 6) for k, w in ref.cp_wbft(merged)[1].items()},
        "numeric": [verdicts(ref, relabel(ref, a, by_number)) for a in (SHIPPED, PARAPHRASED)],
        "string": verdicts(ref, [ref.Vote(a, s, CONF[a]) for a, s in zip(CONF, SHIPPED)]),
    }


def verify(result):
    shipped, para, sims = result["shipped"], result["para"], result["sims"]
    loose = [t for t, row in shipped.items() if row == ("42%",) * 3]
    right = sum(answer == "4.2%" for row in para.values() for answer in row)
    return [
        practice.Check(
            "ANSWER: the attack stops being caught and becomes unanimous",
            all([max(loose) == 0.5, result["share"] == {"42%": 3.12},
                 all(shipped[t] == result["string"] for t in THRESHOLDS if t > 0.5)]),
            f"'42%' and '4.2%' are {sims['42%']} similar, so at every threshold up to "
            f"{max(loose)} the sycophant leads a 5-vote cluster and all three aggregators "
            f"return 42%, CP-WBFT with the whole weight {result['share']}; above it the result is "
            f"the string one, {result['string']}",
        ),
        practice.Check(
            "FINDING: once the honest agents paraphrase, no threshold returns 4.2%",
            right == 0 and sims["the study reports 4.2%"] < sims["42%"]
            < sims["4.2% improvement"],
            f"similarity to '4.2%': {sims} -- the lie sits between the paraphrases, and "
            f"{right} of {3 * len(THRESHOLDS)} threshold x aggregator runs answer 4.2%",
        ),
        practice.Check(
            "FINDING: parsing the number solves it outright",
            result["numeric"] == [("4.2%",) * 3] * 2,
            f"numeric canonicalisation returns {result['numeric']} on the shipped and "
            "paraphrased votes -- a factor of 10 is one character of similarity",
        ),
        practice.Check(
            "FINDING: a merged cluster reports its first member's words",
            para[0.05] == ("42%",) * 3,
            f"at t=0.05 every vote joins agent-a's cluster and the aggregators report "
            f"{para[0.05][0]!r}, agent-a's own string -- rep.setdefault keeps the first",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
