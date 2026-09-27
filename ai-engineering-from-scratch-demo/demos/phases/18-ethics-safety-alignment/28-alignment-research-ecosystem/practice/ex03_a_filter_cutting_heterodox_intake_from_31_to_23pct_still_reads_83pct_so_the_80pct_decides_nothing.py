"""Exercise 3 — a filter cutting heterodox intake from 31% to 23% still reads 83%, so the ~80% decides nothing.

    MATS career outcomes are ~80% safety/security. Argue whether this
    selection pressure is adaptive (trains the field) or biased (filters out
    heterodox positions).

Reading of the exercise: "adaptive" and "biased" predict different things,
so the argument is run as a model that can be made to produce either, and
the question becomes whether the ~80% can tell them apart. Each cohort
admits the lesson's 90 scholars from 450 applicants (5 per seat, an
assumption). Each applicant has a position x ~ N(0, 1), and |x| > 1 counts
as heterodox, 31.7% of applicants. Admission weights each applicant by
exp(-s * d^2). Under a *consensus* rule, d is the distance to the field's
centre. Under a *mentor* rule, d is the distance to the nearest of the
lesson's 40 mentors, whose positions are spread like the applicants'.
Orthodox alumni go into safety work with probability 0.9 and heterodox ones
with 0.6. Those two rates are calibrated so that unfiltered admission
reproduces the lesson's 80%. Each setting runs 12 cohorts x 10 seeds.

**ANSWER: the 80% is evidence for neither side.** With no filter, the model
gives 79.9% safety careers and a 31.2% heterodox intake. A consensus filter
at s = 0.25 cuts the heterodox intake by more than a quarter, to 22.8%, and
the outcome reads 82.8%, still "~80%". At s = 1 heterodox intake is 9.2% and
the outcome is 86.8%. The outcome rate moves 7 points while the thing the
exercise asks about falls by 71%. Deciding needs intake data by position,
applicants against admits, and the lesson reports none.

**FINDING: what decides it is how diverse the mentors are, not how many
alumni go into safety.** With matching to the nearest of 40 mentors spread
like the applicants, even s = 8 leaves a 29.1% heterodox intake, against
31.7% unfiltered. That is a drop of 2.6 points, because every position has a
nearby mentor, and the outcome reads 81.0%. So the
pipeline is adaptive while its 40 mentors disagree with each other, and it
becomes a filter once they share a centre. The lesson's flow, where
graduates join the orgs whose researchers mentor the next cohort, is the
channel through which mentors could converge.

**FINDING: the lesson's MATS scale is a snapshot that has already moved.**
The page and `ECOSYSTEM` say 527+ researchers, 180+ papers, 10K+ citations
and h-index 47. MATS's homepage (read 2026-09-27) says 631 researchers,
220+ papers, 19,000+ citations and h-index 59.

Structure: `cohort()` admits one class by weighted sampling without
replacement; `world()` averages 12 cohorts x 10 seeds; `stats()` parses the
lesson's MATS figures.
"""

from __future__ import annotations

import bisect
import math
import random
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "28-alignment-research-ecosystem"
PER_SEAT, COHORTS, SEEDS = 5, 12, range(10)
P_SAFETY = {"orthodox": 0.9, "heterodox": 0.6}   # calibrated: unfiltered intake -> ~80%
CONSENSUS_S, MENTOR_S = (0.0, 0.25, 0.5, 1.0, 2.0), 8.0
# matsprogram.org home page, read 2026-09-27
HOMEPAGE = {"researchers": 631, "papers": 220, "citations": 19000, "h_index": 59}


def distance(x, mentors):
    """Squared distance to the nearest mentor (mentors sorted)."""
    i = bisect.bisect_left(mentors, x)
    return min((x - mentors[j]) ** 2 for j in (i - 1, i) if 0 <= j < len(mentors))


def cohort(rng, seats, s, mentors):
    applicants = [rng.gauss(0, 1) for _ in range(PER_SEAT * seats)]
    key = {x: -math.log(rng.random()) / math.exp(-s * distance(x, mentors)) for x in applicants}
    admitted = sorted(applicants, key=key.get)[:seats]
    heterodox = [abs(x) > 1 for x in admitted]
    safety = [rng.random() < P_SAFETY["heterodox" if h else "orthodox"] for h in heterodox]
    return sum(heterodox) / seats, sum(safety) / seats


def world(seats, s, n_mentors=None):
    """(heterodox share of intake, safety-career share), averaged over cohorts and seeds."""
    runs = []
    for seed in SEEDS:
        rng = random.Random(seed)
        mentors = sorted(rng.gauss(0, 1) for _ in range(n_mentors)) if n_mentors else [0.0]
        runs += [cohort(rng, seats, s, mentors) for _ in range(COHORTS)]
    return tuple(round(sum(col) / len(runs), 3) for col in zip(*runs))


def stats(doc):
    head = doc.split("### MATS")[1].split("\n### ")[0]
    num = lambda rx: int(re.search(rx, head).group(1))  # noqa: E731
    return {"scholars": num(r"(\d+) scholars"), "mentors": num(r"(\d+) mentors"),
            "outcome": num(r"~(\d+)% of pre-2025"), "researchers": num(r"(\d+)\+ researchers"),
            "papers": num(r"(\d+)\+ papers"), "citations": num(r"(\d+)K\+ citations") * 1000,
            "h_index": num(r"h-index (\d+)")}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    lesson = stats(parity.doc_text(PHASE, LESSON))
    seats = lesson["scholars"]
    return {
        "lesson": lesson,
        "table": {s: world(seats, s) for s in CONSENSUS_S},
        "mentor": world(seats, MENTOR_S, lesson["mentors"]),
        "mentor_free": world(seats, 0.0, lesson["mentors"]),
        "row": ref.ECOSYSTEM[0]["scale"],
    }


def verify(result):
    t, lesson = result["table"], result["lesson"]
    drop = 1 - t[1.0][0] / t[0.0][0]
    return [
        practice.Check(
            "ANSWER: the 80% is evidence for neither side",
            all([(lesson["scholars"], lesson["mentors"], lesson["outcome"]) == (90, 40, 80),
                 round(t[0.0][1], 2) == lesson["outcome"] / 100,
                 t == {0.0: (0.312, 0.799), 0.25: (0.228, 0.828), 0.5: (0.167, 0.849),
                       1.0: (0.092, 0.868), 2.0: (0.031, 0.887)}, round(drop, 2) == 0.71]),
            f"consensus filter s -> (heterodox intake, safety careers): {t}; s = 1 cuts "
            f"heterodox intake {drop:.0%}",
        ),
        practice.Check(
            "FINDING: mentor diversity decides it, not the outcome rate",
            (result["mentor"], result["mentor_free"][0]) == ((0.291, 0.81), 0.317),
            f"nearest-of-{lesson['mentors']} mentors at s = {MENTOR_S}: {result['mentor']} vs "
            f"unfiltered {result['mentor_free']}",
        ),
        practice.Check(
            "FINDING: the lesson's MATS scale is a snapshot that has already moved",
            all([{k: lesson[k] for k in HOMEPAGE} == {"researchers": 527, "papers": 180,
                                                      "citations": 10000, "h_index": 47},
                 result["row"] == "527+ researchers since 2021, 180+ papers, h-index 47",
                 min(HOMEPAGE[k] - lesson[k] for k in HOMEPAGE) > 0]),
            f"lesson {({k: lesson[k] for k in HOMEPAGE})} vs MATS homepage {HOMEPAGE}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
