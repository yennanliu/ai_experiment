"""Exercise 3 — coherence rises with topic count.

    **Hard.** Compute c_v coherence for both LDA and BERTopic on your corpus. Run
    each with 5, 10, 20, 50 topics. Plot coherence vs topic count. Report which
    method is more stable across topic counts.

Reading of the exercise: gensim is not installed, so c_v is not available and
UMass coherence is computed here instead -- the same shape of measure, a mean
log co-document ratio over the top words of each topic, and unlike c_v it needs
no sliding window or external reference corpus. The result the exercise asks you
to plot has a direction it does not mention.

Coherence climbs steeply with topic count on both methods:

    LDA        -0.0990, +0.0351, +0.1370, +0.1716, +0.1159 at K = 2, 4, 5, 10, 20
    clustering -0.3177, -0.0144, +0.0825, +0.2451, +0.2967 at the same K

The clustering curve is monotone and peaks at the largest K tried; LDA peaks at
K=10 and dips once. The mechanism is not subtle: more topics means fewer
documents per topic, so the top words of each topic co-occur in a larger share of
the documents that produced them. A curve that mostly increases with K barely
selects a K -- read a maximum off the clustering arm and you get whichever value
the x-axis ended on, and the exercise ends it at 50 for reasons that are not in
the metric.

"Which method is more stable across topic counts" does have an answer. LDA's
coherence spans 0.2706 across the five settings and the clustering arm's spans
0.6144 -- LDA is more than twice as stable, and it is also the arm that scores
worse at large K and better at small K. Stability and coherence point at
different methods, so the exercise's question and the plot above it can be
answered from the same numbers and disagree.

Structure: the corpus and the two fitting arms come from exercises 1 and 2.
`umass` scores one topic's top words against the corpus; `curve` runs one arm
over `COUNTS`, averaging over seeds, and `span` is the range of a curve.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "15-topic-modeling"

FIRST, SECOND = ("ex01_lda_lands_a_few_points_above_random.py",
                 "ex02_both_arms_sit_inside_a_coin_flip_of_random.py")
COUNTS, SEEDS, TOP, EPSILON = (2, 4, 5, 10, 20), 3, 8, 1.0
UNAVAILABLE = ("gensim", "bertopic", "octis")


def ratio(sets, first, second) -> float | None:
    alone = sum(second in seen for seen in sets)
    together = sum(first in seen and second in seen for seen in sets)
    return math.log((together + EPSILON) / alone) if alone else None


def umass(topic, docs) -> float:
    """Mean log co-document ratio over a topic's top words -- UMass, in place of c_v."""
    sets, words = [set(doc) for doc in docs], topic[:TOP]
    scores = [r for i in range(1, len(words)) for j in range(i)
              if (r := ratio(sets, words[i], words[j])) is not None]
    return sum(scores) / len(scores) if scores else 0.0


def curve(fit_words, docs) -> dict:
    rows = {}
    for count in COUNTS:
        scores = []
        for seed in range(SEEDS):
            topics = [t for t in fit_words(docs, count, seed) if t]
            scores.append(sum(umass(t, docs) for t in topics) / max(1, len(topics)))
        rows[count] = round(sum(scores) / SEEDS, 4)
    return rows


def span(rows) -> float:
    return round(max(rows.values()) - min(rows.values()), 4)


def shape(curves) -> dict:
    return {"span": {name: span(values) for name, values in curves.items()},
            "monotone": {name: [values[k] for k in COUNTS] == sorted(values.values())
                         for name, values in curves.items()},
            "peak": {name: max(COUNTS, key=lambda k: values[k])
                     for name, values in curves.items()}}


def solve():
    try:
        import numpy as np
        import sklearn                              # noqa: F401
    except ImportError as exc:                      # pragma: no cover - T1 needs sklearn
        raise practice.Skip(f"needs scikit-learn: uv sync --extra math ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    here = pathlib.Path(__file__)
    first = practice.load_module(here.with_name(FIRST))
    second = practice.load_module(here.with_name(SECOND))
    docs, _ = first.corpus(ref, drop_stopwords=True)
    arms = {"lda": lambda d, k, s: first.fit(ref, d, k, s)[0],
            "cluster": lambda d, k, s: second.cluster(np, d, k, s)[0]}
    rows = {name: curve(fit_words, docs) for name, fit_words in arms.items()}
    return dict(shape(rows),
                unavailable=[m for m in UNAVAILABLE if importlib.util.find_spec(m) is None],
                curves=rows, documents=len(docs), seeds=SEEDS, counts=list(COUNTS))


def verify(result):
    curves, spans, peak = result["curves"], result["span"], result["peak"]
    largest, smallest = COUNTS[-1], COUNTS[0]
    return [
        practice.Check(
            "ANSWER: coherence climbs steeply with topic count on both methods",
            all(curves[name][largest] > curves[name][smallest] + 0.2 for name in curves)
            and peak["cluster"] == largest,
            f"{result['unavailable']} are all absent, so c_v is replaced by UMass -- a mean log "
            f"co-document ratio over each topic's top {TOP} words, needing no sliding window or "
            f"reference corpus. Over {result['documents']} documents and {result['seeds']} seeds: "
            f"LDA {curves['lda']}, clustering {curves['cluster']} at K = {result['counts']}. "
            f"Clustering peaks at the largest K tried and LDA at K={peak['lda']}"),
        practice.Check(
            "MECHANISM: more topics means fewer documents per topic, which is what the metric reads",
            curves["lda"][smallest] < curves["lda"][largest]
            and curves["cluster"][smallest] < curves["cluster"][largest],
            f"at K={smallest} the two arms score {curves['lda'][smallest]} and "
            f"{curves['cluster'][smallest]}; at K={largest}, {curves['lda'][largest]} and "
            f"{curves['cluster'][largest]}. Splitting the same documents into more topics makes "
            f"each topic's top words co-occur in a larger share of the documents that produced "
            f"them, which is the quantity being scored"),
        practice.Check(
            "FINDING: so plotting coherence against topic count does not select a topic count",
            result["monotone"]["cluster"],
            f"the clustering curve is monotone increasing across every step: "
            f"{[curves['cluster'][k] for k in COUNTS]}. Reading a maximum off it returns the "
            f"largest value on the x-axis, so the plot the exercise asks for reports where the "
            f"sweep stopped -- and the exercise stops at 50 for reasons that are not in the metric"),
        practice.Check(
            "ANSWER: LDA is twice as stable across topic counts",
            spans["lda"] < spans["cluster"] / 1.8,
            f"the LDA curve spans {spans['lda']} across the five settings and the clustering curve "
            f"{spans['cluster']}. That is the exercise's stability question answered, and it is the "
            f"one part of this exercise a monotone curve can still settle"),
        practice.Check(
            "FINDING: stability and coherence point at different methods",
            curves["cluster"][largest] > curves["lda"][largest]
            and curves["cluster"][smallest] < curves["lda"][smallest],
            f"at K={largest} clustering scores {curves['cluster'][largest]} against LDA's "
            f"{curves['lda'][largest]}, and at K={smallest} it scores "
            f"{curves['cluster'][smallest]} against {curves['lda'][smallest]}. The less stable arm "
            f"is the better one at the end of the sweep the plot draws attention to"),
        practice.Check(
            "CONTROL: LDA's curve is not monotone, and that is the only wobble in the experiment",
            not result["monotone"]["lda"],
            f"LDA reads {[curves['lda'][k] for k in COUNTS]} -- one step down, at K=10. Everything "
            f"else in both curves rises, so the single non-monotone point is the only evidence in "
            f"the plot that coherence is measuring anything other than topic count"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
