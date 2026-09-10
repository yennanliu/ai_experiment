"""Exercise 1 — LDA lands a few points above random.

    **Easy.** Fit LDA with 5 topics on the 20 Newsgroups dataset. Print top 10
    words per topic. Label each topic by hand. Did the algorithm find the real
    categories?

Reading of the exercise: the last question has an answer and it needs a baseline
the exercise does not mention. 20 Newsgroups is not downloadable here, so the
corpus is 24 short documents written in four categories -- finance, AI, politics,
sport -- with the labels known by construction. Assigning each document to its
highest-probability topic and scoring cluster purity, LDA at K=4 averages 0.5156
over eight seeds. Random assignment of the same 24 documents to 4 clusters
averages **0.4353**. The algorithm is three to eight points above chance, on a
corpus where the categories are as separable as they get.

The words tell a different story from the assignments, and the difference is
worth keeping. `tokenize` filters only tokens of two characters or fewer, so
`the`, `for`, `was`, `after` and `new` all survive into the vocabulary and into
the top-8 lists -- 6 to 8 of the 32 top words across four topics are function
words. Strip them and the topics become readable at a glance (`cut`, `rates`,
`stocks`, `fed`) while purity *falls*, to 0.4635. Whatever makes a topic look
like finance to a human is not what is separating the documents.

So the honest answer to "did the algorithm find the real categories" is: the
topic word lists look like the categories and the document assignments barely
beat a coin. Labelling topics by hand reads the first and reports it as the
second.

Structure: `CATEGORIES` is the labelled corpus and `STOPWORDS` the list used only
for the ablation. `fit` runs the lesson's own `collapsed_gibbs_lda` and turns the
document mixtures into hard assignments; `purity` scores an assignment against
the known labels, and `expected_purity` is the same score over random
assignments, which is the number the exercise is missing.
"""

from __future__ import annotations

import collections
import random

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "15-topic-modeling"

CATEGORIES = {
    "finance": ("stocks rose after the fed cut interest rates",
                "bond yields fell as investors bought treasuries",
                "the s p 500 hit a new high on earnings reports",
                "the central bank raised rates to slow inflation",
                "investors sold equities amid recession fears",
                "corporate earnings beat analyst estimates this quarter"),
    "ai": ("chip makers reported strong demand for ai accelerators",
           "openai released a new model with multimodal reasoning",
           "deep learning researchers published a paper on efficient attention",
           "the model was trained on a large corpus of text",
           "transformer architectures dominate modern language models",
           "gradient descent optimises the neural network weights"),
    "politics": ("the senate passed a bill on healthcare spending",
                 "the president signed new tariffs on steel imports",
                 "congress debated a tax cut for small businesses",
                 "the election campaign focused on immigration policy",
                 "lawmakers voted against the proposed budget amendment",
                 "the governor announced new education funding"),
    "sport": ("the team won the championship after extra time",
              "the striker scored twice in the second half",
              "the coach was sacked following a losing streak",
              "the tournament final drew a record television audience",
              "the athlete broke the world record in the sprint",
              "the club signed a defender from a rival league")}
STOPWORDS = frozenset("the a an and or but of to in on for with as at by from is are was were be "
                      "been this that these those it its his her their new after over into than "
                      "then".split())
TOPICS, SEEDS, ITERATIONS, TRIALS = 4, 8, 300, 2000


def corpus(ref, drop_stopwords=False) -> tuple:
    """(documents, gold category index) -- the labels are known by construction."""
    gold = [index for index, texts in enumerate(CATEGORIES.values()) for _ in texts]
    return [[w for w in ref.tokenize(text) if not (drop_stopwords and w in STOPWORDS)]
            for texts in CATEGORIES.values() for text in texts], gold


def fit(ref, docs, topics, seed) -> tuple:
    words, mixtures = ref.collapsed_gibbs_lda(docs, n_topics=topics, n_iters=ITERATIONS, seed=seed)
    return words, [max(range(topics), key=lambda k: mix[k]) for mix in mixtures]


def purity(assignment, gold) -> float:
    total = 0
    for cluster in set(assignment):
        members = [g for a, g in zip(assignment, gold) if a == cluster]
        total += collections.Counter(members).most_common(1)[0][1]
    return total / len(gold)


def expected_purity(gold, topics) -> float:
    rng = random.Random(0)
    return round(sum(purity([rng.randrange(topics) for _ in gold], gold)
                     for _ in range(TRIALS)) / TRIALS, 4)


def mean(values) -> float:
    return round(sum(values) / len(values), 4)


def scored(runs, gold) -> dict:
    values = {name: [purity(assignment, gold) for _, assignment in rows]
              for name, rows in runs.items()}
    return {"purity": {name: mean(row) for name, row in values.items()},
            "spread": {name: (round(min(row), 4), round(max(row), 4))
                       for name, row in values.items()}}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    raw_docs, gold = corpus(ref)
    clean_docs, _ = corpus(ref, drop_stopwords=True)
    runs = {name: [fit(ref, docs, TOPICS, seed) for seed in range(SEEDS)]
            for name, docs in (("raw", raw_docs), ("clean", clean_docs))}
    flat = [word for words in runs["raw"][0][0] for word in words]
    return dict(
        scored(runs, gold), random=expected_purity(gold, TOPICS),
        one_cluster=purity([0] * len(gold), gold), gold=purity(gold, gold),
        documents=len(gold), categories=len(CATEGORIES),
        stopword_share=(sum(word in STOPWORDS for word in flat), len(flat)),
        raw_topic=runs["raw"][0][0][0][:5], clean_topic=runs["clean"][0][0][0][:5],
        vocab=len({w for d in raw_docs for w in d}))


def verify(result):
    scores, spread, chance = result["purity"], result["spread"], result["random"]
    share = result["stopword_share"]
    return [
        practice.Check(
            "ANSWER: LDA averages 0.5156 purity where random assignment averages 0.4353",
            chance < scores["raw"] < chance + 0.1,
            f"20 Newsgroups is not downloadable, so the corpus is {result['documents']} documents in "
            f"{result['categories']} categories known by construction. Over {SEEDS} seeds at K="
            f"{TOPICS}, LDA scores {scores['raw']} purity against {chance} for random assignment "
            f"of the same documents to the same number of clusters -- "
            f"{scores['raw'] - chance:+.4f}"),
        practice.Check(
            "MECHANISM: the baseline is high because purity rewards any partition into four",
            result["one_cluster"] < chance < result["gold"] == 1.0,
            f"one cluster containing everything scores {result['one_cluster']}, the true partition "
            f"scores {result['gold']}, and four random clusters score {chance}. The question 'did "
            f"it find the real categories' cannot be read off a purity number without that middle "
            f"value beside it, and the exercise does not ask for it"),
        practice.Check(
            "FINDING: tokenize keeps stopwords, so they fill the top-word lists",
            share[0] > 0,
            f"`tokenize` drops only tokens of two characters or fewer, so `the`, `for`, `was` and "
            f"`after` enter a {result['vocab']}-word vocabulary. {share[0]} of the {share[1]} top "
            f"words across the four topics are function words -- the first topic reads "
            f"{result['raw_topic']}"),
        practice.Check(
            "FINDING: removing them makes the topics readable and the assignments worse",
            scores["clean"] < scores["raw"],
            f"stripped, the first topic reads {result['clean_topic']} -- recognisably finance -- and "
            f"purity falls from {scores['raw']} to {scores['clean']}, below where it started and "
            f"closer to the {chance} baseline. Whatever makes a topic look like a category to a "
            f"reader is not what is separating the documents"),
        practice.Check(
            "MECHANISM: so labelling topics by hand reads the words and reports the assignment",
            scores["clean"] < scores["raw"] and result["clean_topic"] != result["raw_topic"],
            f"the exercise's procedure is: print top words, label by hand, conclude about "
            f"categories. Those are two measurements moving in opposite directions here -- the "
            f"word lists improve as the clustering degrades -- and only one is what the final "
            f"question asks about"),
        practice.Check(
            "CONTROL: the seed moves purity nearly as far as the method does",
            spread["raw"][1] - spread["raw"][0] > (scores["raw"] - chance) / 2,
            f"across {SEEDS} seeds purity runs {spread['raw'][0]} to {spread['raw'][1]} on the raw "
            f"corpus and {spread['clean'][0]} to {spread['clean'][1]} on the stripped one. The "
            f"spread is comparable to the whole margin over random, so a single fit answers the "
            f"exercise's question with whichever seed it drew"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
