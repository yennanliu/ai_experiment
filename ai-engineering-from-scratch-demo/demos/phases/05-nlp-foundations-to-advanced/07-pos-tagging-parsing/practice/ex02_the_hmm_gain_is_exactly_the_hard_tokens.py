"""Exercise 2 — the HMM gain is exactly the hard tokens.

    **Medium.** Train the bigram HMM above and report per-tag precision/recall.
    Which tags does the HMM confuse most?

Reading of the exercise: on exercise 1's corpus the HMM scores 0.9834 against
the baseline's 0.8674 at a 30% training split, and the whole of that difference
sits in two of the three token groups. On words seen with a single tag both
score exactly 1.0000, on all 127 of them -- there is nothing for a sequence
model to add. On ambiguous words the baseline's 0.6364 becomes 0.9773, and on
out-of-vocabulary words its 0.2000 becomes 0.8000. Context buys the tokens where
the word alone is not enough and nothing else, which is the claim the lesson
makes and this is it as an arithmetic identity rather than an assertion.

Per-tag, at that split, precision and recall are above 0.96 everywhere except
ADV recall at 0.909, and the confusions are one apiece -- NOUN read as DET, VERB
as NOUN, ADV as VERB. "Which tags does it confuse most" has no strong answer on
a corpus this size, and saying so is more useful than ranking three singletons.

Two things about the decoder are worth more than the ranking. `train_hmm` counts
a transition into `<EOS>` for every sentence and `viterbi` never reads them, so
nothing stops a decode from ending on a tag that never ends a sentence: on the
lesson's own six-sentence TRAIN, `The the the` comes back `DET ADP DET`, and DET
is not among the tags any training sentence ends on. And `viterbi` starts with
`list(tags)` on a `set`, whose order Python varies between processes, while its
argmax breaks ties by first position -- so on a tie the tagger's output depends
on the run. Constructed, that fires: a two-tag corpus with identical statistics
returns whichever tag is listed first. On the lesson's own model it does not
fire, and all 5040 orderings of its seven tags agree.

Structure: the corpus, its ambiguity set and its bucketing come from exercise 1
via `practice.load_module`. `tag_report` is the per-tag precision/recall table
built from the confusion counts; `decode` runs both taggers over one split and
returns their predictions aligned to the gold tags and the bucket labels.
"""

from __future__ import annotations

import collections
import itertools
import pathlib

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "07-pos-tagging-parsing"

SIBLING = "ex01_the_baseline_measures_the_corpus.py"
SPLITS, BUCKETS = (0.3, 0.7), ("unambiguous", "ambiguous", "oov")
SYMMETRIC = [(["a"], ["X"]), (["a"], ["Y"])]


def decode(ref, sibling, rows, fraction) -> dict:
    cut = int(len(rows) * fraction)
    train_rows, test_rows = rows[:cut], rows[cut:]
    ambiguous = {w for w, tags in sibling.readings(rows).items() if len(tags) > 1}
    vocab = {w.lower() for tokens, _ in train_rows for w in tokens}
    word_best, default = ref.train_mft(train_rows)
    model = ref.train_hmm(train_rows)
    gold, mft, hmm, groups = [], [], [], []
    for tokens, tags in test_rows:
        gold += tags
        mft += ref.predict_mft(tokens, word_best, default)
        hmm += ref.viterbi(tokens, *model)
        groups += [sibling.bucket(w, vocab, ambiguous) for w in tokens]
    return {"gold": gold, "mft": mft, "hmm": hmm, "groups": groups,
            "finals": {tags[-1] for _, tags in train_rows}}


def accuracy(gold, got, groups=None, want=None) -> tuple:
    pairs = [(a, b) for a, b, g in zip(gold, got, groups or gold)
             if want is None or g == want]
    return (round(sum(a == b for a, b in pairs) / len(pairs), 4), len(pairs)) if pairs else (None, 0)


def one_tag(gold, got, tag) -> tuple:
    hit = sum(1 for a, b in zip(gold, got) if a == b == tag)
    false = sum(1 for a, b in zip(gold, got) if a != tag and b == tag)
    missed = sum(1 for a, b in zip(gold, got) if a == tag and b != tag)
    return (round(hit / (hit + false), 3) if hit + false else 0.0,
            round(hit / (hit + missed), 3) if hit + missed else 0.0, hit + missed)


def tag_report(gold, got) -> dict:
    return {tag: one_tag(gold, got, tag) for tag in sorted(set(gold))}


def lesson_model(ref) -> dict:
    """The three decoder facts, all read off the lesson's own six-sentence TRAIN."""
    model = ref.train_hmm(ref.TRAIN)
    every = list(itertools.permutations(sorted(model[2])))
    orders = {tuple(ref.viterbi(["The", "the", "the"], *model[:2], list(o), model[3]))
              for o in every}
    return {"eos": {tag: c["<EOS>"] for tag, c in sorted(model[0].items()) if c["<EOS>"]},
            "lesson_finals": sorted({tags[-1] for _, tags in ref.TRAIN}),
            "lesson_decode": ref.viterbi(["The", "the", "the"], *model),
            "orders": len(orders), "permutations": len(every),
            "tie": [ref.viterbi(["a"], *ref.train_hmm(SYMMETRIC)[:2], order, {"a"})[0]
                    for order in (["X", "Y"], ["Y", "X"])]}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    sibling = practice.load_module(pathlib.Path(__file__).with_name(SIBLING))
    runs = {fraction: decode(ref, sibling, sibling.corpus(), fraction) for fraction in SPLITS}
    low = runs[SPLITS[0]]
    return dict(
        lesson_model(ref), report=tag_report(low["gold"], low["hmm"]),
        accuracy={f: {n: accuracy(runs[f]["gold"], runs[f][n])[0] for n in ("mft", "hmm")}
                  for f in SPLITS},
        buckets={n: {b: accuracy(low["gold"], low[n], low["groups"], b) for b in BUCKETS}
                 for n in ("mft", "hmm")},
        confusions=collections.Counter(
            (a, b) for a, b in zip(low["gold"], low["hmm"]) if a != b).most_common(4))


def verify(result):
    acc, buckets, report = result["accuracy"], result["buckets"], result["report"]
    low = SPLITS[0]
    return [
        practice.Check(
            "ANSWER: per-tag precision and recall are above 0.96 except ADV recall at 0.909",
            min(p for p, _, _ in report.values()) > 0.95 and result["confusions"],
            f"at the {low:.0%} split the HMM's per-tag (precision, recall, support) is {report}, "
            f"and its mistakes are {result['confusions']} -- one apiece. 'Which tags does it confuse "
            f"most' has no strong answer on a corpus this size, and three singletons should not be "
            f"ranked"),
        practice.Check(
            "MECHANISM: on words seen with one tag the two taggers are identical at 1.0000",
            buckets["mft"]["unambiguous"] == buckets["hmm"]["unambiguous"] == (1.0, 127),
            f"both score {buckets['mft']['unambiguous'][0]} over "
            f"{buckets['mft']['unambiguous'][1]} unambiguous seen tokens. A sequence model has "
            f"nothing to add where the word already determines the tag, so the whole of the HMM's "
            f"advantage has to come from the other two groups"),
        practice.Check(
            "FINDING: and it does -- ambiguous 0.6364 -> 0.9773, out-of-vocabulary 0.2000 -> 0.8000",
            buckets["hmm"]["ambiguous"][0] > buckets["mft"]["ambiguous"][0]
            and buckets["hmm"]["oov"][0] > buckets["mft"]["oov"][0],
            f"ambiguous tokens go {buckets['mft']['ambiguous']} -> {buckets['hmm']['ambiguous']} and "
            f"unseen tokens {buckets['mft']['oov']} -> {buckets['hmm']['oov']}, lifting the headline "
            f"{acc[low]['mft']} -> {acc[low]['hmm']}. At the {SPLITS[1]:.0%} split, where nothing is "
            f"out of vocabulary, it is {acc[SPLITS[1]]['mft']} -> {acc[SPLITS[1]]['hmm']}"),
        practice.Check(
            "FINDING: train_hmm counts transitions into <EOS> and viterbi never reads them",
            result["eos"] and "DET" not in result["lesson_finals"]
            and result["lesson_decode"][-1] == "DET",
            f"the lesson's own TRAIN gives {result['eos']} as counts into <EOS>, and its sentences "
            f"end on {result['lesson_finals']}. `viterbi` takes its answer from "
            f"max(V[n-1]) with no final transition, so `The the the` decodes to "
            f"{result['lesson_decode']} -- ending on DET, which no training sentence ends on"),
        practice.Check(
            "CONTROL: the tie-break depends on set order, which Python varies between processes",
            result["tie"] == ["X", "Y"],
            f"`viterbi` opens with `list(tags)` on a set and its argmax keeps the first of equal "
            f"scores. On a two-tag corpus with identical statistics that decides the answer: tag "
            f"order ['X', 'Y'] returns {result['tie'][0]!r} and ['Y', 'X'] returns "
            f"{result['tie'][1]!r}, from the same model on the same token"),
        practice.Check(
            "CONTROL: on the lesson's own model it does not fire -- all 5040 orderings agree",
            result["orders"] == 1,
            f"running the same sentence through every one of the {result['permutations']} orderings "
            f"of the lesson's seven tags gives {result['orders']} distinct output. Exact ties do not "
            f"arise between sums of real-valued log probabilities, so the hazard is latent rather "
            f"than active -- which is the reason it survives"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
