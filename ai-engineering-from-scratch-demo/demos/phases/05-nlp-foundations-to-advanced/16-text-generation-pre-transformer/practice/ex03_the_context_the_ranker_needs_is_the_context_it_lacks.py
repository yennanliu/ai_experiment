"""Exercise 3 — the context the ranker needs is the context it lacks.

    **Hard.** Build a trigram spell corrector: given a misspelled word and its
    context, generate corrections and rank by context probability under the LM.
    Evaluate on the Birkbeck spelling corpus (public).

Reading of the exercise: the Birkbeck corpus is not in this checkout, so the
evaluation set is generated from a corpus of confusable words -- cat, cot, cut,
hat, hit, had, hid, bat, ball, rat, ran -- by corrupting one character of a word
into a non-word that still has three or more real neighbours at edit distance
one. That construction matters: on a corpus without confusable words the
candidate set has one member and no ranker is consulted at all, which is a way of
scoring 1.0 on this exercise without building anything.

Across 25 such cases, averaging 3.68 candidates each, ranking by unigram
frequency scores 0.3200 and ranking by bigram probability under the lesson's
Kneser-Ney scores 0.7200. The language model earns its 10 extra corrections, and
where it earns them is the point. On the 19 cases whose correct bigram appeared
in training it is right 16 times against frequency's 7; on the 6 whose bigram did
not, 2 against 1. The advantage is 0.4737 on contexts the model has seen and
0.1667 on contexts it has not.

That bound is structural rather than a matter of corpus size. A language model
ranks a correction by how ordinary the corrected phrase is, and a phrase it has
never seen is not ordinary to it -- so it has an opinion where the correction was
already reachable and abstains where it was not. This is the reason n-gram spell
correction is normally reported with a unigram back-off: the back-off is not a
detail, it is the half of the system that handles the hard cases.

The lesson provides a bigram model where the exercise asks for a trigram one.
Conditioning on two previous tokens instead of one moves cases from the seen
column to the unseen one, so the same experiment at higher order has a larger
share of the set in the group where the model contributes least.

Structure: `SENTENCES` is the confusable corpus, split 18/6. `edits` and
`candidates` do exhaustive edit-distance-one generation; `corrupt` builds one
evaluation case by finding a non-word neighbour of a training word with at least
`MIN_CANDIDATES` real neighbours of its own. `rank` is the two rankers.
"""

from __future__ import annotations

import collections

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "16-text-generation-pre-transformer"

SENTENCES = (  # 24 sentences of deliberately confusable words, split 18/6
    "the cat sat on the mat .", "the rat sat on the hat .", "a bat hung in the cave .",
    "the dog dug a hole .", "the log fell on the path .", "the cot stood in the corner .",
    "the man cut the rope .", "the cat ran across the room .", "the dog ran after the rat .",
    "the boy hid the ball .", "the girl had the book .", "the ball hit the wall .",
    "the cat sat on the cot .", "the rat ran under the mat .", "the dog sat on the log .",
    "the man had a hat .", "the boy cut the log .", "a bat flew over the wall .",
    "the girl hid the book .", "the cat hit the ball .", "the man ran to the door .",
    "the dog hid the bone .", "the rat hid under the cot .", "the boy had a bat .")
LETTERS, TRAIN, MIN_CANDIDATES, MIN_LEN = "abcdefghijklmnopqrstuvwxyz", 18, 3, 3


def edits(word) -> set:
    """Delete, transpose, replace and insert -- the standard distance-one neighbourhood."""
    cut = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    return ({a + b[1:] for a, b in cut if b} | {a + b[1] + b[0] + b[2:] for a, b in cut if len(b) > 1}
            | {a + c + b[1:] for a, b in cut if b for c in LETTERS}
            | {a + c + b for a, b in cut for c in LETTERS})


def candidates(word, vocab) -> list:
    return sorted(edits(word) & vocab)


def corrupt(word, vocab) -> str | None:
    """A non-word one edit from `word` that has MIN_CANDIDATES real neighbours of its own."""
    for i in range(len(word)):
        for letter in LETTERS:
            typo = word[:i] + letter + word[i + 1:]
            options = candidates(typo, vocab) if typo not in vocab else []
            if len(options) >= MIN_CANDIDATES and word in options:
                return typo
    return None


def evaluation(docs, vocab) -> list:
    """One case per (corruptible word, preceding word) pair, over the whole corpus."""
    pairs = {(previous, word) for doc in docs for previous, word in zip(doc, doc[1:])
             if len(word) >= MIN_LEN and word in vocab}
    cases, seen = [], set()
    for previous, word in sorted(pairs):
        typo = corrupt(word, vocab)
        if typo and (typo, previous) not in seen:
            seen.add((typo, previous))
            cases.append((typo, word, previous))
    return cases


def rank(options, unigrams, kn, prev) -> dict:
    return {"frequency": max(options, key=lambda w: (unigrams[w], w)),  # unigram count, ties by word
            "context": max(options, key=lambda w: (kn(prev, w), w))}


def score(cases, vocab, unigrams, kn, bigrams) -> dict:
    buckets = {True: [0, 0, 0], False: [0, 0, 0]}
    hits, sizes = {"frequency": 0, "context": 0}, []
    for typo, right, previous in cases:
        options = candidates(typo, vocab)
        picked = rank(options, unigrams, kn, previous)
        sizes.append(len(options))
        bucket = buckets[(previous, right) in bigrams]
        bucket[0] += 1
        for index, name in enumerate(("frequency", "context")):
            hits[name] += picked[name] == right
            bucket[index + 1] += picked[name] == right
    return {"accuracy": {n: round(v / len(cases), 4) for n, v in hits.items()},
            "options": round(sum(sizes) / len(sizes), 2), "single": sum(s == 1 for s in sizes),
            "seen": buckets[True], "unseen": buckets[False],
            "edge": {label: round((row[2] - row[1]) / row[0], 4) for label, row in
                     (("seen", buckets[True]), ("unseen", buckets[False]))}}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    docs = [ref.tokenize(text) for text in SENTENCES]
    bigrams, unigrams, contexts = ref.train_bigrams(docs[:TRAIN])
    totals, follow = collections.Counter(), collections.defaultdict(set)
    for (previous, word), count in bigrams.items():
        totals[previous] += count
        follow[previous].add(word)
    unique = sum(len(seen) for seen in contexts.values())
    kn = lambda p, w: ref.kneser_ney_prob(bigrams, contexts, totals, follow, unique, p, w)  # noqa: E731
    vocab = {w for w in unigrams if w not in ("<s>", "</s>")}
    cases = evaluation(docs, vocab)
    return dict(score(cases, vocab, unigrams, kn, bigrams),
                cases=len(cases), vocab=len(vocab), train=TRAIN, held=len(docs) - TRAIN,
                recall=sum(right in candidates(typo, vocab) for typo, right, _ in cases))


def verify(result):
    accuracy, seen, unseen, edge = result["accuracy"], result["seen"], result["unseen"], result["edge"]
    return [
        practice.Check(
            "ANSWER: context ranking scores 0.7200 against frequency's 0.3200",
            accuracy["context"] > 2 * accuracy["frequency"],
            f"the Birkbeck corpus is not in this checkout, so the {result['cases']} cases are "
            f"generated by corrupting a training word into a non-word with {MIN_CANDIDATES}+ real "
            f"neighbours -- {result['options']} candidates on average over "
            f"{result['vocab']} words. Unigram frequency scores {accuracy['frequency']}, bigram "
            f"probability under the lesson's Kneser-Ney {accuracy['context']}"),
        practice.Check(
            "MECHANISM: candidate generation never fails -- every case has its answer in the set",
            result["recall"] == result["cases"] and result["single"] == 0,
            f"edit-distance-one search contains the right word {result['recall']} times in "
            f"{result['cases']} and {result['single']} cases have a single candidate. The half the "
            f"exercise phrases as the work has no errors in it, which is why the corpus had to be "
            f"built from confusable words"),
        practice.Check(
            "FINDING: the model's advantage is three times larger where it has seen the bigram",
            edge["seen"] > 2 * edge["unseen"],
            f"on the {seen[0]} cases whose bigram appeared in the {result['train']} training "
            f"sentences, context ranking is right {seen[2]} times against frequency's {seen[1]}, an "
            f"edge of {edge['seen']}; on the {unseen[0]} whose bigram did not, {unseen[2]} against "
            f"{unseen[1]}, an edge of {edge['unseen']}"),
        practice.Check(
            "MECHANISM: a phrase the model has never seen is not ordinary to it",
            unseen[2] < seen[2] and unseen[0] < seen[0],
            f"the ranker scores a correction by how ordinary the corrected phrase is. On an unseen "
            f"context Kneser-Ney falls back to its continuation term, which ranks by how many "
            f"*different* contexts a word has appeared in -- a unigram-like signal, and the "
            f"frequency ranker already was one"),
        practice.Check(
            "FINDING: so the back-off is the half of the system that handles the hard cases",
            unseen[0] > 0 and unseen[2] <= unseen[0] / 2,
            f"{unseen[0]} of {result['cases']} cases fall outside the model's evidence and it gets "
            f"{unseen[2]}. n-gram spell correction is normally reported with a unigram back-off, "
            f"and that back-off is what runs on the cases the headline is not measuring"),
        practice.Check(
            "CONTROL: the lesson provides a bigram model where the exercise asks for a trigram one",
            seen[0] + unseen[0] == result["cases"],
            f"`kneser_ney_prob` conditions on one previous token, so the split is {seen[0]} seen "
            f"against {unseen[0]} unseen. Conditioning on two moves cases from the first column to "
            f"the second, so the experiment at the order the exercise asks for puts more of the set "
            f"where the model contributes least"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
