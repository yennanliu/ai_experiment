"""Exercise 3 — the choice moves less than one example.

    **Hard.** Train the same corpus with BPE, Unigram, and WordPiece. Measure
    downstream accuracy when using each on a small sentiment classifier. Does the
    choice move the needle by more than 1 point F1?

Reading of the exercise: `code/main.py` ships BPE only. WordPiece is one changed
expression -- score a pair by its frequency over the product of its parts'
frequencies rather than by frequency alone -- so it reuses `init_vocab`,
`pair_counts` and `merge_pair` untouched. Unigram shares no step: it starts from
a large candidate vocabulary and prunes by corpus log-likelihood. Two of the
three arms are buildable here, bracketed by the character floor and the word
ceiling.

The answer is no, and the margin is not close. Over 30 reviews scored by
leave-one-out Naive Bayes on bag-of-pieces, character-level, both algorithms at
25, 50, 100 and 200 merges, and the word-level ceiling all land inside **6.67
points of accuracy** -- two reviews out of thirty, or 7.11 points of F1 from
0.8966 to 0.9677. At matched merge counts BPE and WordPiece never differ by more
than one review, and the sign flips: WordPiece ahead at 25 and 100 merges, BPE at
the ceiling, level everywhere else. A single run reports whichever it picked.

That is not the two algorithms agreeing. Their merge lists diverge at index 0 --
BPE takes `d` + `</w>`, WordPiece takes `f` + `u` -- and share 32 merges out of
146. Two nearly disjoint tokenizers produce the same classifier.

The question is below the resolution of the experiment that would answer it. With
30 held-out items accuracy moves in steps of 1/30, or 3.33 points, so 'more than
1 point' is not a value this measurement can return; one point needs at least 100
held-out items to be on the grid at all. And the character tokenizer -- which has
learned nothing -- already scores 0.9375, so the headroom the algorithm choice
competes for is seven points wide before any algorithm is chosen.

Structure: `wordpiece` mirrors `train_bpe` with the likelihood criterion;
`pieces` encodes a review; `score` runs leave-one-out Naive Bayes and returns
accuracy and positive-class F1; `solve` evaluates both algorithms at each merge
count and at the corpus ceiling.
"""

from __future__ import annotations

import math
from collections import Counter

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "19-subword-tokenization"

REVIEWS = """1 a great film with great acting and a great ending|0 a boring film with terrible acting and a dull ending|1 wonderful acting and a wonderful warm story
0 terrible acting and a boring lifeless story|1 the story is great and the acting is wonderful|0 the story is boring and the acting is terrible
1 a delightful and wonderful little film i loved it|0 a dull and terrible little film i hated it|1 i loved the acting and loved the story too
0 i hated the acting and hated the story too|1 great pacing wonderful score and a delightful cast|0 boring pacing terrible score and a dull cast
1 delightful from start to finish i loved every scene|0 dull from start to finish i hated every scene|1 the cast is great and the story is delightful
0 the cast is boring and the story is dull|1 wonderful film great cast and i loved the ending|0 terrible film boring cast and i hated the ending
1 a great story told with wonderful patient care|0 a boring story told with terrible careless work|1 i loved this delightful and clever little film
0 i hated this dull and clumsy little film|1 great score great cast and a wonderful ending|0 boring score boring cast and a terrible ending
1 the film is delightful and the acting is great|0 the film is dull and the acting is boring|1 wonderful pacing and a story i loved completely
0 terrible pacing and a story i hated completely|1 a delightful film with great acting throughout|0 a dull film with boring acting throughout"""
ROWS = tuple((r[2:], int(r[0])) for line in REVIEWS.splitlines() for r in line.split("|"))
CORPUS = "\n".join(text for text, _ in ROWS)
STEPS = (0, 25, 50, 100, 200)


def wordpiece(ref, num_merges):
    """`train_bpe` with the likelihood criterion in place of raw frequency."""
    vocab, merges = ref.init_vocab(ref.word_counts(CORPUS)), []
    for _ in range(num_merges):
        pairs = ref.pair_counts(vocab)
        if not pairs:
            break
        unigram = Counter()
        for symbols, freq in vocab.items():
            for symbol in symbols:
                unigram[symbol] += freq
        best = max(pairs, key=lambda p: (pairs[p] / (unigram[p[0]] * unigram[p[1]]), p))
        merges.append(best)
        vocab = ref.merge_pair(vocab, best)
    return merges


def pieces(ref, text, merges):
    """The bag of subword pieces for one review."""
    return {p for w in ref.word_counts(text).elements() for p in ref.encode_bpe(w, merges)}


def score(ref, merges):
    """Leave-one-out Naive Bayes over bag-of-pieces: accuracy and positive F1."""
    feats = [pieces(ref, text, merges) for text, _ in ROWS]
    labels = [label for _, label in ROWS]
    hits = Counter()
    for i in range(len(ROWS)):
        seen = {0: Counter(), 1: Counter()}
        for j in range(len(ROWS)):
            if j != i:
                seen[labels[j]].update(feats[j])
        shared = set(seen[0]) | set(seen[1])
        best = max((0, 1), key=lambda c: sum(
            math.log((seen[c][f] + 1) / (sum(seen[c].values()) + len(shared)))
            for f in feats[i] & shared))
        hits[(best, labels[i])] += 1
    tp, fp, fn = hits[(1, 1)], hits[(1, 0)], hits[(0, 1)]
    return (round((tp + hits[(0, 0)]) / len(ROWS), 4),
            round(2 * tp / (2 * tp + fp + fn), 4) if tp else 0.0)


def compare(rows, bpe, wp, ceiling):
    """How the two arms differ -- in scores, in leader, and in the merge lists themselves."""
    won = {n: max(("bpe", "wordpiece"), key=lambda a: rows[n][a][1]) for n in rows}
    accs = [rows[n][a][0] for n in rows for a in rows[n]]
    return {
        "acc_spread": round((max(accs) - min(accs)) * 100, 2),
        "gaps": {n: round(abs(rows[n]["bpe"][0] - rows[n]["wordpiece"][0]) * 100, 2) for n in rows},
        "leader": {n: won[n] if rows[n]["bpe"][1] != rows[n]["wordpiece"][1] else "tie" for n in rows},
        "diverge": next(i for i, (a, b) in enumerate(zip(bpe, wp)) if a != b),
        "first": (bpe[0], wp[0]),
        "shared": len(set(bpe[:ceiling]) & set(wp)),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    ceiling = len(ref.train_bpe(CORPUS, 10**6)[0])
    rows = {n: {"bpe": score(ref, ref.train_bpe(CORPUS, n)[0] if n else []),
                "wordpiece": score(ref, wordpiece(ref, n))} for n in (*STEPS, ceiling)}
    f1s = sorted({row[a][1] for row in rows.values() for a in row})
    return {
        "n": len(ROWS),
        "grid": round(100 / len(ROWS), 2),
        "ceiling": ceiling,
        "rows": rows,
        "spread": round((f1s[-1] - f1s[0]) * 100, 2),
        "floor": rows[0]["bpe"][1],
        "best": f1s[-1],
        **compare(rows, ref.train_bpe(CORPUS, 200)[0], wordpiece(ref, 200), ceiling),
    }


def verify(result):
    rows, gaps = result["rows"], result["gaps"]
    return [
        practice.Check(
            "ANSWER: no -- the whole range is under two examples wide",
            round(result["acc_spread"] / result["grid"]) <= 2,
            f"character-level, both algorithms at {STEPS[1:]} merges and the word-level ceiling "
            f"all land inside {result['acc_spread']} points of accuracy -- "
            f"{round(result['acc_spread'] / result['grid'])} of {result['n']} reviews",
        ),
        practice.Check(
            "MECHANISM: at matched merge counts they differ by at most one review",
            max(gaps.values()) <= result["grid"],
            f"BPE against WordPiece by merge count differs by {list(gaps.values())} points of "
            f"accuracy, topping out at {max(gaps.values())} -- one review out of {result['n']}",
        ),
        practice.Check(
            "FINDING: and the sign flips, so a single run reports a coin toss",
            len(set(result["leader"].values())) > 2,
            f"the leader by merge count is {result['leader']}. Running this experiment once, at "
            "whichever vocabulary size you picked, produces a confident answer with no stable "
            "quantity behind it",
        ),
        practice.Check(
            "MECHANISM: that is not the two algorithms agreeing",
            result["diverge"] == 0 and result["shared"] < result["ceiling"] // 2,
            f"the merge lists differ from the first entry -- BPE takes {result['first'][0]}, "
            f"WordPiece takes {result['first'][1]} -- and share {result['shared']} merges of "
            f"{result['ceiling']}. Two almost disjoint tokenizers give the same classifier",
        ),
        practice.Check(
            "FINDING: one point of F1 is below the grid this experiment runs on",
            result["grid"] > 1.0,
            f"with {result['n']} held-out items accuracy moves in steps of {result['grid']} "
            "points, so 'more than 1 point' is not a value the measurement can return -- one "
            "point needs 100 held-out items to be on the grid at all",
        ),
        practice.Check(
            "CONTROL: a tokenizer that learned nothing is already within six points of the best",
            result["floor"] > 0.85,
            f"zero merges is character-level and scores {result['floor']} F1 against "
            f"{result['best']} for the best of the six -- on a task whose signal is a handful of "
            f"repeated words, the headroom is {result['spread']} points before anything is chosen",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
