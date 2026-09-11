"""Exercise 1 — identical documents rarely score one.

    **Easy.** Implement `cosine_similarity(doc_vec_a, doc_vec_b)` on the
    L2-normalized TF-IDF output. Verify that identical documents score 1.0 and
    disjoint-vocabulary documents score 0.0.

Reading of the exercise: `cosine_similarity` already exists in the lesson's
`code/main.py`, as `sum(x * y for x, y in zip(a, b))` -- a dot product, correct
only on the pre-normalized input the exercise stipulates. So the exercise is
really two claims to check, and they do not fare alike. Over 120 corpora built
from a fixed pool, each containing a duplicated document, `s == 1.0` holds
exactly 21 times: the other 99 land one or two ULPs short. That is not a bug in
the lesson, it is what a float dot product does, and it means the first check
has to be a tolerance check -- written as equality it fails five times in six.
A true cosine, dividing by both norms, is exact on 88 of the same 120, and the
gap that matters is elsewhere: on un-normalized input the lesson's dot product
is off by as much as 0.723 while the true cosine stays inside 2.22e-16, because
one of them is scale-invariant and the other is not.

The second claim is exactly true, and true for a reason the exercise does not
mention. Every idf here is at least 1.0 -- `log((n+1)/(df+1)) + 1` bottoms out
at 1 when a term is in every document -- so no weight is ever zero and every
tf-idf entry is non-negative. A zero dot product therefore means no shared
index and nothing else: over the same corpora, all 109 zero pairs are
token-disjoint. Under the textbook `log(n/df)`, which does zero a universal
term, 368 of 477 zero pairs share tokens. The property the exercise asks you to
verify is a consequence of the smoothing.

Structure: `POOL` is 10 fixed documents; `corpora` yields every 3-subset with
document 0 duplicated, which is the identical pair each check reads. `dot` is
the lesson's own function, `cosine` the two-norm version, and `textbook` is the
unsmoothed idf used only as the ablation for the second claim.
"""

from __future__ import annotations

import itertools
import math

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "02-bag-of-words-tfidf"

POOL = ("the cat sat on the mat", "the dog sat on the mat", "a bird flew over the roof",
        "the cat ran across the room", "rain fell on the quiet street",
        "she read the long report", "the server returned an error",
        "tokens map to integer ids", "the model reads only numbers",
        "he walked home in the dark")
SIZE = 3
ARMS = ("dot/norm", "dot/raw", "cos/norm", "cos/raw")

norm = lambda vec: math.sqrt(sum(x * x for x in vec))                            # noqa: E731
corpora = lambda: [[POOL[c[0]]] + [POOL[i] for i in c]                           # noqa: E731
                   for c in itertools.combinations(range(len(POOL)), SIZE)]


def cosine(a, b) -> float:
    """The two-norm version: scale-invariant, unlike a bare dot product."""
    scale = norm(a) * norm(b)
    return sum(x * y for x, y in zip(a, b)) / scale if scale else 0.0


def prepare(ref, raw) -> tuple:
    docs = [ref.tokenize(text) for text in raw]
    return docs, ref.bag_of_words(docs, ref.build_vocab(docs))


def textbook(ref, bow) -> list:
    """idf = log(n/df): zero for a term in every document, which the lesson's never is."""
    idf = [math.log(len(bow) / d) if d else 0.0 for d in ref.document_frequency(bow)]
    rows = []
    for row in bow:
        total = sum(row)
        rows.append([c / total * i if total else 0.0 for c, i in zip(row, idf)])
    return rows


def measure(ref, raw) -> dict:
    """One corpus: the identical pair under four arms, and every zero pair."""
    docs, bow = prepare(ref, raw)
    unit, plain = ref.l2_normalize(ref.tfidf(bow)), ref.tfidf(bow)
    tokens = [set(doc) for doc in docs]
    pairs = list(itertools.combinations(range(len(raw)), 2))
    return {"dot/norm": ref.cosine_similarity(unit[0], unit[1]),
            "dot/raw": ref.cosine_similarity(plain[0], plain[1]),
            "cos/norm": cosine(unit[0], unit[1]), "cos/raw": cosine(plain[0], plain[1]),
            "zero": [not (tokens[i] & tokens[j]) for i, j in pairs
                     if ref.cosine_similarity(unit[i], unit[j]) == 0.0],
            "tb_zero": [bool(tokens[i] & tokens[j]) for i, j in pairs
                        if ref.cosine_similarity(*[ref.l2_normalize(textbook(ref, bow))[k]
                                                   for k in (i, j)]) == 0.0]}


def degenerate(ref) -> dict:
    """Two ways to a wrong answer with no error: a zero row, and a truncating zip."""
    empty = ref.l2_normalize(ref.tfidf(prepare(ref, ["", "", "the dog ran"])[1]))
    short = ref.l2_normalize(ref.tfidf(prepare(ref, ["cat sat", "dog ran"])[1]))
    wide = ref.l2_normalize(ref.tfidf(prepare(ref, ["cat sat mat hat bat", "dog ran far"])[1]))
    return {"empty": ref.cosine_similarity(empty[0], empty[1]),
            "widths": (len(short[0]), len(wide[0])),
            "truncated": ref.cosine_similarity(short[0], wide[0])}


def tally(runs, key) -> dict:
    return {"exact": {a: sum(run[a] == 1.0 for run in runs) for a in ARMS},
            "worst": {a: max(abs(run[a] - 1.0) for run in runs) for a in ARMS}}[key]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    runs = [measure(ref, raw) for raw in corpora()]
    zeros = [run["zero"] for run in runs]
    textbook_zeros = [run["tb_zero"] for run in runs]
    return dict(degenerate(ref), exact=tally(runs, "exact"), worst=tally(runs, "worst"),
                n=len(runs), idf=ref.inverse_document_frequency([3, 2, 1], 3),
                zero=sum(map(len, zeros)), zero_disjoint=sum(map(sum, zeros)),
                tb_zero=sum(map(len, textbook_zeros)), tb_shared=sum(map(sum, textbook_zeros)))


def verify(result):
    exact, worst, n = result["exact"], result["worst"], result["n"]
    return [
        practice.Check(
            "ANSWER: identical documents score 1.0 exactly in 21 of 120 corpora, not in all of them",
            exact["dot/norm"] == 21 and worst["dot/norm"] < 1e-15,
            f"over {n} corpora each carrying a duplicated document, the lesson's cosine_similarity "
            f"returns exactly 1.0 on {exact['dot/norm']} of them and lands short on "
            f"{n - exact['dot/norm']}, by at most {worst['dot/norm']:.3g} -- one or two ULPs. The "
            f"first check the exercise names has to be written as a tolerance, or it fails 5 times "
            f"in 6 on correct code"),
        practice.Check(
            "MECHANISM: the shipped function is a dot product, so it is not scale-invariant",
            worst["dot/raw"] > 0.5 and worst["cos/raw"] < 1e-15,
            f"`sum(x * y for x, y in zip(a, b))` earns the name only on the pre-normalized input the "
            f"exercise stipulates. Handed the lesson's own un-normalized `tfidf()` output it is off "
            f"by up to {worst['dot/raw']:.3g} on an identical pair, while dividing by both norms "
            f"stays inside {worst['cos/raw']:.3g}. Normalizing first is a precondition, not a habit"),
        practice.Check(
            "FINDING: the two-norm version is exact 4x more often, and neither is always exact",
            exact["cos/norm"] > 4 * exact["dot/norm"] / 1.1 and exact["cos/norm"] < n,
            f"exactly 1.0 on {exact['cos/norm']}/{n} against {exact['dot/norm']}/{n} for the dot "
            f"product, because dot/(|a| |b|) divides a sum of squares by sqrt(S)*sqrt(S) rather than "
            f"trusting the normalization to have landed on 1. It is still not {n}/{n}: no float "
            f"cosine is"),
        practice.Check(
            "FINDING: disjoint scores 0.0 is exactly true, and true because idf is never zero",
            result["zero"] == result["zero_disjoint"] > 0 and min(result["idf"]) >= 1.0,
            f"`log((n+1)/(df+1)) + 1` bottoms out at {min(result['idf']):.1f} for a term in every "
            f"document, so every weight is positive and every tf-idf entry non-negative. A zero dot "
            f"product then means no shared index and nothing else: all {result['zero']} zero pairs "
            f"across the {n} corpora are token-disjoint"),
        practice.Check(
            "CONTROL: under the textbook idf the same claim is false 368 times",
            result["tb_shared"] > 0.7 * result["tb_zero"],
            f"swap in idf = log(n/df), which zeroes a term present in every document, and "
            f"{result['tb_shared']} of {result['tb_zero']} zero pairs share tokens -- two documents "
            f"agreeing on nothing but universal words also score 0.0. The property the exercise says "
            f"to verify is a consequence of the smoothing, not of cosine similarity"),
        practice.Check(
            "FINDING: two ways to get a wrong answer with no error at all",
            result["empty"] == 0 and result["truncated"] > 0,
            f"a pair of identical empty documents scores {result['empty']}, not 1.0: their rows are "
            f"all-zero, and `l2_normalize` maps a zero norm to zeros. And `zip` truncates silently, "
            f"so vectors of width {result['widths'][0]} and {result['widths'][1]} from two different "
            f"corpora return {result['truncated']:.6f} instead of raising -- the same shape bug a "
            f"numpy dot would catch"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
