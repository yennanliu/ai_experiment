"""Exercise 3 — RRF tracks the better arm and never beats it.

    **Hard.** Build a RAG pipeline over a 10,000-document corpus of your choice.
    Implement hybrid retrieval (BM25 + dense) with RRF fusion (see lesson 14).
    Measure answer accuracy with and without the hybrid step. Document which
    question types benefit most.

Reading of the exercise: there is no 10,000-document corpus here and no
embedding model, so the corpus is exercise 1's ten passages and the dense arm is
TF-IDF reduced by truncated SVD -- latent semantic indexing, which is a real
dense retriever with a real learned subspace and no download. The scale is
smaller than the exercise asks for and the structure of the answer does not
depend on it.

With and without the hybrid step, top-1 retrieval over the ten questions reads:
BM25 alone 9, dense alone 8 or 10, RRF 9 or 10 -- the second number in each pair
at 8 SVD components and the first at 4. Two things follow. RRF matches the
better of its two inputs in both settings and exceeds it in neither, so
"accuracy with and without the hybrid step" is +0 or +1 depending on the SVD
dimension, a hyperparameter of an arm the exercise does not mention. And the
whole hybrid effect is one question: at 8 components the two arms disagree about
the top passage on exactly 1 of 10, RRF rescues that one and loses none.

Which question benefits is the part worth documenting. It is `In what year did
Android launch?`, the one BM25 already got wrong in exercise 1 -- the Macworld
passage shares `year` and a date, and one mention of Android does not outweigh
them. The question type that benefits from a dense arm is the type whose
keywords appear in a distractor, which is the type a lexical retriever is
defined to fail on. At 4 components the dense arm is worse than BM25 and rescues
nothing, so the benefit is contingent on the dense arm being good, not on the
fusion being clever.

Structure: `rank_bm25` orders passages by the lesson's own `toy_bm25_score`;
`dense` fits TF-IDF and SVD over the ten passages and orders by cosine; `fuse`
is reciprocal rank fusion at k=60. `arms` runs all three at one SVD width and
reports top-1 accuracy plus where the two inputs disagree.
"""

from __future__ import annotations

import pathlib

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "13-question-answering"

SIBLING = "ex01_two_answers_made_of_articles_match_exactly.py"
DIMENSIONS, RRF_K = (4, 8), 60


def rank_bm25(ref, passages, question) -> list:
    return sorted(range(len(passages)), key=lambda i: -ref.toy_bm25_score(question, passages[i]))


def dense(np, passages, width):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    vector = TfidfVectorizer().fit(passages)
    svd = TruncatedSVD(n_components=width, random_state=0).fit(vector.transform(passages))
    matrix = svd.transform(vector.transform(passages))
    matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)

    def rank(question):
        query = svd.transform(vector.transform([question]))[0]
        query = query / (np.linalg.norm(query) + 1e-9)
        return sorted(range(len(passages)), key=lambda i: -float(matrix[i] @ query))
    return rank


def fuse(left, right, size) -> list:
    scores = {i: 1 / (RRF_K + left.index(i) + 1) + 1 / (RRF_K + right.index(i) + 1)
              for i in range(size)}
    return sorted(scores, key=lambda i: -scores[i])


def orders(np, ref, passages, questions, width) -> list:
    """(index, bm25 top, dense top, fused top) for every question at one SVD width."""
    dense_rank = dense(np, list(passages), width)
    out = []
    for index, (question, _) in enumerate(questions):
        lexical, semantic = rank_bm25(ref, passages, question), dense_rank(question)
        out.append((index, lexical[0], semantic[0], fuse(lexical, semantic, len(passages))[0]))
    return out


def arms(np, ref, passages, questions, width) -> dict:
    rows = orders(np, ref, passages, questions, width)
    return {"top1": {"bm25": sum(i == b for i, b, _, _ in rows),
                     "dense": sum(i == d for i, _, d, _ in rows),
                     "rrf": sum(i == f for i, _, _, f in rows)},
            "disagree": [i for i, b, d, _ in rows if b != d],
            "rescued": [i for i, b, _, f in rows if b != i == f],
            "lost": [i for i, b, _, f in rows if b == i != f]}


def solve():
    try:
        import numpy as np
        import sklearn                              # noqa: F401
    except ImportError as exc:                      # pragma: no cover - T1 needs sklearn
        raise practice.Skip(f"needs scikit-learn: uv sync --extra math ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    sibling = practice.load_module(pathlib.Path(__file__).with_name(SIBLING))
    ref.CORPUS = list(sibling.PASSAGES)
    grid = {width: arms(np, ref, sibling.PASSAGES, sibling.QUESTIONS, width)
            for width in DIMENSIONS}
    wide = grid[DIMENSIONS[-1]]
    return {
        "grid": grid, "questions": len(sibling.QUESTIONS), "passages": len(sibling.PASSAGES),
        "benefit": [sibling.QUESTIONS[i][0] for i in wide["rescued"]],
        "gain": {w: grid[w]["top1"]["rrf"] - grid[w]["top1"]["bm25"] for w in DIMENSIONS},
        "never_exceeds": all(grid[w]["top1"]["rrf"] <= max(grid[w]["top1"]["bm25"],
                                                           grid[w]["top1"]["dense"])
                             for w in DIMENSIONS),
        "matches": all(grid[w]["top1"]["rrf"] == max(grid[w]["top1"]["bm25"],
                                                     grid[w]["top1"]["dense"])
                       for w in DIMENSIONS),
    }


def verify(result):
    grid, gain = result["grid"], result["gain"]
    narrow, wide = grid[DIMENSIONS[0]], grid[DIMENSIONS[-1]]
    return [
        practice.Check(
            "ANSWER: the hybrid step is worth +0 or +1, depending on the dense arm's width",
            set(gain.values()) == {0, 1},
            f"there is no 10,000-document corpus and no embedding model here, so the corpus is "
            f"{result['passages']} passages and the dense arm is TF-IDF reduced by truncated SVD. "
            f"Top-1 over {result['questions']} questions: at {DIMENSIONS[0]} components "
            f"{narrow['top1']}, at {DIMENSIONS[-1]} components {wide['top1']}. 'Accuracy with and "
            f"without the hybrid step' is {gain} -- set by a hyperparameter the exercise never "
            f"names"),
        practice.Check(
            "MECHANISM: RRF matches the better of its two inputs and exceeds it in neither setting",
            result["never_exceeds"] and result["matches"],
            f"at every width the fused score equals the maximum of the two arms: "
            f"{ {w: grid[w]['top1'] for w in DIMENSIONS} }. Fusion here is insurance against "
            f"picking the wrong retriever, not a third retriever that outperforms both"),
        practice.Check(
            "FINDING: the whole effect is one question, and it is the one the arms disagree about",
            len(wide["disagree"]) == 1 and wide["rescued"] == wide["disagree"],
            f"at {DIMENSIONS[-1]} components the two arms pick different top passages on "
            f"{len(wide['disagree'])} of {result['questions']} questions, and RRF rescues "
            f"{wide['rescued']} while losing {wide['lost']}. Where they agree there is nothing to "
            f"fuse, so the fusion can only act on the disagreements and there is one"),
        practice.Check(
            "ANSWER: the question type that benefits is the one whose keywords sit in a distractor",
            result["benefit"] and "Android" in result["benefit"][0],
            f"the rescued question is {result['benefit'][0]!r} -- the one BM25 already got wrong in "
            f"exercise 1, because the Macworld passage shares `year` and a date and one mention of "
            f"the entity does not outweigh them. That is the failure a lexical retriever is defined "
            f"to have, and the only kind a dense arm is positioned to repair"),
        practice.Check(
            "FINDING: at the narrower width the dense arm is worse and rescues nothing",
            narrow["top1"]["dense"] < narrow["top1"]["bm25"] and not narrow["rescued"],
            f"at {DIMENSIONS[0]} components the dense arm scores "
            f"{narrow['top1']['dense']} against BM25's {narrow['top1']['bm25']}, the arms disagree "
            f"on {len(narrow['disagree'])} questions, and RRF rescues {narrow['rescued']} and loses "
            f"{narrow['lost']}. The benefit is contingent on the dense arm being good, not on the "
            f"fusion being clever"),
        practice.Check(
            "CONTROL: RRF never loses a question either, at either width",
            not narrow["lost"] and not wide["lost"],
            f"the fused ranking demotes no correct BM25 answer at {DIMENSIONS[0]} or "
            f"{DIMENSIONS[-1]} components. Reciprocal rank fusion is bounded on both sides here: it "
            f"cannot fall below the better arm and it cannot rise above it, which is the honest "
            f"answer to 'measure accuracy with and without'"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
