"""Exercise 1 — the recipe in the exercise computes half the loss.

    Implement InfoNCE for a batch of 4 pairs by hand. Construct the 4x4
    similarity matrix, run softmax, pick out the diagonal, compute
    cross-entropy. Verify your Python implementation against this hand
    calculation.

Reading of the exercise: the 4x4 matrix is written down as small integers
rather than derived from embeddings, because "verify against this hand
calculation" only means something if the hand calculation is reproducible on
paper. The matrix is built around one generic caption -- text 0 scores 4
against every image, the way "a photo" would -- so that the two directions of
the loss disagree, which is what the exercise's own recipe hides.

**ANSWER: the hand calculation and the lesson agree to 0.0.** Row softmax
diagonals are 0.9479, 0.2619, 0.2619, 0.2619; their cross-entropies 0.0535,
1.3397, 1.3397, 1.3397; averaged with the column direction the loss is
**0.734559**, which is `infonce_loss` to the last bit.

**FINDING: the exercise's recipe is 38.6% above the lesson's loss.** "Pick out
the diagonal, compute cross-entropy" is the image-to-text direction alone --
1.0181. `infonce_loss` averages rows *and* columns, and the column direction
scores 0.4510. Half a symmetric loss is not a symmetric loss.

**FINDING: the two directions disagree about which pairs are right.** At argmax
the rows retrieve **1 of 4** correctly and the columns **3 of 4**: rows 1-3 all
rank the generic caption above their own, while every text but the generic one
ranks its own image first. The averaged number hides a 1-vs-3 split.

**FINDING: the generic caption sits exactly at chance in its own column.**
Column 0 is uniform at 4.0, so its term is log 4 = **1.386294** -- the most a
4-way softmax can charge, and the whole dynamic range a batch of 4 has. The
lesson's own aligned fixture at tau=0.07 scores **6.2e-06**, 0.00045% of that
range, while CLIP's 32,768 batch has 10.397 nats to work with.

Structure: `HAND` is the labelled matrix, `row_terms` and `column_terms` are
the two directions computed from softmax rather than from the lesson's
log-sum-exp, and `retrieved` counts argmax hits in each direction.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "02-clip-contrastive-pretraining"
HAND = ((4.0, 0.0, 0.0, 0.0),
        (4.0, 3.0, 0.0, 0.0),
        (4.0, 0.0, 3.0, 0.0),
        (4.0, 0.0, 0.0, 3.0))
CLIP_BATCH, TAUS = 32768, (0.07, 0.1, 1.0)


def softmax(row):
    shift = max(row)
    weights = [math.exp(value - shift) for value in row]
    total = sum(weights)
    return [weight / total for weight in weights]


def row_terms(matrix):
    """-log of each row's diagonal softmax probability."""
    return [-math.log(softmax(row)[i]) for i, row in enumerate(matrix)]


def column_terms(matrix):
    columns = [[row[j] for row in matrix] for j in range(len(matrix))]
    return [-math.log(softmax(column)[j]) for j, column in enumerate(columns)]


def strict_hit(scores, target):
    """An argmax hit only if the winner is unique -- a tie retrieves nothing."""
    return scores[target] == max(scores) and scores.count(max(scores)) == 1


def retrieved(matrix):
    """Strict argmax hits row-wise and column-wise -- the two retrieval directions."""
    columns = [[row[j] for row in matrix] for j in range(len(matrix))]
    return (sum(strict_hit(row, i) for i, row in enumerate(matrix)),
            sum(strict_hit(col, j) for j, col in enumerate(columns)))


def aligned_batch(ref, tau):
    """The lesson's own 4-pair fixture, rebuilt exactly as demo_infonce does."""
    images = [ref.make_fake_embedding(i) for i in range(4)]
    texts = [[x + 0.05 * ref.make_fake_embedding(i + 100)[k] for k, x in enumerate(vec)]
             for i, vec in enumerate(images)]
    return ref.infonce_loss(ref.similarity_matrix(images, texts, tau=tau))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    matrix = [list(row) for row in HAND]
    rows, columns = row_terms(matrix), column_terms(matrix)
    by_hand = (sum(rows) + sum(columns)) / (2 * len(matrix))
    hits = retrieved(matrix)
    return {
        "diagonal_probs": [round(softmax(row)[i], 6) for i, row in enumerate(matrix)],
        "row_terms": [round(term, 6) for term in rows],
        "row_only": sum(rows) / len(matrix), "column_only": sum(columns) / len(matrix),
        "by_hand": by_hand, "lesson": ref.infonce_loss(matrix),
        "agreement": abs(by_hand - ref.infonce_loss(matrix)),
        "excess_pct": round((sum(rows) / len(matrix) / by_hand - 1) * 100, 1),
        "row_hits": hits[0], "column_hits": hits[1],
        "chance": math.log(len(matrix)),
        "generic_column": [row[0] for row in matrix],
        "generic_term": round(columns[0], 6),
        "clip_chance": math.log(CLIP_BATCH),
        "aligned": {tau: aligned_batch(ref, tau) for tau in TAUS},
    }


def verify(result):
    aligned = result["aligned"]
    return [
        practice.Check(
            "ANSWER: the hand calculation and the lesson agree to 0.0",
            all([result["agreement"] == 0.0, round(result["by_hand"], 6) == 0.734559,
                 result["diagonal_probs"] == [0.947915, 0.261927, 0.261927, 0.261927],
                 result["row_terms"] == [0.05349, 1.339689, 1.339689, 1.339689]]),
            f"row softmax diagonals {result['diagonal_probs']} give cross-entropies "
            f"{result['row_terms']}; averaged with the column direction the loss is "
            f"{result['by_hand']:.6f}, and infonce_loss returns "
            f"{result['lesson']:.6f} -- a difference of {result['agreement']}",
        ),
        practice.Check(
            "FINDING: the exercise's recipe is 38.6% above the lesson's loss",
            all([round(result["row_only"], 4) == 1.0181,
                 round(result["column_only"], 4) == 0.451,
                 result["excess_pct"] == 38.6]),
            f"'pick out the diagonal, compute cross-entropy' is the image-to-text direction "
            f"alone -- {result['row_only']:.4f} -- while infonce_loss averages it with the "
            f"column direction's {result['column_only']:.4f}. The recipe is "
            f"{result['excess_pct']}% high because half a symmetric loss is not one",
        ),
        practice.Check(
            "FINDING: the two directions disagree about which pairs are right",
            all([result["row_hits"] == 1, result["column_hits"] == 3]),
            f"at argmax the rows retrieve {result['row_hits']} of 4 and the columns "
            f"{result['column_hits']} of 4: rows 1-3 rank the generic caption above their "
            "own text, while every text but the generic one ranks its own image first. The "
            "averaged loss hides a 1-vs-3 split, and column 0's four-way tie retrieves "
            "nothing at all",
        ),
        practice.Check(
            "FINDING: the generic caption sits exactly at chance in its own column",
            all([result["generic_column"] == [4.0, 4.0, 4.0, 4.0],
                 round(result["chance"], 6) == 1.386294,
                 result["generic_term"] == 1.386294,
                 round(aligned[0.07] / result["chance"] * 100, 5) == 0.00045,
                 round(result["clip_chance"], 3) == 10.397]),
            f"column 0 is uniform at 4.0, so its term is log 4 = {result['chance']:.6f} -- "
            f"the most a 4-way softmax can charge and the whole range a batch of 4 has. The "
            f"lesson's own aligned fixture at tau=0.07 scores {aligned[0.07]:.2e}, "
            f"{aligned[0.07] / result['chance'] * 100:.5f}% of it, while a "
            f"{CLIP_BATCH:,} batch has {result['clip_chance']:.3f} nats",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
