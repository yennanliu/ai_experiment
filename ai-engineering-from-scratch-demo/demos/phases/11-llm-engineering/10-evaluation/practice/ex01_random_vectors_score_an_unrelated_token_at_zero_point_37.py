"""Exercise 1 — random vectors score an unrelated token at 0.37, and cover a third of the corpus.

    **Add BERTScore.** Implement a simplified BERTScore using word embedding
    cosine similarity. Create a dictionary of 100 common words mapped to random
    50-dimensional vectors. Compute the pairwise cosine similarity matrix
    between reference and hypothesis tokens. Use greedy matching (each
    hypothesis token matches its most similar reference token) to compute
    precision, recall, and F1.

Reading of the exercise: built exactly as specified -- 100 common words, 50
dimensions, Gaussian entries from a fixed seed, greedy max-similarity matching
in both directions -- and then scored against the lesson's own `rouge_l_score`
and `word_overlap_score` on its own test suite, because "add a metric" is only
useful if it says something the existing two do not.

**ANSWER: it works, and on random vectors it is exact match with a floor.**
Two different words have mean cosine -0.002 with a standard deviation of 0.143,
so a matched pair is 1.000 and an unmatched pair is noise -- but greedy matching
takes the *maximum* over the reference tokens, and the maximum of twenty draws
from that distribution averages 0.416. Every hypothesis token, related or not,
is credited two fifths of a match.

**FINDING: 100 common words cover a third of the corpus.** Of the 200 tokens in
the suite's inputs and references, 66 are in the dictionary. The other 134 --
"self-attention", "unsupervised", "cardiovascular" -- have no vector, and every
scheme for handling them (skip, zero, random) changes the score.

**FINDING: the metric it adds says what the lesson's own metric says.** Across
the 8 test cases and 3 models, the random-embedding F1 correlates 0.787 with
`word_overlap_score` -- most of the signal, plus the floor's noise.

**MECHANISM: the floor is a property of the dimension, not of the words.**
Cosines between independent 50-dimensional Gaussians concentrate around 0 with
sd 1/sqrt(50) = 0.141 -- the measured 0.143. Raising the dimension lowers the
floor; adding meaning to the vectors is the only thing that raises the ceiling.

**CONTROL: give the vectors structure and the metric separates.** Placing
related words near each other takes the F1 of a paraphrased hypothesis from
0.534 to 0.983, while `word_overlap_score` scores it 0.333 -- it can only see
the two shared stopwords. That gap is what BERTScore is for, and random vectors
cannot produce it.

Structure: `embeddings` builds the dictionary, `bert_score` is the greedy
matcher, and `structured` is the control embedding.
"""

from __future__ import annotations

import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "10-evaluation"
DIMENSIONS, VOCAB_SIZE, SEED = 50, 100, 0
COMMON = """the of and to a in is it you that he was for on are with as i his they be at one have this
from or had by hot but some what there we can out other were all your when up use how said an each she
which do their time if will way about many then them would write like so these her long make thing see
him two has look more day could go come did my sound no most number who over know water than call
first people may down side been now find any new work part take get place made live where after""".split()
SYNONYMS = (("good", "great"), ("answer", "response"))
PARAPHRASE = ("the answer is good", "the response is great")


def embeddings(seed=SEED):
    """100 common words, the four the paraphrase control needs among them."""
    extra = [w for pair in SYNONYMS for w in pair]
    words = [w for w in COMMON if w not in extra][:VOCAB_SIZE - len(extra)] + extra
    generator = random.Random(seed)
    return {w: [generator.gauss(0, 1) for _ in range(DIMENSIONS)] for w in words}


def structured(base):
    """The control: synonyms placed near each other instead of independently."""
    near = {s: [0.8 * a + 0.2 * b for a, b in zip(base[w], base[s])] for w, s in SYNONYMS}
    return {**base, **near}


def cosine(a, b):
    norms = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / norms if norms else 0.0


def tokens(text, vectors):
    return [w for w in (t.strip(".,()?:;").lower() for t in text.split()) if w in vectors]


def bert_score(reference, hypothesis, vectors):
    """Greedy matching in both directions: precision, recall, F1."""
    ref, hyp = tokens(reference, vectors), tokens(hypothesis, vectors)
    if not ref or not hyp:
        return 0.0, 0.0, 0.0
    p = statistics.mean(max(cosine(vectors[h], vectors[r]) for r in ref) for h in hyp)
    r = statistics.mean(max(cosine(vectors[x], vectors[h]) for h in hyp) for x in ref)
    return round(p, 4), round(r, 4), round(2 * p * r / (p + r), 4) if p + r else 0.0


def correlation(left, right):
    ml, mr = statistics.mean(left), statistics.mean(right)
    cov = sum((a - ml) * (b - mr) for a, b in zip(left, right))
    spread = math.sqrt(sum((a - ml) ** 2 for a in left) * sum((b - mr) ** 2 for b in right))
    return round(cov / spread, 3) if spread else 0.0


def floor_of(vectors, rng):
    """What an unrelated hypothesis token scores after greedy matching."""
    words = list(vectors)
    return statistics.mean(max(cosine(vectors[w], vectors[r])
                               for r in rng.sample(words, 20)) for w in words)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "eval_framework")
    vectors, suite = embeddings(), ref.build_test_suite()
    return {"vocab": len(vectors), "dimensions": DIMENSIONS,
            "expected_sd": round(1 / math.sqrt(DIMENSIONS), 4),
            "floor": round(floor_of(vectors, random.Random(1)), 4),
            **geometry(vectors), **against_lesson(ref, vectors, suite),
            "paraphrase_random": bert_score(*PARAPHRASE, vectors)[2],
            "paraphrase_structured": bert_score(*PARAPHRASE, structured(vectors))[2],
            "paraphrase_overlap": ref.word_overlap_score(*PARAPHRASE),
            "identical": bert_score("the answer", "the answer", vectors)[2]}


def geometry(vectors):
    words = list(vectors)
    pairs = [cosine(vectors[a], vectors[b]) for i, a in enumerate(words) for b in words[i + 1:]]
    return {"mean_pair": round(statistics.mean(pairs), 4),
            "sd_pair": round(statistics.pstdev(pairs), 4)}   # sd is 1/sqrt(dimensions)


def against_lesson(ref, vectors, suite):
    rows = [(case, ref.run_model(model, case.input_text)) for case in suite
            for model in ("gpt-4o", "baseline-v1", "baseline-v2")]
    corpus = [w for case in suite
              for w in tokens(f"{case.reference_output} {case.input_text}", vectors)]
    total = sum(len(f"{c.reference_output} {c.input_text}".split()) for c in suite)
    return {"coverage": round(len(corpus) / total, 3), "in_vocab": len(corpus),
            "tokens": total,
            "correlation": correlation(
                [bert_score(c.reference_output, o, vectors)[2] for c, o in rows],
                [ref.word_overlap_score(c.reference_output, o) for c, o in rows])}


def verify(result):
    return [
        practice.Check(
            "ANSWER: on random vectors it is exact match with a 0.42 floor",
            all([abs(result["mean_pair"]) < 0.01, result["floor"] > 0.3,
                 result["identical"] == 1.0]),
            f"two different words have mean cosine {result['mean_pair']} with sd "
            f"{result['sd_pair']}, and an identical pair scores {result['identical']}. "
            f"Greedy matching takes the maximum over the reference tokens, and the maximum "
            f"of twenty such draws averages {result['floor']}: every hypothesis token is "
            "credited two fifths of a match whether it belongs there or not",
        ),
        practice.Check(
            "FINDING: 100 common words cover a third of the corpus",
            all([result["coverage"] < 0.4, result["vocab"] == VOCAB_SIZE]),
            f"of the {result['tokens']} tokens in the suite's inputs and references, "
            f"{result['in_vocab']} are in the {result['vocab']}-word dictionary, coverage "
            f"{result['coverage']}. Skip, zero and random are three different metrics",
        ),
        practice.Check(
            "FINDING: the metric it adds says what the lesson's own metric says",
            result["correlation"] > 0.7,
            f"across the 8 test cases and 3 models the random-embedding F1 correlates "
            f"{result['correlation']} with `word_overlap_score` -- most of the signal plus "
            "the floor's noise, from a metric already in the file",
        ),
        practice.Check(
            "MECHANISM: the floor is a property of the dimension, not of the words",
            abs(result["sd_pair"] - result["expected_sd"]) < 0.01,
            f"cosines between independent {result['dimensions']}-dimensional Gaussians "
            f"concentrate around 0 with sd 1/sqrt({result['dimensions']}) = "
            f"{result['expected_sd']}; measured {result['sd_pair']}. Dimension lowers the "
            "floor; only meaning raises the ceiling",
        ),
        practice.Check(
            "CONTROL: give the vectors structure and the metric separates",
            all([result["paraphrase_structured"] > result["paraphrase_random"],
                 result["paraphrase_overlap"] < result["paraphrase_random"]]),
            f"on {PARAPHRASE}, random vectors give F1 {result['paraphrase_random']} and "
            f"synonym-aware ones {result['paraphrase_structured']}, against "
            f"{result['paraphrase_overlap']} for word overlap, which sees only the shared "
            "stopwords. That gap is the point of BERTScore",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
