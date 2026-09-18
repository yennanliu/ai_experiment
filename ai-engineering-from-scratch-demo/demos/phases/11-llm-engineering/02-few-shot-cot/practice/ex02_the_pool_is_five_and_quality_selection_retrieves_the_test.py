"""Exercise 2 — the pool is five, and picking well retrieves the test question.

    **Example selection experiment**: For the same 10 problems, compare random
    example selection vs hand-picked similar examples. Measure accuracy
    difference. At what point does example quality matter more than example
    quantity?

Reading of the exercise: accuracy needs a key, so what is measured is the thing
that decides it -- which examples each arm actually puts in the prompt, and what
that costs. Both arms go through the lesson's own `build_cot_prompt`, and
"hand-picked similar" is made mechanical: shared operation words between the
test question and the exemplar question, Jaccard, top three.

**ANSWER: the lesson has no selection hook.** `build_cot_prompt` takes
`examples[:num_examples]` -- a slice, not a choice -- so every "selection"
strategy is a reordering of the pool before the call. The prompt is exactly the
first k exemplars, in order.

**FINDING: the pool is five, and one of them is broken.** Exemplar #5's
reasoning is 1,356 characters against a mean of 179 for the other four, with 5
backtracking markers ("Wait", "Actually", "Hmm") and 2 question marks against 0
everywhere else. It is the only exemplar that models visible self-correction --
the exact reply shape Exercise 1 showed `extract_answer` mis-parses, since it
returns the *first* "the answer is".

**FINDING: random selection draws it 6 times in 10.** Of the ten 3-subsets of a
5-item pool, six contain #5, and those prompts average 2,566 characters against
1,098 -- 2.34x -- for the same three slots.

**FINDING: hand-picking well retrieves the test question itself.** For
TEST_QUESTIONS[4] the nearest exemplar scores similarity 1.0, because it *is*
exemplar #4, answer attached. For TEST_QUESTIONS[1] every exemplar scores 0.0 --
a rate problem with no analogue in the pool -- so the quality arm has nothing to
pick. On five examples the good strategy either leaks or abstains.

**ANSWER: quality dominates from the fourth example, and the crossover is a
property of this pool.** Quantity saturates at 5. k=4 admits the duplicate of a
test question; k=5 takes the prompt from 1,462 to 3,293 characters, +125% for
one exemplar. There is no k above 3 at which more is better.

Structure: `OPS` is the operation vocabulary, `similar` the hand-picked arm,
`subsets` the random arm enumerated exhaustively rather than sampled.
"""

from __future__ import annotations

import itertools
import statistics

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "02-few-shot-cot"
OPS = ("half", "twice", "each", "every", "per", "%", "total", "remaining", "more",
       "cost", "sell", "week", "year", "page", "hour")
MARKERS = ("Wait", "Actually", "Hmm", "No.")
CORRUPT = 4


def signature(text):
    low = text.lower()
    return {word for word in OPS if word in low}


def similar(question, examples, k=3):
    """The hand-picked arm: Jaccard over shared operation words, best k."""
    scored = []
    for index, example in enumerate(examples):
        shared = signature(question) & signature(example["question"])
        union = signature(question) | signature(example["question"])
        scored.append((len(shared) / len(union) if union else 0.0, index))
    scored.sort(reverse=True)
    return scored[:k]


def prompt_chars(ref, examples, k):
    return len(ref.build_cot_prompt("Q?", examples, k)[1])


def pool_shape(ref):
    """What is in the pool: length, backtracking, and which exemplar is the outlier."""
    lengths = [len(e["reasoning"]) for e in ref.GSM8K_EXAMPLES]
    marks = [sum(e["reasoning"].count(m) for m in MARKERS) for e in ref.GSM8K_EXAMPLES]
    others = [n for i, n in enumerate(lengths) if i != CORRUPT]
    return {"pool": len(ref.GSM8K_EXAMPLES), "lengths": lengths, "markers": marks,
            "others_mean": round(statistics.mean(others)),
            "questions": [e["reasoning"].count("?") for e in ref.GSM8K_EXAMPLES]}


def random_arm(ref):
    """Every 3-subset of the pool, not a sample of them."""
    pool = ref.GSM8K_EXAMPLES
    rows = [(s, prompt_chars(ref, [pool[i] for i in s], 3))
            for s in itertools.combinations(range(len(pool)), 3)]
    with_it = [n for s, n in rows if CORRUPT in s]
    without = [n for s, n in rows if CORRUPT not in s]
    return {"subsets": len(rows), "draws": len(with_it),
            "with_corrupt": round(statistics.mean(with_it)),
            "without_corrupt": round(statistics.mean(without)),
            "ratio": round(statistics.mean(with_it) / statistics.mean(without), 2)}


def quality_arm(ref):
    picks = [similar(t["question"], ref.GSM8K_EXAMPLES) for t in ref.TEST_QUESTIONS]
    return {"best": [round(p[0][0], 2) for p in picks],
            "identical": [i for i, p in enumerate(picks) if p[0][0] == 1.0],
            "undefined": [i for i, p in enumerate(picks) if p[0][0] == 0.0]}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "advanced_prompting")
    pool = ref.GSM8K_EXAMPLES
    rendered = ref.build_cot_prompt("Q?", pool, 3)[1]
    return {**pool_shape(ref), **random_arm(ref), **quality_arm(ref),
            "sliced": [e["question"][:20] in rendered for e in pool],
            "by_k": [prompt_chars(ref, pool, k) for k in range(1, len(pool) + 1)]}


def verify(result):
    by_k, lengths = result["by_k"], result["lengths"]
    jump = round(100 * (by_k[4] - by_k[3]) / by_k[3])
    return [
        practice.Check(
            "ANSWER: there is no selection hook -- build_cot_prompt slices",
            all([result["sliced"] == [True, True, True, False, False]]),
            f"`examples[:num_examples]` puts exactly the first three exemplars in the "
            f"prompt and leaves the other two out ({result['sliced']}). Every selection "
            "strategy is therefore a reordering of the pool before the call",
        ),
        practice.Check(
            "FINDING: the pool is five and one exemplar is the outlier on every axis",
            all([result["pool"] == 5, lengths[CORRUPT] > 4 * result["others_mean"],
                 result["markers"][CORRUPT] == 5, sum(result["markers"]) == 5]),
            f"reasoning lengths {lengths} against a mean of {result['others_mean']} for the "
            f"other four; backtracking markers {result['markers']}; question marks "
            f"{result['questions']}. Exemplar #5 is the only one that models visible "
            "self-correction -- the shape Exercise 1 showed the extractor mis-parses",
        ),
        practice.Check(
            "FINDING: random selection draws the broken exemplar 6 times in 10",
            all([result["subsets"] == 10, result["draws"] == 6, result["ratio"] > 2]),
            f"{result['draws']} of the {result['subsets']} 3-subsets of a 5-item pool "
            f"contain it, and those prompts average {result['with_corrupt']} characters "
            f"against {result['without_corrupt']} -- {result['ratio']}x for the same three "
            "slots. The variance in a random arm here is one exemplar wide",
        ),
        practice.Check(
            "FINDING: hand-picking well retrieves the test question itself",
            all([result["identical"] == [4], result["undefined"] == [1]]),
            f"best-match similarity per test question: {result['best']}. Question "
            f"{result['identical']} scores 1.0 because its nearest exemplar is itself, "
            f"answer attached; question {result['undefined']} scores 0.0 against all five, "
            "a rate problem with no analogue. The quality arm either leaks or abstains",
        ),
        practice.Check(
            "ANSWER: quality dominates from the fourth example, on this pool",
            all([len(by_k) == 5, by_k[4] > 2 * by_k[3]]),
            f"prompt characters by k: {by_k}. Quantity saturates at 5; k=4 admits the "
            f"exemplar that duplicates a test question, and k=5 costs +{jump}% "
            f"({by_k[3]} -> {by_k[4]}) for the one broken exemplar. Above k=3 there is no "
            "count at which more is better, which is a fact about the pool, not the method",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
