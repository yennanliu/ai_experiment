"""Exercise 2 — two thirds of token-level masks leak from a surviving sibling.

    **Medium.** Implement whole-word masking: if a word is tokenized into
    subwords, mask all subwords together or none. Measure whether this improves
    MLM accuracy on a 500-sentence corpus.

Reading of the exercise: `whole_word_mlm` already exists in `code/main.py`, so
"implement" is read as *exercise it against the token-level path and measure what
separates them*. The measurement the exercise names cannot separate anything --
the lesson's only model is `toy_predict`, which returns a uniform distribution --
so the quantity whole-word masking was invented to remove is measured instead:
how often a masked subword still has an unmasked sibling inside its own word.
The corpus is 500 synthetic sentences with word lengths drawn 55/25/13/7 over
1-4 subwords, mean 1.71, 11,085 tokens.

**ANSWER: whole-word masking removes a 64.27% leak, and the exercise's own metric
cannot see it.** Under `create_mlm_batch`, **64.27%** of masked tokens sit in a
word that still has an unmasked subword -- the answer is spelled out beside the
question. Under `whole_word_mlm` it is **0.00%**, by construction: the branch is
drawn per span and applied to every token in it.

**FINDING: `toy_predict` makes the prescribed measurement identically
uninformative.** It returns `1/V` for every position and every vocabulary entry,
so MLM accuracy is exactly `1/V` under both schemes, for any corpus, at any mask
probability. The exercise asks whether whole-word masking improves a number that
provably cannot move.

**CONTROL: both schemes select the same share of tokens.** 14.16% token-level
against 13.89% word-level on the same corpus -- selection is drawn per word and
is independent of word length, so the token rate is unbiased and the two arms are
comparable. Whole-word masking is not harder because it masks more; it is harder
because it masks *together*.

**FINDING: the variance is what actually changes.** Masking whole words makes the
count of masked tokens per sentence lumpier, because one draw now commits up to
four positions. That is the real cost of the scheme, and it is the reason the
mask rate is the same while the gradient is noisier.

Structure: `corpus` builds the sentences and their spans; `picked` labels one
sentence under one scheme; `leak` reports rate, sibling-leak and spread.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "06-bert-masked-language-modeling"
VOCAB, SENTENCES, PROB = 200, 500, 0.15
LENGTHS, WEIGHTS = (1, 2, 3, 4), (0.55, 0.25, 0.13, 0.07)


def corpus(ref, seed=1):
    """500 sentences of subword-tokenised words, each with its (start, end) spans."""
    rng = random.Random(seed)
    out = []
    for _ in range(SENTENCES):
        tokens, spans = [ref.CLS_ID], [(0, 1)]
        for _ in range(rng.randint(8, 16)):
            start = len(tokens)
            tokens += [rng.randrange(3, VOCAB)
                       for _ in range(rng.choices(LENGTHS, weights=WEIGHTS)[0])]
            spans.append((start, len(tokens)))
        spans.append((len(tokens), len(tokens) + 1))
        tokens.append(ref.SEP_ID)
        out.append((tokens, spans))
    return out


def picked(ref, tokens, spans, whole, rng):
    """The labelled positions of each word, under one masking scheme."""
    labels = (ref.whole_word_mlm(tokens, spans, VOCAB, PROB, rng) if whole
              else ref.create_mlm_batch(tokens, VOCAB, PROB, rng))[1]
    return [[i for i in range(s, e) if labels[i] != ref.IGNORE_INDEX] for s, e in spans]


def leak(ref, data, whole, seed=9):
    """(selected share of tokens, share of masks with a surviving sibling, spread)."""
    rng = random.Random(seed)
    masked = leaked = total = 0
    counts = []
    for tokens, spans in data:
        words = picked(ref, tokens, spans, whole, rng)
        total += len(tokens)
        masked += sum(len(w) for w in words)
        leaked += sum(len(w) for w, (s, e) in zip(words, spans) if w and len(w) < e - s)
        counts.append(sum(len(w) for w in words))
    return masked / total, leaked / masked, statistics.pstdev(counts) / statistics.fmean(counts)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    data = corpus(ref)
    lengths = [e - s for _, spans in data for s, e in spans[1:-1]]
    flat = ref.toy_predict([0, 1, 2], list(range(VOCAB)))
    return {
        "token": leak(ref, data, whole=False), "word": leak(ref, data, whole=True),
        "tokens": sum(len(t) for t, _ in data), "mean_word": statistics.fmean(lengths),
        "multi": sum(1 for n in lengths if n > 1) / len(lengths),
        "uniform": len({round(p, 12) for row in flat for p in row}) == 1,
        "accuracy": flat[0][0], "vocab": VOCAB,
    }


def verify(result):
    token, word = result["token"], result["word"]
    return [
        practice.Check(
            "ANSWER: token-level masking leaks from a sibling 64% of the time; whole-word, never",
            token[1] > 0.6 and word[1] == 0.0,
            f"over {SENTENCES} sentences and {result['tokens']:,} tokens, mean word length "
            f"{result['mean_word']:.2f} with {result['multi']:.0%} of words multi-subword: "
            f"{token[1]:.2%} of create_mlm_batch's masked tokens sit in a word that still has an "
            f"unmasked subword, against {word[1]:.2%} for whole_word_mlm. The branch is drawn per "
            "span and applied to every token in it, so the leak is zero by construction",
        ),
        practice.Check(
            "FINDING: the prescribed metric cannot move, because toy_predict is uniform",
            result["uniform"] and result["accuracy"] == 1 / result["vocab"],
            f"toy_predict returns {result['accuracy']} = 1/V at every position and every "
            f"vocabulary entry, so MLM accuracy is exactly {1 / result['vocab']:.4f} under both "
            "schemes, for any corpus and any mask probability. The exercise asks whether "
            "whole-word masking improves a number that provably cannot change",
        ),
        practice.Check(
            "CONTROL: both schemes select the same share of tokens",
            abs(token[0] - word[0]) < 0.01,
            f"{token[0]:.3%} token-level against {word[0]:.3%} word-level on the same corpus. "
            "Selection is drawn per word and is independent of word length, so the token rate is "
            "unbiased. Whole-word masking is not harder because it masks more of the sentence",
        ),
        practice.Check(
            "FINDING: what actually changes is the variance, not the rate",
            word[2] > token[2],
            f"the per-sentence mask count has a coefficient of variation of {token[2]:.3f} "
            f"token-level and {word[2]:.3f} word-level, {word[2] / token[2]:.2f}x, because one "
            "draw now commits up to four positions. Same rate, lumpier gradient -- which is the "
            "cost the scheme pays for removing the leak",
        ),
        practice.Check(
            "CONTROL: whole_word_mlm already exists, so the implementation is a reading",
            hasattr(parity.load_reference(PHASE, LESSON, "main"), "whole_word_mlm"),
            "code/main.py ships whole_word_mlm(tokens, word_spans, vocab_size, mask_prob, rng) "
            "with the 80/10/10 branch drawn per span. The work the exercise describes is already "
            "done; what is not done is measuring the one thing that separates it from the other "
            "path, which the lesson's uniform model is incapable of showing",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
