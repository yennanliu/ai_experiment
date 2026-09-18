"""Exercise 5 — no generation ever puts text after the image.

    Extend the toy decoder to emit a mixed-modality response given a text-only
    prompt. Measure how often the model picks image-first vs text-first given
    training-data distribution 60% text-first / 40% image-first.

Reading of the exercise: the extension is a sampling loop over the lesson's own
bigram, run a thousand times rather than three, because "how often" is a
frequency and the lesson's demo draws three samples. The corpus's own split is
measured first, since 60/40 is a parameter of `make_dataset` and not an
assumption, and then the question is put to the generations -- where it turns
out to have no answer, for a reason worth more than the frequency would have
been.

**ANSWER: from a text-only prompt, 100% of generations are text-first.** That is
forced, not learned: the prompt is text. The measurable question is what comes
after, and **1,000 of 1,000** generations open an image and **0** emit any text
once it closes.

**FINDING: `generate` stops at the first `</image>`.** Its loop breaks on
`SEP_CLOSE` whenever the output contains any text token -- and a text prompt
always does -- so the terminating condition fires at the end of the first image
for every text-prompted sample. The 40% of the corpus that is image-then-text
can contribute its image-to-text bigrams and never be reproduced.

**FINDING: the corpus split is 37.9%, not 40%.** `make_dataset` flips
`random.random() < 0.4` per sequence, so over 1,000 draws the realised
image-first share is **0.379**. At the lesson's own n=40 the same coin gives a
sample whose standard error is about **7.7 points** -- a fifth of the parameter
being estimated.

**FINDING: generation length is bounded by the image, not by `max_len`.**
Samples run from **7** to **25** tokens against a `max_len` of 30, and the upper
end is set by how long the bigram wanders inside the image block. The one knob
the caller has does not bind.

Structure: `corpus_split` measures the realised image-first share, `sample`
draws one generation from the lesson's own bigram, and `after_image` looks for
any text token past the closing separator.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "11-chameleon-early-fusion-tokens"
CORPUS_N, DRAWS, MAX_LEN = 1000, 1000, 30
PROMPT = [1, 5]
TARGET_IMAGE_FIRST, LESSON_N = 0.4, 40


def corpus_split(ref, size=CORPUS_N, seed=42):
    random.seed(seed)
    corpus = ref.make_dataset(size)
    return corpus, sum(1 for seq in corpus if seq[0] == ref.SEP_OPEN) / len(corpus)


def after_image(ref, tokens):
    """Any text token after the first closing separator."""
    if ref.SEP_CLOSE not in tokens:
        return False
    tail = tokens[tokens.index(ref.SEP_CLOSE) + 1:]
    return any(token < ref.VOCAB_TEXT for token in tail)


def standard_error(share, size):
    return round(math.sqrt(share * (1 - share) / size) * 100, 1)


def tally(ref, samples):
    return {
        "draws": len(samples),
        "text_first": sum(row[0] < ref.VOCAB_TEXT for row in samples),
        "opens_image": sum(ref.SEP_OPEN in row for row in samples),
        "ends_closed": sum(row[-1] == ref.SEP_CLOSE for row in samples),
        "text_after": sum(after_image(ref, row) for row in samples),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    corpus, share = corpus_split(ref)
    bigram = ref.train_bigram(corpus)
    random.seed(7)
    samples = [ref.generate(bigram, list(PROMPT), max_len=MAX_LEN) for _ in range(DRAWS)]
    return {
        **tally(ref, samples),
        "corpus_share": round(share, 3), "target": TARGET_IMAGE_FIRST,
        "corpus_error": round(abs(share - TARGET_IMAGE_FIRST) * 100, 1),
        "lesson_stderr": standard_error(TARGET_IMAGE_FIRST, LESSON_N),
        "lengths": (min(len(row) for row in samples), max(len(row) for row in samples)),
        "max_len": MAX_LEN,
        "hit_max": sum(len(row) == MAX_LEN for row in samples),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 100% text-first, and 0 of 1,000 emit text after the image",
            all([result["text_first"] == result["draws"] == DRAWS,
                 result["opens_image"] == DRAWS, result["text_after"] == 0]),
            f"all {result['draws']:,} generations from the text prompt {PROMPT} are "
            f"text-first -- forced, since the prompt is text -- and all "
            f"{result['opens_image']:,} open an image. {result['text_after']} emit any text "
            "after it closes, so the mixed-modality response the exercise asks for is "
            "text-then-image and nothing else",
        ),
        practice.Check(
            "FINDING: generate stops at the first </image>",
            all([result["ends_closed"] == DRAWS, result["text_after"] == 0]),
            f"the loop breaks on SEP_CLOSE whenever the output holds any text token, and a "
            f"text prompt always does, so {result['ends_closed']:,} of {result['draws']:,} "
            "samples terminate at the end of the first image. The 40% of the corpus that is "
            "image-then-text contributes its bigrams and can never be reproduced",
        ),
        practice.Check(
            "FINDING: the corpus split is 37.9%, not 40%",
            all([result["corpus_share"] == 0.379, result["corpus_error"] == 2.1,
                 result["lesson_stderr"] == 7.7]),
            f"make_dataset flips random.random() < {TARGET_IMAGE_FIRST} per sequence, so "
            f"over {CORPUS_N:,} draws the realised image-first share is "
            f"{result['corpus_share']} -- {result['corpus_error']} points off. At the "
            f"lesson's own n={LESSON_N} the same coin has a standard error of "
            f"{result['lesson_stderr']} points, a fifth of the parameter being estimated",
        ),
        practice.Check(
            "FINDING: generation length is bounded by the image, not by max_len",
            all([result["lengths"] == (7, 25), result["hit_max"] == 0,
                 result["lengths"][1] < MAX_LEN]),
            f"samples run {result['lengths'][0]} to {result['lengths'][1]} tokens against a "
            f"max_len of {result['max_len']}, and {result['hit_max']} reach it. The upper end "
            "is set by how long the bigram wanders inside the image block, so the one knob "
            "the caller has does not bind",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
