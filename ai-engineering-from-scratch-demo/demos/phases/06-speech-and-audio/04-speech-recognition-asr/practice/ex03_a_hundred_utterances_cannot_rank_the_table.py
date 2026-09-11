"""Exercise 3 — a hundred utterances cannot rank the lesson's own table.

    **Hard.** Use `whisper-large-v3-turbo` on [LibriSpeech
    test-clean](https://www.openslr.org/12). Compute WER on the first 100
    utterances. Compare with published numbers.

Reading of the exercise: `whisper`, `torch`, `transformers`, `datasets` and
`soundfile` are absent and LibriSpeech is not here, so the transcription half
cannot run. The comparison half can, in full, and it is where the exercise
decides its own answer: **100 utterances is not enough to compare with the
published numbers.** This is the `DESIGN D11` scaled-down run -- a 2,620-utterance
corpus at 20 words each, degraded at each rate in the lesson's own Step 6 table,
scored with the lesson's own `wer`, then sampled 2,000 times at n=100. What is
simulated is the acoustics; the sampling, the metric and the published rates are
real. The command and cost for the full run are in the lesson README.

The generator lands within 0.07 pp of every published rate on the full corpus, so
the sampling below is about sample size and nothing else.

| | |
|---|---|
| Whisper-L-v3-turbo, published | **1.58%** |
| the same, over 100 utterances (95% of draws) | **1.10% - 2.15%** |
| width of that interval | **1.05 pp** |
| width of the lesson's whole five-model table | **0.52 pp** |

The entire published table fits inside one sample's confidence interval, twice
over. So a 100-utterance measurement cannot tell any two of these models apart,
and it usually cannot even order them: over 2,000 draws the published ranking
comes back **7.4%** of the time and the best model is picked **45.1%** of the
time, against 20% for a coin.

How many words would it take? From `n = 2 * 1.96^2 * p(1-p) / (p1-p2)^2`:

| comparison | reference words needed |
|---|---:|
| best vs worst (1.40 vs 1.92) | **4,638** (~232 utterances) |
| Seamless vs wav2vec (1.70 vs 1.92) | 28,212 |
| Whisper vs Seamless (1.58 vs 1.70) | 86,068 |
| Parakeet vs Canary (1.40 vs 1.48) | **170,382** |

test-clean holds about 52,400 words. The two best rows of the lesson's own table
cannot be separated on the whole of test-clean, let alone on the first hundred
utterances of it -- they differ by 0.08 pp and would need **3.3x** the corpus.

Structure: `degrade` turns a reference into a hypothesis at a target word error
rate; `build` scores every utterance with the lesson's `wer` and keeps the error
counts; `sample` draws 100-utterance subsets; `words_needed` is the two-sample
size formula.
"""

from __future__ import annotations

import importlib.util
import random

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "04-speech-recognition-asr"
TABLE = (("Parakeet-TDT-1.1B", 1.40), ("Canary-1B Flash", 1.48),
         ("Whisper-L-v3-turbo", 1.58), ("Seamless M4T v2", 1.70), ("wav2vec 2.0 Large", 1.92))
UTTERANCES, WORDS, SAMPLE, DRAWS, Z = 2620, 20, 100, 2000, 1.96
ABSENT = ("whisper", "torch", "transformers", "datasets", "soundfile")
LEXICON = [f"w{i:03d}" for i in range(300)]


def degrade(sentence, rate, rnd):
    """A hypothesis at the target word error rate: 60% substitutions, 20% each del/ins."""
    out = []
    for word in sentence.split():
        draw = rnd.random()
        if draw < rate * 0.6:
            out.append(rnd.choice(LEXICON))
        elif draw < rate * 0.8:
            continue
        elif draw < rate:
            out.extend([word, rnd.choice(LEXICON)])
        else:
            out.append(word)
    return " ".join(out)


def build(ref, corpus):
    """Per-utterance error counts for every model, scored by the lesson's own `wer`."""
    counts = {}
    for index, (name, rate) in enumerate(TABLE):
        rnd = random.Random(100 + index)
        counts[name] = [ref.wer(s, degrade(s, rate / 100, rnd)) * WORDS for s in corpus]
    return counts


def rate_of(errors, indices):
    return sum(errors[i] for i in indices) / (len(indices) * WORDS) * 100


def sample(counts, rnd):
    """2,000 draws of 100 utterances: how often the published order comes back."""
    published = [name for name, _ in TABLE]
    exact, best, watched = 0, 0, []
    for _ in range(DRAWS):
        picked = rnd.sample(range(UTTERANCES), SAMPLE)
        scored = {name: rate_of(counts[name], picked) for name in published}
        order = sorted(scored, key=scored.get)
        exact += order == published
        best += order[0] == published[0]
        watched.append(scored["Whisper-L-v3-turbo"])
    watched.sort()
    return {"exact": exact / DRAWS, "best": best / DRAWS,
            "lo": watched[int(0.025 * DRAWS)], "hi": watched[int(0.975 * DRAWS)]}


def words_needed(first, second):
    """Two-sample size for separating two word error rates at 95%."""
    pooled = (first + second) / 2
    return 2 * Z**2 * pooled * (1 - pooled) / (first - second) ** 2


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rnd = random.Random(0)
    corpus = [" ".join(rnd.choice(LEXICON) for _ in range(WORDS)) for _ in range(UTTERANCES)]
    counts = build(ref, corpus)
    whole = list(range(UTTERANCES))
    return {
        "absent": [m for m in ABSENT if importlib.util.find_spec(m) is None],
        "full": {name: rate_of(counts[name], whole) for name, _ in TABLE},
        "draws": sample(counts, random.Random(3)),
        "needed": {f"{a[0]} vs {b[0]}": words_needed(a[1] / 100, b[1] / 100)
                   for a, b in ((TABLE[0], TABLE[-1]), (TABLE[3], TABLE[4]),
                                (TABLE[2], TABLE[3]), (TABLE[0], TABLE[1]))},
        "corpus_words": UTTERANCES * WORDS,
    }


def verify(result):
    draws, full = result["draws"], result["full"]
    span = TABLE[-1][1] - TABLE[0][1]
    width = draws["hi"] - draws["lo"]
    closest = result["needed"][f"{TABLE[0][0]} vs {TABLE[1][0]}"]
    widest = result["needed"][f"{TABLE[0][0]} vs {TABLE[-1][0]}"]
    return [
        practice.Check(
            "CONTROL: the transcription half cannot run, and the generator is calibrated",
            len(result["absent"]) == len(ABSENT)
            and max(abs(full[n] - p) for n, p in TABLE) < 0.10,
            f"find_spec is None for {result['absent']} and LibriSpeech is not here, so the "
            f"simulated corpus stands in: over all {UTTERANCES} utterances it reproduces every "
            f"published rate to {max(abs(full[n] - p) for n, p in TABLE):.3f} pp, which leaves "
            "sample size as the only thing the numbers below are about",
        ),
        practice.Check(
            "ANSWER: the whole published table fits inside one 100-utterance interval",
            width > 2 * span,
            f"Whisper's 1.58% published rate comes back as {draws['lo']:.2f}-{draws['hi']:.2f}% "
            f"across 95% of {DRAWS} draws of {SAMPLE} utterances -- {width:.2f} pp wide against "
            f"the {span:.2f} pp that separates the best model in the table from the worst. "
            "One sample cannot tell any two of them apart",
        ),
        practice.Check(
            "ANSWER: at n=100 the published ranking comes back 7% of the time",
            draws["exact"] < 0.15 and draws["best"] < 0.5,
            f"over {DRAWS} draws the five models land in the published order "
            f"{draws['exact'] * 100:.1f}% of the time and the best model is picked "
            f"{draws['best'] * 100:.1f}% of the time, against {100 / len(TABLE):.0f}% for a coin. "
            "'Compare with published numbers' has no answer at this sample size",
        ),
        practice.Check(
            "FINDING: the two best rows cannot be separated on the whole of test-clean",
            closest > 3 * result["corpus_words"] > widest,
            f"separating {TABLE[0][1]}% from {TABLE[1][1]}% at 95% needs {closest:,.0f} reference "
            f"words, {closest / result['corpus_words']:.1f}x the ~{result['corpus_words']:,} in "
            f"test-clean. Even the widest gap in the table, {TABLE[0][1]}% against "
            f"{TABLE[-1][1]}%, needs {widest:,.0f} words -- about "
            f"{widest / WORDS:.0f} utterances, more than the 100 the exercise names",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
