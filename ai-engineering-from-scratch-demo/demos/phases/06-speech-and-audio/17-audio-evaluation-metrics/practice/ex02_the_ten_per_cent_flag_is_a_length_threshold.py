"""Exercise 2 — the ten per cent flag is a prompt-length threshold.

    **Medium.** Build a TTS round-trip WER harness. Run your Kokoro or F5-TTS
    output through Whisper. Compute WER over 50 prompts. Flag prompts with WER
    &gt; 10%.

Reading of the exercise: `kokoro`, `f5_tts`, `whisper`, `torch`, `transformers`,
`soundfile` and `jiwer` are all absent and no audio ships anywhere in the
reference tree, so the synthesise-and-transcribe arm cannot run. The harness
around it can, scored with the lesson's own `wer`, and the harness is where the
exercise decides its own answer.

**`WER > 10%` is not a quality threshold, it is a length threshold.** A prompt of
`N` reference words has WER on the grid `{0, 1/N, 2/N, ...}`, so the flag fires
on a single word error whenever `1/N > 0.10`. Over 50 prompts of 4 to 12 words,
**31 of 50** are short enough that "more than 10%" and "not perfect" are the same
predicate; for the other 19 the flag silently becomes "at least two errors". At a
2% per-word error rate the harness flags **3** prompts while **6** contain an
error at all.

**Two different numbers are both "WER over 50 prompts".** Pooling edits over all
words (micro) and averaging the per-prompt rates (macro) are not the same
statistic, because the macro average weights a four-word prompt like a twelve-word
one:

| per-word error rate | corpus (micro) WER | mean of per-prompt WER | flagged | any error |
|---:|---:|---:|---:|---:|
| 0.02 | **0.0167** | 0.0163 | 3 | 6 |
| 0.05 | **0.0646** | **0.0700** | 14 | 20 |

**And 50 prompts is 418 reference words.** At the measured corpus WER that is a
binomial standard deviation of **0.63 pp** and a 95% interval of **±1.23 pp**, so
the harness cannot separate two systems a point apart -- which is the whole
spread of the lesson's own Open ASR row between Whisper-large-v3-turbo at 1.58%
and everything near it.

Structure: `corpus` builds the 50 prompts; `degrade` turns a prompt into a
hypothesis at a target per-word error rate; `harness` scores one run and returns
both aggregations plus the flags; `interval` is the binomial 95% half-width.
"""

from __future__ import annotations

import importlib.util
import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "17-audio-evaluation-metrics"
PROMPTS, FLAG, Z = 50, 0.10, 1.96
RATES = (0.02, 0.05)
WORDS = ("lights kitchen weather timer jazz volume bedroom alarm forecast playlist "
         "tomorrow morning").split()
ABSENT = ("kokoro", "f5_tts", "whisper", "torch", "transformers", "soundfile", "jiwer")


def corpus(count=PROMPTS, seed=7):
    rnd = random.Random(seed)
    return [" ".join(rnd.choice(WORDS) for _ in range(rnd.randint(4, 12)))
            for _ in range(count)]


def degrade(text, rate, rnd):
    """A hypothesis at the target per-word rate: 60% substitutions, 20% each del/ins."""
    out = []
    for word in text.split():
        draw = rnd.random()
        if draw < rate * 0.6:
            out.append(rnd.choice(WORDS))
        elif draw < rate * 0.8:
            continue
        elif draw < rate:
            out.extend([word, rnd.choice(WORDS)])
        else:
            out.append(word)
    return " ".join(out)


def harness(ref, prompts, rate, seed=11):
    """One round trip: both aggregations, the flags, and the error count."""
    rnd = random.Random(seed)
    scores = [ref.wer(text, degrade(text, rate, rnd)) for text in prompts]
    lengths = [len(text.split()) for text in prompts]
    edits = sum(score * length for score, length in zip(scores, lengths))
    return {"micro": edits / sum(lengths), "macro": statistics.mean(scores),
            "flagged": sum(1 for s in scores if s > FLAG),
            "any": sum(1 for s in scores if s > 0), "words": sum(lengths)}


def interval(rate, words, z=Z):
    return z * math.sqrt(rate * (1 - rate) / words)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    prompts = corpus()
    lengths = [len(text.split()) for text in prompts]
    runs = {rate: harness(ref, prompts, rate) for rate in RATES}
    low = runs[RATES[0]]
    return {
        "absent": [m for m in ABSENT if importlib.util.find_spec(m) is None],
        "runs": runs, "prompts": len(prompts), "words": sum(lengths),
        "short": sum(1 for n in lengths if 1 / n > FLAG),
        "span": (min(lengths), max(lengths)),
        "half": interval(low["micro"], low["words"]),
        "sd": interval(low["micro"], low["words"]) / Z,
    }


def verify(result):
    runs, low, high = result["runs"], result["runs"][RATES[0]], result["runs"][RATES[1]]
    return [
        practice.Check(
            "CONTROL: nothing here synthesises or transcribes, so the harness is the subject",
            len(result["absent"]) == len(ABSENT),
            f"find_spec is None for {result['absent']} and no audio ships anywhere in the "
            "reference tree, so the round trip itself cannot run. Everything wrapped around it "
            "can, scored with the lesson's own `wer`",
        ),
        practice.Check(
            "ANSWER: the 10% flag is a length threshold, not a quality threshold",
            result["short"] == 31,
            f"a prompt of N words has WER on the grid 1/N, so the flag fires on a single error "
            f"whenever 1/N > {FLAG}: {result['short']} of {result['prompts']} prompts (lengths "
            f"{result['span'][0]}-{result['span'][1]}) are short enough that 'over 10%' and "
            f"'not perfect' are the same predicate, and for the other "
            f"{result['prompts'] - result['short']} it quietly becomes 'at least two errors'",
        ),
        practice.Check(
            "ANSWER: at a 2% per-word rate it flags 3 prompts and 6 contain an error",
            low["flagged"] == 3 and low["any"] == 6,
            f"corpus WER {low['micro']:.4f} over {low['words']} reference words, "
            f"{low['flagged']} flagged against {low['any']} with any error at all. Half the "
            "erroring prompts are invisible to the flag because they are long enough to absorb "
            "one mistake",
        ),
        practice.Check(
            "FINDING: two different numbers are both 'WER over 50 prompts'",
            abs(high["macro"] - high["micro"]) > 0.004,
            f"pooling edits over all words gives {high['micro']:.4f} at a "
            f"{RATES[1]} per-word rate and averaging the per-prompt rates gives "
            f"{high['macro']:.4f}; at {RATES[0]} they are {low['micro']:.4f} and "
            f"{low['macro']:.4f}. The macro average weights a four-word prompt like a "
            "twelve-word one, and the exercise does not say which it wants",
        ),
        practice.Check(
            "FINDING: 50 prompts is 418 words, which resolves about a point",
            result["words"] < 500 and result["half"] > 0.01,
            f"{result['prompts']} prompts carry {result['words']} reference words, so at a "
            f"corpus WER of {low['micro']:.4f} the binomial standard deviation is "
            f"{result['sd'] * 100:.2f} pp and the 95% interval is "
            f"+/-{result['half'] * 100:.2f} pp. Two systems a point apart are one sample",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
