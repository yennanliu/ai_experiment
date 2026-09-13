"""Exercise 3 — BLEU reads 68 for a system that is wrong on more than half.

    **Hard.** Fine-tune `flan-t5-small` on a tiny English -> pig-Latin corpus
    (200 pairs). Measure BLEU on a held-out 50-pair set. Compare against
    fine-tuning `Llama-3.2-1B` on the same data with the same compute.

Reading of the exercise: `torch`, `transformers`, `datasets`, `sacrebleu` and
`sentencepiece` all return None from `find_spec` and there is no network, so
neither model can be fetched or trained. What *is* buildable is the corpus and
the metric, and running the metric against systems whose errors are known
exactly says more about the experiment than either fine-tune would. 250
pronounceable words are generated, 200 train and 50 held out with zero overlap,
and character-level corpus BLEU-4 is implemented here.

**ANSWER: BLEU cannot separate the outcomes this task has.**

| system | BLEU-4 | exact matches |
|---|---:|---:|
| the rule, correct | 100.00 | 50 / 50 |
| **vowel branch wrong** (`way` -> `ay`) | **98.07** | 48 / 50 |
| moves one letter, not the onset cluster | 68.47 | 22 / 50 |
| copy the input | 23.37 | **0 / 50** |
| memorised all 200 training pairs | 23.37 | **0 / 50** |

**FINDING: the floor is 23, not 0.** A system that gets *nothing* right scores
23.37, because pig-Latin is a permutation of the input's characters plus a
suffix, and character n-grams do not care about order at that scale. A quarter
of the scale is unreachable-from-below noise.

**FINDING: the top of the scale is blunter than the error rate it reports.** The
one-letter system is wrong on **28 of 50** items -- every multi-letter onset --
and still reads **68.47**. The wrong-vowel-branch system is wrong on every case
where that branch fires and loses **1.93 BLEU**. Exact match -- which a
deterministic transduction actually admits -- reads 48/50, then 22/50, then 0/50,
cleanly, on the same outputs.

**FINDING: memorisation scores exactly the copy baseline.** The held-out words
have zero overlap with the 200 training pairs, so a model that memorised the
table perfectly is indistinguishable from one that learned nothing. 200 pairs
either teach the rule or teach nothing measurable; there is no partial credit in
the task, only in the metric.

**CONTROL: "the same compute" is a 16x step ratio.** flan-t5-small's 77M
parameters against Llama-3.2-1B's 1.24B is 16.1x, so equal FLOPs buys the larger
model about a sixteenth of the steps -- and the comparison also puts an
instruction-tuned checkpoint against a base one, confounding architecture with
pretraining recipe.

Structure: `pig` is the rule; `corpus` builds the split; `bleu` is character-level
corpus BLEU-4 with a brevity penalty; `systems` collects the five candidates.
"""

from __future__ import annotations

import collections
import importlib.util
import math
import random

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "08-t5-bart-encoder-decoder"
VOWELS, TRAIN, HELD = "aeiou", 200, 50
ONSETS = ("", *"b c d f g h j k l m n p r s t v w br cl dr fl gr pl st str tr sp sn thr sc sw pr bl gl sl sh ch".split())
CODAS = (*"n t d m k s l p r ng st nd ck ll sh ft mp nt rk".split(), "")
NEEDED, SIZES = ("torch", "transformers", "datasets", "sacrebleu", "sentencepiece"), (77e6, 1.24e9)


def onset(word):
    """Index of the first vowel -- where pig-Latin cuts a consonant cluster."""
    return next((i for i, c in enumerate(word) if c in VOWELS), len(word))


def pig(word):
    """The rule: vowel-initial takes 'way', otherwise move the onset cluster and add 'ay'."""
    return word + "way" if word[0] in VOWELS else word[onset(word):] + word[:onset(word)] + "ay"


def wrong_branch(word):
    """Learned 'ay' for vowel-initial words too -- wrong on every vowel-initial item."""
    return word + "ay" if word[0] in VOWELS else pig(word)


def one_letter(word):
    """Moves a single consonant instead of the whole onset cluster."""
    return pig(word) if word[0] in VOWELS else word[1:] + word[0] + "ay"


def corpus(seed=7, total=TRAIN + HELD):
    """Pronounceable words, deduplicated, split train/held with no overlap."""
    rng, seen = random.Random(seed), []
    while len(seen) < total:
        word = rng.choice(ONSETS) + rng.choice(VOWELS) + rng.choice(CODAS)
        if len(word) >= 3 and word not in seen:
            seen.append(word)
    return seen[:TRAIN], seen[TRAIN:total]


def counts(text, order):
    """Character n-gram counts of one string."""
    return collections.Counter(tuple(text[i:i + order]) for i in range(len(text) - order + 1))


def bleu(hypotheses, references, order=4):
    """Character-level corpus BLEU-4 with a brevity penalty, on a 0-100 scale."""
    clipped, total, lengths = [0] * order, [0] * order, [0, 0]
    for hypothesis, reference in zip(hypotheses, references):
        lengths[0], lengths[1] = lengths[0] + len(hypothesis), lengths[1] + len(reference)
        for k in range(1, order + 1):
            seen, want = counts(hypothesis, k), counts(reference, k)
            clipped[k - 1] += sum(min(v, want[g]) for g, v in seen.items())
            total[k - 1] += max(0, len(hypothesis) - k + 1)
    if min(clipped) == 0 or min(total) == 0:      # no 4-gram matched anywhere
        return 0.0
    precision = sum(math.log(clipped[k] / total[k]) for k in range(order)) / order
    penalty = 1.0 if lengths[0] > lengths[1] else math.exp(1 - lengths[1] / max(lengths[0], 1))
    return 100 * penalty * math.exp(precision)


def shape(held):
    """How many held-out items each known-wrong rule can possibly get wrong."""
    return (sum(1 for w in held if w[0] in VOWELS),
            sum(1 for w in held if w[0] not in VOWELS and onset(w) > 1))


def systems(train, held):
    """Five candidate outputs: the rule, two known-wrong rules, copying, and memorising."""
    table = {word: pig(word) for word in train}
    return {"rule": [pig(w) for w in held], "wrong_branch": [wrong_branch(w) for w in held],
            "one_letter": [one_letter(w) for w in held],
            "copy": list(held), "memorised": [table.get(w, w) for w in held]}


def solve():
    parity.load_reference(PHASE, LESSON, "main")
    train, held = corpus()
    outputs, references = systems(train, held), [pig(word) for word in held]
    vowel_initial, clusters = shape(held)
    return {
        "scores": {name: bleu(out, references) for name, out in outputs.items()},
        "exact": {name: sum(1 for a, b in zip(out, references) if a == b)
                  for name, out in outputs.items()},
        "overlap": len(set(train) & set(held)), "held": len(held),
        "vowel_initial": vowel_initial, "clusters": clusters,
        "absent": [m for m in NEEDED if importlib.util.find_spec(m) is None],
        "compute": SIZES[1] / SIZES[0],
    }


def verify(result):
    scores, exact = result["scores"], result["exact"]
    return [
        practice.Check(
            "ANSWER: BLEU cannot separate the outcomes this task has",
            scores["rule"] == 100.0 and scores["wrong_branch"] > 95,
            f"BLEU-4 / exact out of {result['held']}: rule {scores['rule']:.2f}/{exact['rule']}, "
            f"wrong vowel branch {scores['wrong_branch']:.2f}/{exact['wrong_branch']}, one-letter "
            f"onset {scores['one_letter']:.2f}/{exact['one_letter']}, copy "
            f"{scores['copy']:.2f}/{exact['copy']} -- 77 of BLEU's 100 points sit between "
            "'wrong about everything' and 'wrong about nothing'",
        ),
        practice.Check(
            "FINDING: the floor is 23, not 0",
            scores["copy"] > 20 and exact["copy"] == 0,
            f"copying the input scores {scores['copy']:.2f} with {exact['copy']}/"
            f"{result['held']} correct, because pig-Latin is a permutation of the input's "
            "characters plus a suffix. A quarter of the scale is unreachable from below",
        ),
        practice.Check(
            "FINDING: the top of the scale is worse than the error rate it reports",
            100 - scores["wrong_branch"] < 3 and exact["wrong_branch"] < result["held"],
            f"the one-letter system is wrong on {result['clusters']} of {result['held']} items "
            f"-- every multi-letter onset -- and still reads {scores['one_letter']:.2f}; the "
            f"wrong-branch system misses all {result['vowel_initial']} vowel-initial items for "
            f"{100 - scores['wrong_branch']:.2f}. Exact match reads {exact['wrong_branch']}, "
            f"{exact['one_letter']}, {exact['copy']}",
        ),
        practice.Check(
            "FINDING: memorising all 200 pairs scores exactly the copy baseline",
            result["overlap"] == 0 and scores["memorised"] == scores["copy"],
            f"the held-out words share {result['overlap']} entries with the {TRAIN} training "
            f"pairs, so memorising the table perfectly scores {scores['memorised']:.2f} -- "
            "identical to copying. The partial credit is in the metric, not in the task",
        ),
        practice.Check(
            "CONTROL: neither fine-tune is buildable, and 'same compute' is a 16x step ratio",
            result["absent"] == list(NEEDED),
            f"find_spec is None for {result['absent']} and there is no network. flan-t5-small's "
            f"77M against Llama-3.2-1B's 1.24B is {result['compute']:.1f}x, so equal FLOPs buys "
            "the larger model a sixteenth of the steps -- and the comparison puts an "
            "instruction-tuned checkpoint against a base one",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
