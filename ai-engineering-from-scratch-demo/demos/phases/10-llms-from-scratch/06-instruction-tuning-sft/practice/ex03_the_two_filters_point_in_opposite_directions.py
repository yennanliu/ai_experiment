"""Exercise 3 — the diversity filter drops the four longest responses, which is the other filter backwards.

    Build a data quality scorer. For each instruction-response pair, compute:
    (a) response length in tokens, (b) instruction-to-response ratio, (c)
    vocabulary diversity (unique tokens / total tokens). Filter out examples
    with response length < 10 tokens or diversity < 0.3. Show how filtering
    affects the final loss.

Reading of the exercise: "tokens" means the lesson's tokens, which are bytes --
`tokenize_instruction_pair` calls `.encode("utf-8")` -- so all three metrics are
computed on byte tokens and the two thresholds are applied as written. The same
diversity is then recomputed on word tokens, because that is the unit the 0.3
threshold comes from and the two units do not give the same answer.

**ANSWER: the filter drops 4 of the lesson's own 8 examples**, and none of them
for being short. Every response is at least 26 bytes, so `length < 10` fires
never; `diversity < 0.3` fires four times.

**FINDING: the two filters point in opposite directions.** Sorted by response
length, the four survivors are the four *shortest* (26, 27, 31, 61 bytes) and
the four dropped are the four *longest* (71, 78, 84, 90). Byte diversity is
`unique/total` over an alphabet of about 40 characters that English actually
uses, so it falls as a response gets longer -- the correlation between length
and diversity here is **-0.99**. The length filter removes short answers and
the diversity filter, standing beside it, removes long ones.

**MECHANISM: 0.3 is a word-level threshold on a byte-level tokenizer.** The same
eight responses score 0.909 to 1.000 on word tokens, all far above 0.3, against
0.233 to 0.654 on bytes. On words the filter passes everything; on bytes it
halves the dataset. The number is not wrong, it is in the wrong units, and
nothing in the exercise says which unit it belongs to.

**FINDING: the "final loss" difference is which example was measured last.**
The two runs end at 5.5011 and 5.5042, a gap of 3.1e-03 against a within-run
spread of 1.7e-02 across the examples themselves. The last logged loss is
whichever example the permutation put last. `sft_train` discards its gradient
and updates with `lr * np.random.randn(...)` (Exercise 5), so filtering changes
only how many random steps get taken.

Structure: `score` is the three metrics on byte tokens; `word_diversity` is (c)
recomputed on the unit the threshold came from.
"""

from __future__ import annotations

import contextlib
import io
import statistics

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "06-instruction-tuning-sft"
SEED, EPOCHS = 1, 3
MIN_LENGTH, MIN_DIVERSITY = 10, 0.3
SHAPE = dict(vocab_size=256, embed_dim=64, num_heads=4, num_layers=2,
             max_seq_len=64, ff_dim=256)


def score(example):
    """(a) response length, (b) instruction-to-response ratio, (c) diversity — on bytes."""
    response = list(example["response"].encode("utf-8"))
    instruction = list(example["instruction"].encode("utf-8"))
    return {
        "length": len(response),
        "ratio": len(instruction) / len(response),
        "diversity": len(set(response)) / len(response),
    }


def word_diversity(example):
    """(c) recomputed on word tokens, the unit the 0.3 threshold comes from."""
    words = example["response"].lower().split()
    return len(set(words)) / len(words)


def keeps(scored):
    return scored["length"] >= MIN_LENGTH and scored["diversity"] >= MIN_DIVERSITY


def run(ref, dataset):
    """The full loss trace of one `sft_train` run from a fixed initialisation."""
    np.random.seed(SEED)
    model = ref.MiniGPT(**SHAPE)
    np.random.seed(SEED + 100)
    with contextlib.redirect_stdout(io.StringIO()):
        _, losses = ref.sft_train(model, dataset, num_epochs=EPOCHS)
    return losses


def split(scored):
    """Lengths of the survivors and of the dropped, sorted, plus the short-filter count."""
    return (sorted(s["length"] for s in scored if keeps(s)),
            sorted(s["length"] for s in scored if not keeps(s)),
            sum(1 for s in scored if s["length"] < MIN_LENGTH))


def spread(values):
    return min(values), max(values)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    data = ref.INSTRUCTION_DATA
    scored = [score(example) for example in data]
    survivors = [example for example, s in zip(data, scored) if keeps(s)]
    survived, dropped, too_short = split(scored)
    return {
        "kept": len(survivors),
        "total": len(data),
        "too_short": too_short,
        "lengths": (survived, dropped),
        "correlation": statistics.correlation([s["length"] for s in scored],
                                              [s["diversity"] for s in scored]),
        "byte_range": spread([s["diversity"] for s in scored]),
        "word_range": spread([word_diversity(e) for e in data]),
        "traces": (run(ref, data), run(ref, survivors)),
    }


def verify(result):
    kept, total = result["kept"], result["total"]
    survived, dropped = result["lengths"]
    byte_lo, byte_hi = result["byte_range"]
    word_lo, word_hi = result["word_range"]
    plain_trace, filtered_trace = result["traces"]
    plain, filtered = plain_trace[-1], filtered_trace[-1]
    spread = statistics.stdev(plain_trace)
    return [
        practice.Check(
            f"ANSWER: the filter drops {total - kept} of {total}, none of them for being short",
            kept == total - 4 and result["too_short"] == 0,
            f"every response is at least {min(survived + dropped)} bytes, so "
            f"length < {MIN_LENGTH} fires {result['too_short']} times and "
            f"diversity < {MIN_DIVERSITY} fires {total - kept}. The dataset the lesson ships is "
            f"halved by one of the two conditions the exercise names and untouched by the other",
        ),
        practice.Check(
            "FINDING: the two filters point in opposite directions",
            max(survived) <= min(dropped) and result["correlation"] < -0.8,
            f"sorted by response length, the survivors are {survived} bytes and the dropped are "
            f"{dropped} -- the four shortest against the four longest. Byte diversity is "
            f"unique/total over the ~40 characters English actually uses, so it falls as a "
            f"response grows: the correlation between length and diversity here is "
            f"{result['correlation']:.2f}. The length filter removes short answers and the "
            "diversity filter, standing beside it, removes long ones",
        ),
        practice.Check(
            "MECHANISM: 0.3 is a word-level threshold applied to a byte-level tokenizer",
            word_lo > MIN_DIVERSITY > byte_lo and byte_hi < word_lo,
            f"the same eight responses score {word_lo:.3f} to {word_hi:.3f} on word tokens, all "
            f"far above {MIN_DIVERSITY}, against {byte_lo:.3f} to {byte_hi:.3f} on the bytes the "
            "lesson's tokenizer actually emits. On words the filter passes everything; on bytes "
            "it halves the dataset. The threshold is not wrong, it is in the wrong units, and "
            "the exercise says only 'tokens'",
        ),
        practice.Check(
            "FINDING: the 'final loss' difference is which example was measured last",
            abs(filtered - plain) < spread,
            f"{EPOCHS} epochs on all {total} examples ends at {plain:.4f} and on the {kept} "
            f"survivors at {filtered:.4f}, a difference of {abs(filtered - plain):.1e} against a "
            f"within-run spread of {spread:.1e} across the examples themselves -- the last "
            "logged loss is whichever example the permutation put last, not what was learned. "
            "sft_train discards its gradient and updates with lr * np.random.randn(...), so "
            "filtering changes only how many random steps get taken",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
