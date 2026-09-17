"""Exercise 2 — 5% of examples is 9.5% of trained tokens, and truncation eats half the rest.

    Implement data mixing. Create a function that takes an SFT dataset and a raw
    text corpus, then produces training batches where 5% of examples are raw
    text (no masking) and 95% are instruction pairs (masked). Run 3 epochs and
    compare forgetting metrics against pure SFT training.

Reading of the exercise: the mix is built at exactly the ratio the exercise
names -- 5% of *examples*, which needs the eight pairs repeated 19 times so that
8 raw examples is exactly 5% of 160 -- and the quantity that actually reaches the
optimiser is then counted on the batches `sft_train` receives, after its own
`seq_len=64` truncation and its own shift. Forgetting is the lesson's own
`measure_forgetting` on text neither arm was shown, run for both arms.

**ANSWER: 5% of examples is 9.5% of trained tokens.** A raw example carries
**61 of its 63** tokens through the mask; an instruction pair carries **245 of
493**, or 49.7%. So the raw share of what receives gradient is
`8 x 61 / (8 x 61 + 19 x 245)` = **9.5%**, nearly double the ratio that was set.

**FINDING: the trainer's truncation removes 48% of the supervised signal.**
`sft_train` cuts each example to `seq_len=64` tokens, and it cuts from the end,
which is where the response is. Untruncated, the eight pairs are 724 tokens of
which **468** are response. After truncation they are 493 tokens of which
**245** are. Half the tokens SFT exists to train on never reach the loss, and
the cut is invisible in the dataset -- it happens inside the trainer.

**MECHANISM: the ratio is specified on examples and the effect lands on
tokens.** Masking is the whole point of SFT and it is also what makes the two
example types incomparable units. Setting "5%" on the countable thing sets 9.5%
on the thing that matters, and the gap widens as the supervised share falls --
which truncation is what makes fall.

**FINDING: the comparison the exercise ends on has no signal.** Three epochs of
`sft_train` move the lesson's own forgetting metric by less than 1e-04 on either
arm, and the two arms agree to the same precision, because the update is
`lr * np.random.randn(...)` and the gradient is discarded (Exercise 5). Pure SFT
and mixed SFT are the same random walk, so "compare forgetting metrics against
pure SFT training" compares two numbers that differ in their sixth decimal, on a
model whose attention and embeddings never move at all.

Structure: `effective` applies the trainer's own truncation and shift to one
example; `mix` builds the dataset at an exact example ratio; `tally` sums
`effective` over a dataset.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "06-instruction-tuning-sft"
SEED, EPOCHS, RAW_SHARE, SEQ_LEN = 1, 3, 0.05, 64
REPEATS, RAW_COUNT = 19, 8
SHAPE = dict(vocab_size=256, embed_dim=64, num_heads=4, num_layers=2,
             max_seq_len=64, ff_dim=256)
RAW = "Machine learning is a field of study. " * 20
HELD = "The transformer architecture relies on attention over a sequence. " * 8


def effective(ref, example):
    """Supervised and total tokens after `sft_train`'s own truncation and shift."""
    tokens = ref.tokenize_instruction_pair(example["instruction"], example["response"])
    mask = ref.create_loss_mask(tokens)
    if len(tokens) < 3:
        return 0, 0
    tokens, mask = tokens[:SEQ_LEN], mask[:SEQ_LEN]
    return int(mask[1:].sum()), len(tokens) - 1


def raw_example():
    """Raw text as this trainer can carry it: no instruction, so nothing is masked out."""
    return {"instruction": "", "response": RAW}


def tally(ref, dataset):
    """Supervised and total trained tokens across a whole dataset."""
    counted = [effective(ref, example) for example in dataset]
    return sum(s for s, _ in counted), sum(t for _, t in counted)


def untruncated(ref):
    """The same two counts before the trainer cuts anything."""
    total = supervised = 0
    for example in ref.INSTRUCTION_DATA:
        tokens = ref.tokenize_instruction_pair(example["instruction"], example["response"])
        total += len(tokens)
        supervised += int(ref.create_loss_mask(tokens).sum())
    return supervised, total


def mix(ref, repeats=REPEATS, raw_count=RAW_COUNT):
    """Exactly `raw_count` raw examples in `raw_count + 8 * repeats` -- 5% when both default."""
    return list(ref.INSTRUCTION_DATA) * repeats + [raw_example()] * raw_count


def forgetting(ref, dataset):
    """`measure_forgetting` before and after three epochs of the lesson's own trainer."""
    np.random.seed(SEED)
    model = ref.MiniGPT(**SHAPE)
    np.random.seed(SEED + 100)
    with contextlib.redirect_stdout(io.StringIO()):
        before = ref.measure_forgetting(model, HELD)
        trained, _ = ref.sft_train(model, dataset, num_epochs=EPOCHS)
        after = ref.measure_forgetting(trained, HELD)
    return before, after


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    pair_sup, pair_tot = tally(ref, ref.INSTRUCTION_DATA)
    raw_sup, raw_tot = effective(ref, raw_example())
    batches = mix(ref)
    mixed_raw = RAW_COUNT * raw_sup
    return {
        "pairs": (pair_sup, pair_tot),
        "raw": (raw_sup, raw_tot),
        "untruncated": untruncated(ref),
        "examples": (RAW_COUNT, len(batches)),
        "token_share": mixed_raw / (mixed_raw + REPEATS * pair_sup),
        "pure": forgetting(ref, ref.INSTRUCTION_DATA),
        "mixed": forgetting(ref, batches),
    }


def verify(result):
    pair_sup, pair_tot = result["pairs"]
    raw_sup, raw_tot = result["raw"]
    full_sup, full_tot = result["untruncated"]
    raw_n, total_n = result["examples"]
    share = result["token_share"]
    pure_before, pure_after = result["pure"]
    mixed_before, mixed_after = result["mixed"]
    return [
        practice.Check(
            f"ANSWER: {RAW_SHARE:.0%} of examples is {share:.1%} of trained tokens",
            abs(raw_n / total_n - RAW_SHARE) < 1e-9 and share > 1.8 * RAW_SHARE,
            f"the mix is {raw_n} raw examples in {total_n}, exactly {raw_n / total_n:.1%}. Of the "
            f"tokens that reach the optimiser a raw example carries {raw_sup} of its {raw_tot} "
            f"through the mask and an instruction pair {pair_sup} of {pair_tot}, "
            f"{pair_sup / pair_tot:.1%}, so the raw share of what receives gradient is "
            f"{share:.1%} -- {share / RAW_SHARE:.1f} times the ratio the exercise sets",
        ),
        practice.Check(
            "FINDING: the trainer's truncation removes 48% of the supervised signal",
            pair_sup < 0.6 * full_sup and pair_tot < full_tot,
            f"sft_train cuts each example to seq_len={SEQ_LEN} tokens and it cuts from the end, "
            f"which is where the response is. Untruncated the eight pairs are {full_tot} tokens "
            f"of which {full_sup} are response, {full_sup / full_tot:.1%}; after truncation they "
            f"are {pair_tot} of which {pair_sup} are, {pair_sup / pair_tot:.1%}. "
            f"{100 * (1 - pair_sup / full_sup):.0f}% of the tokens SFT exists to train on never "
            "reach the loss, and the cut happens inside the trainer where the dataset cannot show "
            "it",
        ),
        practice.Check(
            "MECHANISM: the ratio is set on examples and the effect lands on tokens",
            share > raw_n / total_n,
            f"masking is the point of SFT and it is also what makes the two example types "
            f"incomparable units: a raw example is {raw_sup / raw_tot:.1%} supervised and a pair "
            f"{pair_sup / pair_tot:.1%}. Setting {RAW_SHARE:.0%} on the countable thing sets "
            f"{share:.1%} on the thing that matters, and the gap widens as the supervised share "
            "falls -- which is exactly what the truncation above does to it",
        ),
        practice.Check(
            "FINDING: both arms move in the sixth decimal, and agree with each other there",
            abs(pure_after - pure_before) < 1e-4 and abs(mixed_after - mixed_before) < 1e-4,
            f"{EPOCHS} epochs of sft_train move the lesson's own forgetting metric from "
            f"{pure_before:.6f} to {pure_after:.6f} on pure SFT and from {mixed_before:.6f} to "
            f"{mixed_after:.6f} on the mix -- {pure_after - pure_before:+.2e} and "
            f"{mixed_after - mixed_before:+.2e}. The update is lr * np.random.randn(...) and the "
            "gradient is discarded, so the two arms are the same random walk and the comparison "
            "the exercise ends on has nothing in it",
        ),
        practice.Check(
            "FINDING: the metric could not show forgetting even with a real gradient here",
            pure_before > 5.0,
            f"the held-out loss sits at {pure_before:.4f} against ln(256) = 5.5452, so the model "
            "is at the uniform baseline before training and stays there. And it is measured on a "
            "model whose attention stack and embedding table never move -- the only weights that "
            "could forget anything are two FFN matrices taking Gaussian steps",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
