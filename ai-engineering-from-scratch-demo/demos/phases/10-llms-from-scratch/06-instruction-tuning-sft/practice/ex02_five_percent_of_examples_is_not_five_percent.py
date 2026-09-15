"""Exercise 2 — 5% of examples is 7.5% of trained tokens, and the forgetting delta is 2.8e-06.

    Implement data mixing. Create a function that takes an SFT dataset and a raw
    text corpus, then produces training batches where 5% of examples are raw
    text (no masking) and 95% are instruction pairs (masked). Run 3 epochs and
    compare forgetting metrics against pure SFT training.

Reading of the exercise: the mix is built at the ratio the exercise names -- 5%
of *examples*, which is what it says -- and then the quantity that actually
reaches the optimiser is counted, because a raw example trains on all its tokens
while an instruction pair trains only on its response. Forgetting is the
lesson's own `measure_forgetting` on text neither arm was shown.

**ANSWER: 5% of examples is 7.5% of trained tokens.** Across the lesson's eight
pairs, **468 of 724** tokens are response tokens -- 64.6%. A raw example
contributes 100% of its tokens and an instruction pair 64.6% of its, so the raw
share of what carries gradient is `0.05 / (0.05 + 0.95 x 0.646)` = **7.5%**,
half again what the ratio was set to.

**MECHANISM: the ratio is specified on examples and the effect lands on
tokens.** Masking is the whole point of SFT and it is also what makes the two
example types incomparable units. Setting "5%" on the countable thing sets 7.5%
on the thing that matters, and the gap widens as responses get shorter relative
to instructions -- at a 50% response share it would be 9.5%.

**FINDING: the comparison the exercise ends on has no signal.** Three epochs of
`sft_train` move the lesson's own forgetting metric by **2.83e-06** -- from
5.524290 to 5.524293, a relative change of 5e-07 -- because the update is
`lr * np.random.randn(...)` and the gradient is discarded (Exercise 5). Pure SFT
and mixed SFT are the same random walk, so "compare forgetting metrics against
pure SFT training" compares two numbers that differ in their sixth decimal.

**FINDING: the metric could not show forgetting anyway.** It is measured on a
model whose attention and embeddings never move, so the only weights that could
forget anything are two FFN matrices taking Gaussian steps.

Structure: `mix` builds the batch list at a given example ratio; `trained_share`
converts an example ratio into the token ratio it implies.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "06-instruction-tuning-sft"
SEED, EPOCHS, RAW_SHARE = 1, 3, 0.05
SHAPE = dict(vocab_size=256, embed_dim=64, num_heads=4, num_layers=2,
             max_seq_len=64, ff_dim=256)
RAW = "Machine learning is a field of study. " * 20
HELD = "The transformer architecture relies on attention over a sequence. " * 8


def response_share(ref):
    """Of every token in the SFT set, the fraction that carries gradient."""
    total = supervised = 0
    for example in ref.INSTRUCTION_DATA:
        tokens = ref.tokenize_instruction_pair(example["instruction"], example["response"])
        total += len(tokens)
        supervised += int(ref.create_loss_mask(tokens).sum())
    return supervised, total


def trained_share(raw_ratio, masked_share):
    """The raw-text share of *trained tokens* implied by a raw share of examples."""
    return raw_ratio / (raw_ratio + (1 - raw_ratio) * masked_share)


def mix(ref, raw_ratio):
    """Batches at `raw_ratio` raw examples: raw text unmasked, pairs masked."""
    pairs = [(ref.tokenize_instruction_pair(e["instruction"], e["response"]), True)
             for e in ref.INSTRUCTION_DATA]
    count = max(1, round(raw_ratio / (1 - raw_ratio) * len(pairs)))
    chunk = list(RAW.encode("utf-8"))[:64]
    return pairs + [(chunk, False) for _ in range(count)]


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
    supervised, total = response_share(ref)
    masked = supervised / total
    before, after = forgetting(ref, ref.INSTRUCTION_DATA)
    batches = mix(ref, RAW_SHARE)
    return {
        "supervised": (supervised, total),
        "masked_share": masked,
        "token_share": trained_share(RAW_SHARE, masked),
        "half_share": trained_share(RAW_SHARE, 0.5),
        "raw_examples": sum(1 for _, is_pair in batches if not is_pair),
        "batches": len(batches),
        "forgetting": (before, after),
    }


def verify(result):
    supervised, total = result["supervised"]
    before, after = result["forgetting"]
    share, masked = result["token_share"], result["masked_share"]
    return [
        practice.Check(
            f"ANSWER: {RAW_SHARE:.0%} of examples is {result['token_share']:.1%} of trained tokens",
            share > 1.4 * RAW_SHARE,
            f"{supervised} of {total} tokens across the lesson's eight pairs are response tokens, "
            f"{masked:.1%}, so a raw example contributes 100% of its tokens where an instruction "
            f"pair contributes {masked:.1%} of its. The raw share of what carries gradient is "
            f"{RAW_SHARE} / ({RAW_SHARE} + {1 - RAW_SHARE} x {masked:.3f}) = {share:.1%}, half "
            "again the ratio the exercise sets",
        ),
        practice.Check(
            "MECHANISM: the ratio is set on examples and the effect lands on tokens",
            result["half_share"] > share and result["raw_examples"] >= 1,
            f"masking is the point of SFT and it is also what makes the two example types "
            f"incomparable units. The mix built here is {result['raw_examples']} raw of "
            f"{result['batches']} batches; the gap between the set ratio and the effective one "
            f"widens as responses shorten -- at a 50% response share it would be "
            f"{result['half_share']:.1%} rather than {share:.1%}",
        ),
        practice.Check(
            "FINDING: the comparison the exercise ends on differs in the sixth decimal",
            abs(after - before) < 1e-4,
            f"{EPOCHS} epochs of sft_train move the lesson's own forgetting metric from "
            f"{before:.6f} to {after:.6f}, a change of {after - before:+.2e} or "
            f"{abs(after - before) / before:.0e} relative, because the update is "
            "lr * np.random.randn(...) and the gradient is discarded. Pure SFT and mixed SFT "
            "are the same random walk",
        ),
        practice.Check(
            "FINDING: the metric could not show forgetting even with a real gradient here",
            before > 5.0,
            f"the held-out loss sits at {before:.4f} against ln(256) = 5.5452, so the model is "
            "at the uniform baseline before training and stays there. And it is measured on a "
            "model whose attention stack and embedding table never move -- the only weights that "
            "could forget anything are two FFN matrices taking Gaussian steps",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
