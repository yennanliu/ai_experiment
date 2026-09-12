"""Exercise 3 — the decoder-only baseline gets 6.7x the supervised positions.

    **Hard.** Train a tiny (2-layer, d=64) BERT on 10,000 sentences from a public
    dataset. Fine-tune the `[CLS]` token for SST-2 sentiment. Compare against a
    decoder-only baseline at matched params -- which wins?

Reading of the exercise: nothing in the setup exists. `torch`, `datasets` and
`transformers` all return None from `find_spec`, there is no SST-2 and no network
to fetch one, the lesson has no encoder, no training loop and no `[CLS]`
representation -- `CLS_ID` appears twice in the whole module, in its own
definition and inside `SPECIAL_IDS`, and is only ever *skipped*. So the
comparison is done where it is decided rather than where it is measured: on the
supervision arithmetic of the two objectives, at the lesson's own `mask_prob`.

**ANSWER: at matched parameters the decoder-only model sees 6.67x the supervised
positions per sequence.** MLM labels `0.15 * L` positions; causal LM labels every
one of `L`. At L=128 that is 19.2 against 128.

**FINDING: BERT's compensation is context, and it is worth 1.97x, not 6.67x.**
Each masked position conditions on all `L - 1` other tokens; a causal position `i`
conditions on `i`. Averaged over the sequence that is 127 against 64.5. Multiply
the two effects and the decoder still leads on supervised context-tokens per
sequence by **3.39x** -- 8,256 against 2,438. That ratio, not architecture, is
why decoder-only pretraining won the scaling race.

**FINDING: 12% of pretraining tokens carry a symbol that never occurs
downstream.** `15% x 80%` become `[MASK]`, and a fine-tuned SST-2 input contains
zero of them. The 80/10/10 rule exists to shrink that mismatch and it does not
remove it; a causal LM has no such gap at all.

**CONTROL: the `[CLS]` the exercise fine-tunes has no representation here.**
`CLS_ID` is a member of `SPECIAL_IDS` and every masking path `continue`s on it.
The lesson never pools it, never trains it, and `toy_predict` ignores position
entirely.

**CONTROL: the comparison is not free for BERT even on its own terms.** Only 15%
of positions carry a label, so **85%** of every sequence is pushed through the
encoder to produce no gradient at all -- the compute is spent, the supervision is
not. A causal LM labels every position it computes.

Structure: `supervision` is the per-sequence arithmetic; `absent` probes the
packages; `mentions` counts `CLS_ID` in the lesson's own source.
"""

from __future__ import annotations

import importlib.util
import inspect

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "06-bert-masked-language-modeling"
LENGTH, PROB = 128, 0.15
NEEDED = ("torch", "datasets", "transformers")


def supervision(length=LENGTH, prob=PROB):
    """(labelled positions, mean context per labelled position) for MLM and causal LM."""
    return {"mlm": (prob * length, length - 1),
            "causal": (float(length), (length + 1) / 2)}


def absent(names=NEEDED):
    """Which of the exercise's stated requirements are not installed."""
    return [name for name in names if importlib.util.find_spec(name) is None]


def mentions(ref, name):
    """How many lines of the lesson's own source name this symbol."""
    return [line.strip() for line in inspect.getsource(ref).splitlines() if name in line]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    counts = supervision()
    product = {key: positions * context for key, (positions, context) in counts.items()}
    return {
        "counts": counts, "product": product,
        "positions": counts["causal"][0] / counts["mlm"][0],
        "context": counts["mlm"][1] / counts["causal"][1],
        "net": product["causal"] / product["mlm"],
        "absent": absent(), "cls": mentions(ref, "CLS_ID"),
        "maskable": PROB * 0.8, "corrupted": PROB * 0.9,
        "symbols": [name for name in dir(ref) if not name.startswith("_")],
    }


def verify(result):
    mlm, causal = result["counts"]["mlm"], result["counts"]["causal"]
    return [
        practice.Check(
            "ANSWER: the decoder-only baseline gets 6.67x the supervised positions",
            abs(result["positions"] - 1 / PROB) < 1e-9,
            f"MLM labels {PROB:.0%} * L = {mlm[0]:.1f} positions per sequence at L={LENGTH}; a "
            f"causal LM labels all {causal[0]:.0f}. The ratio is 1/mask_prob = "
            f"{result['positions']:.2f}x and it does not depend on model size, so 'at matched "
            "params' fixes everything except the one quantity that differs",
        ),
        practice.Check(
            "FINDING: BERT's compensation is context, and it is worth 1.97x",
            abs(result["context"] - 1.97) < 0.02,
            f"each masked position conditions on all L-1 = {mlm[1]:.0f} other tokens; a causal "
            f"position i conditions on i, averaging (L+1)/2 = {causal[1]:.1f}. That is "
            f"{result['context']:.2f}x more context per supervised position -- real, and a third "
            "of what the position count gives away",
        ),
        practice.Check(
            "FINDING: on supervised context-tokens the decoder still leads 3.4x",
            abs(result["net"] - 3.39) < 0.02,
            f"positions x context: {result['product']['mlm']:,.0f} for MLM against "
            f"{result['product']['causal']:,.0f} for causal, {result['net']:.2f}x. That ratio, "
            "and not the architecture, is the reason decoder-only pretraining won the scaling "
            "race -- both models are the same transformer with a different mask",
        ),
        practice.Check(
            "CONTROL: the whole setup is absent, including the [CLS] to fine-tune",
            result["absent"] == list(NEEDED) and len(result["cls"]) == 2,
            f"find_spec is None for {result['absent']}, there is no SST-2 and no network to fetch "
            f"one, and the module's {len(result['symbols'])} symbols include no encoder and no "
            f"training loop. CLS_ID appears on exactly {len(result['cls'])} lines -- "
            f"{result['cls']} -- its own definition and SPECIAL_IDS, so every masking path skips "
            "it and nothing ever pools it. There is no [CLS] representation to fine-tune",
        ),
        practice.Check(
            "CONTROL: 12% of pretraining tokens carry a symbol that never occurs downstream",
            abs(result["maskable"] - 0.12) < 1e-12,
            f"{PROB:.0%} x 80% = {result['maskable']:.0%} of tokens become [MASK], and an SST-2 "
            f"input contains none. Only {result['corrupted']:.1%} are corrupted at all, so "
            f"{1 - PROB:.1%} of every sequence is pushed through the encoder for no gradient: the "
            "compute is spent and the supervision is not. A causal LM has neither gap",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
