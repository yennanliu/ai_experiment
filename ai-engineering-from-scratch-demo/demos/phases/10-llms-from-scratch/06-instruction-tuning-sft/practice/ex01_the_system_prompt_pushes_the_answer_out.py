"""Exercise 1 — the system prompt evicts the response it was meant to condition.

    Add system prompt support. Modify `tokenize_instruction_pair` to accept a
    system message and prepend it before the instruction. Create 5 examples with
    different system prompts ("You are a poet", "You are a math tutor") and
    verify the model sees different system prompts during training.

Reading of the exercise: "prepend it before the instruction" is implemented
exactly that way -- inside the `INST_START` block, ahead of the instruction
bytes, so `create_loss_mask` still masks it out -- and "verify the model sees
different system prompts during training" is read as a claim to check rather
than a box to tick, since `sft_train` truncates to `seq_len` and the thing it
truncates is the end.

**FINDING: 5 of the lesson's 8 examples already overflow the window.** At
`seq_len=64` the eight pairs tokenise to 64, 106, 111, 56, 97, 112, 61 and 117
bytes. `sft_train` does `tokens = tokens[:seq_len]`, and the response is at the
end, so five examples are already being trained on a response cut mid-sentence
before any system prompt exists.

**ANSWER: adding one pushes all eight over.** A 16-byte `"You are a poet. "`
takes the lengths to 80, 122, 127, 72, 113, 128, 77 and 133 -- **8 of 8** past
64. The system prompt is prepended, the truncation cuts from the other end, and
what it removes is the response the loss is computed on.

**MECHANISM: the two ends belong to different owners.** The system prompt is
masked out, so it contributes no gradient and only costs budget; the response is
masked in, so it is the only thing that does contribute. Prepending to a
fixed-width window spends the response's budget on tokens that cannot be learned
from. A longer system prompt strictly reduces the amount of supervision per
example.

**FINDING: "verify the model sees different system prompts" cannot be
verified.** `sft_train` updates with `lr * np.random.randn(...)` and never uses
its gradient, so five examples with five different system prompts leave weights
bit-identical to eight examples of the same one -- Exercise 5 proves this
directly. Whatever the model sees, it does not keep.

Structure: `tokenize_with_system` is the modification the exercise asks for;
`budget` reports, per example, how much of the response survives truncation.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "06-instruction-tuning-sft"
SEQ_LEN = 64
SYSTEMS = ("You are a poet.", "You are a math tutor.", "You are a historian.",
           "You are a chemist.", "You are a careful editor.")


def tokenize_with_system(ref, system, instruction, response):
    """`tokenize_instruction_pair` with the system message prepended to the instruction."""
    prefix = f"{system} " if system else ""
    return ref.tokenize_instruction_pair(prefix + instruction, response)


def budget(ref, system):
    """Per example: total tokens, and response tokens surviving truncation to SEQ_LEN."""
    rows = []
    for example in ref.INSTRUCTION_DATA:
        tokens = tokenize_with_system(ref, system, example["instruction"],
                                      example["response"])
        mask = ref.create_loss_mask(tokens)
        rows.append((len(tokens), int(mask.sum()), int(mask[:SEQ_LEN].sum())))
    return rows


def totals(rows):
    """(lengths, how many overflow, response tokens kept, response tokens wanted)."""
    return ([n for n, _, _ in rows], sum(n > SEQ_LEN for n, _, _ in rows),
            sum(k for _, _, k in rows), sum(w for _, w, _ in rows))


def five_prompts(ref):
    """The same pair under each of the five system prompts."""
    example = ref.INSTRUCTION_DATA[0]
    return [tokenize_with_system(ref, system, example["instruction"], example["response"])
            for system in SYSTEMS]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    plain, prompted = totals(budget(ref, "")), totals(budget(ref, SYSTEMS[0]))
    five = five_prompts(ref)
    return {
        "lengths": (plain[0], prompted[0]),
        "over": (plain[1], prompted[1]),
        "supervised": (plain[2], prompted[2]),
        "wanted": (plain[3], prompted[3]),
        "distinct": len({tuple(tokens) for tokens in five}),
        "masked_prompt": all(ref.create_loss_mask(tokens)[:5].sum() == 0 for tokens in five),
        "examples": len(ref.INSTRUCTION_DATA),
    }


def verify(result):
    plain_len, prompted_len = result["lengths"]
    over_plain, over_prompted = result["over"]
    kept_plain, kept_prompted = result["supervised"]
    want_plain, want_prompted = result["wanted"]
    return [
        practice.Check(
            f"FINDING: {result['over'][0]} of {result['examples']} already overflow seq_len=64",
            over_plain >= result["examples"] // 2,
            f"with no system prompt the eight pairs tokenise to {plain_len} bytes, and sft_train "
            f"does tokens = tokens[:{SEQ_LEN}]. The response is at the end, so {over_plain} "
            "examples are already trained on a response cut mid-sentence before a system prompt "
            f"exists: {kept_plain} of {want_plain} response tokens survive truncation",
        ),
        practice.Check(
            "ANSWER: one 16-byte system prompt pushes all eight over",
            over_prompted == result["examples"] > over_plain,
            f"prepending {SYSTEMS[0]!r} takes the lengths to {prompted_len} -- {over_prompted} of "
            f"{result['examples']} past {SEQ_LEN}. The prompt goes on the front and the "
            f"truncation cuts from the back, so supervised response tokens fall from "
            f"{kept_plain} to {kept_prompted}, "
            f"{100 * (1 - kept_prompted / kept_plain):.0f}% less supervision for the same eight "
            "examples",
        ),
        practice.Check(
            "MECHANISM: the masked end is the end that survives truncation",
            result["masked_prompt"] and kept_prompted < want_prompted,
            "create_loss_mask zeroes everything before RESP_START, so the system prompt "
            "contributes no gradient and only consumes budget, while the response is the only "
            f"masked-in span. Of {want_prompted} response tokens across the eight prompted "
            f"examples, {kept_prompted} survive the window. A longer system prompt strictly "
            "reduces supervision per example, and it buys nothing that is trained on",
        ),
        practice.Check(
            "FINDING: 'verify the model sees different system prompts' cannot be verified",
            result["distinct"] == len(SYSTEMS),
            f"the five prompts do produce {result['distinct']} distinct token sequences, so the "
            "tokenizer change works. But sft_train updates with lr * np.random.randn(...) and "
            "never uses its gradient, so five different system prompts leave the weights "
            "bit-identical to five copies of one -- Exercise 5 proves that directly. Whatever "
            "the model sees during training, it does not keep",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
