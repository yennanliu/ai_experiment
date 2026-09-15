"""Exercise 4 — the mask covers all three assistant turns, and all three user turns with them.

    Implement multi-turn conversation training. Extend the tokenization to
    handle 3-turn conversations (user-assistant-user-assistant-user-assistant).
    The loss mask should cover all three assistant turns. Verify the mask is
    correct by printing the token-mask alignment for one example.

Reading of the exercise: the tokenization is extended the obvious way --
concatenate three `tokenize_instruction_pair` blocks -- and then the lesson's
own `create_loss_mask` is run over the result, because "verify the mask is
correct" is an instruction to check rather than to assume. The alignment is
printed as the exercise asks, and then read.

**ANSWER: the mask does cover all three assistant turns.** It also covers the
second and third user turns, both `INST_START` markers after the first, and both
`INST_END` markers. On the conversation A/B, C/D, E/F it masks in every token
from the first response onward:

    tokens  253  65 254 255  66 | 253  67 254 255  68 | 253  69 254 255  70
    mask      0   0   0   0   1 |   1   1   1   0   1 |   1   1   1   0   1
                              B         C                     E

**MECHANISM: `in_response` is set and never cleared.** The loop sets it `True`
at the first `RESP_START` and has no branch that sets it back, so from token 5
onward the only thing that can zero the mask is the `continue` on a later
`RESP_START` -- which is why positions 8 and 13 are the only zeros after the
first turn. The flag is a latch, and single-turn data never shows it, because a
single-turn conversation has nothing after the response.

**FINDING: 6 of the 9 masked-in positions are not assistant text.** The mask
selects 9 tokens: 3 are the assistant bytes B, D and F that the exercise wants,
2 are the user bytes C and E, and 4 are turn markers. The model is trained to
predict the user's next message and the structural tokens that introduce it.

**FINDING: the loss this produces is the wrong objective, not a noisier one.**
Training on the user turn teaches the model to write both halves of the
conversation -- the exact failure that instruction masking exists to prevent,
reintroduced by the function that implements the masking.

Structure: `conversation` concatenates three pairs; `alignment` is the printed
token-mask table the exercise asks for, kept as data so it can be asserted on.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "06-instruction-tuning-sft"
TURNS = (("A", "B"), ("C", "D"), ("E", "F"))


def conversation(ref, turns):
    """Three user/assistant pairs, tokenised the way the lesson tokenises one."""
    tokens = []
    for instruction, response in turns:
        tokens += ref.tokenize_instruction_pair(instruction, response)
    return tokens


def spans(ref, turns):
    """Per turn, the token index ranges of its instruction bytes and response bytes."""
    user, assistant, offset = [], [], 0
    for instruction, response in turns:
        block = ref.tokenize_instruction_pair(instruction, response)
        user += [offset + 1 + i for i in range(len(instruction.encode("utf-8")))]
        assistant += [offset + len(block) - len(response.encode("utf-8")) + i
                      for i in range(len(response.encode("utf-8")))]
        offset += len(block)
    return user, assistant


def alignment(ref, tokens, mask):
    """The token-mask table the exercise asks to print, as rows rather than output."""
    return [(i, int(token), int(flag)) for i, (token, flag) in enumerate(zip(tokens, mask))]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    tokens = conversation(ref, TURNS)
    mask = ref.create_loss_mask(tokens)
    user, assistant = spans(ref, TURNS)
    selected = [i for i, flag in enumerate(mask) if flag]
    markers = set(ref.SPECIAL_TOKENS.values())
    single = conversation(ref, TURNS[:1])
    return {
        "rows": alignment(ref, tokens, mask),
        "selected": selected,
        "assistant_covered": all(i in selected for i in assistant),
        "user_leaked": [i for i in user if i in selected],
        "marker_leaked": [i for i in selected if tokens[i] in markers],
        "assistant": assistant,
        "single_clean": [i for i, f in enumerate(ref.create_loss_mask(single)) if f]
                        == spans(ref, TURNS[:1])[1],
        "tokens": tokens,
    }


def verify(result):
    selected, assistant = result["selected"], result["assistant"]
    user_leaked, marker_leaked = result["user_leaked"], result["marker_leaked"]
    tokens = result["tokens"]
    return [
        practice.Check(
            "ANSWER: all three assistant turns are covered -- and so is everything after the first",
            result["assistant_covered"] and len(selected) > len(assistant),
            f"the mask selects positions {selected} of {len(tokens)}. The {len(assistant)} "
            f"assistant bytes at {assistant} are all in, which is what the exercise asks for, "
            f"and so are {len(selected) - len(assistant)} positions that are not assistant text. "
            "Everything from the first response onward is masked in",
        ),
        practice.Check(
            "MECHANISM: in_response is a latch -- it is set once and never cleared",
            [i for i, t, f in result["rows"] if not f] == [0, 1, 2, 3, 8, 13],
            "create_loss_mask sets in_response = True at the first RESP_START and has no branch "
            "that sets it back, so after position 4 the only zeros left are the `continue` on a "
            "later RESP_START: positions "
            + str([i for i, t, f in result["rows"] if not f][-2:])
            + ". Single-turn data never shows this, because a single-turn conversation has "
            "nothing after its response",
        ),
        practice.Check(
            "FINDING: the user's own turns are masked in, along with the turn markers",
            len(user_leaked) == 2 and len(marker_leaked) == 4,
            f"of the {len(selected)} masked-in positions, {len(assistant)} are the assistant "
            f"bytes the exercise wants, {len(user_leaked)} are user bytes at {user_leaked}, and "
            f"{len(marker_leaked)} are turn markers at {marker_leaked}. The model is trained to "
            "predict the user's next message and the structural tokens that introduce it",
        ),
        practice.Check(
            "CONTROL: the single-turn mask is exactly right, which is why this survives",
            result["single_clean"],
            "on one turn the mask selects the response bytes and nothing else, so every test the "
            "lesson runs passes. The latch only becomes visible at the second turn -- and the "
            "objective it produces is not a noisier version of instruction tuning but the "
            "opposite one: a model trained to write both halves of the conversation, which is "
            "the failure response masking exists to prevent",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
