"""Exercise 1 — 18 of the 50 merged tokens represent nothing that survives.

    **Easy:** Add a `get_token_bytes(id)` method that shows the raw bytes for
    any token ID. Use it to inspect what your most common merged tokens actually
    represent.

Reading of the exercise: the method is already shipped -- `ProductionTokenizer`
carries `get_token_bytes` -- so the exercise is read as *use it, on the lesson's
own corpus at its own 50 merges*, which is what its second sentence asks for
anyway. `demo_full_tokenizer`'s corpus and merge count are used throughout.

**ANSWER: the most common merged tokens are short and generic.** `b're'` x5,
`b'in'` x4, then `b' la'`, `b' d'`, `b' learning'`, `b' p'`, `b'ge'`, `b' tra'`,
`b'ata'`, `b'):'` at 3 apiece. Only one of the top ten is a whole word.

**FINDING: 18 of the 50 merged tokens are never used, on the corpus they were
trained on.** And they are not random: every one of them was eaten as an operand
of a later merge -- `b' l'`, `b'mo'`, `b'mode'`, `b'he'`, `b' le'`, `b' lea'`,
`b' learn'`, `b' learnin'`. BPE reaches `b' learning'` one byte at a time and
each rung becomes a vocabulary entry the next merge makes unreachable. Being
eaten is not on its own fatal -- 13 of the 32 survivors were eaten too, `b'he'`
going into both `b'The'` and `b' the'` -- but it is the only way to die here.
36% of what was learned is scaffolding.

**FINDING: the class has two unknown-ID policies and this method is the honest
one.** `get_token_bytes(999999)` returns `b'<?>'`; `decode([999999])` returns
`''`. `decode` skips any ID it does not recognise, so `decode([72, 999999, 73])`
is `'HI'` -- a bad ID in the middle of a sequence leaves no trace at all, not a
replacement character and not an exception. The method the exercise asks you to
add is the only place in the class where an unknown ID is reported.

Structure: `consumed` is the set of token IDs some later merge used as an
operand; `trained` is the lesson's own tokenizer at its own settings.
"""

from __future__ import annotations

import collections
import contextlib
import importlib.util
import io

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "02-building-a-tokenizer"
MERGES, MISSING_ID = 50, 999_999
CORPUS = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Machine learning transforms how we build software. "
    "Deep learning models need large datasets to train. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
    "for i in range(100): print(i) "
)


def trained(ref):
    """The lesson's own tokenizer at its own settings, training output swallowed."""
    if importlib.util.find_spec("regex") is None:
        raise practice.Skip("uv sync --extra llm  # pre_tokenize's regex branch, not its "
                            "ASCII fallback, which drops every non-ASCII letter")
    tokenizer = ref.ProductionTokenizer()
    with contextlib.redirect_stdout(io.StringIO()):
        tokenizer.train(CORPUS, num_merges=MERGES)
    return tokenizer


def consumed(tokenizer):
    """Token IDs that a later merge used as one of its two operands."""
    return {token for pair in tokenizer.merges for token in pair}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    tokenizer = trained(ref)
    counts = collections.Counter(tokenizer.encode(CORPUS))
    merged, eaten = list(tokenizer.merges.values()), consumed(tokenizer)
    unused = [i for i in merged if i not in counts]
    return {
        "shipped": hasattr(ref.ProductionTokenizer, "get_token_bytes"),
        "top": [(tokenizer.get_token_bytes(i), c)
                for i, c in counts.most_common() if i >= 256][:10],
        "learned": len(merged),
        "unused": [tokenizer.get_token_bytes(i) for i in unused],
        "all_eaten": all(i in eaten for i in unused),
        "used_eaten": sum(1 for i in merged if i in counts and i in eaten),
        "used": len(merged) - len(unused),
        "missing_bytes": tokenizer.get_token_bytes(MISSING_ID),
        "missing_decode": tokenizer.decode([MISSING_ID]),
        "sandwiched": tokenizer.decode([72, MISSING_ID, 73]),
    }


def verify(result):
    unused, top = result["unused"], result["top"]
    return [
        practice.Check(
            "ANSWER: the most common merged tokens are short, generic, and mostly not words",
            result["shipped"] and top[0][0] == b"re" and len(top) == 10,
            "the method the exercise asks for is already on ProductionTokenizer, so this is "
            "its output on the lesson's own corpus at its own 50 merges: "
            + ", ".join(f"{b!r} x{c}" for b, c in top[:6])
            + f", and four more at {top[-1][1]}. One of the top ten is a whole word",
        ),
        practice.Check(
            "FINDING: 18 of the 50 merged tokens are never used on their own training corpus",
            len(unused) > MERGES // 3,
            f"{len(unused)} of the {result['learned']} merges the tokenizer learned appear in no "
            f"encoding of the text they were learned from. The vocabulary is "
            f"{result['learned']} merged entries wide and {result['learned'] - len(unused)} of "
            "them carry the corpus",
        ),
        practice.Check(
            "MECHANISM: every one of them was eaten as an operand of a later merge",
            result["all_eaten"] and result["used_eaten"] < result["used"],
            f"all {len(unused)} appear as the left or right half of some later merge: "
            + ", ".join(repr(b) for b in unused[:7])
            + f". BPE reaches b' learning' one byte at a time and each rung becomes a vocabulary "
            f"entry the next merge makes unreachable. Being eaten is not on its own fatal -- "
            f"{result['used_eaten']} of the {result['used']} surviving tokens were eaten too, "
            "b'he' going into both b'The' and b' the' -- but it is the only way to die here",
        ),
        practice.Check(
            "FINDING: the class has two unknown-ID policies, and this method is the honest one",
            (result["missing_bytes"], result["missing_decode"], result["sandwiched"])
            == (b"<?>", "", "HI"),
            f"get_token_bytes({MISSING_ID}) returns {result['missing_bytes']!r}, but "
            f"decode([{MISSING_ID}]) returns {result['missing_decode']!r} -- decode skips any ID "
            f"it does not recognise, so decode([72, {MISSING_ID}, 73]) is "
            f"{result['sandwiched']!r}. A bad ID mid-sequence leaves no replacement character "
            "and raises nothing. The method the exercise asks you to add is the only place in "
            "the class where an unknown ID is reported at all",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
