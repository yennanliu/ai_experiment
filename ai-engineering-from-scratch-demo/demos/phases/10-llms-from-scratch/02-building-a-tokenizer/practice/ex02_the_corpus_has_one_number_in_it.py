"""Exercise 2 — the comparison the exercise asks for has one number to compare on.

    **Medium:** Implement the Llama-style pre-tokenizer that splits on
    whitespace and digits but keeps leading spaces. Compare its vocabulary with
    the GPT-2 regex approach on the same corpus.

Reading of the exercise: "splits on whitespace and digits but keeps leading
spaces" is implemented literally -- `\\s*\\S+` for the unit, then each digit cut
off on its own with its leading space -- and the GPT-2 arm is the lesson's own
`pre_tokenize`, not a reimplementation. "The same corpus" is
`demo_full_tokenizer`'s, 50 merges, and because the comparison turns out to be
empty there a second, number-bearing corpus is run beside it.

**FINDING: the GPT-2 arm is not GPT-2's regex unless `regex` is installed.**
`code/main.py` compiles `\\p{L}`/`\\p{N}` behind a `try`, and falls back to
`[a-zA-Z]`/`[0-9]` on `ImportError`. Under the fallback, CJK and Hangul match
*no* alternative -- they are `\\w`, so the punctuation class excludes them -- and
`finditer` simply does not return them. "你好世界 Hello World" loses 4
characters, "빠른 갈색 여우" loses 6, and "Café naïve" comes back as
`'Caf'`, `' na'`, `'ve'`. The lesson declares `regex` nowhere, and its own
`demo_full_tokenizer` prints `Round-trip: FAIL` when it is absent.

**ANSWER: on the lesson's corpus the two pre-tokenizers are the same
tokenizer.** 49 of 50 merges identical, 188 tokens against 187. That is not a
finding about Llama; it is a finding about the corpus. Digit handling is the
whole difference between these two, and the lesson's corpus is **3 digit
characters in 358** -- the `100` in `range(100)`, once.

**CONTROL: give it numbers and the vocabularies split.** On a corpus that is 64
digits in 235 characters, shared merges fall to 37 of 50. GPT-2 learns 15
digit-bearing merges -- `' 2024'`, `' 1250'`, `' 1375'`, `'0000'` -- and Llama
learns 3, all single digits. GPT-2 compresses to 131 tokens and Llama to 152,
**16% worse**, which is the trade Llama takes deliberately: ten digit tokens
that always mean the same thing, instead of `' 2024'` being one token while
`' 2025'` is two.

Structure: `llama_units` and `ref.pre_tokenize` are the two pre-tokenizers;
`train` is one merge loop over whatever units it is given, on the lesson's own
`apply_merge`; `FALLBACK` is main.py's own `except ImportError` pattern.
"""

from __future__ import annotations

import collections
import importlib.util
import re

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "02-building-a-tokenizer"
MERGES = 50
FALLBACK = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+""")
NON_ASCII = ("你好世界 Hello World", "빠른 갈색 여우", "Café naïve", "🔥🌍🚀")
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
NUMERIC = (
    "Revenue rose to 1250 in 2023 and 1375 in 2024, up 10 percent. "
    "The model has 7000000000 parameters and 128256 vocabulary entries. "
    "Order 1250 units at 2024 each for a total of 2530000. "
    "Version 3.14 shipped on 2024-01-15 with 1375 fixes. "
)


def llama_units(text):
    """Split on whitespace and on digits, keeping the leading space on each unit."""
    units = []
    for word in re.findall(r"\s*\S+|\s+", text):
        units += re.findall(r"\s*\d|\s*\D+", word)
    return units


def count_pairs(chunks):
    """Adjacent pairs across every chunk, never crossing a chunk boundary."""
    pairs = collections.Counter()
    for seq in chunks:
        pairs.update(zip(seq, seq[1:]))
    return pairs


def train(ref, units, text):
    """`MERGES` merges over pre-tokenised units, on the lesson's own `apply_merge`."""
    chunks = [list(unit.encode("utf-8")) for unit in units(text)]
    vocab, learned = {i: bytes([i]) for i in range(256)}, []
    for step in range(MERGES):
        pairs = count_pairs(chunks)
        if not pairs:
            break
        best, new = max(pairs, key=pairs.get), 256 + step
        vocab[new] = vocab[best[0]] + vocab[best[1]]
        learned.append(vocab[new].decode("utf-8", "replace"))
        chunks = [ref.apply_merge(seq, best, new) for seq in chunks]
    return learned, sum(len(seq) for seq in chunks)


def arm(ref, text):
    """Both pre-tokenizers on one corpus: their merge lists, sizes, and digit merges."""
    gpt, gpt_tokens = train(ref, ref.pre_tokenize, text)
    llama, llama_tokens = train(ref, llama_units, text)
    return {
        "shared": len(set(gpt) & set(llama)),
        "tokens": (gpt_tokens, llama_tokens),
        "digits": (sum(c.isdigit() for c in text), len(text)),
        "gpt_digit": [s for s in gpt if any(c.isdigit() for c in s)],
        "llama_digit": [s for s in llama if any(c.isdigit() for c in s)],
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    if importlib.util.find_spec("regex") is None:
        raise practice.Skip("uv sync --extra llm  # the GPT-2 arm needs main.py's regex branch")
    lost = [(probe, len(probe) - sum(len(m.group()) for m in FALLBACK.finditer(probe)))
            for probe in NON_ASCII]
    return {
        "is_gpt2": type(ref.GPT2_PATTERN).__module__.lstrip("_") == "regex",
        "fallback_loses": sum(n for _, n in lost) > 0,
        "lost": lost,
        "reference_keeps": all("".join(ref.pre_tokenize(p)) == p for p in NON_ASCII),
        "lesson": arm(ref, CORPUS),
        "numeric": arm(ref, NUMERIC),
        "split": (ref.pre_tokenize("range(100)"), llama_units("range(100)")),
    }


def verify(result):
    lesson, numeric = result["lesson"], result["numeric"]
    digits, size = lesson["digits"]
    cost = (numeric["tokens"][1] - numeric["tokens"][0]) / numeric["tokens"][0]
    return [
        practice.Check(
            "FINDING: the GPT-2 arm is not GPT-2's regex unless `regex` happens to be installed",
            result["is_gpt2"] and result["reference_keeps"] and result["fallback_loses"],
            "main.py compiles the p{L}/p{N} pattern behind a try and falls back to ASCII classes "
            "on ImportError. Under the fallback CJK and Hangul match no alternative at all -- "
            "they are \\w, so the punctuation class excludes them -- and finditer drops them: "
            + ", ".join(f"{p!r} loses {n}" for p, n in result["lost"] if n)
            + ". The lesson declares regex nowhere, and its own demo prints Round-trip: FAIL "
            "without it. This run has the real pattern, so the comparison below is the real one",
        ),
        practice.Check(
            "ANSWER: on the lesson's corpus the two pre-tokenizers are the same tokenizer",
            lesson["shared"] >= MERGES - 2 and abs(lesson["tokens"][0] - lesson["tokens"][1]) < 5,
            f"{lesson['shared']} of {MERGES} merges are identical and the corpus encodes to "
            f"{lesson['tokens'][0]} tokens against {lesson['tokens'][1]}. The two arms differ "
            "only in where they cut, and on this text they cut in almost the same places",
        ),
        practice.Check(
            "FINDING: that is a fact about the corpus -- it has one number in it",
            digits < 0.02 * size and not lesson["gpt_digit"] + lesson["llama_digit"],
            f"digit handling is the whole difference between these two pre-tokenizers, and the "
            f"lesson's corpus is {digits} digit characters in {size} -- the 100 in range(100), "
            f"once. Neither arm learns a single digit-bearing merge in {MERGES}. GPT-2 cuts "
            f"range(100) as {result['split'][0]} and the Llama rule as {result['split'][1]}, and "
            "the corpus gives that distinction one chance to matter",
        ),
        practice.Check(
            "CONTROL: give it numbers and the vocabularies split, at a 16% compression cost",
            numeric["shared"] < lesson["shared"] - 10
            and len(numeric["gpt_digit"]) > 4 * len(numeric["llama_digit"]),
            f"on a corpus that is {numeric['digits'][0]} digits in {numeric['digits'][1]} "
            f"characters, shared merges fall to {numeric['shared']} of {MERGES}. GPT-2 learns "
            f"{len(numeric['gpt_digit'])} digit-bearing merges -- {numeric['gpt_digit'][:4]} -- "
            f"and Llama {len(numeric['llama_digit'])}, {numeric['llama_digit']}, all single "
            f"digits. GPT-2 compresses to {numeric['tokens'][0]} tokens and Llama to "
            f"{numeric['tokens'][1]}, {100 * cost:.0f}% worse: the trade is ten digit tokens "
            "that always mean the same thing, against ' 2024' being one token and ' 2025' two",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
