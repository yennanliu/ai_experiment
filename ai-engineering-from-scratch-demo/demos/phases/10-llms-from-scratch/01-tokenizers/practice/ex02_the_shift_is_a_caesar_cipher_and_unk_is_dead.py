"""Exercise 2 — the shift is a Caesar cipher, <unk> is unreachable, and the split is lossy.

    Add special tokens (`<pad>`, `<eos>`, `<unk>`) to the BPE tokenizer. Assign
    them IDs 0, 1, 2 and shift all other tokens accordingly. Implement a
    pre-tokenization step that splits on whitespace before running BPE.

Reading of the exercise: both halves are implemented exactly as written and then
measured against the lesson's own `compression_ratio` and its Step 3 roundtrip
test, because both are instructions whose literal execution breaks something the
lesson already checks. Corpus and merge count are `main.py`'s
`demo_bpe_training`: 8 sentences, 50 merges.

**ANSWER: done correctly the shift is a pure relabelling.** Every ID rises by
exactly 3, token counts are unchanged on all four probes, the roundtrip holds,
and the vocabulary goes 306 -> 309. Nothing measurable moves, which is the
point: special tokens buy addressability, not compression.

**FINDING: done as written it is a Caesar cipher.** "Shift all other tokens
accordingly" names the vocabulary and the merge table, but `encode` also starts
from raw byte values and must be shifted too. Shift the first two and not the
third and no merge can fire -- the shifted pair IDs no longer match unshifted
bytes -- so a 23-byte sentence encodes to 23 tokens and decodes to
`'Qeb\\x1d`^q...'`, every byte down by 3. Same length, no exception, no `<unk>`.

**FINDING: `<unk>` can never be emitted.** The base vocabulary covers all 256
bytes, so Japanese, emoji, Arabic and NUL all roundtrip exactly and token 2
appears in no encoding of anything. The lesson says this itself two sections
earlier -- byte-level BPE "never produces an unknown token" -- so one of the
three tokens the exercise asks for is an embedding row that gets no gradient.

**FINDING: the pre-tokeniser as specified is lossy, and its compression win is
the deleted text.** `text.split()` discards whitespace: 387 bytes in, 319 out,
all 68 spaces gone, so Step 3's `PASS` becomes `FAIL` on every sentence. It
scores 0.4367 against flat BPE's 0.5013 -- over a denominator of 387 it no
longer reproduces. Attach the space to the word instead (GPT-2's fix) and the
roundtrip returns and the ratio is 0.5323, **6.2% worse** than no
pre-tokenisation at all, because word boundaries forbid exactly the phrase
tokens Exercise 1 found. It is a correctness fix bought with compression.

Structure: `Shifted` is the shift done right; `caesar` is the shift done as
written; `units` is the pre-tokeniser under both readings, `split` being the
exercise's and `attach` the lossless one.
"""

from __future__ import annotations

import collections
import contextlib
import io
import re

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "01-tokenizers"
MERGES, OFFSET = 50, 3
SPECIAL = ("<pad>", "<eos>", "<unk>")
CORPUS = (
    "The cat sat on the mat. The cat ate the rat. The dog sat on the log. "
    "The dog ate the frog. Natural language processing is the study of how "
    "computers understand and generate human language. Tokenization is the "
    "first step in any NLP pipeline. Language models read tokens, not words. "
    "The tokenizer converts text into a sequence of integers. Each integer "
    "maps to a subword in the vocabulary."
)
PROBES = (CORPUS, "The cat sat on the mat.", "unhappiness", "Hello, world!")
EXOTIC = ("日本語", "\U0001f642", "الع", "\x00\x01")


class Shifted:
    """The exercise's shift, applied to `encode` as well as to the vocabulary."""

    def __init__(self, ref):
        self.ref = ref
        self.vocab = {i: s.encode("utf-8") for i, s in enumerate(SPECIAL)}
        self.vocab.update({k + OFFSET: v for k, v in ref.vocab.items()})
        self.merges = {(a + OFFSET, b + OFFSET): t + OFFSET for (a, b), t in ref.merges.items()}

    def encode(self, text, shift=OFFSET):
        """`shift=0` is the exercise done as written: the vocabulary moved, the bytes not."""
        tokens = [b + shift for b in text.encode("utf-8")]
        for pair, new in self.merges.items():
            tokens = self.ref._merge_pair(tokens, pair, new)
        return tokens

    def decode(self, tokens):
        return b"".join(self.vocab[t] for t in tokens).decode("utf-8", "replace")


def units(text, mode):
    """Whitespace pre-tokenisation: `split` as the exercise says, `attach` losslessly."""
    return text.split() if mode == "split" else re.findall(r"\s*\S+|\s+", text)


def pretokenised(shim, mode):
    """`MERGES` merges that never cross a unit boundary, on the lesson's `_merge_pair`."""
    pieces = collections.Counter(units(CORPUS, mode))
    seqs = {p: list(p.encode("utf-8")) for p in pieces}
    vocab = {i: bytes([i]) for i in range(256)}
    for step in range(MERGES):
        pairs = collections.Counter()
        for piece, count in pieces.items():
            for pair in zip(seqs[piece], seqs[piece][1:]):
                pairs[pair] += count
        if not pairs:
            break
        best, new = max(pairs, key=pairs.get), 256 + step
        vocab[new] = vocab[best[0]] + vocab[best[1]]
        seqs = {p: shim._merge_pair(s, best, new) for p, s in seqs.items()}
    return seqs, vocab


def arm(shim, mode):
    """`CORPUS` under `mode`'s pre-tokeniser: its token count, and what decode recovers."""
    seqs, vocab = pretokenised(shim, mode)
    encoded = [t for piece in units(CORPUS, mode) for t in seqs[piece]]
    return {"ratio": len(encoded) / len(CORPUS.encode("utf-8")),
            "recovered": b"".join(vocab[t] for t in encoded).decode("utf-8", "replace")}


def compare(base, shifted):
    """Everything the shift done right claims, measured over PROBES and EXOTIC."""
    return {
        "offsets": sorted({i - j for p in PROBES
                          for i, j in zip(shifted.encode(p), base.encode(p))}),
        "counts": all(len(shifted.encode(p)) == len(base.encode(p)) for p in PROBES),
        "shift_roundtrip": all(shifted.decode(shifted.encode(p)) == p for p in PROBES),
        "unk_fires": any(2 in shifted.encode(p) for p in PROBES + EXOTIC),
        "exotic_roundtrip": all(shifted.decode(shifted.encode(x)) == x for x in EXOTIC),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    base = ref.BPETokenizer()
    with contextlib.redirect_stdout(io.StringIO()):
        base.train(CORPUS, num_merges=MERGES)
    shifted = Shifted(base)
    arms = {mode: arm(base, mode) for mode in ("split", "attach")}
    naive = shifted.encode(PROBES[1], shift=0)
    return dict(compare(base, shifted), arms=arms,
                vocab=(base.vocab_size(), len(shifted.vocab)),
                caesar=(len(naive), shifted.decode(naive)),
                flat_ratio=ref.compression_ratio(base, CORPUS),
                bytes=len(CORPUS.encode("utf-8")))


def verify(result):
    split, attach = result["arms"]["split"], result["arms"]["attach"]
    fired, garbled, flat = *result["caesar"], result["flat_ratio"]
    lost = result["bytes"] - len(split["recovered"].encode())
    cost = (attach["ratio"] - flat) / flat
    return [
        practice.Check(
            "ANSWER: the shift, done to encode() as well, is a pure relabelling",
            result["offsets"] == [OFFSET] and result["shift_roundtrip"],
            f"every ID rises by exactly {OFFSET} on all {len(PROBES)} probes, token counts are "
            f"unchanged ({result['counts']}), the roundtrip holds and the vocabulary goes "
            f"{result['vocab'][0]} -> {result['vocab'][1]}: the rows buy addressability, not "
            "compression",
        ),
        practice.Check(
            "FINDING: shifting the vocabulary but not encode() turns the tokenizer into ROT-3",
            garbled != PROBES[1] and fired == len(PROBES[1].encode("utf-8")),
            f"the phrase names the vocabulary and the merge table; encode() also starts from raw "
            f"bytes. Shift the first two only and {PROBES[1]!r} encodes to {fired} tokens for "
            f"{fired} bytes -- no merge fires, their shifted pair IDs no longer matching the "
            f"bytes -- and decodes to {garbled!r}: no exception, no <unk>, a silent ID bug",
        ),
        practice.Check(
            "FINDING: <unk> is unreachable -- a third of the exercise's special tokens is dead",
            not result["unk_fires"] and result["exotic_roundtrip"],
            "the base vocabulary covers all 256 bytes, so Japanese, emoji, Arabic and NUL all "
            "roundtrip exactly and token 2 appears in no encoding of anything -- which the lesson "
            "states two sections earlier. <pad> and <eos> are real; <unk> is a row with no gradient",
        ),
        practice.Check(
            "FINDING: the split is lossy, and the honest pre-tokeniser costs 6.2%",
            split["recovered"] != CORPUS == attach["recovered"] and 0.05 < cost < 0.08,
            f"text.split() drops every separator: {result['bytes']} bytes in, "
            f"{len(split['recovered'].encode())} out, all {lost} spaces gone, so Step 3's PASS "
            f"becomes FAIL on every sentence and its {split['ratio']:.4f} against flat BPE's "
            f"{flat:.4f} is scored over a denominator it no longer reproduces. Attach the space "
            f"to the word, as GPT-2 does, and it returns at {attach['ratio']:.4f}, "
            f"{100 * cost:.1f}% worse than none, word boundaries forbidding the phrase tokens "
            "Exercise 1 found",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
