"""Exercise 1 — "the" is assembled from the right, and "th" is never a token.

    Modify the BPE tokenizer to print the vocabulary at each merge step. Watch
    how "t" + "h" becomes "th", then "th" + "e" becomes "the". Track how common
    English words get assembled piece by piece.

Reading of the exercise: the modification is already shipped -- `code/bpe.py` is
`code/main.py`'s tokenizer plus a `print` in the training loop -- so "modify the
BPE tokenizer to print" is read as *instrument the lesson's own training run and
check the narrative it asks you to watch for*. The corpus and merge count are
`main.py`'s `demo_bpe_training`: 8 sentences, 50 merges, the canonical run.

**ANSWER: the narrated path does not occur.** The three merges that build "the"
are #1 `'e'+' ' -> 'e '`, #2 `'h'+'e ' -> 'he '` and #11 `' t'+'he ' -> ' the '`.
The word is built right to left, from its ending, and what the vocabulary gets
is `' the '` -- with a space on both sides.

**FINDING: neither `'th'` nor `'the'` is ever a token.** Not at 50 merges, not at
the 235 where the corpus runs dry. `'t'+'h'` is rank 8 of the initial pairs at 7
occurrences and merge 2 consumes the `'h'` before its turn comes, so
`encode("the")` returns three single bytes -- no better than the character-level
tokenizer of Step 1. What was learned is `' the '`, delimiters included.

**MECHANISM: capitalisation splits the prefix bond and not the suffix bond.**
`'h'+'e'` occurs 12 times because it is "the" (7) *plus* "The" (5); `'t'+'h'`
gets only the lowercase half. The suffix bond of a sentence-initial word always
outranks its prefix bond, so BPE builds such words from the right.

**FINDING: without pre-tokenisation BPE learns phrases, not words.** 26 of the
50 merges contain a space and 7 span a word boundary, including `' sat on the '`.
Run to exhaustion it memorises: 235 merges turn all 387 bytes into **one token**.

**CONTROL: whitespace pre-tokenisation produces `'the'` -- still not via `'th'`.**
The same corpus merged word-internally gives `'he'` at merge 1 and `'the'` at
merge 3, with no token containing a space anywhere in the run. The exercise's
path is wrong under both readings, and wrong the same way each time.

Structure: `merge_log` names each merge as a (left, right, result) triple of
strings; `pretokenised` is the control, the same greedy criterion restricted to
word interiors.
"""

from __future__ import annotations

import collections
import contextlib
import io

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "01-tokenizers"
MERGES = 50
CORPUS = (
    "The cat sat on the mat. The cat ate the rat. "
    "The dog sat on the log. The dog ate the frog. "
    "Natural language processing is the study of how computers "
    "understand and generate human language. "
    "Tokenization is the first step in any NLP pipeline. "
    "Language models read tokens, not words. "
    "The tokenizer converts text into a sequence of integers. "
    "Each integer maps to a subword in the vocabulary."
)


def train(ref, num_merges):
    """The lesson's own `train`, with its per-merge printing swallowed."""
    tokenizer = ref.BPETokenizer()
    with contextlib.redirect_stdout(io.StringIO()):
        tokenizer.train(CORPUS, num_merges=num_merges)
    return tokenizer


def merge_log(tokenizer):
    """Each merge as (left, right, result), decoded — the vocabulary at each step."""
    return [(tokenizer.token_to_str(a), tokenizer.token_to_str(b), tokenizer.token_to_str(new))
            for (a, b), new in tokenizer.merges.items()]


def _apply(seq, pair):
    out, i = [], 0
    while i < len(seq):
        if tuple(seq[i:i + 2]) == pair:
            out.append(pair[0] + pair[1])
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def pretokenised(num_merges):
    """The control: the same greedy criterion, but pairs never cross a space."""
    words = collections.Counter(CORPUS.split())
    seqs = {w: [bytes([b]) for b in w.encode("utf-8")] for w in words}
    log = []
    for _ in range(num_merges):
        pairs = collections.Counter()
        for word, count in words.items():
            seq = seqs[word]
            for i in range(len(seq) - 1):
                pairs[(seq[i], seq[i + 1])] += count
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        log.append(b"".join(best).decode("utf-8", "replace"))
        seqs = {w: _apply(s, best) for w, s in seqs.items()}
    return log


def initial_ranking():
    """Adjacent byte pairs of the raw corpus, and their order, most frequent first."""
    stream = list(CORPUS.encode("utf-8"))
    counts = collections.Counter(zip(stream, stream[1:]))
    return counts, [bytes(p).decode("utf-8", "replace") for p, _ in counts.most_common()]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    tokenizer, full = train(ref, MERGES), train(ref, 10 * MERGES)
    log, saturated = merge_log(tokenizer), [r for _, _, r in merge_log(full)]
    results = [r for _, _, r in log]
    counts, ranked = initial_ranking()
    return {
        "path": [(i, log[i - 1]) for i in (1, 2, 11)],
        "th_rank": ranked.index("th") + 1,
        "th_count": counts[(ord("t"), ord("h"))],
        "he_count": counts[(ord("h"), ord("e"))],
        "cases": (CORPUS.count("th"), CORPUS.count("Th")),
        "ever": [t for t in ("th", "the") if t in saturated],
        "encoded_the": [tokenizer.token_to_str(t) for t in tokenizer.encode("the")],
        "spaced": sum(1 for r in results if " " in r),
        "phrases": sorted((r for r in results if " " in r.strip()), key=len),
        "saturation": len(full.merges),
        "collapsed": len(full.encode(CORPUS)),
        "bytes": len(CORPUS.encode("utf-8")),
        "control": pretokenised(MERGES),
    }


def verify(result):
    path, control = result["path"], result["control"]
    steps = ", ".join(f"#{i} {a!r}+{b!r} -> {c!r}" for i, (a, b, c) in path)
    lower, upper = result["cases"]
    return [
        practice.Check(
            "ANSWER: 'the' is assembled from the right, as ' the ', in merges 1, 2 and 11",
            [(a, b) for _, (a, b, _) in path] == [("e", " "), ("h", "e "), (" t", "he ")],
            f"the lesson's own train() builds the word in the order {steps} -- the ending "
            "first, the 't' last, and the vocabulary holds ' the ', spaced on either side",
        ),
        practice.Check(
            "FINDING: neither 'th' nor 'the' is ever a token, at any merge count",
            not result["ever"] and result["encoded_the"] == ["t", "h", "e"],
            f"'t'+'h' is rank {result['th_rank']} of the initial pairs at {result['th_count']} "
            f"occurrences and never wins -- not in the {MERGES} merges the lesson runs, not in "
            f"the {result['saturation']} that exhaust the corpus -- because merge 2 takes the "
            f"'h' first. Nor does 'the': encode('the') returns {result['encoded_the']}, three "
            "single bytes, no better than Step 1. What was learned is ' the ', delimiters in",
        ),
        practice.Check(
            "MECHANISM: capitalisation splits the prefix bond and leaves the suffix bond whole",
            result["he_count"] == lower + upper > result["th_count"],
            f"'h'+'e' occurs {result['he_count']} times because it is 'the' ({lower}) plus "
            f"'The' ({upper}); 't'+'h' gets only the lowercase half, {result['th_count']}. A "
            "sentence-initial word has its first letter split across two cases and its interior "
            "not, so the suffix bond outranks the prefix bond and BPE builds from the right",
        ),
        practice.Check(
            "FINDING: with no pre-tokenisation BPE learns phrases, and at the limit memorises",
            result["spaced"] > MERGES // 3 and result["collapsed"] == 1,
            f"{result['spaced']} of the {MERGES} merges contain a space and "
            f"{len(result['phrases'])} span a word boundary, including {result['phrases'][-1]!r}. "
            f"Space is a mergeable byte, so nothing stops a token crossing words: run to "
            f"exhaustion, {result['saturation']} merges compress all {result['bytes']} bytes "
            f"into {result['collapsed']} token, which is what Exercise 2's step prevents",
        ),
        practice.Check(
            "CONTROL: pre-tokenised, 'the' does form -- via 'he', still never via 'th'",
            control[:3] == ["he", "at", "the"] and "th" not in control,
            f"restricting merges to word interiors gives {control[:3]} as the first three, and "
            "no token containing a space anywhere in the run. The exercise's path is wrong under "
            "both readings and wrong the same way: the intermediate token is the suffix",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
