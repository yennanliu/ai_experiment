"""Exercise 3 — the likelihood ratio is a rarity detector, and BPE wins the comparison.

    Implement the WordPiece merge criterion (likelihood ratio instead of
    frequency). Train both BPE and WordPiece on the same corpus with the same
    number of merges. Compare the resulting vocabularies -- which one produces
    more linguistically meaningful subwords?

Reading of the exercise: "the WordPiece merge criterion" is taken verbatim from
the lesson's own Key Terms row, `count(AB)/(count(A)*count(B))`, and both arms
run through the lesson's own `_get_pairs` and `_merge_pair` -- the BPE arm is
checked merge-for-merge against the reference's `train`, so only the scoring
function differs. "More linguistically meaningful" is scored mechanically or it
is an opinion: a merged token counts as **reusable** if its stripped form occurs
inside more than one distinct word of the corpus, which is what a subword is for.

**ANSWER: BPE, decisively, and the exercise implies the opposite.** 34 of BPE's
50 merges are reusable against WordPiece's 3, and BPE compresses the corpus to
0.5013 tokens per byte against 0.8527 -- 70% worse. WordPiece's vocabulary here
is `'LP'`, `'bw'`, `'qu'`, `'ubw'`, `'bul'`, `'s,'`; BPE's is `'e '`, `'he '`,
`'at'`, `'. '`, `'an'`, `'in'`, `'er'`.

**MECHANISM: dividing by `count(A)*count(B)` removes the frequency floor.** The
ratio is maximised at 1.0 by a pair whose two parts each occur once and always
together -- a score no frequent pair can reach. So the criterion selects hapaxes:
46 of its 50 merges fire on a pair occurring exactly **once**, mean 1.14 against
BPE's 3.86, and the first merge is `'LP'`, from the single occurrence of "NLP".

**FINDING: the formula is right and the setting is wrong.** Real WordPiece
carries a minimum-count threshold. Restore one and the degeneracy goes: at
`count(AB) >= 2` the hapax merges drop from 46 to **0**, reusable merges rise
from 3 to 30, and compression improves 0.8527 -> 0.6176. It is still 23% worse
than BPE, which is the honest trade -- the criterion buys a different notion of
subword and pays compression for it -- but it is no longer a rarity detector.

**FINDING: the merge path Exercise 1 attributes to BPE is this criterion's.**
At a floor of 5 the ratio yields `'T'+'h' -> 'Th'`, `'Th'+'e' -> 'The'`, then
`'t'+'h' -> 'th'` and `'th'+'e' -> 'the'` -- consecutively, exactly the sequence
the lesson narrates for BPE, which BPE produces at no merge count on this corpus.
`'t'` is rarer than `'e'` or `' '`, so dividing by the parts is what promotes the
t-h bond that raw frequency never does.

Structure: `merge_run` is one training loop parameterised by scoring function and
count floor; `RATIO` and `FREQUENCY` are the two criteria; `reusable` is the
mechanical reading of "linguistically meaningful".
"""

from __future__ import annotations

import collections
import contextlib
import io

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "01-tokenizers"
MERGES = 50
CORPUS = (
    "The cat sat on the mat. The cat ate the rat. The dog sat on the log. "
    "The dog ate the frog. Natural language processing is the study of how "
    "computers understand and generate human language. Tokenization is the "
    "first step in any NLP pipeline. Language models read tokens, not words. "
    "The tokenizer converts text into a sequence of integers. Each integer "
    "maps to a subword in the vocabulary."
)
WORDS = [w.strip(".,") for w in CORPUS.split()]


def FREQUENCY(pairs, units, pair):
    """BPE: raw count of the pair."""
    return pairs[pair]


def RATIO(pairs, units, pair):
    """WordPiece, as the lesson's Key Terms row states it."""
    return pairs[pair] / (units[pair[0]] * units[pair[1]])


def merge_run(ref, score, floor=1, num_merges=MERGES):
    """One training loop on the lesson's own `_get_pairs` / `_merge_pair`."""
    shim = ref.BPETokenizer()
    tokens = list(CORPUS.encode("utf-8"))
    vocab = {i: bytes([i]) for i in range(256)}
    log = []
    for step in range(num_merges):
        pairs = {p: c for p, c in shim._get_pairs(tokens).items() if c >= floor}
        if not pairs:
            break
        units = collections.Counter(tokens)
        best = max(pairs, key=lambda p: score(pairs, units, p))
        new = 256 + step
        vocab[new] = vocab[best[0]] + vocab[best[1]]
        log.append((vocab[best[0]], vocab[best[1]], vocab[new], pairs[best]))
        tokens = shim._merge_pair(tokens, best, new)
    return log, len(tokens) / len(CORPUS.encode("utf-8"))


def text(entry, field=2):
    return entry[field].decode("utf-8", "replace")


def reusable(log):
    """Merged tokens whose stripped form occurs inside more than one corpus word."""
    return [e for e in log
            if text(e).strip()
            and len({w for w in WORDS if text(e).strip() in w}) > 1]


def arm(ref, score, floor=1, num_merges=MERGES):
    log, ratio = merge_run(ref, score, floor, num_merges)
    path = [(text(e, 0), text(e, 1), text(e)) for e in log]
    return {
        "tokens": [text(e) for e in log],
        "head": [(a, b) for a, b, _ in path[:4]],
        "narrated": " then ".join(f"{a!r}+{b!r} -> {c!r}" for a, b, c in path[:4]),
        "ratio": ratio,
        "hapax": sum(1 for e in log if e[3] == 1),
        "mean_count": sum(e[3] for e in log) / len(log),
        "reusable": len(reusable(log)),
        "merges": len(log),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    shipped = ref.BPETokenizer()
    with contextlib.redirect_stdout(io.StringIO()):
        shipped.train(CORPUS, num_merges=MERGES)
    bpe = arm(ref, FREQUENCY)
    return {
        "bpe": bpe,
        "wordpiece": arm(ref, RATIO),
        "floored": arm(ref, RATIO, floor=2),
        "strict": arm(ref, RATIO, floor=5),
        "agrees": bpe["tokens"] == [shipped.token_to_str(t) for t in shipped.merges.values()],
    }


def verify(result):
    bpe, wp = result["bpe"], result["wordpiece"]
    floored, strict = result["floored"], result["strict"]
    gain = (wp["ratio"] - bpe["ratio"]) / bpe["ratio"]
    residual = (floored["ratio"] - bpe["ratio"]) / bpe["ratio"]
    return [
        practice.Check(
            "ANSWER: BPE produces the more meaningful subwords, by 34 reusable merges to 3",
            result["agrees"] and bpe["reusable"] > 10 * wp["reusable"]
            and bpe["ratio"] < wp["ratio"],
            f"of {MERGES} merges each, {bpe['reusable']} of BPE's occur inside more than one "
            f"distinct corpus word against {wp['reusable']} of WordPiece's, and BPE "
            f"compresses to {bpe['ratio']:.4f} tokens per byte against {wp['ratio']:.4f}, "
            f"{100 * gain:.0f}% worse. BPE's first merges are {bpe['tokens'][:6]}; "
            f"WordPiece's are {wp['tokens'][:6]}. The BPE arm is the lesson's own train(), "
            "merge for merge -- only the scoring function differs between the arms",
        ),
        practice.Check(
            "MECHANISM: dividing by the parts removes the frequency floor, so it selects hapaxes",
            wp["hapax"] > 0.8 * MERGES and bpe["hapax"] == 0
            and wp["mean_count"] < 1.5 < bpe["mean_count"],
            f"{wp['hapax']} of WordPiece's {MERGES} merges fire on a pair occurring exactly "
            f"once, against {bpe['hapax']} of BPE's; mean count at merge time is "
            f"{wp['mean_count']:.2f} against {bpe['mean_count']:.2f}. count(AB)/(count(A)*"
            "count(B)) reaches its maximum of 1.0 when both parts occur once and always "
            f"together, which no frequent pair can match -- so the first merge is "
            f"{wp['tokens'][0]!r}, from the single occurrence of 'NLP'. The criterion "
            "measures surprise per occurrence and is then asked to rank by it, unweighted",
        ),
        practice.Check(
            "FINDING: a minimum-count threshold removes the degeneracy and leaves the real trade",
            floored["hapax"] == 0 and floored["reusable"] > 8 * wp["reusable"]
            and wp["ratio"] > floored["ratio"] > bpe["ratio"],
            f"requiring count(AB) >= 2 -- the threshold real WordPiece implementations carry "
            f"and the lesson's one-line formula omits -- takes hapax merges from "
            f"{wp['hapax']} to {floored['hapax']}, reusable merges from {wp['reusable']} to "
            f"{floored['reusable']}, and compression from {wp['ratio']:.4f} to "
            f"{floored['ratio']:.4f}. That is still {100 * residual:.0f}% worse than BPE, "
            "which is the honest answer to the exercise: the criterion buys a different "
            "notion of subword and pays compression for it. The collapse was the setting",
        ),
        practice.Check(
            "FINDING: Exercise 1's narrated path is this criterion's, not BPE's",
            strict["head"] == [("T", "h"), ("Th", "e"), ("t", "h"), ("th", "e")],
            f"at a floor of 5 the ratio yields {strict['narrated']}. The lesson narrates "
            "'t'+'h' -> 'th', 'th'+'e' -> 'the' as what BPE does, and BPE produces neither "
            "token at any merge count -- but this criterion produces both, consecutively. 't' "
            "is rarer than 'e' or ' ', so dividing by the parts promotes the t-h bond that raw "
            "frequency never reaches",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
