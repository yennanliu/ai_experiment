"""Exercise 3 -- the lesson's bleu4 never crashes on an empty caption; it crashes on 521 of 2,000 inputs with no references.

    Add a NaN-safe variant of `bleu4` that handles empty generated sequences without crashing.

Reading of the exercise: `bleu4_safe` wraps the lesson's `bleu4` (not a copy
of it) and makes it total: `None` or empty captions score 0.0, empty
reference captions are dropped, no usable reference scores 0.0, and a
non-finite result becomes 0.0. It is tested on four named edge cases and on
2,000 seeded random inputs whose captions, references and reference lists may
each be empty, beside the unwrapped `bleu4`. The smoothing it inherits is then
checked against the definition the lesson names, on the captions the
lesson's own `main()` scores.

**ANSWER: `bleu4_safe` below raises on 0 of the 2,000 inputs and returns a
value in [0, 1] on all of them.** The lesson's `bleu4` already handles an
empty caption: it returns 0.0 and cannot produce NaN, because it only takes
the log of a positive ratio. What it crashes on is an empty reference list:
521 of the 2,000 raise `ValueError`, all from that cause, and 52 of those
also have an empty caption. With only empty references it scores
an 8-token caption 0.1349 against nothing; `bleu4_safe` gives 0.0.

**FINDING: the smoothing is not Chen and Cherry method 1.** The lesson says
'Chen and Cherry "method 1" (add 1 to numerator and denominator ...)'. In
NLTK's implementation of that paper
(https://raw.githubusercontent.com/nltk/nltk/develop/nltk/translate/bleu_score.py,
read 2026-09-29), method 1 adds epsilon = 0.1 to a zero numerator only; add-1
to both is method 2 (Lin and Och). On the shipped run, true method 1 scores
the untrained captions 0.0188 and the trained ones 0.0955, against the
lesson's 0.122 and 0.173. A caption sharing no token with its references
scores 0.0137 instead of 0.1186. On captions with no zero precision the two
agree exactly.

**FINDING: a perfect caption shorter than 4 tokens scores 0.** `[5, 6, 7]`
against `[[5, 6, 7]]` is 0.0 with smoothing on, because a missing 4-gram order
returns before smoothing applies. The lesson's own run never hits this,
because it always generates 8 tokens.

Structure: `bleu4_safe` is the answer; `outcome`/`fuzz` classify each call as
a value or an exception name; `bleu_method1` reuses the lesson's
`_ngrams`/`_count` and changes only the smoothing.
"""


from __future__ import annotations

import contextlib
import io
import math
import random
from collections import Counter

from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
torch.set_num_threads(1)

PHASE, LESSON = "19-capstone-projects", "63-multimodal-eval"


def bleu4_safe(ref, generated, references, smoothing=True):
    """The lesson's bleu4, but total: never raises, always a finite float in [0, 1]."""
    refs = [list(r) for r in (references or []) if r]
    if not generated or not refs:
        return 0.0
    score = ref.bleu4(list(generated), refs, smoothing)
    return score if math.isfinite(score) else 0.0


def outcome(fn, gen, refs):
    try:
        return round(fn(gen, refs), 4)  # a NaN would fail the [0, 1] test in fuzz()
    except Exception as exc:  # noqa: BLE001 - the outcome is what is measured
        return type(exc).__name__


def cases(n=2000, seed=0):
    rng = random.Random(seed)
    seq = lambda: [rng.randint(1, 6) for _ in range(rng.randint(0, 9))]  # noqa: E731
    return [(seq(), [seq() for _ in range(rng.randint(0, 3))]) for _ in range(n)]


def fuzz(fn):
    """(raised, of those with no references, of those with an empty caption, out of [0, 1])."""
    res = [(outcome(fn, g, r), g, r) for g, r in cases()]
    bad = [(g, r) for x, g, r in res if isinstance(x, str)]
    return (len(bad), sum(not r for _, r in bad), sum(not g for g, _ in bad),
            sum(not (isinstance(x, str) or 0 <= x <= 1) for x, _, _ in res))


def bleu_method1(ref, gen, refs, eps=0.1):
    """Chen and Cherry method 1 as NLTK defines it: eps added to a zero numerator only."""
    logs = []
    for n in range(1, 5):
        counts = Counter(ref._count(ref._ngrams(gen, n)))
        best = Counter()
        for r in refs:
            best |= Counter(ref._count(ref._ngrams(r, n)))  # max count over references
        if not counts:
            return 0.0
        logs.append(math.log((sum((counts & best).values()) or eps) / sum(counts.values())))
    r_len = len(min(refs, key=lambda r: (abs(len(r) - len(gen)), len(r))))
    return min(1.0, math.exp(1 - r_len / len(gen))) * math.exp(sum(logs) / 4)


def shipped_captions(ref):
    """The (caption, references) pairs the lesson's main() scores, via a restored spy."""
    calls, saved = [], ref.bleu4
    ref.bleu4 = lambda g, r, smoothing=True: calls.append((list(g), r)) or saved(g, r, smoothing)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ref.main()
    finally:
        ref.bleu4 = saved
    return calls


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    edge = {"empty": ([], [[1, 2, 3, 4]]), "no refs": ([1, 2, 3, 4], []), "empty refs": ([1] * 8, [[]]),
            "3 tokens": ([5, 6, 7], [[5, 6, 7]])}
    safe, method1 = (lambda g, r: bleu4_safe(ref, g, r)), (lambda g, r: bleu_method1(ref, g, r))
    calls = shipped_captions(ref)
    runs = {"before": calls[:50], "after": calls[50:], "disjoint": [([999] * 8, r) for _, r in calls[:50]]}
    return {
        "edge": {k: (outcome(ref.bleu4, *v), outcome(safe, *v)) for k, v in edge.items()},
        "fuzz": {"ref": fuzz(ref.bleu4), "safe": fuzz(safe)},
        "lesson": {k: round(sum(ref.bleu4(*c) for c in v) / 50, 4) for k, v in runs.items()},
        "method1": {k: round(sum(method1(*c) for c in v) / 50, 4) for k, v in runs.items()},
        "gold_parity": max(abs(method1(r[0], r) - ref.bleu4(r[0], r)) for _, r in calls),
        "doc_says_method1": '"method 1" (add 1 to numerator and denominator' in parity.doc_text(PHASE, LESSON),
    }


def verify(result):
    r, e, f = result, result["edge"], result["fuzz"]
    return [
        practice.Check(
            "ANSWER: bleu4_safe raises on 0 of 2,000 inputs; bleu4 raises on 521, all with no references",
            (f, e["empty"], e["no refs"], e["empty refs"])
            == ({"ref": (521, 521, 52, 0), "safe": (0, 0, 0, 0)}, (0.0, 0.0), ("ValueError", 0.0), (0.1349, 0.0)),
            f"fuzz (raised, no refs, empty caption, out of range): {f}; edges (lesson, safe): {e}",
        ),
        practice.Check(
            "FINDING: the smoothing is not Chen and Cherry method 1",
            (r["doc_says_method1"], r["gold_parity"], r["lesson"], r["method1"])
            == (True, 0.0, {"before": 0.122, "after": 0.173, "disjoint": 0.1186},
                {"before": 0.0188, "after": 0.0955, "disjoint": 0.0137}),
            f"lesson's smoothing {r['lesson']}; NLTK method 1 {r['method1']}; else off by {r['gold_parity']}",
        ),
        practice.Check("FINDING: a perfect caption shorter than 4 tokens scores 0", e["3 tokens"] == (0.0, 0.0),
                       f"[5, 6, 7] vs [[5, 6, 7]] (lesson, safe): {e['3 tokens']}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
