"""Exercise 3 — a 10% text excerpt keeps z near 8 at every context width; scattered edits fail because each one spoils K+1 positions.

    Read Kirchenbauer et al. 2023 Section 6 on robustness. Why do text
    watermarks fail under paraphrase but image watermarks survive cropping?

Reading of the exercise: the question compares two different edits, so the
answer is tested by running both edits on the lesson's text watermark.
"Paraphrase" is the reference's `paraphrase` (30% of tokens replaced at
scattered positions). "Cropping" is its text analogue: keep a contiguous 10%
of the text, as the lesson's Stable Signature claim keeps 10% of an image. The
context width K is varied by setting the reference's `K`, and a context-free
variant (one fixed green list, `green_set` replaced) closes the sweep. Twenty
seeded 1000-token texts for each setting.

**ANSWER: a text position only carries signal while its K+1-token window is
intact, and scattered edits break most windows; cropping breaks none.** The
detector rehashes the K tokens before each position to find its green list, so
one substituted token spoils its own position and the next K. At 30% and the
reference's K = 4 only 0.7^5 = 16.8% of windows survive, and mean z falls to
4.68. A contiguous 10% excerpt keeps every window inside it, so z is
0.8 x sqrt(100) = 8 in expectation at every K (measured 7.82 to 8.10). The
asymmetry is contiguous versus scattered, not image versus text: an excerpt of
watermarked text survives just as a crop of a watermarked image does.

**FINDING: widening the context makes the watermark more fragile.** Mean z
after the 30% attack is 12.46 / 8.93 / 4.68 / 1.14 at K = 1 / 2 / 4 / 8, against
the closed form 0.8 sqrt(1000) x 0.7^(K+1) = 12.40 / 8.68 / 4.25 / 1.02. A
fixed green list (no context) keeps 17.72 (closed form 17.71).

**FINDING: the robust end of the sweep gives the key away.** With one fixed
list, the 100 most frequent tokens across the twenty texts are 100% green, so
counting output tokens recovers the green list. At K = 4 that count finds
48%, chance. The paper's context hashing trades paraphrase robustness for
secrecy.

**FINDING: `K = 0` does not mean "no context" in the reference.**
`green_set` slices `prev_tokens[-K:]`, and `[-0:]` is the whole list, so K = 0
hashes the entire history: one substituted token at position 100 leaves only
the 100 positions before it scoring, and z drops from 24.94 to 4.65.

Structure: `configured()` sets K or the green function and restores both;
`attack_and_crop()` measures one setting; `frequency_leak()` counts tokens.
"""

from __future__ import annotations

import collections
import contextlib
import inspect
import math
import random

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "23-watermarking-synthid-stable-signature-c2pa"
N, TEXTS, RATIO, CROP, WIDTHS = 1000, 20, 0.3, 100, (1, 2, 4, 8)


@contextlib.contextmanager
def configured(ref, k=None, green=None, seed=0):
    saved = ref.K, ref.green_set, ref.random
    ref.K = saved[0] if k is None else k
    ref.green_set = green or saved[1]
    ref.random = random.Random(seed)
    try:
        yield
    finally:
        ref.K, ref.green_set, ref.random = saved


def fixed_green(ref):
    evens = {t for t in range(ref.VOCAB) if t % 2 == 0}
    return lambda prev: evens


def attack_and_crop(ref, **setting):
    """Mean z after the 30% attack and on a contiguous 10% crop, plus the texts."""
    marked, attacked, cropped = [], [], []
    for s in range(TEXTS):
        with configured(ref, seed=s, **setting):
            text = ref.watermarked_sample(N, [ref.random.randrange(ref.VOCAB) for _ in range(ref.K)])
            attacked.append(ref.detect(ref.paraphrase(text, RATIO)))
            start = ref.random.randrange(N - CROP)
            cropped.append(ref.detect(text[start:start + CROP + ref.K]))
        marked.append(text)
    return round(sum(attacked) / TEXTS, 2), round(sum(cropped) / TEXTS, 2), marked


def frequency_leak(ref, marked):
    """Share of the 100 most frequent tokens that are even (the fixed list's green)."""
    counts = collections.Counter(t for text in marked for t in text)
    top = sorted(counts, key=lambda t: (-counts[t], t))[:100]
    return sum(t % 2 == 0 for t in top) / 100


def whole_history(ref):
    """K = 0: z of a watermarked text before and after one edit at position 100."""
    with configured(ref, k=0, seed=0):
        text = ref.watermarked_sample(N, [ref.random.randrange(ref.VOCAB)])
        edited = text[:100] + [(text[100] + 1) % ref.VOCAB] + text[101:]
        return round(ref.detect(text), 2), round(ref.detect(edited), 2)


def predicted(k):
    return round(0.8 * math.sqrt(N) * (1 - RATIO) ** (k + 1), 2)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    sweep = {k: attack_and_crop(ref, k=k) for k in WIDTHS}
    fixed = attack_and_crop(ref, green=fixed_green(ref))
    return {
        "attacked": {k: v[0] for k, v in sweep.items()},
        "cropped": {k: v[1] for k, v in sweep.items()},
        "predicted": {k: predicted(k) for k in WIDTHS},
        "fixed": fixed[:2], "fixed_predicted": round(0.8 * math.sqrt(N) * (1 - RATIO), 2),
        "leak_fixed": frequency_leak(ref, fixed[2]), "leak_k4": frequency_leak(ref, sweep[4][2]),
        "slices_minus_k": "prev_tokens[-K:]" in inspect.getsource(ref.green_set),
        "k0": whole_history(ref), "window": round((1 - RATIO) ** (ref.K + 1), 3), "K": ref.K,
    }


def verify(result):
    att, crop, pred = result["attacked"], result["cropped"], result["predicted"]
    return [
        practice.Check(
            "ANSWER: scattered edits break the K+1 windows; a contiguous 10% crop keeps z near 8",
            (result["K"], result["window"], att[4], crop)
            == (4, 0.168, 4.68, {1: 7.82, 2: 7.92, 4: 8.1, 8: 7.96}),
            f"K = {result['K']}: {result['window']:.1%} of windows survive 30%, mean z {att[4]}; "
            f"10% crop mean z by K {crop}",
        ),
        practice.Check(
            "FINDING: widening the context makes the watermark more fragile",
            att == {1: 12.46, 2: 8.93, 4: 4.68, 8: 1.14}
            and pred == {1: 12.4, 2: 8.68, 4: 4.25, 8: 1.02}
            and result["fixed"][0] == 17.72 and result["fixed_predicted"] == 17.71,
            f"mean z after 30% by K {att}; closed form {pred}; fixed list {result['fixed'][0]} "
            f"(closed form {result['fixed_predicted']})",
        ),
        practice.Check(
            "FINDING: the robust end of the sweep gives the key away",
            result["leak_fixed"] == 1.0 and result["leak_k4"] == 0.48,
            f"top-100 tokens green: fixed list {result['leak_fixed']:.0%}, K = 4 "
            f"{result['leak_k4']:.0%}",
        ),
        practice.Check(
            "FINDING: K = 0 hashes the whole history",
            result["slices_minus_k"] and result["k0"] == (24.94, 4.65),
            f"green_set slices prev_tokens[-K:]; at K = 0, z before/after one edit at "
            f"position 100: {result['k0']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
