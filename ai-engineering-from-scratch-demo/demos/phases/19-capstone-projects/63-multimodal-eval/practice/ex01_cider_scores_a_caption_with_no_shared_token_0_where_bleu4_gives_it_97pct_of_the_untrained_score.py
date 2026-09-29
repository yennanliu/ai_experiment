"""Exercise 1 -- CIDEr scores a caption with no shared token 0 where the lesson's BLEU-4 gives it 97% of the untrained model's score.

    Add CIDEr to the captioning metrics. CIDEr uses TF-IDF weighting on n-grams, which rewards informative tokens.

Reading of the exercise: "CIDEr" is taken as CIDEr-D, the variant the
MS-COCO caption evaluation ships (pycocoevalcap `cider/cider_scorer.py`,
https://raw.githubusercontent.com/salaniz/pycocoevalcap/master/cider/cider_scorer.py,
read 2026-09-29): 1- to 4-gram tf-idf vectors with document frequency counted
over the eval set's references, hypothesis counts clipped to the reference,
cosine per order, a Gaussian length penalty on bigram counts (sigma 6),
averaged over orders and references and scaled by 10. It is added next to the
lesson's `bleu4` and scored on exactly the captions the lesson's `main()`
generates: `bleu4` is wrapped (and restored) so every greedy caption and its
references are recorded as the shipped run scores them, before and after the
50 training steps.

**ANSWER: CIDEr-D is `cider_d()` below; on the shipped run it moves from 0.0354
to 0.335, against a ceiling of 5.79.** The ceiling is each image's first
reference used as its caption (the other two references are token-shifted
variants, so even a perfect copy does not reach 10). A self-check confirms
the scale: a caption equal to its only reference, with no n-gram shared across
images, scores exactly 10.

**FINDING: BLEU-4's smoothing floor makes the untrained model look 70% as good
as the trained one; CIDEr says 11%.** A caption of eight tokens that occur in
no reference (`[999] * 8`) gets mean BLEU-4 0.1186, which is 97% of the
untrained model's 0.1220. CIDEr gives it 0. On CIDEr the untrained model is at
0.0354, 10.6% of the trained 0.335; on BLEU it is at 70.5% (0.122 of 0.173).

**FINDING: the trained captioner has collapsed to 4 captions for 50 images.**
22 of its 50 captions share no n-gram with their references and score CIDEr 0.
CIDEr also cannot score a one-image eval: with N = 1 every idf is log 1 = 0,
so even a perfect caption scores 0.

Structure: `run_lesson()` runs the shipped `main()` with the bleu4 spy;
`grams`/`tfidf`/`sim`/`cider_d` follow the cider_scorer functions of the same
roles.
"""


from __future__ import annotations

import contextlib
import io
import math
from collections import Counter

from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
torch.set_num_threads(1)

PHASE, LESSON = "19-capstone-projects", "63-multimodal-eval"


def run_lesson(ref):
    """The lesson's main() as shipped, recording every (caption, references) bleu4 scores."""
    calls, saved = [], ref.bleu4

    def spy(gen, refs, smoothing=True):
        calls.append((list(gen), refs))
        return saved(gen, refs, smoothing)

    ref.bleu4 = spy
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            ref.main()
    finally:
        ref.bleu4 = saved
    return calls[:50], calls[50:]


def grams(seq):
    return [Counter(tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)) for n in range(1, 5)]


def tfidf(counts, df, log_n):
    """counts2vec: per-order tf-idf vectors, their norms, and the bigram count as length."""
    vec = [{g: tf * (log_n - math.log(max(1.0, df[g]))) for g, tf in c.items()} for c in counts]
    return vec, [math.sqrt(sum(x * x for x in v.values())) for v in vec], sum(counts[1].values())


def sim(hyp, ref_, sigma=6.0):
    (vh, nh, lh), (vr, nr, lr) = hyp, ref_
    val = [sum(min(x, vr[n].get(g, 0.0)) * vr[n].get(g, 0.0) for g, x in vh[n].items())
           / (nh[n] * nr[n]) if nh[n] and nr[n] else 0.0 for n in range(4)]
    return sum(val) / 4 * math.exp(-((lh - lr) ** 2) / (2 * sigma**2))


def doc_freq(docs):
    return Counter(g for refs in docs for g in {g for rg in refs for c in rg for g in c})


def cider_d(hyps, refsets):
    """CIDEr-D as pycocoevalcap computes it; document frequency over the eval set's references."""
    docs = [[grams(r) for r in refs] for refs in refsets]
    df = doc_freq(docs)
    log_n = math.log(len(refsets))
    out = []
    for hyp, refs in zip(hyps, docs):
        h = tfidf(grams(hyp), df, log_n)
        out.append(10.0 * sum(sim(h, tfidf(rg, df, log_n)) for rg in refs) / len(refs))
    return out


def mean(xs):
    return round(sum(xs) / len(xs), 4)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    before, after = run_lesson(ref)
    refsets = [r for _, r in before]
    runs = {"before": [g for g, _ in before], "after": [g for g, _ in after], "disjoint": [[999] * 8] * 50}
    score = {k: cider_d(v, refsets) for k, v in runs.items()}
    bleu = {k: mean(list(map(ref.bleu4, v, refsets))) for k, v in runs.items()}
    return {"cider": {k: mean(v) for k, v in score.items()}, "bleu": bleu,
            "oracle": mean(cider_d([r[0] for r in refsets], refsets)),
            "distinct_after": len(set(map(tuple, runs["after"]))),
            "zero_cider_after": score["after"].count(0.0),
            "unit": cider_d([[1, 2, 3, 4, 5], [9, 8, 7]], [[[1, 2, 3, 4, 5]], [[6, 6, 6]]])[0],
            "one_image": cider_d([[1, 2, 3, 4, 5]], [[[1, 2, 3, 4, 5]]])[0]}


def verify(result):
    r, c, b = result, result["cider"], result["bleu"]
    return [
        practice.Check(
            "ANSWER: CIDEr-D moves from 0.0354 to 0.335 on the shipped run, ceiling 5.79",
            (c["before"], c["after"], r["oracle"]) == (0.0354, 0.335, 5.7911)
            and abs(r["unit"] - 10.0) < 1e-9,
            f"CIDEr-D before {c['before']}, after {c['after']}, first-reference ceiling "
            f"{r['oracle']}; a copy of the only reference scores {r['unit']:.6f}",
        ),
        practice.Check(
            "FINDING: BLEU-4's smoothing floor makes the untrained model look 70% as good as "
            "the trained one; CIDEr says 11%",
            (b["disjoint"], b["before"], b["after"], c["disjoint"]) == (0.1186, 0.122, 0.173, 0.0),
            f"no-shared-token caption: BLEU {b['disjoint']} vs untrained {b['before']} "
            f"({b['disjoint'] / b['before']:.0%}), CIDEr {c['disjoint']}; before/after "
            f"BLEU {b['before'] / b['after']:.1%}, CIDEr {c['before'] / c['after']:.1%}",
        ),
        practice.Check(
            "FINDING: the trained captioner has collapsed to 4 captions for 50 images",
            (r["distinct_after"], r["zero_cider_after"], r["one_image"]) == (4, 22, 0.0),
            f"{r['distinct_after']} distinct captions; {r['zero_cider_after']}/50 score CIDEr 0; "
            f"a perfect caption in a one-image eval scores {r['one_image']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
