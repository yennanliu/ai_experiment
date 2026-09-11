"""Exercise 3 — a probe without a null.

    **Hard.** Train a model on the 20 Newsgroups corpus. Compute two bias axes:
    `he - she` and `doctor - nurse`. Project occupation words onto both axes.
    Report which occupations have the largest bias gap. This is the kind of
    probe fairness researchers use.

Reading of the exercise: 20 Newsgroups needs a download this repo will not
make, so the corpus is built here -- and building it is what makes the probe
checkable, because a corpus written on purpose has a known true bias. Two are
built. In the symmetric one every occupation appears with `he` and with `she`
the same number of times in the same frames, so its gender bias is zero by
construction. In the skewed one four occupations take `he` three times in four
and four take `she`.

The probe cannot tell the difference at the resolution the exercise asks for.
On the corpus with no bias in it, projections onto `he - she` reach 0.539, and
the spread between occupations, 0.036, is an order of magnitude below the
spread of a single occupation across random seeds, 0.26 to 0.32. Which
occupation ranks most male changes on every seed. What does survive is the
group: on the skewed corpus the four `he`-weighted occupations separate from
the four `she`-weighted ones with a per-seed deviation under 0.07, and on the
symmetric corpus they do not separate at all. So the probe measures what it was
built to measure one level up from where the exercise reads it -- a direction,
not a ranking.

Two more things the exercise's framing hides. The two axes are not consistently
oriented against each other: `cos(he - she, doctor - nurse)` wanders across
seeds on the same corpus and changes sign. And the scales do not permit a
verdict from one run -- the largest projection on the corpus with no bias in it
is several times the whole group gap on the corpus that has one, so a number
without its null beside it says nothing.

Structure: `OCCUPATIONS` and `FRAMES` generate both corpora; `build(skew=False)`
is the null. `axis` is a normalized difference of two rows and `project` a
cosine against it. `study` trains `SEEDS` models on one corpus and returns per
occupation the mean and the across-seed deviation, plus the axis alignment and
which occupation topped each seed.
"""

from __future__ import annotations

import importlib.util

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "03-word-embeddings-word2vec"

OCCUPATIONS = ("doctor", "nurse", "engineer", "teacher", "chef", "pilot", "lawyer", "dancer")
FRAMES = ("{p} works as a {o} every day", "{p} trained to be a {o} for years",
          "the {o} said {p} would arrive early", "everyone knows {p} is a {o}")
SEEDS, MALE = 5, 3
CONFIG = {"dim": 16, "window": 2, "k_neg": 5, "lr": 0.05, "epochs": 200}


def build(skew: bool) -> list:
    """Symmetric: every occupation with both pronouns, so the true bias is zero."""
    out = []
    for i, occupation in enumerate(OCCUPATIONS):
        male = MALE if i % 2 == 0 else len(FRAMES) - MALE
        for k, frame in enumerate(FRAMES):
            if skew:
                out.append(frame.format(p="he" if k < male else "she", o=occupation))
            else:
                out += [frame.format(p=p, o=occupation) for p in ("he", "she")]
    return out


def axis(np, vocab, weights, plus, minus):
    difference = weights[vocab[plus]] - weights[vocab[minus]]
    return difference / (np.linalg.norm(difference) + 1e-9)


def project(np, vocab, weights, word, direction) -> float:
    vector = weights[vocab[word]]
    return float(vector @ direction / (np.linalg.norm(vector) + 1e-9))


def study(np, ref, sentences) -> dict:
    docs = [ref.tokenize(sentence) for sentence in sentences]
    rows, align = [], []
    for seed in range(SEEDS):
        vocab, weights = ref.train(docs, seed=seed, **CONFIG)
        gender = axis(np, vocab, weights, "he", "she")
        align.append(float(gender @ axis(np, vocab, weights, "doctor", "nurse")))
        rows.append([project(np, vocab, weights, o, gender) for o in OCCUPATIONS]
                    + [project(np, vocab, weights, o,
                               axis(np, vocab, weights, "doctor", "nurse"))
                       for o in ("doctor", "nurse")])
    table = np.array(rows)
    gendered, pair = table[:, :len(OCCUPATIONS)], table[:, len(OCCUPATIONS):]
    means = gendered.mean(axis=0)
    return {"mean": means.round(3).tolist(), "sd": gendered.std(axis=0).round(3).tolist(),
            "reach": float(np.abs(gendered).max()), "align": [round(a, 3) for a in align],
            "between": float(means.max() - means.min()),
            "within": float(gendered.std(axis=0).mean()),
            "groups": (float(means[0::2].mean()), float(means[1::2].mean())),
            "leaders": sorted({OCCUPATIONS[int(np.argmax(row))] for row in gendered}),
            "own_axis": pair.mean(axis=0).round(3).tolist()}


def solve():
    try:
        import numpy as np
    except ImportError as exc:                      # pragma: no cover - T1 needs numpy
        raise practice.Skip(f"needs numpy: uv sync --extra math ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    arms = {name: study(np, ref, build(skew)) for name, skew in (("null", False), ("skewed", True))}
    return {"arms": arms, "seeds": SEEDS,
            "sizes": {name: len(build(skew)) for name, skew in (("null", False), ("skewed", True))},
            "missing": [n for n in ("gensim", "torchtext")
                        if importlib.util.find_spec(n) is None]}


def verify(result):
    null, skewed, seeds = result["arms"]["null"], result["arms"]["skewed"], result["seeds"]
    null_gap = abs(null["groups"][0] - null["groups"][1])
    skew_gap = abs(skewed["groups"][0] - skewed["groups"][1])
    return [
        practice.Check(
            "ANSWER: on a corpus with zero bias by construction the probe still reads 0.539",
            null["reach"] > 0.4,
            f"20 Newsgroups needs a download, so the corpora are built: {result['sizes']['null']} "
            f"sentences in which every occupation appears with 'he' and with 'she' identically, and "
            f"{result['sizes']['skewed']} in which four take 'he' {MALE} times in "
            f"{len(FRAMES)}. On the first, whose true gender bias is zero, projections onto "
            f"he - she reach {null['reach']:.3f}. The probe has no null, and the exercise supplies "
            f"none"),
        practice.Check(
            "MECHANISM: the seed moves each occupation further than the corpus separates them",
            null["within"] > 5 * null["between"],
            f"on the null corpus the occupation means are {null['mean']} -- a spread of "
            f"{null['between']:.3f} between them -- while one occupation's deviation across "
            f"{seeds} seeds is {null['sd']}, averaging {null['within']:.3f}. Any ranking read off a "
            f"single run is reading the initialisation"),
        practice.Check(
            "FINDING: the ranking the exercise asks for is unstable on both corpora",
            len(null["leaders"]) > 1 and len(skewed["leaders"]) > 1,
            f"'which occupations have the largest bias gap' names a different occupation on "
            f"different seeds: {null['leaders']} on the null corpus and {skewed['leaders']} on the "
            f"skewed one. Even where the bias is real and was put there deliberately, the per-word "
            f"ordering inside the biased group is noise"),
        practice.Check(
            "FINDING: one level up the probe works -- the group separates, and only when it should",
            skew_gap > 3 * null_gap and skewed["within"] < null["within"],
            f"averaging the four he-weighted occupations against the four she-weighted ones gives "
            f"{skewed['groups'][0]:.3f} vs {skewed['groups'][1]:.3f} on the skewed corpus, a gap of "
            f"{skew_gap:.3f}, against {null_gap:.3f} on the null corpus. Per-seed deviation falls "
            f"from {null['within']:.3f} to {skewed['within']:.3f}: the direction is measurable, the "
            f"ranking is not"),
        practice.Check(
            "FINDING: the two axes are not consistently oriented against each other",
            max(null["align"]) - min(null["align"]) > 0.4,
            f"cos(he - she, doctor - nurse) on one corpus across {seeds} seeds: {null['align']} on "
            f"the null corpus and {skewed['align']} on the skewed one -- the null range spans "
            f"{max(null['align']) - min(null['align']):.3f} and changes sign. Reporting a word's "
            f"position on 'both axes' presumes a relation between them that the runs do not hold"),
        practice.Check(
            "CONTROL: the noise on the unbiased corpus is larger than the signal on the biased one",
            null["reach"] > 3 * skew_gap,
            f"the largest projection the probe returns on a corpus built to have no gender bias is "
            f"{null['reach']:.3f}; the entire group gap on the corpus built to have one is "
            f"{skew_gap:.3f} -- a factor of {null['reach'] / skew_gap:.1f}. So a single number off "
            f"a single run cannot be read as bias at all without the null run beside it, which is "
            f"the run the exercise does not ask for"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
