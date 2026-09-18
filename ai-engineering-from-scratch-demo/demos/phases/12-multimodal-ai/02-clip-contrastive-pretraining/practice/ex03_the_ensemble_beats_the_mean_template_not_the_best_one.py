"""Exercise 3 — the ensemble beats the mean template, not the best one.

    Build a zero-shot classifier for cats vs dogs. Try two prompt templates: `a
    photo of a {class}` and `a picture of a {class}`. Measure accuracy on 100
    test images. Does the ensemble of templates beat single?

Reading of the exercise: the lesson ships no text encoder, so the text model is
stated here rather than borrowed -- a template embeds its class prototype plus
its own template-specific noise at weight ALPHA, and a test image is its class
prototype plus image noise at weight BETA. Both prototypes are the lesson's own
`make_fake_embedding(42)` and `(43)`, the seeds its zero-shot demo uses for cat
and dog. The headline is one cell of a sweep, because a single cell cannot say
whether the ensemble helps or the fixture was chosen.

**ANSWER: yes, by 8 points -- 91% against 83% and 80%** on 100 images at
ALPHA = BETA = 2.0, and 9.5 points above the mean of its two templates.

**FINDING: the ensemble beats the mean of its templates, not the best of
them.** Over the 15-cell sweep it is at or above the mean of its two singles in
**15 of 15** cells but above the better single in only **6**. Where the two
templates are far apart it drags the good one down: at ALPHA = 4, BETA = 2 the
singles are 51% and 73% and the ensemble is 64%.

**FINDING: all of the gain is template variance.** At ALPHA = 0 the two
templates are the same vector, and single and ensemble agree exactly -- 98%,
98%, 98% -- at every image-noise level. Prompt ensembling is a variance
reduction over prompts, so a benchmark that varies only the images cannot
measure it.

**FINDING: in the lesson's own text model, applying a template destroys the
class.** `demo_prompt_ensemble` embeds a prompt as
`make_fake_embedding(sum(ord(c) for c in prompt))`, and the resulting "a photo
of a cat" vector scores **-0.1254** against the lesson's cat prototype and
**+0.0046** against its dog prototype -- nearer the wrong class. `demo_zero_shot`
sidesteps this by using the prototype itself as the text vector, and
`demo_prompt_ensemble` never classifies anything.

**FINDING: that seed is anagram-invariant.** `sum(ord(c))` gives "a photo of a
cat" and "a photo of a act" the same seed, 1401, and byte-identical embeddings
-- while "a photo of a dog" is 1403, two away, and independent.

Structure: `mix` is the shared perturbation, `fixture` builds the two template
vectors per class, their ensemble and the 100 labelled images, `accuracy`
scores one set of text vectors, and `GRID` is the (ALPHA, BETA) sweep.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "02-clip-contrastive-pretraining"
TEMPLATES = ("a photo of a {class}", "a picture of a {class}")
CLASSES = {"cat": 42, "dog": 43}
DIM, IMAGES, SEED = 64, 100, 7
ALPHA, BETA = 2.0, 2.0
GRID = tuple((alpha, beta) for alpha in (0.0, 1.0, 2.0, 3.0, 4.0)
             for beta in (2.0, 3.0, 4.0))


def mix(ref, base, noise, weight):
    return ref.normalize([b + weight * n for b, n in zip(base, noise)])


def text_vectors(ref, protos, noise, alpha):
    """One vector per (template, class), plus the mean-of-templates ensemble."""
    texts = {t: {c: mix(ref, protos[c], noise(), alpha) for c in protos} for t in TEMPLATES}
    ensemble = {c: ref.normalize([sum(texts[t][c][k] for t in TEMPLATES) / len(TEMPLATES)
                                  for k in range(DIM)]) for c in protos}
    return texts, ensemble


def image_set(ref, protos, noise, beta):
    """100 labelled images, balanced across the classes."""
    labels = [name for _ in range(IMAGES // len(protos)) for name in protos]
    return [(name, mix(ref, protos[name], noise(), beta)) for name in labels]


def fixture(ref, alpha, beta, seed=SEED):
    """Two template vectors per class, their ensemble, and 100 labelled images."""
    rng = random.Random(seed)

    def noise():
        return [rng.gauss(0, 1) for _ in range(DIM)]

    protos = {name: ref.make_fake_embedding(s) for name, s in CLASSES.items()}
    texts, ensemble = text_vectors(ref, protos, noise, alpha)
    return texts, ensemble, image_set(ref, protos, noise, beta)


def accuracy(ref, images, text_vectors):
    hits = sum(1 for label, vec in images
               if max(text_vectors, key=lambda c: ref.cosine(vec, text_vectors[c])) == label)
    return round(hits / len(images) * 100, 1)


def cell(ref, alpha, beta):
    texts, ensemble, images = fixture(ref, alpha, beta)
    singles = [accuracy(ref, images, texts[t]) for t in TEMPLATES]
    return singles, accuracy(ref, images, ensemble)


def prompt_vector(ref, prompt):
    """The lesson's own text embedding: seeded by the sum of the prompt's code points."""
    return ref.normalize(ref.make_fake_embedding(sum(ord(char) for char in prompt)))


def lesson_prompt_scores(ref):
    """The lesson's own templated prompt vector against its own class prototypes."""
    templated = prompt_vector(ref, TEMPLATES[0].format(**{"class": "cat"}))
    protos = {name: ref.normalize(ref.make_fake_embedding(s)) for name, s in CLASSES.items()}
    return {f"to_{name}": round(ref.cosine(templated, vec), 4) for name, vec in protos.items()}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    sweep = {pair: cell(ref, *pair) for pair in GRID}
    singles, ensemble = sweep[(ALPHA, BETA)]
    seeds = {p: sum(ord(c) for c in p) for p in
             ("a photo of a cat", "a photo of a act", "a photo of a dog")}
    return {
        "singles": singles, "ensemble": ensemble,
        "over_best": round(ensemble - max(singles), 1),
        "over_mean": round(ensemble - sum(singles) / len(singles), 1),
        "beats_mean": sum(e >= sum(s) / len(s) - 1e-9 for s, e in sweep.values()),
        "beats_best": sum(e > max(s) for s, e in sweep.values()),
        "cells": len(sweep),
        "worst_drag": min(round(e - max(s), 1) for s, e in sweep.values()),
        "no_template_noise": sorted({(tuple(s), e) for (a, _), (s, e) in sweep.items()
                                     if a == 0.0}),
        **lesson_prompt_scores(ref),
        "seeds": seeds,
        "anagram": (ref.make_fake_embedding(seeds["a photo of a cat"])
                    == ref.make_fake_embedding(seeds["a photo of a act"])),
    }


def verify(result):
    singles, seeds = result["singles"], result["seeds"]
    return [
        practice.Check(
            "ANSWER: yes, by 8 points -- 91% against 83% and 80%",
            all([sorted(singles) == [80.0, 83.0], result["ensemble"] == 91.0,
                 result["over_best"] == 8.0, result["over_mean"] == 9.5]),
            f"on {IMAGES} images at ALPHA=BETA={ALPHA:g}, the two templates score {singles} "
            f"and their ensemble {result['ensemble']}% -- {result['over_best']} points over "
            f"the better template and {result['over_mean']} over their mean",
        ),
        practice.Check(
            "FINDING: the ensemble beats the mean of its templates, not the best of them",
            all([result["beats_mean"] == 15, result["beats_best"] == 6,
                 result["cells"] == 15, result["worst_drag"] == -9.0]),
            f"over {result['cells']} (ALPHA, BETA) cells the ensemble is at or above the mean "
            f"of its singles in {result['beats_mean']} and above the better single in only "
            f"{result['beats_best']}. Where the templates disagree it drags the good one "
            f"down -- worst cell {result['worst_drag']} points below the best template",
        ),
        practice.Check(
            "FINDING: all of the gain is template variance",
            all([len(result["no_template_noise"]) == 3,
                 all(e == s[0] == s[1] for s, e in result["no_template_noise"])]),
            f"at ALPHA=0 the two templates are the same vector, and single and ensemble "
            f"agree exactly at every image-noise level: {result['no_template_noise']}. "
            "Prompt ensembling reduces variance over prompts, so a benchmark that varies "
            "only the images cannot measure it",
        ),
        practice.Check(
            "FINDING: in the lesson's own text model, applying a template destroys the class",
            all([result["to_cat"] == -0.1254, result["to_dog"] == 0.0046,
                 result["to_dog"] > result["to_cat"]]),
            f"make_fake_embedding(sum(ord(c) for c in 'a photo of a cat')) scores "
            f"{result['to_cat']} against the lesson's cat prototype and {result['to_dog']} "
            "against its dog prototype -- nearer the wrong class. demo_zero_shot sidesteps "
            "this by using the prototype as the text vector, and demo_prompt_ensemble never "
            "classifies anything",
        ),
        practice.Check(
            "FINDING: that seed is anagram-invariant",
            all([result["anagram"], seeds["a photo of a cat"] == 1401,
                 seeds["a photo of a act"] == 1401, seeds["a photo of a dog"] == 1403]),
            f"sum(ord(c)) gives {seeds} -- 'cat' and 'act' collide at 1401 and produce "
            "byte-identical embeddings, while 'dog' is two away at 1403 and independent. "
            "Seed distance carries no meaning in either direction",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
