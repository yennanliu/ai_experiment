"""Exercise 3 — mean pool discards position.

    **(Hard)** Replace the VLM's image encoder with DINOv3 instead of its default
    SigLIP/CLIP. Re-train only the projector (frozen LLM + frozen DINOv3). Measure
    whether dense-prediction tasks (counting, spatial reasoning) improve.

Reading of the exercise: DINOv3 is not reachable -- `timm`, `transformers`,
`open_clip` and `dinov2` all raise ModuleNotFoundError, and no weights may be
downloaded -- so the encoder swap is staged with three frozen 32x32 linear
stand-ins applied to the lesson's own vision tokens, chosen so that rank and
class-direction retention vary independently. The lesson gives no dense task, so
one is built out of `synthetic_vision_class_data` itself: scenes of 16 patches
where k of them are the object class, placed in the first or the second half.
Counting asks for k, spatial reasoning asks which half. The answer to "do dense
tasks improve" then splits. Counting moves with the encoder and the move has
nothing to do with rank -- a rank-2 encoder that keeps the class direction scores
within 0.050 of the full-rank one, while a rank-2 encoder that removes it lands
below chance. Spatial reasoning does not move at all, for any encoder, and cannot:
`ToyVLM.forward` mean-pools over patches before the head, which makes it exactly
permutation-invariant, so patch order never reaches the classifier. That is the
lesson's own "Spatial reasoning is still weak" section reproduced in one line of
its own code, and no encoder can repair it.

Structure: `caught` is exercise 1's import probe and `fit` is exercise 2's
training recipe, both reached through `practice.load_module` rather than copied.
`scenes` builds the dense-task fixture from the lesson's own generator -- one
bank of background sequences, one of object sequences, spliced by a mask -- so
the noise model is the lesson's and not a new one. `encoders` returns the three
stand-ins: a random rotation, the rank-2 projector onto a plane containing the
class direction, and the rank-2 projector onto a plane orthogonal to it, both
built by `plane`. `frozen` returns a ToyVLM with its head frozen -- the
exercise's "frozen LLM", leaving only the projector trainable -- and `flat` is
the order-reading linear control that keeps the patch axis the pooled model
throws away. At 145 lines of code the file is over D14's 120-line target and
under its 150-line ceiling; the overrun is three encoders x two tasks plus two
controls.
"""

from __future__ import annotations

import importlib
import pathlib

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "25-vision-language-models"

STACK = ("timm", "transformers", "open_clip", "dinov2")
PATCHES, DIM, SCENES, HALF = 16, 32, 300, 8
TRAIN, LLM, DEPTHS, STEPS = 240, 64, 3, 200      # DEPTHS: how many ViT levels DeepStack stacks

listing = lambda got: ", ".join(f"{k} {v}" for k, v in got.items())                     # noqa: E731
absent = lambda got: all(v == "ModuleNotFoundError" for v in got.values())              # noqa: E731
stack = lambda: [(n, lambda n=n: importlib.import_module(n)) for n in STACK]            # noqa: E731
table = lambda got: " ".join(f"{k} {v:.3f}" for k, v in got.items())                    # noqa: E731
spatials = lambda acc, keys: [acc[f"{k}/spatial"] for k in keys]                        # noqa: E731
by_task = lambda got, keys, task: {k: got[f"{k}/{task}"] for k in keys}                 # noqa: E731
ranks = lambda torch, enc: {k: int(torch.linalg.matrix_rank(w)) for k, w in enc.items()}  # noqa: E731
kept = lambda torch, enc, d: {k: float(torch.linalg.vector_norm(w @ d)) for k, w in enc.items()}  # noqa: E731
flat = lambda torch, n: torch.nn.Sequential(torch.nn.Flatten(1), torch.nn.Linear(PATCHES * DIM, n))  # noqa: E731
kit = practice.load_module(pathlib.Path(__file__).with_name("ex02_val_split_is_one_class.py"))
caught = kit.sibling.caught                      # exercise 1's ImportError probe, via exercise 2


def scenes(torch, ref):
    gen = torch.Generator().manual_seed(1)
    bank, tag = ref.synthetic_vision_class_data(2, PATCHES, DIM, SCENES, seed=0)
    background, objects = bank[tag == 0], bank[tag == 1]
    counts = torch.randint(1, HALF + 1, (SCENES,), generator=gen)
    side = torch.randint(0, 2, (SCENES,), generator=gen)
    where, start = torch.arange(PATCHES)[None, :], (side * HALF)[:, None]
    mask = ((where >= start) & (where < start + counts[:, None]))[..., None]
    delta = objects.mean((0, 1)) - background.mean((0, 1))
    return torch.where(mask, objects, background), counts - 1, side, delta


def plane(torch, first, second):
    basis = torch.linalg.qr(torch.stack([first, second], 1))[0][:, :2]
    return basis @ basis.T


def encoders(torch, functional, delta):
    gen = torch.Generator().manual_seed(2)
    unit = functional.normalize(delta, dim=0)
    off = [functional.normalize(v - (v @ unit) * unit, dim=0) for v in torch.randn(2, DIM, generator=gen)]
    return {"dense_r32": torch.linalg.qr(torch.randn(DIM, DIM, generator=gen))[0],
            "keeps_delta_r2": plane(torch, unit, off[0]), "blind_r2": plane(torch, *off)}


def frozen(torch, ref, classes):
    torch.manual_seed(0)
    model = ref.ToyVLM(DIM, LLM, classes)
    for parameter in model.head.parameters():
        parameter.requires_grad_(False)
    return model


def solve():
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    scene, counts, side, delta = scenes(torch, ref)
    enc = encoders(torch, functional, delta)
    tasks, got, guess = {"count": (counts, HALF), "spatial": (side, 2)}, {}, {}
    for name, matrix in enc.items():
        seen = scene @ matrix.T
        for task, (label, classes) in tasks.items():
            split = ((seen[:TRAIN], label[:TRAIN]), (seen[TRAIN:], label[TRAIN:]))
            model = frozen(torch, ref, classes)
            got[f"{name}/{task}"] = kit.fit(torch, functional, model, *split, STEPS)[-1]
            guess[f"{name}/{task}"] = model(seen[TRAIN:]).argmax(-1).bincount(minlength=classes).tolist()
    seen = scene @ enc["dense_r32"].T
    torch.manual_seed(0)
    order = {task: kit.fit(torch, functional, flat(torch, classes), (seen[:TRAIN], label[:TRAIN]),
                           (seen[TRAIN:], label[TRAIN:]), STEPS)[-1]
             for task, (label, classes) in tasks.items()}
    shuffled, pooled = torch.randperm(PATCHES, generator=torch.Generator().manual_seed(3)), frozen(
        torch, ref, HALF)
    with torch.no_grad():
        swing = float((pooled(seen[:8]) - pooled(seen[:8][:, shuffled])).abs().max())
    try:
        ref.Projector()(ref.deepstack_features([seen[:4]] * DEPTHS))
        widened = "accepted"
    except RuntimeError as exc:                     # pragma: no cover - 96 into a 32-wide Linear
        widened = str(exc)
    return {"missing": caught(stack(), ImportError), "acc": got, "guess": guess, "order": order,
            "rank": ranks(torch, enc), "kept": kept(torch, enc, functional.normalize(delta, dim=0)),
            "swing": swing, "widened": widened, "in_features": ref.Projector().net[0].in_features,
            "majority": {t: float(y[TRAIN:].bincount(minlength=n).max()) / (SCENES - TRAIN) for t, (y, n) in tasks.items()}}


def verify(result):
    acc, rank, kept_, order = result["acc"], result["rank"], result["kept"], result["order"]
    major, spatial = result["majority"], spatials(result["acc"], result["rank"])
    return [
        practice.Check(
            "ANSWER: DINOv3 is not reachable, so the swap is staged with frozen linear stand-ins",
            absent(result["missing"]),
            f"importing the encoder stack gives {listing(result['missing'])} and no weights may be downloaded, "
            f"so three frozen {DIM}x{DIM} maps stand in over the lesson's own tokens: ranks {listing(rank)}, "
            f"retention {table(kept_)}. Rank and retention vary independently, which is what separates them below"),
        practice.Check(
            "MECHANISM: an encoder swap is not a drop-in -- the projector's width is hardcoded",
            "cannot be multiplied" in result["widened"] and result["in_features"] == DIM,
            f"`Projector` opens with Linear(in_features={result['in_features']}), and the lesson's own DeepStack "
            f"trick concatenates {DEPTHS} ViT levels along the channel axis, so feeding `deepstack_features` back "
            f"in raises: {result['widened']}. A new encoder width forces a new projector; it is not optional"),
        practice.Check(
            "ANSWER: counting moves with the encoder, and rank is not the axis it moves on",
            acc["keeps_delta_r2/count"] > acc["dense_r32/count"] - 0.1
            and acc["blind_r2/count"] < major["count"] and kept_["blind_r2"] < 1e-5,
            f"counting accuracy {table(by_task(acc, rank, 'count'))} against a majority baseline of "
            f"{major['count']:.3f} and a uniform {1 / HALF:.3f}. Dropping 30 of 32 dimensions costs "
            f"{acc['dense_r32/count'] - acc['keeps_delta_r2/count']:.3f} while the class direction "
            f"survives ({kept_['keeps_delta_r2']:.3f}); removing it at the same rank costs everything"),
        practice.Check(
            "MECHANISM: spatial reasoning improves by 0.000 for any encoder -- the pooling ate it",
            max(spatial) == min(spatial) and abs(max(spatial) - major["spatial"]) < 1e-6
            and result["swing"] < 1e-5,
            f"all three encoders score {max(spatial):.3f} on which-half -- exactly the validation majority "
            f"rate {major['spatial']:.3f} -- and all three answer with a constant: "
            f"{listing(by_task(result['guess'], rank, 'spatial'))}. `ToyVLM.forward` runs "
            f"`pooled = projected.mean(dim=1)`, so shuffling the {PATCHES} patches moves the logits by at "
            f"most {result['swing']:.2e} -- float32 round-off on a reordered sum, not a tolerance. A mean "
            f"is symmetric in its arguments; position never reaches the head, whatever feeds it"),
        practice.Check(
            "CONTROL: the spatial signal is in the features -- only the pooling loses it",
            order["spatial"] > 0.95 and order["count"] > 0.95,
            f"a plain Linear({PATCHES}*{DIM}, c) on the same dense_r32 features, differing from ToyVLM only in "
            f"keeping the patch axis, scores {order['spatial']:.3f} on which-half and {order['count']:.3f} on "
            f"counting, against the pooled model's {acc['dense_r32/spatial']:.3f} and {acc['dense_r32/count']:.3f}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
