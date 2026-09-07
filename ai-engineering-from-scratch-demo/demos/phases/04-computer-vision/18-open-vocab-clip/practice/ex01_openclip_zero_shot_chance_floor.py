"""Exercise 1 — openclip zero shot chance floor.

    **(Easy)** Use a pretrained OpenCLIP ViT-B/32 and do zero-shot classification on CIFAR-10 with the 80-template prompt set. Report top-1 accuracy; it should be around 85-90%.

Reading of the exercise: both halves of the setup are absent, and this file
measures their absence rather than routing around it. `open_clip`, `clip` and
`transformers` are in no dependency group of this repo, so "pretrained" cannot
happen; `torchvision.datasets.CIFAR10(download=False)` raises, and no weights or
datasets may be fetched. The 85-90% figure is therefore a claim about a
checkpoint and a dataset that are not here (OpenCLIP ViT-B/32 on LAION-2B is
reported in that band; not measured here). What *is* here is the lesson's own
`TwoTower` plus a CIFAR-10-shaped stand-in: 10 classes of `synthetic_cifar` from
Phase 4 Lesson 4, standardised and pushed through a frozen seeded random
projection into the 128-d pre-extracted image features the lesson's image tower
expects, with one 64-d unit vector per class in the slot a text tower would fill.
That arrangement contradicts the exercise's framing twice. Zero-shot top-1 is an
argmax of cosine similarity, so 1/`CLASSES` = 0.100 is the floor an untrained head
sits at -- measured 0.094 mean over 8 inits, and *below* the floor on half of
them, because a fresh two-tower ever names only 4-7 of the 10 classes. And once
trained the number saturates: top-1 reaches 1.000 by step 10 and stays there, so
it ranks nothing and the runner-up margin is measured beside it. The 80-template
prompt set is Exercise 2's subject and is deliberately not built here; this file
uses one prompt per class.

Structure: `probe_import` returns the `ModuleNotFoundError` text for one CLIP
package; `stand_in` builds the 10-class feature bank through the frozen
projection together with the one-per-class prompt table; `step_ladder` trains one
seed under the lesson's own `clip_loss` and, at each rung of `LADDER`, scores the
lesson's own `zero_shot_classify` and records top-1, the top-1-minus-runner-up
cosine margin, and how many distinct classes the head ever names -- so rung 0 of
the eight ladders doubles as the random-init sweep; `scale_and_null` checks argmax
invariance to `logit_scale` and measures the shuffled-prompt null over 2,000
permutations. At 142 code lines this sits above D14's 120-line target and 8 clear
of the ceiling: five checks over three probes, one of which is a six-rung
training curve run for eight seeds.
"""

from __future__ import annotations

import importlib

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "18-open-vocab-clip"

IMG_IN, TXT_IN, CLASSES, CHANCE = 128, 64, 10, 0.1
PER_CLASS, TRAIN, BATCH, LR, JITTER = 120, 1000, 64, 3e-3, 0.05
PROJ_SEED, PROMPT_SEED, NULL_SEED, PERMS = 1234, 7, 77, 2000
MISSING, SCALES = ("open_clip", "clip", "transformers"), (0.01, 1.0, 14.285, 100.0)
LADDER, SEEDS, SHOWN = (0, 2, 5, 10, 25, 100), tuple(range(8)), 3
NAMES = [f"class_{c}" for c in range(CLASSES)]

seeded = lambda torch, seed: torch.Generator().manual_seed(seed)    # noqa: E731 - a constructor
column = lambda rows, at, which: [rows[s][at][which] for s in SEEDS]  # noqa: E731 - a projection
fixed3 = lambda values: ", ".join(f"{v:.3f}" for v in values)       # noqa: E731 - a formatter
curve = lambda row: "  ".join(f"{n}:{a:.3f}/m{m:.3f}" for n, (a, m, _) in zip(LADDER, row))  # noqa: E731


def probe_import(name) -> str:
    try:
        importlib.import_module(name)
        return f"{name} -> present"
    except ModuleNotFoundError as exc:
        return f"{name} -> {type(exc).__name__}: {exc}"


def stand_in(torch, functional) -> tuple:
    lesson04 = parity.load_reference(PHASE, "04-image-classification", "main")
    images, labels = lesson04.synthetic_cifar(num_per_class=PER_CLASS, num_classes=CLASSES, seed=0)
    pixels = torch.from_numpy(images).reshape(len(images), -1).float()
    pixels = (pixels - pixels.mean()) / pixels.std()
    frozen = torch.randn(pixels.shape[1], IMG_IN, generator=seeded(torch, PROJ_SEED))
    prompts = torch.randn(CLASSES, TXT_IN, generator=seeded(torch, PROMPT_SEED))
    return pixels @ frozen / pixels.shape[1] ** 0.5, torch.from_numpy(labels).long(), functional.normalize(prompts, dim=-1)


def step_ladder(ref, torch, train, test, table, seed) -> tuple:
    torch.manual_seed(seed)
    model = ref.TwoTower(img_in=IMG_IN, txt_in=TXT_IN)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    gen, row, done, last = seeded(torch, 100 + seed), [], 0, 0.0
    for target in LADDER:
        for _ in range(target - done):
            pick = torch.randint(0, len(train[0]), (BATCH,), generator=gen)
            caption = table[train[1][pick]] + JITTER * torch.randn(BATCH, TXT_IN, generator=gen)
            loss = ref.clip_loss(*model(train[0][pick], caption))
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = float(loss.detach())
        done = target
        picked = torch.tensor([NAMES.index(n) for n in ref.zero_shot_classify(model, test[0], table, NAMES)])
        with torch.no_grad():
            best = (model.encode_image(test[0]) @ model.encode_text(table).T).topk(2, -1).values
        row.append(((picked == test[1]).float().mean().item(), (best[:, 0] - best[:, 1]).mean().item(), int(picked.unique().numel())))
    return model, row, last


def scale_and_null(torch, model, test, table) -> dict:
    with torch.no_grad():
        sim = model.encode_image(test[0]) @ model.encode_text(table).T
    picked, gen = sim.argmax(dim=-1), seeded(torch, NULL_SEED)
    null = sum((torch.randperm(CLASSES, generator=gen)[picked] == test[1]).float().mean().item() for _ in range(PERMS)) / PERMS
    return {"null": null, "test": len(picked), "stable": all(torch.equal(picked, (s * sim).argmax(dim=-1)) for s in SCALES)}


def solve():
    try:
        import torch
        import torch.nn.functional as functional
        import torchvision
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    try:
        torchvision.datasets.CIFAR10(root="./_no_such_cifar_root", train=False, download=False)
        cifar = "present"
    except RuntimeError as exc:
        cifar = f"{type(exc).__name__}: {exc}"
    feats, labels, table = stand_in(torch, functional)
    train, test = (feats[:TRAIN], labels[:TRAIN]), (feats[TRAIN:], labels[TRAIN:])
    runs = {seed: step_ladder(ref, torch, train, test, table, seed) for seed in SEEDS}
    rows = {seed: runs[seed][1] for seed in SEEDS}
    return {"backends": "; ".join(probe_import(name) for name in MISSING), "cifar": cifar,
            "init": column(rows, 0, 0), "distinct": column(rows, 0, 2), "final": column(rows, -1, 0),
            "curves": " | ".join(curve(rows[s]) for s in SEEDS[:SHOWN]), "start": column(rows, 0, 1),
            "margin": column(rows, -1, 1), "loss": [runs[s][2] for s in SEEDS],
            **scale_and_null(torch, runs[SEEDS[0]][0], test, table)}


def verify(result):
    init, margin, start, seen = result["init"], result["margin"], result["start"], result["distinct"]
    mean, low = sum(init) / len(init), sum(a < CHANCE for a in init)
    return [
        practice.Check(
            "ANSWER: 85-90% is not reproducible here — neither the checkpoint nor CIFAR-10 exists",
            "present" not in result["backends"] and result["cifar"].startswith("RuntimeError"),
            f"every CLIP package the exercise implies is absent: {result['backends']}; and "
            f"`torchvision.datasets.CIFAR10(download=False)` gives {result['cifar']}, with no download "
            "permitted. OpenCLIP ViT-B/32 on LAION-2B is reported in that band; not measured here"),
        practice.Check(
            "FINDING: an untrained head sits at 1/10 and half the inits fall below it, by collapse",
            abs(mean - CHANCE) < 0.02 and min(init) < CHANCE and max(seen) < CLASSES,
            f"the lesson's own `zero_shot_classify` on {len(SEEDS)} fresh `TwoTower` inits over "
            f"{result['test']} held-out stand-in images: top-1 {fixed3(init)}, mean {mean:.3f} against "
            f"the 1/{CLASSES} = {CHANCE:.3f} floor, and {low} of {len(SEEDS)} land *under* chance. Those "
            f"inits name only {seen} distinct classes of {CLASSES} -- both towers are near-arbitrary "
            f"linear maps, so a prompt or two wins nearly every image, and a head that never names a "
            f"class cannot score its images: hence {min(init):.3f}, not {CHANCE:.3f}"),
        practice.Check(
            "FINDING: top-1 saturates at 1.000 by step 10 and ranks nothing — the margin still moves",
            all(a == 1.0 for a in result["final"]) and min(margin) > 10 * max(start),
            f"top-1 / mean top-1-minus-runner-up cosine margin at {LADDER} steps of the lesson's own "
            f"`clip_loss`, first {SHOWN} of {len(SEEDS)} seeds: {result['curves']}. All {len(SEEDS)} "
            f"finish at 1.000 for a spread of 0.000, overshooting the exercise's band, while the margin "
            f"climbs on from {min(start):.3f}-{max(start):.3f} to {min(margin):.3f}-{max(margin):.3f}"),
        practice.Check(
            "MECHANISM: top-1 cannot see the temperature at all — argmax is scale-invariant",
            result["stable"],
            f"scaling the similarity matrix by {SCALES} -- a span bracketing the lesson's own "
            f"`logit_scale.exp()` of ~14.3 -- leaves all {result['test']} predictions bit-identical: a "
            f"positive scalar cannot reorder a row. `logit_scale` reaches zero-shot accuracy only "
            f"through the loss, which ends at {min(result['loss']):.3f}-{max(result['loss']):.3f}"),
        practice.Check(
            "CONTROL: the chance floor is exactly 1/C, measured by shuffling the prompt-to-class map",
            abs(result["null"] - CHANCE) < 0.01,
            f"the trained head's predictions relabelled through {PERMS:,} random permutations of the "
            f"{CLASSES} prompts score {result['null']:.5f} against 1/{CLASSES} = {CHANCE:.5f}: every "
            f"image has probability 1/{CLASSES} of being relabelled right whatever the class balance, "
            "so this is the null every zero-shot number in the lesson reads against"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
