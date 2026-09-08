"""Exercise 2 — template averaging gap.

    **(Medium)** Compare single-template ("a photo of a {}") vs 80-template averaged embeddings on the same CIFAR-10 task. Quantify the gap and explain why templates help.

Reading of the exercise: there is no text encoder and no tokenizer here --
`open_clip` is in no dependency group, so the literal string "a photo of a {}"
cannot be turned into a vector at all, and the lesson's own text tower consumes
pre-extracted 64-d features rather than tokens. A template is therefore modelled
as what a template *is* in feature space: a per-template direction shared across
all classes (the "blurry photo of" style) plus per-(class, template) content
noise, both added to the class prototype before L2-normalising. That model is
what makes the exercise answerable, and it immediately splits its "explain why
templates help" into two claims that do not fare alike. The shared style
direction is nearly free -- at style 1.2, three times the setting used for the
main arm, the single-template arm still loses at most 0.065 -- because adding the
same vector to every class barely reorders an argmax. The per-class noise is what
costs, and averaging is precisely its cure: the mean of K such draws shrinks by
1/sqrt(K) to within 4%. Two traps sit in the implementation. `emb` and `txt_in`
are both 64 in the lesson's `TwoTower`, so feeding an already-averaged embedding
table back through `encode_text` type-checks and silently answers the wrong
question; and the 80 must be averaged *after* the tower, since `text_proj` is a
ReLU MLP and the two orders give class tables only 0.92-0.96 cosine apart. Top-1
also saturates: the gap closes at K = 4, so 76 of the 80 templates buy no
accuracy at all, and the runner-up margin is reported alongside it.

Structure: `stand_in` builds the CIFAR-10-shaped feature bank and the class
prototypes through a frozen random projection; `train_tower` trains the lesson's
own `TwoTower` under its own `clip_loss`; `template_bank` synthesises the 80
templates at a given style and noise and returns the noise draw with them;
`table_score` runs CLIP's own recipe -- encode K templates, mean, renormalise --
and reports top-1, the top-1-minus-runner-up margin and the class table itself;
`k_ladder` walks `KS` for each of three template seeds and is called three times,
once for the mixed arm, once style-only, once with neither, while the pre-tower
averaging arm reaches `table_score` as a one-template bank of already-averaged
features. At 146 code lines this sits above D14's 120-line target and 4 clear of
the ceiling: five checks over three ladders, a direct measurement of the
1/sqrt(K) law, and the pre- versus post-tower averaging control.
"""

from __future__ import annotations

import importlib

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "18-open-vocab-clip"

IMG_IN, TXT_IN, CLASSES, TEMPLATES = 128, 64, 10, 80
PER_CLASS, TRAIN, BATCH, LR, JITTER = 120, 1000, 64, 3e-3, 0.05
PROJ_SEED, PROMPT_SEED, STEPS, STYLE, LOUD, NOISE = 1234, 7, 300, 0.4, 1.2, 0.3
KS, BANKS = (1, 2, 4, 8, 20, TEMPLATES), (11, 12, 13)

col = lambda rows, at, which: [rows[s][at][which] for s in BANKS]   # noqa: E731 - a projection
gaps = lambda rows: [rows[s][-1][0] - rows[s][0][0] for s in BANKS]  # noqa: E731 - a projection
sqrt_law = lambda draw: [(k, draw[:, :k].mean(dim=1).pow(2).mean().sqrt().item(), NOISE / k ** 0.5) for k in KS]  # noqa: E731
ladder = lambda rows: " | ".join("  ".join(f"K={k}:{a:.3f}/m{m:.3f}" for k, (a, m) in zip(KS, rows[s])) for s in BANKS)  # noqa: E731
law_row = lambda law: "  ".join(f"K={k}:{got:.5f} vs {want:.5f} ({got / want:.3f}x)" for k, got, want in law)  # noqa: E731


def stand_in(torch, functional) -> tuple:
    lesson04 = parity.load_reference(PHASE, "04-image-classification", "main")
    images, labels = lesson04.synthetic_cifar(num_per_class=PER_CLASS, num_classes=CLASSES, seed=0)
    raw = torch.from_numpy(images).reshape(len(images), -1).float()
    pixels, width = (raw - raw.mean()) / raw.std(), raw.shape[1]
    frozen = torch.randn(width, IMG_IN, generator=torch.Generator().manual_seed(PROJ_SEED))
    prompts = functional.normalize(torch.randn(CLASSES, TXT_IN, generator=torch.Generator().manual_seed(PROMPT_SEED)), dim=-1)
    return pixels @ frozen / width ** 0.5, torch.from_numpy(labels).long(), prompts


def train_tower(ref, torch, train, protos) -> tuple:
    torch.manual_seed(0)
    model = ref.TwoTower(img_in=IMG_IN, txt_in=TXT_IN)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(100)
    for _ in range(STEPS):
        pick = torch.randint(0, len(train[0]), (BATCH,), generator=gen)
        caption = protos[train[1][pick]] + JITTER * torch.randn(BATCH, TXT_IN, generator=gen)
        loss = ref.clip_loss(*model(train[0][pick], caption))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model, float(loss.detach())


def template_bank(torch, functional, protos, style, noise, seed) -> tuple:
    gen = torch.Generator().manual_seed(seed)
    shared = style * torch.randn(TEMPLATES, TXT_IN, generator=gen)
    content = noise * torch.randn(CLASSES, TEMPLATES, TXT_IN, generator=gen)
    return functional.normalize(protos[:, None, :] + shared[None] + content, dim=-1), content


def table_score(torch, functional, model, bank, k, split) -> tuple:
    with torch.no_grad():
        flat = model.encode_text(bank[:, :k].reshape(-1, TXT_IN))
    table = functional.normalize(flat.reshape(CLASSES, k, -1).mean(dim=1), dim=-1)
    sim = split[0] @ table.T
    best = sim.topk(2, dim=-1).values
    return (sim.argmax(dim=-1) == split[1]).float().mean().item(), (best[:, 0] - best[:, 1]).mean().item(), table


def k_ladder(torch, functional, model, protos, split, style, noise) -> tuple:
    rows, draw, bank = {}, None, None
    for seed in BANKS:
        bank, draw = template_bank(torch, functional, protos, style, noise, seed)
        rows[seed] = [table_score(torch, functional, model, bank, k, split)[:2] for k in KS]
    return rows, draw, bank


def solve():
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    try:
        importlib.import_module("open_clip")
        backend = "open_clip -> present"
    except ModuleNotFoundError as exc:
        backend = f"open_clip -> {type(exc).__name__}: {exc}"
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    feats, labels, protos = stand_in(torch, functional)
    train, test = (feats[:TRAIN], labels[:TRAIN]), (feats[TRAIN:], labels[TRAIN:])
    model, loss = train_tower(ref, torch, train, protos)
    with torch.no_grad():
        split = (model.encode_image(test[0]), test[1])
    mixed, draw, _ = k_ladder(torch, functional, model, protos, split, STYLE, NOISE)
    styled = k_ladder(torch, functional, model, protos, split, LOUD, 0.0)[0]
    plain, _, same = k_ladder(torch, functional, model, protos, split, 0.0, 0.0)
    bank, _ = template_bank(torch, functional, protos, STYLE, NOISE, BANKS[0])
    post = table_score(torch, functional, model, bank, TEMPLATES, split)
    pre = table_score(torch, functional, model, bank.mean(dim=1)[:, None, :], 1, split)
    law = sqrt_law(draw)
    cosine = torch.diagonal(post[2] @ pre[2].T)
    return {"backend": backend, "loss": loss, "test": len(test[1]), "law": law, "mixed": mixed, "styled": styled,
            "plain": plain, "pre": pre[:2], "post": post[:2], "clean": plain[BANKS[0]][0], "identical": torch.equal(same[:, 0], same[:, TEMPLATES - 1]),
            "law_ok": all(abs(got / want - 1) < 0.05 for _, got, want in law), "worst": max(1.0 - a for a in col(styled, 0, 0)),
            "saturated": all(a == 1.0 for a in col(mixed, -1, 0) + col(mixed, 2, 0)), "cosine": (cosine.min().item(), cosine.max().item())}


def verify(result):
    mixed, styled, plain = result["mixed"], result["styled"], result["plain"]
    gap, single, margin, style_arm = gaps(mixed), col(mixed, 0, 0), col(mixed, -1, 1), col(styled, 0, 0)
    return [
        practice.Check(
            "ANSWER: 80 templates beat one by 0.095-0.295 top-1 — the real templates are unavailable",
            min(gap) > 0.05 and result["saturated"],
            f"{result['backend']}, so \"a photo of a {{}}\" cannot be tokenised or embedded here; a template is modelled instead as a "
            f"shared style direction ({STYLE}) plus per-class content noise ({NOISE}) on the prototype. Over {len(BANKS)} banks and "
            f"{result['test']} held-out images, K=1 scores {min(single):.3f}-{max(single):.3f} against 1.000 at K={TEMPLATES}: a gap of "
            f"{min(gap):.3f}-{max(gap):.3f}. OpenAI report 1-3% top-1 for the real 80 on ImageNet; not measured here"),
        practice.Check(
            "FINDING: the accuracy gap closes at K=4 — 76 of the 80 templates buy nothing measurable",
            result["saturated"] and min(margin) > 0.3,
            f"top-1 / mean top-1-minus-runner-up margin along K in {KS}: {ladder(mixed)}. All {len(BANKS)} banks reach 1.000 by K=4 and "
            f"stay, so top-1 ranks nothing past that rung; the margin keeps moving, ending at {min(margin):.3f}-{max(margin):.3f} "
            f"against {result['clean'][1]:.3f} for one noiseless prompt -- averaging recovers all the accuracy, about half the margin"),
        practice.Check(
            "MECHANISM: averaging K templates shrinks the noise term by 1/sqrt(K), to within 4%",
            result["law_ok"],
            f"per-dimension RMS of the mean of K content-noise draws against the predicted {NOISE}/sqrt(K): {law_row(result['law'])}. "
            f"That is the whole of \"why templates help\": {TEMPLATES} draws cut the nuisance component "
            f"{result['law'][0][2] / result['law'][-1][2]:.1f}x and leave the class prototype behind"),
        practice.Check(
            "CONTROL: a shared style direction is nearly free, and identical templates gain exactly 0",
            result["worst"] < 0.1 and result["identical"] and max(gaps(plain)) == 0.0,
            f"with noise off and only a style direction of {LOUD} -- 3x the main arm's -- the single-template arm scores "
            f"{min(style_arm):.3f}-{max(style_arm):.3f}, costing at most {result['worst']:.3f} against the {max(gap):.3f} per-class "
            f"noise costs; at style and noise 0 the {TEMPLATES} bank rows are bitwise identical and the gap is exactly "
            f"{max(gaps(plain)):.3f} at {result['clean'][0]:.3f} top-1 -- variance reduction, not extra wording"),
        practice.Check(
            "CONTROL: averaging embeddings is not averaging features — and top-1 cannot tell them apart",
            result["cosine"][0] < 0.97 and result["pre"][0] == result["post"][0],
            f"CLIP averages *after* the tower. Doing it before gives class tables only {result['cosine'][0]:.4f}-"
            f"{result['cosine'][1]:.4f} cosine apart per class, because `text_proj` is a ReLU MLP not a linear map. Both score "
            f"{result['post'][0]:.3f} top-1, so accuracy cannot rank the recipes; the margins differ, {result['post'][1]:.3f} after "
            f"against {result['pre'][1]:.3f} before, on a tower whose loss ended at {result['loss']:.3f}"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
