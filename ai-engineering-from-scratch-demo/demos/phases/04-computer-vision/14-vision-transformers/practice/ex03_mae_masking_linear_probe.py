"""Exercise 3 — mae masking linear probe.

    **(Hard)** Implement MAE pretraining for the tiny ViT: mask 75% of patches, train the encoder + a small decoder to reconstruct the masked patches. Evaluate linear-probe accuracy on the synthetic data before and after pretraining.

Reading of the exercise: the pretraining half works and the evaluation half
cannot. Neither `code/main.py` nor `docs/en.md` ships an MAE, so one is built here
by subclassing the lesson's own `ViT` -- its `patch`, `cls_token`, `pos_embed`,
`blocks` and `ln` are the encoder unchanged, only re-entered through an `encode`
that gathers the 4 surviving tokens first, and the additions are a 96-wide
one-block decoder fed a shared mask token, with the loss on the 12 hidden patches
only. Its inherited `head` is dead weight in this model and is never called. It
learns, on all three seeds. The evaluation the exercise then asks for has no
headroom at all: Lesson 4's `synthetic_cifar` writes class c as the spatial
frequency 2 + c, which is linearly separable in raw pixel space, so a logistic
regression on the 3072 raw pixels is already perfect, the untrained encoder's
frozen CLS features are already perfect, and "before" and "after" are the same
number by construction rather than by accident. That is the answer, and a control
shows the saturation belongs to the dataset rather than to the probe. What the run
does measure honestly is the objective itself -- how far below the
constant-prediction floor reconstruction gets, that the encoder demonstrably never
touches a hidden pixel, and that the same forward pass is far worse on the patches
it *was* handed, because MAE never puts those in the loss. Nothing is downloaded;
the encoder is random-init before pretraining and MAE-pretrained after, and no
published weights exist for a ViT this small in any case.

Structure: `mae_class` returns the MAE subclass, built inside `solve` because `nn`
only exists after the guarded import: `encode` gathers the kept tokens and their
positions before the blocks, `forward` refills the sequence with mask tokens,
unshuffles it with the inverse permutation and decodes. `masks` draws one random
75% mask per image with the argsort trick, returning the kept indices, the restore
permutation and a 0/1 hidden mask; `pretrain` runs the MAE objective, recording the
masked-patch loss beside the loss the same predictions earn on the patches the
encoder was handed; `leak_probe` overwrites every hidden patch with noise and
re-encodes; `seed_run` is one seed end to end. At 146 code lines this sits above
D14's 120-line target: the exercise names three deliverables, and the three seeds
are what make the finding a finding rather than an anecdote.
"""

from __future__ import annotations

from statistics import fmean

from harness import parity, practice

PHASE, LESSON, CLASSIFICATION = "04-computer-vision", "14-vision-transformers", "04-image-classification"

SIZE, PATCH, DIM, HEADS, DEPTH, DEC_DIM = 32, 8, 192, 3, 4, 96
PATCHES, PIXELS, KEEP, GRID = (SIZE // PATCH) ** 2, PATCH * PATCH * 3, 4, SIZE // PATCH
CLASSES, PER_CLASS, SPLIT, BATCH = 10, 60, 480, 64
STEPS, LR, SEEDS, TAIL, MEAN, STD = 200, 1.5e-3, (0, 1, 2), 20, 0.5, 0.25   # MEAN/STD: Lesson 4's own

patch_mse = lambda pred, want, mask: (((pred - want) ** 2).mean(-1) * mask).sum() / mask.sum()   # noqa: E731
span = lambda values: f"{min(values):.3f} to {max(values):.3f}"                 # noqa: E731
cls_of = lambda model, view: model.encode(view)[:, 0].detach().numpy()          # noqa: E731
fit = lambda regression, table, y: float(regression(max_iter=2000)              # noqa: E731
    .fit(table[:SPLIT], y[:SPLIT]).score(table[SPLIT:], y[SPLIT:]))


def mae_class(torch, nn, ref):
    class MAE(ref.ViT):
        def __init__(self):
            super().__init__(image_size=SIZE, patch_size=PATCH, dim=DIM, depth=DEPTH, num_heads=HEADS)
            self.to_dec, self.dec_ln = nn.Linear(DIM, DEC_DIM), nn.LayerNorm(DEC_DIM)
            self.mask_token, self.dec_pos = (nn.Parameter(nn.init.trunc_normal_(torch.zeros(*s), std=0.02))
                                             for s in ((1, 1, DEC_DIM), (1, PATCHES, DEC_DIM)))
            self.dec_block, self.dec_out = ref.Block(DEC_DIM, HEADS), nn.Linear(DEC_DIM, PIXELS)
        def encode(self, x, keep=None):
            tokens = self.patch(x) + self.pos_embed[:, 1:]
            if keep is not None:
                tokens = tokens.gather(1, keep[..., None].expand(-1, -1, DIM))
            tokens = torch.cat([self.cls_token.expand(x.size(0), -1, -1) + self.pos_embed[:, :1], tokens], dim=1)
            for block in self.blocks:
                tokens = block(tokens)
            return self.ln(tokens)
        def forward(self, x, keep, restore):
            seen = self.to_dec(self.encode(x, keep))[:, 1:]
            full = torch.cat([seen, self.mask_token.expand(x.size(0), PATCHES - KEEP, -1)], dim=1)
            full = full.gather(1, restore[..., None].expand(-1, -1, DEC_DIM)) + self.dec_pos
            return self.dec_out(self.dec_ln(self.dec_block(full)))
    return MAE


def masks(torch, count, gen):
    order = torch.rand(count, PATCHES, generator=gen).argsort(dim=1)
    return order[:, :KEEP], order.argsort(dim=1), torch.zeros(count, PATCHES).scatter(1, order[:, KEEP:], 1.0)


def pretrain(torch, model, images, targets, seed) -> dict:
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    gen, masked, seen = torch.Generator().manual_seed(seed), [], []
    for _ in range(STEPS):
        idx, (keep, restore, hidden) = torch.randint(0, SPLIT, (BATCH,), generator=gen), masks(torch, BATCH, gen)
        predicted, wanted = model(images[idx], keep, restore), targets[idx]
        loss = patch_mse(predicted, wanted, hidden)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        masked.append(loss.item())
        seen.append(patch_mse(predicted.detach(), wanted, 1.0 - hidden).item())
    return {"first": masked[0], "masked": fmean(masked[-TAIL:]), "seen": fmean(seen[-TAIL:])}


def leak_probe(torch, model, images, gen) -> dict:
    keep, _restore, hidden = masks(torch, len(images), gen)
    covered = hidden.reshape(-1, 1, GRID, GRID).repeat_interleave(PATCH, 2).repeat_interleave(PATCH, 3)
    dirty = torch.where(covered > 0, torch.randn(images.shape, generator=gen), images)
    with torch.no_grad():
        clean, whole, spoiled = model.encode(images, keep), model.encode(images), model.encode(dirty, keep)
    return {"gap": (clean - spoiled).abs().max().item(), "scale": clean.abs().max().item(),
            "kept": clean.size(1), "whole": whole.size(1), "hidden": int(hidden[0].sum())}


def seed_run(torch, regression, mae_cls, clean, targets, labels, seed) -> dict:
    torch.manual_seed(seed)
    model = mae_cls()
    before = fit(regression, cls_of(model, clean), labels)
    history = pretrain(torch, model, clean, targets, seed)
    return {"before": before, "after": fit(regression, cls_of(model, clean), labels), "model": model, **history}


def solve():
    try:
        import torch
        import torch.nn.functional as functional
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    images, labels = parity.load_reference(PHASE, CLASSIFICATION, "main").synthetic_cifar(
        num_per_class=PER_CLASS, num_classes=CLASSES)
    clean = (torch.from_numpy(images).permute(0, 3, 1, 2).float() - MEAN) / STD
    targets, build = functional.unfold(clean, PATCH, stride=PATCH).transpose(1, 2), mae_class(torch, torch.nn, ref)
    runs = [seed_run(torch, LogisticRegression, build, clean, targets, labels, seed) for seed in SEEDS]
    seen = targets[:SPLIT]
    return {"runs": runs, "val": len(clean) - SPLIT, "spread": float(seen.std()),
            "pixels": fit(LogisticRegression, clean.reshape(len(clean), -1).numpy(), labels),
            "floor": ((seen - seen.reshape(-1, PIXELS).mean(0)) ** 2).mean().item(),
            "leak": leak_probe(torch, runs[0]["model"], clean[:16], torch.Generator().manual_seed(7))}


def verify(result):
    runs, leak, column = result["runs"], result["leak"], lambda key: [run[key] for run in runs]
    return [
        practice.Check(
            "ANSWER: linear-probe accuracy before and after MAE is the same number on every seed",
            max(run["after"] - run["before"] for run in runs) == 0.0 and min(column("before")) == 1.0,
            f"frozen-CLS probe over seeds {SEEDS}, fitted on {SPLIT} images, scored on {result['val']} held out: "
            f"before {span(column('before'))}, after {span(column('after'))} -- 1.000 both sides of every run, so "
            f"it resolves nothing, while {STEPS} steps took masked MSE {span(column('first'))} -> "
            f"{span(column('masked'))}"),
        practice.Check(
            "FINDING: reconstruction is genuinely learned, well past the constant-prediction floor",
            max(column("masked")) < result["floor"] / 4,
            f"the loss covers the {leak['hidden']} hidden patches only, and predicting the training set's mean "
            f"patch for each costs {result['floor']:.4f} (pixel std {result['spread']:.3f}); the model reaches "
            f"{span(column('masked'))}, {result['floor'] / max(column('masked')):.1f}x or more below that floor"),
        practice.Check(
            "MECHANISM: the encoder provably never sees a masked pixel — its sequence is 5 tokens, not 17",
            leak["gap"] < 1e-05 < leak["scale"] and leak["kept"] == KEEP + 1,
            f"masking 75% of {PATCHES} patches keeps {KEEP}, so the encoder sees {leak['kept']} tokens (CLS + "
            f"{KEEP}) not {leak['whole']}; the mask is an argsort permutation, so exactly {leak['hidden']} hide "
            f"each time, and overwriting them with noise moves the latent {leak['gap']:.1e} at scale {leak['scale']:.2f}"),
        practice.Check(
            "CONTROL: the probe was saturated before any encoder existed — raw pixels already score 1.000",
            result["pixels"] >= min(column("before")),
            f"the same logistic regression on the {SIZE * SIZE * 3} un-featurised pixels scores "
            f"{result['pixels']:.3f} on the same split: `synthetic_cifar` writes class c as the spatial frequency "
            f"2 + c, which a linear map reads off directly, so no pretraining recipe could gain here"),
        practice.Check(
            "CONTROL: only the hidden patches are supervised — the same predictions are far worse on the seen ones",
            min(column("seen")) > 4 * max(column("masked")),
            f"the identical forward pass, scored on the {KEEP} patches the encoder was handed instead of the "
            f"{leak['hidden']} it was not, gives {span(column('seen'))} against {span(column('masked'))}, at least "
            f"{min(column('seen')) / max(column('masked')):.1f}x worse -- those outputs never enter the loss, so a "
            f"copying decoder could not cheat the objective"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
