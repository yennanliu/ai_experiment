"""Exercise 2 — without warm-up, post-LN collapses to one class on 3 of 3 seeds and pre-LN reaches 67%.

    Swap pre-LN for post-LN and train for one epoch on a synthetic shape classifier. Observe which one trains stably without LR warm-up.

Reading of the exercise: the lesson's `Block` has no post-LN switch, so
post-LN is a replacement `forward` bound onto each block,
`x = ln1(x + attn(x)); x = ln2(x + ffn(x))`, reusing the block's own
modules and weights. The model is the lesson's `VisionEncoder` at the
lesson's depth of 12 and 4x MLP, scaled to CPU: 32x32 images, patch 8
(17 tokens), width 64, 4 heads, plus a linear head on the CLS vector. The
shape set is 4,096 noisy images of a square, a disc or a plus (3 classes),
with 512 held out. One epoch is 128 Adam steps of 32 at a constant lr of
1e-3 with no warm-up. Pre-LN and post-LN are each trained from seeds 0, 1
and 2. As a control, post-LN on seed 0 is also run with a 32-step linear
warm-up.

**ANSWER: pre-LN trains stably and post-LN does not.** Pre-LN ends the
epoch at 67.4%, 65.6% and 67.8% held-out accuracy on the three seeds
(chance is 33%). Post-LN collapses on all three: its loss settles at
ln 3 = 1.099 and it predicts one class for all 512 test images, 31.4%.
Its worst batch loss is also higher, 3.45 on seed 0 against
pre-LN's 2.78.

**FINDING: a 32-step warm-up rescues the same post-LN run.** The
seed-0 post-LN model at the same lr, with only a 32-step linear warm-up
added, reaches 64.1% and does not collapse. This matches the lesson's claim that pre-LN "trains stably
without learning-rate warm-up tricks", and shows the failure is at the
start of training: the post-LN weights are not unusable.

Structure: `shapes()` draws the labelled images; `post_ln()` is the
swapped block forward; `train()` runs one epoch and scores the held-out set.
"""

from __future__ import annotations

import math

from harness import parity, practice

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - depends on the installed extras
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "59-vit-transformer"
SIZE, N_TRAIN, N_TEST, BATCH, LR, WARM = 32, 4096, 512, 32, 1e-3, 32
SEEDS = (0, 1, 2)


def shapes(n, seed):
    """n noisy 3x32x32 images of a square (0), a disc (1) or a plus (2)."""
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(SIZE), torch.arange(SIZE), indexing="ij")
    labels = torch.randint(0, 3, (n,), generator=g)
    imgs = torch.zeros(n, 3, SIZE, SIZE)
    for i in range(n):
        r = int(torch.randint(4, 9, (1,), generator=g))
        cy, cx = (int(v) for v in torch.randint(r, SIZE - r, (2,), generator=g))
        dy, dx = (yy - cy).abs(), (xx - cx).abs()
        masks = [(dy <= r) & (dx <= r), dy**2 + dx**2 <= r * r,
                 ((dy <= r) & (dx <= 1)) | ((dx <= r) & (dy <= 1))]
        imgs[i] = masks[int(labels[i])].float()
    return imgs + 0.1 * torch.randn(imgs.shape, generator=g), labels


def post_ln(self, x, store_attn=False):
    x = self.ln1(x + self.attn(x, store_attn=store_attn))
    return self.ln2(x + self.ffn(x))


def train(ref, post, seed, warm=0):
    torch.manual_seed(seed)
    cfg = ref.ViTConfig(image_size=SIZE, patch_size=8, hidden=64, depth=12, heads=4)
    enc, head = ref.VisionEncoder(cfg), torch.nn.Linear(64, 3)
    if post:
        for block in enc.vit.blocks:
            block.forward = post_ln.__get__(block)
    opt = torch.optim.Adam([*enc.parameters(), *head.parameters()], lr=LR)
    x, y = shapes(N_TRAIN, 0)
    losses = []
    for step, s in enumerate(range(0, N_TRAIN, BATCH)):
        for group in opt.param_groups:
            group["lr"] = LR * min(1.0, (step + 1) / warm) if warm else LR
        loss = F.cross_entropy(head(enc(x[s:s + BATCH])[1]), y[s:s + BATCH])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    xt, yt = shapes(N_TEST, 1)
    with torch.no_grad():
        pred = head(enc(xt)[1]).argmax(-1)
    acc, classes, end = (pred == yt).float().mean().item(), int(pred.unique().numel()), sum(losses[-4:]) / 4
    return {"acc": acc, "classes": classes, "end_loss": end, "max_loss": max(losses),
            "trained": acc > 0.6 and classes == 3,
            "collapsed": classes == 1 and abs(end - math.log(3)) < 0.02}


def column(runs, key, digits=None):
    return [r[key] if digits is None else round(r[key], digits) for r in runs]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    pre, post = [train(ref, False, s) for s in SEEDS], [train(ref, True, s) for s in SEEDS]
    return {"pre": pre, "post": post, "warm": train(ref, True, 0, warm=WARM),
            "pre_ok": all(column(pre, "trained")), "post_collapsed": all(column(post, "collapsed")),
            "acc": {"pre": column(pre, "acc", 3), "post": column(post, "acc", 3)},
            "post_classes": column(post, "classes"), "post_end": column(post, "end_loss", 3)}


def verify(result):
    r, pre, post, warm = result, result["pre"], result["post"], result["warm"]
    return [
        practice.Check(
            "ANSWER: pre-LN trains stably and post-LN does not",
            all([r["pre_ok"], r["post_collapsed"], post[0]["max_loss"] > pre[0]["max_loss"]]),
            f"held-out accuracy pre-LN {r['acc']['pre']}; post-LN {r['acc']['post']}, predicting "
            f"{r['post_classes']} class(es), end loss {r['post_end']} (ln 3 = {math.log(3):.3f}); "
            f"seed-0 max loss "
            f"post {post[0]['max_loss']:.2f} vs pre {pre[0]['max_loss']:.2f}",
        ),
        practice.Check(
            "FINDING: a 32-step warm-up rescues the same post-LN run",
            all([warm["acc"] > 0.6, warm["classes"] == 3]),
            f"post-LN seed 0 with a {WARM}-step warm-up: {warm['acc']:.1%}, "
            f"{warm['classes']} classes predicted, end loss {warm['end_loss']:.3f}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
