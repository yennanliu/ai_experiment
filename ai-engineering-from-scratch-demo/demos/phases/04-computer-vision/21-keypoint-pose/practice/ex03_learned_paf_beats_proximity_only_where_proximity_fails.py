"""Exercise 3 — the learned PAF beats proximity only where proximity fails.

    **(Hard)** Build a 2-person synthetic dataset where each image shows two
    instances of the 4-keypoint pattern. Train a bottom-up pipeline with PAFs that
    predict which keypoint belongs to which instance, and evaluate OKS.

Reading of the exercise: the "4-keypoint pattern" is a chain 0-1-2-3 with 8 px
limbs, so there are 3 PAFs (6 channels). Keypoint k is drawn in its own colour,
because exercise 1 shows the lesson's identical black squares give a heatmap no
identity at all. The lesson's own `TinyKeypointNet(num_keypoints=10)` predicts the
4 heatmaps (the lesson's `gaussian_heatmap`) and the 6 PAF channels under the
lesson's plain MSE, 600 steps. Decoding is bottom-up: two peaks per type (the
lesson's `heatmap_to_coords`, then again after suppressing a 7x7 window), then
each limb's 2x2 pairing chosen by the higher PAF line integral. OKS uses
`exp(-d^2 / 2 s^2 k^2)` with s = 24 px (the chain's length) and k = 0.1, matched
to the better of the two instance assignments. Two layouts are scored:
`random` (both chains anywhere, same or opposite direction) and `antiparallel`
(the second chain runs back along the first, 4 px to the side). A PAF only has
something to prove against grouping by proximity, so that is scored too.

**ANSWER: OKS 0.880 (random) and 0.963 (antiparallel)** with the learned PAF.

**FINDING: where proximity fails, the PAF is decisive.** In the antiparallel
layout the middle limb's wrong pairs are 4 px long and the right ones 8 px, so
nearest-neighbour grouping gets **0%** of images right (OKS **0.489**), and the
learned PAF **100%** (OKS 0.963).

**FINDING: everywhere else the learned PAF is the worse grouper.** Under random
placement it groups **73.4%** of images correctly against **87.5%** for plain
proximity (OKS 0.880 vs 0.937). The field it learned is weak: its dot with the
true limb direction averages **0.41** on limb pixels, against 1.0 in the target,
so where the two chains happen to lie near each other a wrong pair can collect
as much of it as the right one.

**CONTROL: the decoder is not the problem.** Fed the target heatmaps and PAFs,
the same decoder groups **100%** of both layouts (OKS 1.000), while proximity on
the same targets groups **92.2%** of random layouts and **0%** of antiparallel ones.

The file is 29 lines over D14's 120-line target. What they buy is the
exercise's own deliverables at the smallest scale that still says something:
a dataset with heatmap *and* PAF targets (`place`, `limb_field`, `sample`), a
bottom-up decoder (`paf`, `group`, `score`), and the proximity baseline and
target-field control without which the OKS number would have nothing to be
compared against. The four checks are 30 lines, most of them measured numbers.

Structure: `place` lays out the two chains; `sample` renders image, heatmaps
and PAFs; `group` chains limbs by `paf` or by proximity; `score` decodes two
peaks per type, groups them, and returns mean OKS and the fraction grouped right.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "21-keypoint-pose"
S, LIMB, STEPS, EVAL, SCALE, KAPPA = 64, 8, 600, 64, 24.0, 0.1
LIMBS, LAYOUTS = ((0, 1), (1, 2), (2, 3)), ("random", "antiparallel")
COLOURS, LINK = ((0, 1, 1), (1, 0, 1), (1, 1, 0), (0, 0, 0)), ("paf", "near")


def place(np, rng, layout):
    """Two 4-keypoint chains, no two keypoints of different chains within 4 px."""
    while True:
        t = rng.uniform(0, 2 * np.pi)
        u, n = np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])
        steps = np.outer(np.arange(4) * LIMB, u)
        a = rng.uniform(20, 44, size=2) - 12 * u + steps
        sign, other = np.sign(rng.uniform(-1, 1)), rng.uniform(20, 44, 2) - 12 * u
        b = a[::-1] + 4 * n if layout == "antiparallel" else other + sign * steps
        inst = [np.round(a).astype(int), np.round(b).astype(int)]
        inside = all(((k >= 4) & (k < S - 4)).all() for k in inst)
        if inside and np.abs(inst[0][:, None] - inst[1][None]).max(-1).min() >= 4:
            return inst


def limb_field(np, p, q):
    """Unit vector p->q on pixels within 1.5 px of the segment, and that mask."""
    (yy, xx), u = np.mgrid[0:S, 0:S], (q - p) / np.linalg.norm(q - p)
    along = (xx - p[0]) * u[0] + (yy - p[1]) * u[1]
    near = np.abs((xx - p[0]) * u[1] - (yy - p[1]) * u[0]) <= 1.5
    on = ((along >= 0) & (along <= np.linalg.norm(q - p)) & near).astype("float32")
    return np.stack([on * u[0], on * u[1]]), on


def sample(np, ref, rng, layout):
    inst, img = place(np, rng, layout), np.ones((3, S, S), "float32")
    for kp in inst:
        for k, (x, y) in enumerate(kp):
            img[:, y - 2 : y + 2, x - 2 : x + 2] = np.reshape(COLOURS[k], (3, 1, 1))
    hm = [np.maximum(*(ref.gaussian_heatmap(S, *kp[k]) for kp in inst)) for k in range(4)]
    for a, b in LIMBS:
        (f0, m0), (f1, m1) = (limb_field(np, 1.0 * kp[a], 1.0 * kp[b]) for kp in inst)
        hm.extend((f0 + f1) / np.maximum(m0 + m1, 1))
    return img, np.stack(hm).astype("float32"), np.stack(inst).astype("float32")


def paf(field, p, q):
    """Mean of field . unit(q - p) at 10 points along p->q: the PAF line integral."""
    v = q - p
    u = v / max(float(v.norm()), 1e-6)
    pts = [(p + t / 9 * v).round().long().clamp(0, S - 1) for t in range(10)]
    return sum(float(field[0, y, x] * u[0] + field[1, y, x] * u[1]) for x, y in pts) / 10


def group(torch, cand, field, how):
    """Chain limbs 0-1-2-3, choosing each 2x2 pairing by the higher total link."""
    a, b = [cand[0, 0]], [cand[0, 1]]
    link = paf if how == "paf" else lambda _f, p, q: -float((p - q).norm())
    for i, (_, k) in enumerate(LIMBS):
        c0, c1, f = cand[k, 0], cand[k, 1], field[2 * i : 2 * i + 2]
        keep = link(f, a[-1], c0) + link(f, b[-1], c1) >= link(f, a[-1], c1) + link(f, b[-1], c0)
        a, b = a + [c0 if keep else c1], b + [c1 if keep else c0]
    return torch.stack([torch.stack(a), torch.stack(b)])


def score(torch, ref, out, gts, how):
    """(mean OKS, fraction of images whose instances are grouped correctly)."""
    first, rest, total, right = ref.heatmap_to_coords(out[:, :4]), out[:, :4].clone(), 0.0, 0
    for n in range(len(gts)):  # second peak per type: argmax again after a 7x7 suppression
        for k, (x, y) in enumerate(first[n].long().tolist()):
            rest[n, k, max(y - 3, 0) : y + 4, max(x - 3, 0) : x + 4] = -1e9
    cand = torch.stack([first, ref.heatmap_to_coords(rest)], 2)
    for n, gt in enumerate(gts):
        pred = group(torch, cand[n], out[n, 4:], how)
        d2 = ((pred - gt[torch.tensor([[0, 1], [1, 0]])]) ** 2).sum(-1)  # both instance assignments
        total += float(torch.exp(-d2 / (2 * (SCALE * KAPPA) ** 2)).mean((1, 2)).max())
        owner = (pred[None] - gt[:, None]).norm(dim=-1).argmin(0)  # nearest true instance
        right += bool((owner == owner[:, :1]).all() and owner[0, 0] != owner[1, 0])
    return round(total / len(gts), 3), right / len(gts)


def solve():
    try:
        import numpy as np
        import torch
    except ImportError as exc:  # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    torch.set_num_threads(2)
    torch.manual_seed(0)
    ref, rng = parity.load_reference(PHASE, LESSON, "main"), np.random.default_rng(0)
    batch = lambda r, n, lay: map(
        torch.from_numpy, map(np.stack, zip(*[sample(np, ref, r, lay) for _ in range(n)]))
    )
    opt = torch.optim.Adam((model := ref.TinyKeypointNet(num_keypoints=10)).parameters(), lr=3e-3)
    for step in range(STEPS):
        img, target, _ = batch(rng, 16, LAYOUTS[step % 2])
        loss = torch.nn.functional.mse_loss(model(img), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    result = {}
    for layout in LAYOUTS:
        img, target, gts = batch(np.random.default_rng(1), EVAL, layout)
        runs = {"pred": model.eval()(img).detach(), "true": target}
        result[layout] = {
            f"{s}_{h}": score(torch, ref, runs[s], gts, h) for s in runs for h in LINK
        }
        pred, true = (x[:, 4:].reshape(EVAL, 3, 2, S, S) for x in runs.values())
        result[layout]["strength"] = float((pred * true).sum(2)[true.norm(dim=2) > 0.5].mean())
    return result


def verify(result):
    rnd, anti = result["random"], result["antiparallel"]
    oks = lambda r, k: f"{r[k][1]:.1%} grouped, OKS {r[k][0]:.3f}"
    return [
        practice.Check(
            "ANSWER: the learned bottom-up pipeline reaches OKS above 0.85 on both layouts",
            rnd["pred_paf"][0] > 0.85 and anti["pred_paf"][0] > 0.85,
            f"OKS (s={SCALE:.0f} px, k={KAPPA}): random {oks(rnd, 'pred_paf')}, "
            f"antiparallel {oks(anti, 'pred_paf')}",
        ),
        practice.Check(
            "FINDING: where proximity fails, the PAF is decisive",
            anti["pred_near"][1] < 0.1 and anti["pred_paf"][1] > 0.9,
            f"antiparallel middle limbs put wrong pairs 4 px apart and right ones {LIMB} px: "
            f"proximity {oks(anti, 'pred_near')}, learned PAF {oks(anti, 'pred_paf')}",
        ),
        practice.Check(
            "FINDING: everywhere else the learned PAF is the worse grouper",
            rnd["pred_paf"][1] < rnd["pred_near"][1] and rnd["strength"] < 0.6,
            f"random layouts: learned PAF {oks(rnd, 'pred_paf')}, proximity "
            f"{oks(rnd, 'pred_near')}; the learned field's dot with the true limb direction "
            f"is {rnd['strength']:.2f} on limb pixels, against 1.0 in the target",
        ),
        practice.Check(
            "CONTROL: on target fields the same decoder groups every image",
            rnd["true_paf"] == anti["true_paf"] == (1.0, 1.0) and anti["true_near"][1] == 0,
            f"target PAFs: random {oks(rnd, 'true_paf')}, antiparallel {oks(anti, 'true_paf')}; "
            f"proximity on the same targets: {oks(rnd, 'true_near')}, {oks(anti, 'true_near')}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
