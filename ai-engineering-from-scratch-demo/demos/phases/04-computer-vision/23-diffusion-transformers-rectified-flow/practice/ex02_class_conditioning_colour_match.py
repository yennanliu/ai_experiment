"""Exercise 2 — class conditioning colour match.

    **(Medium)** Add text conditioning by concatenating a learned class embedding to the time embedding (10 blob "classes" by colour). Sample with class 0, 5, and 9 and verify colours match.

Reading of the exercise: two things in it do not exist yet. `synthetic_blobs`
draws `rng.uniform(-1, 1, 3)` per image, so there are no "10 classes by colour" to
condition on -- the labels are built here by repainting the lesson's own blob
masks from a fixed 10-entry palette, which keeps its shapes and adds nothing else.
And "concatenating" cannot be done as written: `TinyDiT.__init__` hardcodes
`DiTBlock(dim, heads, cond_dim=dim)`, so a 2*dim concatenation does not fit the
`AdaLNZero(dim, cond_dim)` linear it feeds. Concatenation therefore has to be
projected back to dim, which is a learned mixture of concat and add, and the DiT
paper's actual choice -- plain addition -- is run as the second arm; at this
budget the projection turns out to earn its parameters several times over. Both
are wired in by *replacing* `time_mlp` rather than editing the reference:
`TinyDiT.forward` calls `self.time_mlp(...)` and nothing else on the conditioning
path, so a module that wraps the original and folds a class embedding into its
output conditions every AdaLN in every block. "Verify colours match" is then a
measurement rather than a look: the unit hue of a sample's brightest pixels,
classified against the palette by cosine, which recovers the labels of the real
data at 1.000 and so is exact on anything blob-shaped.

The last check is about the first optimiser step rather than the samples, because
that is where adaLN-Zero shows: its zero-initialised modulation MLP makes every
DiT block *exactly* the identity at init, so the class embedding this exercise
adds receives exactly zero gradient on the step that follows.

Structure: `classed_blobs` builds the palette and repaints the lesson's own blobs
with it; `hue` reduces an image to the unit colour direction of its brightest
pixels; `make_conditional` builds a TinyDiT whose `time_mlp` has been wrapped to
fold in a class embedding by either route, and `run_arm` trains and samples one
such model with the lesson's own `rectified_flow_train_step` and
`rectified_flow_sample`, setting the labels on the wrapper first;
`at_initialisation` measures a fresh model before any training and then reads
`.grad` after exactly one of those steps, which is legal because the lesson's step
zero-grads before it backprops and only mutates the parameters afterwards.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "23-diffusion-transformers-rectified-flow"

CLASSES, DIM, DEPTH, HEADS, SIZE = 10, 64, 2, 4, 12
TRAIN_STEPS, BATCH, LR, IMAGES = 1000, 32, 3e-4, 200
DRAWN, EULER, BRIGHT, ASKED = 12, 20, 8, (0, 5, 9)

rate = lambda hits: sum(hits) / CLASSES                                                             # noqa: E731
per = lambda hits: " ".join(f"{c}:{hits[c]:.2f}" for c in ASKED)                                    # noqa: E731
matched = lambda hits, chance: rate(hits) > 0.8 and min(hits[c] for c in ASKED) > 3 * chance        # noqa: E731
frozen = lambda g, t: all(g[n] == 0.0 for n in g if "attn" in n or "time_mlp" in n) and t[2] > 0.0  # noqa: E731


def classed_blobs(torch, ref):
    index = torch.arange(CLASSES, dtype=torch.float32)
    angle = 2 * torch.pi * index / CLASSES
    colours = torch.stack([angle.cos(), angle.sin(), torch.where(index % 2 == 0, 0.8, -0.8)], dim=-1)
    images, labels = ref.synthetic_blobs(num=IMAGES, size=SIZE, seed=0), torch.arange(IMAGES) % CLASSES
    painted = colours[labels][:, :, None, None].expand_as(images)
    return torch.where((images != 0).any(1, keepdim=True), painted, torch.zeros(1)), labels, colours


def hue(torch, images):
    where = images.abs().sum(1).flatten(1).topk(BRIGHT, dim=1).indices[:, None, :].expand(-1, 3, -1)
    return torch.nn.functional.normalize(images.flatten(2).gather(2, where).mean(2), dim=-1)


def make_conditional(torch, ref, mode, seed=0):
    class CondTime(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base, self.labels, self.embed = base, None, torch.nn.Embedding(CLASSES, DIM)
            self.project = torch.nn.Linear(2 * DIM, DIM) if mode == "concat" else None
            torch.nn.init.normal_(self.embed.weight, std=0.02)

        def forward(self, stamp):
            timing, cond = self.base(stamp), self.embed(self.labels)
            return self.project(torch.cat([timing, cond], -1)) if self.project else timing + cond

    torch.manual_seed(seed)
    model = ref.TinyDiT(image_size=SIZE, patch_size=2, in_channels=3, dim=DIM, depth=DEPTH, heads=HEADS)
    model.time_mlp = CondTime(model.time_mlp)
    return model


def run_arm(torch, ref, numpy, mode, data, labels, colours):
    model = make_conditional(torch, ref, mode)
    optimiser, generator = torch.optim.Adam(model.parameters(), lr=LR), numpy.random.default_rng(0)
    torch.manual_seed(1)
    history = []
    for _ in range(TRAIN_STEPS):
        picked = generator.choice(IMAGES, BATCH)
        model.time_mlp.labels = labels[picked]
        history.append(ref.rectified_flow_train_step(model, data[picked], optimiser, "cpu"))
    torch.manual_seed(5)
    unit, hits = torch.nn.functional.normalize(colours, dim=-1), []
    for label in range(CLASSES):
        model.time_mlp.labels = torch.full((DRAWN,), label)
        drawn = ref.rectified_flow_sample(model, (DRAWN, 3, SIZE, SIZE), steps=EULER, device="cpu")
        hits.append(float(((hue(torch, drawn) @ unit.T).argmax(1) == label).float().mean()))
    return {"hits": hits, "loss": sum(history[-50:]) / 50, "params": sum(p.numel() for p in model.parameters())}


def at_initialisation(torch, ref, model, data, labels):
    model.time_mlp.labels = torch.zeros(4, dtype=torch.long)
    probe, tokens = torch.randn(4, 3, SIZE, SIZE), torch.randn(4, model.num_patches, DIM)
    with torch.no_grad():
        identity = float((model.blocks[0](tokens, torch.randn(4, DIM)) - tokens).abs().max())
        early = model(probe, torch.zeros(4))
        blind = [float((early - model(probe, torch.ones(4))).abs().max())]
        model.time_mlp.labels = torch.full((4,), CLASSES - 1)
        blind.append(float((early - model(probe, torch.zeros(4))).abs().max()))
    model.time_mlp.labels = labels[:BATCH]
    ref.rectified_flow_train_step(model, data[:BATCH], torch.optim.Adam(model.parameters()), "cpu")
    grads = {name: float(p.grad.abs().max()) for name, p in model.named_parameters()}
    return {"identity": identity, "blind": blind, "grads": grads, "tensors": len(grads),
            "quiet": len([n for n in grads if grads[n] == 0.0]), "still": sum(
                p.numel() for n, p in model.named_parameters() if grads[n] == 0.0),
            "thirds": [float(g.abs().max()) for g in model.blocks[0].adaln1.mlp.weight.grad.chunk(3)]}


def solve():
    try:
        import numpy
        import torch
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    data, labels, colours = classed_blobs(torch, ref)
    arms = {m: run_arm(torch, ref, numpy, m, data, labels, colours) for m in ("concat", "add")}
    fresh = at_initialisation(torch, ref, make_conditional(torch, ref, "concat"), data, labels)
    scored = hue(torch, data) @ torch.nn.functional.normalize(colours, dim=-1).T
    return dict(fresh, arms=arms, marginal=float(data.mean((0, 2, 3)).norm()),
                recovered=float((scored.argmax(1) == labels).float().mean()))


def verify(result):
    arms, thirds, best = result["arms"], result["thirds"], result["arms"]["concat"]
    return [
        practice.Check(
            "ANSWER: classes 0, 5 and 9 come out the colour they were asked for",
            matched(best["hits"], 1 / CLASSES),
            f"{DRAWN} images per class at {EULER} Euler steps, each scored by the hue of its {BRIGHT} brightest "
            f"pixels against the {CLASSES}-colour palette: classes {per(best['hits'])}, ten-class mean "
            f"{rate(best['hits']):.3f} vs {1 / CLASSES:.3f} chance, final loss {best['loss']:.4f}"),
        practice.Check(
            "FINDING: 'concatenating' needs a projection the lesson has no room for -- and it pays",
            best["params"] > arms["add"]["params"] and rate(best["hits"]) > 2 * rate(arms["add"]["hits"]),
            f"`TinyDiT.__init__` hardcodes `DiTBlock(dim, heads, cond_dim=dim)`, so a 2*{DIM} concat does not "
            f"fit the `AdaLNZero` linear it feeds and must be projected back, at "
            f"{best['params'] - arms['add']['params']:,} parameters over add's {arms['add']['params']:,}: "
            f"{rate(best['hits']):.3f} vs {rate(arms['add']['hits']):.3f} for plain addition, on near-equal "
            f"losses ({best['loss']:.4f} vs {arms['add']['loss']:.4f})"),
        practice.Check(
            "CONTROL: the colour cannot be coming from the marginal, which has none",
            result["marginal"] < 0.02 and result["recovered"] == 1.0,
            f"the palette sums to a mean colour of norm {result['marginal']:.4f} over the training set -- nothing "
            f"for an unconditional model to prefer -- yet the same classifier reads the data's own labels at "
            f"{result['recovered']:.3f}"),
        practice.Check(
            "MECHANISM: adaLN-Zero makes every DiT block exactly the identity at initialisation",
            result["identity"] == 0.0 and max(result["blind"]) == 0.0,
            f"`AdaLNZero.__init__` zeroes its modulation linear, so scale = shift = gate = 0 and `x + gate*attn(..)` "
            f"returns x: block output minus input is {result['identity']:.1f} exactly, and the model inherits it -- "
            f"bit-identical for t=0 vs 1 ({result['blind'][0]:.1f}), class 0 vs {CLASSES - 1} ({result['blind'][1]:.1f})"),
        practice.Check(
            "MECHANISM: so the class embedding gets exactly zero gradient on the first step",
            frozen(result["grads"], thirds),
            f"one `rectified_flow_train_step` on that fresh model leaves {result['still']:,} of {best['params']:,} "
            f"parameters at grad exactly 0.0 -- {result['quiet']} of {result['tensors']} tensors: every attention "
            f"and MLP weight, the wrapped `time_mlp`, and the `embed` this exercise adds. Inside "
            f"`self.mlp(cond).chunk(3)` -- scale, shift, gate -- only the gate is alive ({thirds[0]:.1f}, "
            f"{thirds[1]:.1f}, {thirds[2]:.3e}); the rest sit behind a zeroed gate"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
