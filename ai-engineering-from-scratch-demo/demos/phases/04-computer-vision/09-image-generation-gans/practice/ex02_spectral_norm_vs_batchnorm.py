"""Exercise 2 — spectral norm vs batchnorm.

    **(Medium)** Replace the discriminator's batch norm with spectral norm. Train
    both versions side by side. Which one converges faster? Which one has lower
    variance across three seeds?

Reading of the exercise: the lesson's own `Discriminator(use_sn=...)` flag is
exactly the swap being asked for -- with `use_sn=True` the BatchNorm2d layers are
dropped and every conv is wrapped in `spectral_norm` -- so both arms run the
lesson's code with one argument changed, three seeds each. "Converges" needs a
referent, because a GAN's losses are a score in a game, not a distance to an
optimum: it is read here as "the discriminator stops winning", measured by the
D and G losses over the last three epochs together with how much of the frame
the generator is still painting (the real circles fill 0.198 of it). The second
half of the question hides a trap. Lower seed-to-seed variance is scored as if it
were a virtue, and the batch-norm arm wins it -- because all three of its seeds
fail in the same place. Variance is only informative next to the level it is
measured around, so both are reported. Scaled to feat=16 and 256 images so the
six runs fit a CI core in ~30 s. Every claim below was re-run on seeds 3-5
offline and all six held there, though one of that draw's three spectral-norm
runs finished at the batch-norm arm's level -- which is the spread the third and
fourth checks are about.
"""

from __future__ import annotations

import statistics

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "09-image-generation-gans"

Z_DIM, FEAT, NUM, BATCH, EPOCHS = 64, 16, 256, 32, 20
SEEDS, TAIL, EVAL_N, EVAL_SEED = (0, 1, 2), 3, 32, 1234
THRESHOLD = 0.175                   # background is -1, every circle colour >= -0.3

mean = lambda values: sum(values) / len(values)                                  # noqa: E731
tail = lambda run, key: mean(run[key][-TAIL:])                                   # noqa: E731
spread = lambda values: statistics.pstdev(values)                                # noqa: E731
row = lambda runs, key: " ".join(f"{tail(r, key):.2f}" for r in runs)            # noqa: E731
ends = lambda arms, key: {k: [tail(r, key) for r in v] for k, v in arms.items()}  # noqa: E731


def area_frac(torch, imgs) -> float:
    """Fraction of the frame the generator has painted above the background."""
    return float((imgs.max(1).values > THRESHOLD).float().mean())


def run(torch, ref, loader_cls, dataset_cls, real, use_sn, seed) -> dict:
    """One arm, one seed: the lesson's own loop with `use_sn` the only change."""
    torch.manual_seed(seed)
    G = ref.Generator(z_dim=Z_DIM, img_channels=3, feat=FEAT)
    D = ref.Discriminator(img_channels=3, feat=FEAT, use_sn=use_sn)
    opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    loader = loader_cls(dataset_cls(real), batch_size=BATCH, shuffle=True)
    out = {"d": [], "g": [], "area": [], "params": sum(p.numel() for p in D.parameters())}
    for _ in range(EPOCHS):
        losses = [ref.train_step(G, D, batch, torch.randn(batch.size(0), Z_DIM), opt_g, opt_d, "cpu")
                  for (batch,) in loader]
        with torch.random.fork_rng():
            torch.manual_seed(EVAL_SEED)
            out["area"].append(area_frac(torch, ref.sample(G, n=EVAL_N, z_dim=Z_DIM)))
        out["d"].append(mean([ld for ld, _ in losses]))
        out["g"].append(mean([lg for _, lg in losses]))
    return out


def solve():
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    real = ref.synthetic_circles(num=NUM)
    arms = {name: [run(torch, ref, DataLoader, TensorDataset, real, use_sn, seed) for seed in SEEDS]
            for name, use_sn in (("bn", False), ("sn", True))}
    return {"arms": arms, "real_area": area_frac(torch, (real + 1) / 2)}


def verify(result):
    bn, sn = result["arms"]["bn"], result["arms"]["sn"]
    real = result["real_area"]
    d_end, g_end, a_end = ends(result["arms"], "d"), ends(result["arms"], "g"), ends(result["arms"], "area")
    lowest = min(a_end["bn"] + a_end["sn"])
    return [
        practice.Check(
            "ANSWER: spectral norm converges faster, on the mean of three seeds",
            mean(g_end["sn"]) < mean(g_end["bn"]) - 0.75 and mean(d_end["sn"]) > 2 * mean(d_end["bn"]),
            f"over the last {TAIL} of {EPOCHS} epochs, G loss averages {mean(g_end['sn']):.2f} with spectral "
            f"norm against {mean(g_end['bn']):.2f} with batch norm, and D loss {mean(d_end['sn']):.2f} "
            f"against {mean(d_end['bn']):.2f} -- {mean(d_end['sn']) / mean(d_end['bn']):.1f}x. Per seed, "
            f"SN G {row(sn, 'g')} vs BN G {row(bn, 'g')}"),
        practice.Check(
            "MECHANISM: with batch norm the discriminator wins outright, in all three seeds",
            max(d_end["bn"]) < 0.15 and min(g_end["bn"]) > 3.0 and min(a_end["bn"]) > 0.9,
            f"every batch-norm seed ends with D loss below 0.15 ({row(bn, 'd')}) and G loss above 3.0 "
            f"({row(bn, 'g')}) -- D separates real from fake almost perfectly, so G's gradient is a small "
            f"number times a saturated sigmoid. Its samples still cover {min(a_end['bn']):.0%}+ of the frame "
            f"against {real:.1%} for the data: nothing has been learned"),
        practice.Check(
            "ANSWER: batch norm has the lower seed-to-seed variance -- and that is the trap",
            2 * spread(a_end["bn"]) < spread(a_end["sn"]),
            f"standard deviation of the painted area across the three seeds is {spread(a_end['bn']):.4f} for "
            f"batch norm and {spread(a_end['sn']):.4f} for spectral norm, "
            f"{spread(a_end['sn']) / spread(a_end['bn']):.0f}x wider. But BN's three seeds cluster at "
            f"{row(bn, 'area')} and SN's at {row(sn, 'area')}: the tight arm is tight because all three of "
            "its runs are sitting on the same failure"),
        practice.Check(
            "FINDING: read around a level, the same spread is the SN arm's escape, not its noise",
            min(a_end["sn"]) < min(a_end["bn"]) - 0.02 and max(a_end["bn"]) - min(a_end["bn"]) < 0.1,
            f"the batch-norm arm spans {max(a_end['bn']) - min(a_end['bn']):.3f} in painted area, the "
            f"spectral-norm arm {max(a_end['sn']) - min(a_end['sn']):.3f}; SN's best seed reaches "
            f"{min(a_end['sn']):.3f} where BN's best is {min(a_end['bn']):.3f}. The spread appears because "
            "some SN seeds get moving and others do not -- a variance you want, at a level you want"),
        practice.Check(
            "CONTROL: the swap is not a capacity change",
            abs(bn[0]["params"] - sn[0]["params"]) < 0.01 * bn[0]["params"],
            f"the batch-norm discriminator holds {bn[0]['params']:,} parameters and the spectral-norm one "
            f"{sn[0]['params']:,}, a difference of {abs(bn[0]['params'] - sn[0]['params'])} "
            f"({abs(bn[0]['params'] - sn[0]['params']) / bn[0]['params']:.2%}). `use_sn=True` drops the two "
            "BatchNorm2d layers and adds a power-iteration buffer per conv; the convs are untouched"),
        practice.Check(
            "CONTROL: winning this comparison is not the same as solving the task",
            lowest > 3 * real,
            f"the best of all six runs still paints {lowest:.1%} of the frame against {real:.1%} for the real "
            f"circles, {lowest / real:.1f}x too much, after {EPOCHS} epochs at feat={FEAT}. Spectral norm "
            "moves the arm that is losing more slowly; neither arm has generated a circle"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
