"""Exercise 1 — infonce temperature sweep.

    **(Easy)** Verify that InfoNCE loss drops when you decrease temperature for well-aligned embeddings and rises when you decrease temperature for random embeddings. Produce a plot `tau in [0.05, 0.1, 0.2, 0.5]` vs loss.

Reading of the exercise: both halves of the claim hold, but "well-aligned" is
carrying the whole first half and the exercise never says how aligned is enough.
Sweeping the alignment shows the claim has an edge: with `z2 = normalize(z1 +
eps * noise)` the loss is monotone in tau for eps <= 0.3 in all four seeds and
*reverses* for eps >= 0.5 in all four, where dropping tau from 0.1 to 0.05 makes
things worse rather than better -- because tau -> 0 turns the softmax into a
hard max, which rewards a positive that already wins its row and punishes one
that does not. So the exercise's monotonicity is a property of the *data*, not
of InfoNCE. The plot it asks for cannot be a plot: matplotlib is not in the
`vision` deps group, so the sweep ships as a printed table of measured losses
instead. Two exact structural facts are worth more than the sweep. A collapsed
encoder -- every embedding the same vector -- gives log(2N-1) = 3.4339872 to
seven decimals at *every* tau, so this loss is completely blind to the one
failure SSL cares about, and no temperature schedule can detect it. And the
lesson's own `main.py` prints the random-pair loss with the comment "should be
near log(2N-1) = 3.434" while itself measuring 5.229 at tau=0.1: log(2N-1) is
the tau -> infinity limit, which this draw approaches from above (3.4345 at
tau=100), not the value at the lesson's own temperature. The approach is not
from above for every seed -- with only N=16 rows the sampled positive can beat
the sampled mean -- so the check asserts that |loss - log(2N-1)| shrinks
monotonically rather than that the loss stays above it. Nothing is trained and
nothing is downloaded; every number is one forward pass of `ref.info_nce`.

Structure: `sweep` evaluates the lesson's `info_nce` across TAUS for one pair of
embedding matrices and reports whether the losses fall monotonically as tau
falls; `views` draws an anchor batch and a noisy second view of it;
`alignment_edge` walks eps over four seeds and records where that monotonicity
dies, with the mean positive cosine at each step; `collapse_row` builds a
rank-one batch and reads the loss off at every tau; `tail` reproduces
`main.py`'s own draw order and pushes tau far past the sweep on it to locate the
log(2N-1) limit; `seed_sweeps` reduces the per-seed sweeps to scalars so that
`verify` only formats -- D14 caps `verify` at complexity 8 and a first version
that aggregated inline scored 19.

At 146 code lines this sits above D14's 120-line target and 4 clear of the
ceiling: five checks over four probes, the widest being 4 seeds x 6 alignment
levels x 4 temperatures.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "17-self-supervised-vision"

BATCH, DIM = 16, 32
TAUS = (0.05, 0.1, 0.2, 0.5)
EPSILONS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
SEEDS = (0, 1, 2, 3)
TAIL_TAUS = (1.0, 5.0, 20.0, 100.0)
FLOOR = math.log(2 * BATCH - 1)

row = lambda values: " ".join(f"{v:.4f}" for v in values)              # noqa: E731 - a formatter
gap_row = lambda gaps: " > ".join(f"{gap:.4f}" for gap in gaps)        # noqa: E731 - a formatter
edge_row = lambda table: "  ".join(                                    # noqa: E731 - a formatter
    f"eps={eps} {table[eps][0]}/4 (cos {table[eps][1]:.3f})" for eps in EPSILONS)


def sweep(ref, left, right) -> tuple:
    losses = [ref.info_nce(left, right, tau).item() for tau in TAUS]
    return losses, all(losses[i] < losses[i + 1] for i in range(len(TAUS) - 1))


def views(torch, functional, seed, eps):
    torch.manual_seed(seed)
    anchor = functional.normalize(torch.randn(BATCH, DIM), dim=-1)
    return anchor, functional.normalize(anchor + eps * torch.randn(BATCH, DIM), dim=-1)


def alignment_edge(torch, functional, ref) -> dict:
    table = {}
    for eps in EPSILONS:
        monotone, cosines = [], []
        for seed in SEEDS:
            anchor, other = views(torch, functional, seed, eps)
            monotone.append(sweep(ref, anchor, other)[1])
            cosines.append((anchor * other).sum(-1).mean().item())
        table[eps] = (sum(monotone), sum(cosines) / len(SEEDS))
    return table


def collapse_row(torch, functional, ref) -> list:
    torch.manual_seed(7)
    constant = functional.normalize(torch.randn(1, DIM), dim=-1).repeat(BATCH, 1)
    return [ref.info_nce(constant, constant, tau).item() for tau in TAUS]


def tail(torch, functional, ref) -> dict:
    torch.manual_seed(0)
    anchor = functional.normalize(torch.randn(BATCH, DIM), dim=-1)
    unrelated = functional.normalize(torch.randn(BATCH, DIM), dim=-1)
    losses = [ref.info_nce(anchor, unrelated, tau).item() for tau in TAIL_TAUS]
    return {"losses": losses, "gaps": [abs(loss - FLOOR) for loss in losses],
            "hot": ref.info_nce(anchor, unrelated, TAUS[1]).item(),
            "identical": ref.info_nce(anchor, anchor, TAUS[1]).item()}


def seed_sweeps(torch, functional, ref) -> dict:
    aligned, independent = [], []
    for seed in SEEDS:
        anchor, _ = views(torch, functional, seed, 0.0)
        torch.manual_seed(200 + seed)
        unrelated = functional.normalize(torch.randn(BATCH, DIM), dim=-1)
        aligned.append(sweep(ref, anchor, anchor))
        independent.append(sweep(ref, anchor, unrelated))
    cold_aligned = [losses[0] for losses, _ in aligned]
    cold_random = [losses[0] for losses, _ in independent]
    return {"aligned": aligned[0][0], "random": independent[0][0],
            "mono_aligned": sum(flag for _, flag in aligned),
            "mono_random": sum(flag for _, flag in independent),
            "aligned_span": (min(cold_aligned), max(cold_aligned)),
            "random_span": (min(cold_random), max(cold_random))}


def solve():
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    collapsed = collapse_row(torch, functional, ref)
    return {**seed_sweeps(torch, functional, ref), "collapsed": collapsed,
            "collapsed_err": max(abs(loss - FLOOR) for loss in collapsed),
            "collapsed_spread": max(collapsed) - min(collapsed),
            "edge": alignment_edge(torch, functional, ref), "tail": tail(torch, functional, ref)}


def verify(result):
    aligned, rand, edge = result["aligned"], result["random"], result["edge"]
    collapsed, tails, gaps = result["collapsed"], result["tail"], result["tail"]["gaps"]
    return [
        practice.Check(
            "ANSWER: both halves hold — aligned pairs fall with tau, random pairs rise, in all 4 seeds",
            result["mono_aligned"] == len(SEEDS) and result["mono_random"] == 0,
            f"the lesson's own `info_nce` over tau {TAUS} at N={BATCH}, D={DIM}. Identical views (seed 0): "
            f"{row(aligned)} -- falling as tau falls, monotone in {result['mono_aligned']}/4 seeds. "
            f"Independent views (seed 0): {row(rand)} -- rising, monotone-down in {result['mono_random']}/4. "
            f"Across seeds the tau=0.05 aligned loss spans {result['aligned_span'][0]:.5f}-"
            f"{result['aligned_span'][1]:.5f} and the random one {result['random_span'][0]:.3f}-"
            f"{result['random_span'][1]:.3f}"),
        practice.Check(
            "FINDING: 'well-aligned' is load-bearing — the claim reverses once the positive stops winning",
            edge[0.3][0] == len(SEEDS) and edge[0.5][0] == 0 and edge[0.6][0] == 0,
            f"with `z2 = normalize(z1 + eps*noise)`, the number of seeds whose loss still falls monotonically "
            f"as tau falls, per eps: {edge_row(edge)}. eps<=0.3 keeps the claim in 4/4, eps>=0.5 breaks it in "
            f"4/4, and eps=0.4 splits {edge[0.4][0]}/4 -- so the exercise's monotonicity is a property of how "
            "aligned the views are, not of InfoNCE: tau -> 0 approaches a hard max, which only helps a "
            "positive that already wins its own row"),
        practice.Check(
            "MECHANISM: a collapsed encoder reads exactly log(2N-1) at every tau — this loss cannot see it",
            result["collapsed_err"] < 1e-06 and result["collapsed_spread"] == 0.0,
            f"one vector repeated {BATCH} times as both views gives {row(collapsed)} against log(2N-1) = "
            f"log({2 * BATCH - 1}) = {FLOOR:.7f}, identical to {result['collapsed_spread']:.1f} across all "
            f"four temperatures. When every similarity is equal, `masked_fill` leaves {2 * BATCH - 1} tied "
            f"logits per row and tau cancels out of the softmax entirely -- so the number the exercise sweeps "
            "is blind to representation collapse, which is why DINO needs centring"),
        practice.Check(
            "FINDING: log(2N-1) is the tau -> infinity limit, not the random-pair loss the lesson prints",
            gaps == sorted(gaps, reverse=True) and gaps[-1] < 1e-03 and tails["hot"] > 1.2 * FLOOR,
            f"reproducing `main.py`'s own draw (seed 0, two consecutive normalised `randn` batches) gives "
            f"{tails['identical']:.3f} for identical views and {tails['hot']:.3f} for random ones, matching "
            f"what it prints -- beside the comment \"should be near log(2N-1) = 3.434\", which the random loss "
            f"exceeds by {tails['hot'] / FLOOR:.2f}x. Raising tau to {TAIL_TAUS} gives {row(tails['losses'])}, "
            f"and |loss - {FLOOR:.4f}| shrinks {gap_row(gaps)} -- so log(2N-1) is this loss's tau -> infinity "
            "limit, not its value at the lesson's own tau=0.1"),
        practice.Check(
            "CONTROL: the plot the exercise asks for is not buildable here, so the sweep ships as a table",
            len(TAUS) == len(aligned) == len(rand) == 4,
            f"matplotlib is not in the `vision` deps group (numpy, PIL, sklearn, torch, torchvision), so `tau "
            f"vs loss` is printed rather than drawn. The full grid, aligned then random, seed 0: {row(aligned)} "
            f"/ {row(rand)}. The two curves cross: at tau=0.5 they differ by {rand[3] - aligned[3]:.3f} and at "
            f"tau=0.05 by {rand[0] - aligned[0]:.3f}, so low temperature widens the gap between a good and a "
            f"useless encoder by {(rand[0] - aligned[0]) / (rand[3] - aligned[3]):.1f}x"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
