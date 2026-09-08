"""Exercise 3 — lora rank step budget.

    **(Hard)** Train a LoRA on 10-20 images of a single subject (a pet, a logo, a character) and generate novel scenes with that subject in them. Report the LoRA rank and training steps that produced the best identity preservation without overfitting to the input images.

Reading of the exercise: Stable Diffusion's weights are not downloadable here and
a CPU could not fine-tune them if they were (exercise 1 measures both), so the
frozen base model is lesson 10's own TinyUNet, trained by lesson 10's own
`train_step` on its own `synthetic_circles` at its own T=200 schedule -- 37,395
parameters instead of 860M, but the same DDPM objective this lesson's LoRA
pseudocode writes down. The adapter is the real construction: a rank-r
down-projection into every `nn.Conv2d`, plus a zero-initialised 1x1
up-projection, which is what kohya and `peft` inject into SD's attention layers.
The subject is 12 circles from the reference generator repainted one colour, held
against 12 more of the same subject the adapter never sees.

Two things the exercise asks for are not here, and both are deliberate. It asks
to "generate novel scenes", but the honest readout of a 37k-parameter stand-in is
not a picture: identity is scored instead as the improvement in the lesson's own
eps-MSE on the 12 held-out images of the subject, which is generation's own
training objective and cannot be inflated by memorising the inputs. And the
sweep runs one base-model seed rather than two -- D14's line ceiling paid for the
grid instead -- so the replication that backs the monotonicity claims is the four
ranks, each an independent adapter initialisation, not a second foundation model.

The premise that breaks is "without overfitting to the input images", which
points at the wrong quantity. The train-minus-held-out gap never exceeds 0.007
anywhere in the 4x3 grid, while what the adapter *forgets* about everything else
runs 5.5% to 38.6%; forgetting, not memorisation of the 12 inputs, is what sets
the budget. Score a setting as affordable when it forgets less than it gains and
the exercise finally has an answer -- rank 8 at 400 steps -- where raw identity,
monotone in both knobs, would have picked rank 8 at 1200 and lost more than it
won. `inject` freezes a model and wraps its convs; `eps_mse` inside `readout`
scores a fixed set of (image, t, noise) triples, drawn from a private generator
so that measuring never perturbs the training stream; `train_arm` is one rank x
steps sweep. Real training, scaled to about 16 seconds: 60 base epochs then 4
ranks x 1200 adapter steps. At 147 lines of code this file is over D14's
120-line target and under its ceiling.
"""

from __future__ import annotations

import copy

from harness import parity, practice

PHASE, LESSON, DIFFUSION = "04-computer-vision", "11-stable-diffusion", "10-image-generation-diffusion"

T, SIZE, EPOCHS, BATCH, SHOTS, LORA_LR = 200, 16, 60, 32, 12, 2e-3
RANKS, SNAPSHOTS, SUBJECT, SEED = (1, 2, 4, 8), (100, 400, 1200), (1.0, -0.3, -0.3), 0
GPU = "accelerate launch train_dreambooth_lora.py --rank=8 --max_train_steps=1200 (SD 1.5, batch 1)"

grid = lambda: [(r, s) for r in RANKS for s in SNAPSHOTS]                                   # noqa: E731
rat = lambda a, cell: a[cell]["forget"] / a[cell]["gain"]                                   # noqa: E731
ratio = lambda a, step: " ".join(f"{rat(a, (r, step)):.2f}" for r in RANKS)                 # noqa: E731
worst = lambda a, step: max(rat(a, (r, step)) for r in RANKS)                               # noqa: E731
cheap = lambda a, rank: max(rat(a, (rank, s)) for s in SNAPSHOTS)                           # noqa: E731
climbs = lambda a, k: all(a[(r, s)][k] < a[(r, n)][k] for r in RANKS for s, n in PAIRS)      # noqa: E731
afford = lambda a: [c for c in grid() if a[c]["forget"] < a[c]["gain"]]                     # noqa: E731
pick = lambda a: max(afford(a), key=lambda c: a[c]["gain"])                                 # noqa: E731
greedy = lambda a: max(grid(), key=lambda c: a[c]["gain"])                                  # noqa: E731
show = lambda a, key, spec=".3f": " ".join(f"r{r}@{s}:{a[(r, s)][key]:{spec}}" for r, s in grid())  # noqa: E731
span = lambda a, key: (min(a[c][key] for c in grid()), max(a[c][key] for c in grid()))      # noqa: E731
cost = lambda a, one, two: a[one]["forget"] / a[two]["forget"]                               # noqa: E731
repaint = lambda torch, imgs, hue: torch.where((imgs > -0.999).all(1, keepdim=True), hue, imgs)  # noqa: E731
PAIRS, BY_STEPS, BY_RANK = list(zip(SNAPSHOTS, SNAPSHOTS[1:])), ((1, 1200), (1, 100)), ((8, 100), (1, 100))


def inject(nn, model, rank):
    model.requires_grad_(False)
    trainable = []
    for name, conv in [(n, c) for n, c in model.named_children() if isinstance(c, nn.Conv2d)]:
        part = nn.Module()
        part.base, part.up = conv, nn.Conv2d(rank, conv.out_channels, 1, bias=False)
        part.down = nn.Conv2d(conv.in_channels, rank, conv.kernel_size, conv.stride, conv.padding, bias=False)
        nn.init.zeros_(part.up.weight)
        part.forward = lambda x, held=part: held.base(x) + held.up(held.down(x))
        setattr(model, name, part)
        trainable += [part.down.weight, part.up.weight]
    return trainable


def readout(torch, diffusion, schedule, model, sets):
    scores = []
    for seed, images in enumerate(sets[:3]):
        draw, batch = torch.Generator().manual_seed(seed), images.repeat(8, 1, 1, 1)
        stamps = torch.randint(0, T, (batch.size(0),), generator=draw)   # a private generator, so
        with torch.no_grad():                                            # measuring cannot perturb
            noise = torch.randn(batch.shape, generator=draw)             # the training draw
            noisy = diffusion.q_sample(batch, stamps, noise, schedule)
            scores.append(float(torch.nn.functional.mse_loss(model(noisy, stamps), noise)))
    return scores


def train_arm(torch, diffusion, loader_cls, dataset_cls, schedule, sets, base):
    floors, rows = readout(torch, diffusion, schedule, base, sets), {}
    for rank in RANKS:
        fitted = copy.deepcopy(base)
        torch.manual_seed(100 + rank)
        trained = inject(torch.nn, fitted, rank)
        tuner, width = torch.optim.Adam(trained, lr=LORA_LR), sum(t.numel() for t in trained)
        shots, step = loader_cls(dataset_cls(sets[0]), batch_size=4, shuffle=True), 0
        while step < max(SNAPSHOTS):
            for (batch,) in shots:
                diffusion.train_step(fitted, batch, schedule, tuner, "cpu", T=T)
                step += 1
                if step in SNAPSHOTS:
                    won = [(f - n) / f for f, n in zip(floors, readout(torch, diffusion, schedule, fitted, sets))]
                    rows[(rank, step)] = {"gain": won[1], "ovft": won[0] - won[1], "params": width,
                                          "forget": -won[2]}
    return rows


def solve():
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    diffusion = parity.load_reference(PHASE, DIFFUSION, "main")
    torch.set_num_threads(2)
    schedule = diffusion.precompute_schedule(
        diffusion.linear_beta_schedule(T=T, beta_start=1e-4, beta_end=0.04))
    data = diffusion.synthetic_circles(num=200, size=SIZE, seed=0)
    subject = repaint(torch, diffusion.synthetic_circles(2 * SHOTS, SIZE, 5), torch.tensor(SUBJECT).view(1, 3, 1, 1))
    sets = (subject[:SHOTS], subject[SHOTS:], data[:64])
    loader = DataLoader(TensorDataset(data), batch_size=BATCH, shuffle=True)
    torch.manual_seed(SEED)                         # the frozen foundation model
    base = diffusion.TinyUNet(img_channels=3, base=16)
    optimiser = torch.optim.Adam(base.parameters(), lr=1e-3)
    for _ in range(EPOCHS):
        [diffusion.train_step(base, batch, schedule, optimiser, "cpu", T=T) for (batch,) in loader]
    rows = train_arm(torch, diffusion, DataLoader, TensorDataset, schedule, sets, base)
    stamp, weights = torch.zeros(8, dtype=torch.long), sum(p.numel() for p in base.parameters())
    with torch.no_grad():                           # inject() mutates in place, so this compares the
        before = base(data[:8], stamp)              # very same model before and after the adapter
        inject(torch.nn, base, RANKS[-1])
        gap = float((base(data[:8], stamp) - before).abs().max())
    return {"rows": rows, "base_params": weights, "noop": gap}


def verify(result):
    rows = result["rows"]
    return [
        practice.Check(
            "ANSWER: rank 8 at 400 steps -- the top of the affordable frontier, not what raw identity picks",
            all([pick(rows)[0] == RANKS[-1], pick(rows)[1] == 400, len(afford(rows)) >= 8,
                 greedy(rows) == (RANKS[-1], SNAPSHOTS[-1]), greedy(rows) not in afford(rows)]),
            f"affordable = forgets less of the base distribution than it gains on unseen subject images. Raw "
            f"identity picks r{greedy(rows)[0]}@{greedy(rows)[1]} (gain {rows[greedy(rows)]['gain']:.3f}), which "
            f"is not; the frontier's argmax is r{pick(rows)[0]}@{pick(rows)[1]} at {rows[pick(rows)]['gain']:.3f}"
            f", 1 of {len(afford(rows))} affordable in {len(grid())}. Held-out gain {show(rows, 'gain')}"),
        practice.Check(
            "FINDING: every setting works, so identity on its own cannot pick one",
            span(rows, "gain")[1] > 2 * span(rows, "gain")[0] > 0.10,
            f"all {len(grid())} adapters improve eps-MSE on the 12 subject images none of them saw, by "
            f"{span(rows, 'gain')[0]:.1%} to {span(rows, 'gain')[1]:.1%}, so identity alone just picks the "
            "biggest, longest-trained adapter"),
        practice.Check(
            "MECHANISM: gain and forgetting both climb with steps, and their ratio crosses 1 only at the top",
            all([climbs(rows, "gain"), climbs(rows, "forget"), cheap(rows, RANKS[0]) < 1,
                 worst(rows, SNAPSHOTS[0]) < 0.8 < 1 < worst(rows, SNAPSHOTS[-1])]),
            f"both rise at every snapshot for all {len(RANKS)} independently initialised ranks -- forgetting "
            f"{show(rows, 'forget', '.1%')}. Forgetting over gain by rank {'/'.join(str(r) for r in RANKS)}: "
            f"{ratio(rows, SNAPSHOTS[0])} at {SNAPSHOTS[0]} steps and {ratio(rows, SNAPSHOTS[-1])} at "
            f"{SNAPSHOTS[-1]}, rank 1 peaking at {cheap(rows, 1):.2f}: the product binds, not either knob"),
        practice.Check(
            "FINDING: memorising the 12 inputs is not the constraint -- rank is cheap and steps are dear",
            all([span(rows, "ovft")[1] < 0.05, cost(rows, *BY_RANK) < cost(rows, *BY_STEPS)]),
            f"the gap the exercise names peaks at {span(rows, 'ovft')[1]:.3f} ({show(rows, 'ovft')}) against "
            f"forgetting's {span(rows, 'forget')[0]:.1%} to {span(rows, 'forget')[1]:.1%}; 8x the rank costs "
            f"{cost(rows, *BY_RANK):.2f}x it, 12x the steps {cost(rows, *BY_STEPS):.2f}x"),
        practice.Check(
            "CONTROL: the zero-initialised adapter is an exact no-op before the first step",
            all([result["noop"] == 0.0, result["base_params"] == 37395,
                 rows[(RANKS[0], SNAPSHOTS[0])]["params"] == 942]),
            f"injecting rank {RANKS[-1]} moves eps by {result['noop']:.1f} because `up` starts at zero, so every "
            f"number above is the adapter's: 942 to {rows[(RANKS[-1], 100)]['params']:,} parameters against "
            f"{result['base_params']:,} (real LoRAs are reported at 10-50 MB on SD 1.5's 860M U-Net, not measured "
            f"here). `{GPU}`, an estimated 15-25 min on a 4090"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
