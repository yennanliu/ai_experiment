"""Exercise 3 — action separates frames.

    **(Hard)** Build a minimal latent-action video model: take a dataset of
    (frame_t, action_t, frame_{t+1}) triples (any simple 2D game), train a tiny
    video DiT conditioned on action embeddings, and show that different actions
    produce different next frames.

Reading of the exercise: "show that different actions produce different next
frames" is a claim that needs a scale to be worth anything, because two
predictions always differ by something -- and here the demonstration succeeds
while the model learns nothing. Both the conditioned and the action-blind arm
converge onto the all-zero frame, whose MSE is 4 lit pixels of 256 = 0.015625,
and training ten times longer makes the action separation *smaller*, not larger.
So the exercise can be passed, and a picture of four visibly different frames
produced, by a network that has not learned the game at all. It is measured here against the two
numbers that bound it -- the separation the ground truth itself has, and the
separation an identical model with the action removed can reach, which is the
best any action-blind predictor can do. The game is a dot on a 16x16 grid under
four moves, chosen because its next frame is a deterministic function of frame
and action, so the achievable separation is exact rather than estimated. The
model is the lesson's own VideoPatch3D and DividedAttentionBlock with an action
embedding added to every token; TinyVideoDiT itself takes no conditioning
argument, so it cannot be used unmodified and that is recorded as a check.

Structure: `game` generates the (frame_t, action, frame_t+1) triples for a dot that moves
two cells per action; `world_model` builds the conditioned
network from the lesson's own parts; `fit` trains one arm; `rollout` predicts
the next frame for every action from one starting frame; `spread` is the mean
pairwise distance across those predictions.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "28-world-models-video-diffusion"

SIDE, ACTIONS, DIM, STEPS, LONG, BATCH, LR = 16, 4, 32, 300, 3000, 32, 3e-3
MOVES = ((0, -2), (0, 2), (-2, 0), (2, 0))
SAMPLES = 256

spread = lambda fr: float(sum((fr[i] - fr[j]).pow(2).sum().sqrt()               # noqa: E731
                              for i in range(len(fr)) for j in range(i + 1, len(fr))) / 6)
count = lambda module: sum(p.numel() for p in module.parameters())              # noqa: E731
signature = lambda fn: fn.__code__.co_varnames[:fn.__code__.co_argcount]        # noqa: E731


def game(torch) -> tuple:
    gen = torch.Generator().manual_seed(0)
    starts = torch.randint(2, SIDE - 4, (SAMPLES, 2), generator=gen)
    acts = torch.randint(0, ACTIONS, (SAMPLES,), generator=gen)
    now, nxt = torch.zeros(SAMPLES, 1, SIDE, SIDE), torch.zeros(SAMPLES, 1, SIDE, SIDE)
    for i in range(SAMPLES):
        row, col = int(starts[i, 0]), int(starts[i, 1])
        moved = [(row + MOVES[int(acts[i])][0]) % SIDE, (col + MOVES[int(acts[i])][1]) % SIDE]
        now[i, 0, row:row + 2, col:col + 2] = 1.0
        nxt[i, 0, moved[0]:moved[0] + 2, moved[1]:moved[1] + 2] = 1.0
    return now, acts, nxt


def world_model(torch, nn, ref, conditioned=True):
    class LatentAction(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch = ref.VideoPatch3D(in_channels=1, dim=DIM, patch_t=1, patch_h=2, patch_w=2)
            self.block, self.embed = ref.DividedAttentionBlock(DIM, heads=2), nn.Embedding(ACTIONS, DIM)
            self.out, self.conditioned = nn.Linear(DIM, 4), conditioned

        def forward(self, frame, action):
            tokens, grid = self.patch(frame.unsqueeze(2))
            if self.conditioned:
                tokens = tokens + self.embed(action).unsqueeze(1)
            tokens = self.block(tokens, grid)
            cells = self.out(tokens).view(-1, grid[1], grid[2], 2, 2)
            return cells.permute(0, 3, 1, 4, 2).reshape(-1, 1, SIDE, SIDE)

    return LatentAction()


def fit(torch, nn, ref, data, conditioned, steps=STEPS) -> tuple:
    torch.manual_seed(1)
    model = world_model(torch, nn, ref, conditioned)
    optimiser = torch.optim.Adam(model.parameters(), lr=LR)
    now, acts, nxt = data
    gen, losses = torch.Generator().manual_seed(0), []
    for _ in range(steps):
        pick = torch.randint(0, SAMPLES, (BATCH,), generator=gen)
        loss = ((model(now[pick], acts[pick]) - nxt[pick]) ** 2).mean()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        losses.append(float(loss.detach()))
    return model, losses


def rollout(torch, model, frame) -> list:
    with torch.no_grad():
        return [model(frame, torch.tensor([a])) for a in range(ACTIONS)]


def solve():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    data = game(torch)
    now, acts, nxt = data
    model, losses = fit(torch, nn, ref, data, conditioned=True)
    blind, blind_losses = fit(torch, nn, ref, data, conditioned=False)
    longer, long_losses = fit(torch, nn, ref, data, conditioned=True, steps=LONG)
    frame = now[:1]
    first = [int((acts == a).nonzero()[0, 0]) for a in range(ACTIONS)]
    truth = [nxt[i:i + 1] for i in first]
    zeroed = world_model(torch, nn, ref, True)
    zeroed.load_state_dict(model.state_dict())
    with torch.no_grad():
        zeroed.embed.weight.zero_()
        grid = model.patch(frame.unsqueeze(2))[1]
    return {"loss": (losses[0], losses[-1]), "blind_loss": (blind_losses[0], blind_losses[-1]),
            "long_loss": long_losses[-1], "long_spread": spread(rollout(torch, longer, frame)),
            "empty": float((nxt ** 2).mean()), "grid": tuple(int(g) for g in grid),
            "spread": spread(rollout(torch, model, frame)), "params": count(model),
            "blind_spread": spread(rollout(torch, blind, frame)), "embed_params": count(model.embed),
            "zeroed_spread": spread(rollout(torch, zeroed, frame)), "truth_spread": spread(truth),
            "forward_args": list(signature(ref.TinyVideoDiT.forward))}


def verify(result):
    spread_, blind_, truth_ = result["spread"], result["blind_spread"], result["truth_spread"]
    zeroed_, loss, empty = result["zeroed_spread"], result["loss"], result["empty"]
    return [
        practice.Check(
            "ANSWER: the four actions produce different next frames, at 17% of the real separation",
            spread_ > 0.1 * truth_ and blind_ == 0.0 and loss[-1] < 0.2 * loss[0],
            f"loss {loss[0]:.4f} -> {loss[-1]:.4f} over {STEPS} steps; the four predictions from one "
            f"frame sit {spread_:.3f} apart against a ground truth of {truth_:.3f} "
            f"({spread_ / truth_:.0%}), and exactly {blind_} apart with the action removed"),
        practice.Check(
            "FINDING: and it means nothing -- both arms end on the empty-frame predictor",
            abs(loss[-1] - empty) < 0.001 and abs(result["blind_loss"][-1] - empty) < 0.001
            and result["long_spread"] < spread_,
            f"the dot is four lit pixels of {SIDE * SIDE}, so an all-zero frame scores MSE "
            f"{empty:.5f}; the conditioned arm ends at {loss[-1]:.5f} and the blind arm "
            f"{result['blind_loss'][-1]:.5f}, both within 0.0003 of it. Nor is it undertraining: at "
            f"{LONG} steps the loss is {result['long_loss']:.5f} and the separation *falls* to "
            f"{result['long_spread']:.3f} -- the visible difference is largest when it knows least"),
        practice.Check(
            "MECHANISM: 128 numbers carry the whole demonstration",
            zeroed_ < 0.02 * spread_,
            f"the action is an additive bias on every token, so zeroing the trained "
            f"{result['embed_params']}-parameter embedding -- nothing else of {result['params']:,} "
            f"touched -- drops the spread {spread_:.3f} -> {zeroed_:.2e}: those "
            f"{result['embed_params']} numbers are the demonstration, not learned dynamics"),
        practice.Check(
            "MECHANISM: on a single frame the video block's time attention mixes nothing",
            result["grid"][0] == 1,
            f"patch_t=1 over one frame gives grid {result['grid']}, so the time attention runs on "
            f"sequences of length 1: {result['grid'][1] * result['grid'][2]} self-pairs against "
            f"space's {(result['grid'][1] * result['grid'][2]) ** 2:,}. Half the block cannot move "
            "information -- a video architecture answering a question posed with one image"),
        practice.Check(
            "CONTROL: TinyVideoDiT cannot be conditioned without editing it",
            result["forward_args"] == ["self", "x"],
            f"TinyVideoDiT.forward is declared {result['forward_args']} -- no action argument -- so "
            "'conditioned on action embeddings' cannot be done with it as shipped. This model is "
            "assembled from the lesson's own VideoPatch3D and DividedAttentionBlock instead"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
