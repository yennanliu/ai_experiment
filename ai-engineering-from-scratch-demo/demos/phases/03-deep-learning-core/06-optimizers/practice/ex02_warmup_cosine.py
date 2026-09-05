"""Exercise 2 — linear warmup then cosine decay, against constant-lr Adam.

    Implement a learning rate warmup schedule: linear ramp from 0 to max_lr over
    the first 10% of training steps, then cosine decay to 0. Train with Adam +
    warmup vs Adam without warmup. Measure how many epochs it takes to reach 90%
    accuracy on the circle dataset.

Reading of the exercise: "training steps" is per sample, not per epoch — the lesson
trains online, so 300 epochs over 200 points is 60,000 steps and the ramp covers
6,000 of them. Check 1 is the literal measurement. Checks 2-4 ask whether the answer
survives contact: the schedule loses at every max_lr tried, and the control separates
"the shape is wrong here" from "the schedule simply spends less learning rate".
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "06-optimizers"
EPOCHS, MAX_LR, WARM = 300, 0.001, 0.10


def schedule(step, total, max_lr):
    """Linear 0 -> max_lr over the first 10% of steps, then cosine max_lr -> 0."""
    ramp = int(total * WARM)
    if step < ramp:
        return max_lr * step / ramp
    return max_lr * 0.5 * (1 + math.cos(math.pi * (step - ramp) / (total - ramp)))


def train(ref, data, lr, scheduled, epochs=EPOCHS):
    """The lesson's loop with the lesson's Adam; `opt.lr` is the only thing rewritten."""
    opt = ref.Adam(lr=lr)
    net = ref.OptimizerTestNetwork(opt, hidden_size=8)
    history, step, total = [], 0, epochs * len(data)
    for _epoch in range(epochs):
        loss, correct = 0.0, 0
        for point, label in data:
            opt.lr = schedule(step, total, lr) if scheduled else lr
            pred = net.forward(point)
            params = net.get_params()
            opt.step(params, net.compute_grads(label))
            net.set_params(params)
            p = max(1e-15, min(1 - 1e-15, pred))
            loss -= label * math.log(p) + (1 - label) * math.log(1 - p)
            correct += (pred >= 0.5) == (label >= 0.5)
            step += 1
        history.append((loss / len(data), 100.0 * correct / len(data)))
    return history


def reached(history, threshold=90.0):
    return next((i for i, (_l, acc) in enumerate(history) if acc >= threshold), None)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    data = ref.make_circle_data()
    total = EPOCHS * len(data)
    sweep = {lr: (train(ref, data, lr, False), train(ref, data, lr, True))
             for lr in (MAX_LR, 0.003, 0.01)}
    mean = sum(schedule(s, total, MAX_LR) for s in range(total)) / total
    return {"sweep": sweep, "mean": mean, "total": total,
            "half": train(ref, data, mean, False)}


def digest(result) -> tuple:
    """The per-arm summaries `verify` compares, so that stays a list of comparisons."""
    sweep = result["sweep"]
    named = [("plain", sweep[MAX_LR][0]), ("warm", sweep[MAX_LR][1]), ("half", result["half"])]
    return ({n: f"{h[-1][1]:.1f}%" for n, h in named},
            [(lr, reached(a), reached(b)) for lr, (a, b) in sorted(sweep.items())],
            {lr: (reached(sweep[lr][1], 100.0), reached(sweep[lr][0], 100.0)) for lr in sweep})


def verify(result):
    sweep, mean = result["sweep"], result["mean"]
    plain, warm, half = *sweep[MAX_LR], result["half"]
    tail, late, fast = digest(result)
    return [
        practice.Check("ANSWER: warmup + cosine needs 54 epochs to 90%, constant Adam 39",
                       reached(warm) == 54 and reached(plain) == 39,
                       f"at the lesson's default max_lr = {MAX_LR}, 300 epochs = "
                       f"{result['total']} steps and a {int(result['total'] * WARM)}-step "
                       f"ramp: 90% accuracy at epoch {reached(warm)} scheduled against "
                       f"{reached(plain)} constant, ending {tail['warm']} against {tail['plain']}"),
        practice.Check("FINDING: the schedule is slower at every max_lr tried, by 15-11 epochs",
                       all(w > p for _lr, p, w in late),
                       "epochs to 90%, constant then scheduled — " + ", ".join(
                           f"max_lr {lr}: {p} vs {w}" for lr, p, w in late) +
                       ". Warmup is insurance against early instability, and a 33-parameter "
                       "net trained one sample at a time has none to insure against"),
        practice.Check("MECHANISM: the schedule spends exactly half the learning-rate budget",
                       abs(mean - MAX_LR / 2) < 1e-6,
                       f"mean lr over the run is {mean:.6f}, exactly max_lr/2, so the run "
                       f"integrates to {mean * result['total']:.1f} lr-steps against "
                       f"{MAX_LR * result['total']:.1f} constant. Both halves cost: the ramp "
                       f"averages max_lr/2 and so does the cosine tail"),
        practice.Check("CONTROL: halving alone does not explain the delay — the shape helps",
                       reached(half) == 76 and reached(half) > reached(warm),
                       f"constant lr = {mean:.5f}, the schedule's own mean, reaches 90% at "
                       f"epoch {reached(half)} — 22 epochs *later* than the schedule's "
                       f"{reached(warm)}. Its endpoint {tail['half']} matches the schedule's "
                       f"{tail['warm']}, so the shape buys speed while the budget sets the finish"),
        practice.Check("CONTROL: the decay half does earn its keep once max_lr is aggressive",
                       fast[0.01][0] is not None and fast[0.01][1] is None,
                       f"at max_lr = 0.01 the scheduled run reaches 100% training accuracy at "
                       f"epoch {fast[0.01][0]} and the constant run never does (best "
                       f"{max(a for _p, a in sweep[0.01][0]):.1f}%). Annealing to 0 is what the "
                       f"cosine tail is for; the ramp is the part this problem does not need"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
