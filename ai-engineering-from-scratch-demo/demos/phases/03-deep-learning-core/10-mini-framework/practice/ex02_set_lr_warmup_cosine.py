"""Exercise 2 — set_lr(), warmup + cosine, and the state a rebuilt optimizer throws away.

    Implement learning rate scheduling in the optimizer: add a `set_lr()` method and wire in
    the cosine schedule from Lesson 09. Train the circle classifier with warmup + cosine and
    compare to constant LR.

Reading of the exercise: `set_lr` is one assignment, so the question worth asking is what it buys
and what the obvious alternative costs. Check 1 compares warmup+cosine to constant LR over three
seeds, check 2 reshuffles that multiset to separate the schedule's shape from its mean, check 3 is
the parity control on the plumbing, and check 4 changes the LR by rebuilding the optimizer instead.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON, SCHED = "03-deep-learning-core", "10-mini-framework", "09-learning-rate-schedules"
LR, EPOCHS, N, SEEDS = 0.01, 20, 300, (1, 2, 3)
TRAIN, STEPS, WARM = int(0.8 * N), EPOCHS * int(0.8 * N), int(0.04 * EPOCHS * N)


def set_lr(self, lr):
    self.lr = lr                   # `Adam.step` reads `self.lr` fresh, so this is the whole method


def flat(model, slot=0):        # every scalar parameter (slot 0) or its gradient (slot 3)
    return [e[slot][e[1]][e[2]] if e[2] is not None else e[slot][e[1]] for e in model.parameters()]


def probe(model, before, lr):
    live = [(abs(a - b) / lr, abs(g)) for a, b, g in zip(flat(model), before, flat(model, 3))
            if abs(g) > 1e-5]
    return {"lo": min(q for q, _ in live), "hi": max(q for q, _ in live), "live": len(live),
            "total": len(before), "gmin": min(g for _, g in live), "gmax": max(g for _, g in live)}


def retune(ref, opt, model, lrs, step, rebuild):
    opt = ref.Adam(model.parameters(), lr=LR) if rebuild else opt  # the other way to change the LR
    return opt if lrs is None else (opt.set_lr(lrs[step]), opt)[1]


def fit(ref, lrs, seed, rebuild=False):
    data = ref.make_circle_data(N, seed=7)
    random.seed(seed)              # the reference Linear draws its init from the global RNG
    model = ref.Sequential(ref.Linear(2, 16), ref.ReLU(), ref.Linear(16, 8), ref.ReLU(),
                           ref.Linear(8, 1), ref.Sigmoid())
    crit, opt = ref.BCELoss(), ref.Adam(model.parameters(), lr=LR)
    loader, step, unit, run = ref.DataLoader(data[:TRAIN], 16, True), 0, {}, 0.0
    for _ in range(EPOCHS):
        run = 0.0
        for inputs, targets in loader:
            for x, t in zip(inputs, targets):
                opt = retune(ref, opt, model, lrs, step, rebuild)
                before = flat(model) if step == 0 else None
                run += crit(model.forward(x), t)
                opt.zero_grad(), model.backward(crit.backward()), opt.step()
                unit = probe(model, before, opt.lr) if step == 0 and opt.lr else unit
                step += 1
    hits = sum((model.forward(x)[0] >= 0.5) == (t[0] == 1.0) for x, t in data[TRAIN:])
    return {"loss": run / TRAIN, "acc": 100.0 * hits / (N - TRAIN), "t": opt.t, "unit": unit}


fmt = lambda values, spec=".4f": " / ".join(format(v, spec) for v in values)   # noqa: E731


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    ref.Adam.set_lr = set_lr                                  # the method the exercise asks for
    sch = parity.load_reference(PHASE, SCHED, "main")
    cos = [sch.warmup_cosine_schedule(s, lr=LR, total_steps=STEPS, warmup_steps=WARM, lr_min=1e-5)
           for s in range(STEPS)]
    const, perm = [LR] * STEPS, list(cos)
    random.Random(9).shuffle(perm)
    out = {"mean_lr": sum(cos) / STEPS, "held": N - TRAIN, "steps": STEPS, "warm": WARM,
           "plain": fit(ref, None, SEEDS[0]), "rebuild": fit(ref, const, SEEDS[0], rebuild=True)}
    for name, lrs in (("const", const), ("cosine", cos), ("shuffled", perm)):
        runs = [fit(ref, lrs, s) for s in SEEDS]
        out[name], out[name + "_acc"] = [v["loss"] for v in runs], [v["acc"] for v in runs]
    return out


def verify(r):
    u, gap = r["rebuild"]["unit"], [abs(a - b) for a, b in zip(r["cosine_acc"], r["const_acc"])]
    return [
        practice.Check("ANSWER: warmup + cosine buys optimization, not generalization",
                       all(c < k for c, k in zip(r["cosine"], r["const"])) and max(gap) <= 2.0,
                       f"final mean training loss over seeds {SEEDS} ({EPOCHS} epochs of Adam, circle "
                       f"classifier): constant lr={LR} {fmt(r['const'])} vs warmup({r['warm']} steps)"
                       f"+cosine {fmt(r['cosine'])} — lower on every seed; on the {r['held']} held-out "
                       f"rows {fmt(r['const_acc'], '.1f')} vs {fmt(r['cosine_acc'], '.1f')}, within "
                       f"{max(gap):.1f} points"),
        practice.Check("CONTROL: the ordering of the learning rates does the work, not their mean",
                       all(s > c for s, c in zip(r["shuffled"], r["cosine"])),
                       f"the same {r['steps']} rates shuffled — identical multiset, identical mean "
                       f"{r['mean_lr']:.6f} — give {fmt(r['shuffled'])}, landing with constant LR "
                       f"{fmt(r['const'])}, not cosine {fmt(r['cosine'])}. MECHANISM: only a monotone "
                       f"tail spends the END of training at a small LR"),
        practice.Check("CONTROL: the set_lr plumbing itself changes nothing",
                       r["const"][0] == r["plain"]["loss"],
                       f"set_lr on a constant schedule reproduces the reference loop that never touches "
                       f"lr, bit for bit: {r['const'][0]!r} vs {r['plain']['loss']!r}, {r['steps']} steps"),
        practice.Check("FINDING: changing the LR by rebuilding Adam silently turns it into signSGD",
                       r["rebuild"]["t"] == 1 and r["rebuild"]["loss"] > 10 * r["const"][0],
                       f"a fresh Adam per step resets `t`, pinning bias correction at t=1 where "
                       f"m_hat/(sqrt(v_hat)+eps) = g/(|g|+eps): there all {u['live']} of {u['total']} "
                       f"parameters with |g| > 1e-5 moved by {u['lo']:.6f}-{u['hi']:.6f} x lr although their "
                       f"gradients span {u['gmin']:.2e} to {u['gmax']:.2e}. After {r['steps']} "
                       f"steps it still reports t={r['rebuild']['t']}, at loss {r['rebuild']['loss']:.4f}"
                       f" / {r['rebuild']['acc']:.1f}% against {r['const'][0]:.4f} / "
                       f"{r['const_acc'][0]:.1f}% for set_lr"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
