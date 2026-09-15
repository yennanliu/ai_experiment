"""Exercise 2 — neither, and the target net was never stale enough to matter.

    **Medium.** Disable the target network (use the online net for both sides of
    the Bellman target). Measure training instability — does return oscillate or
    diverge?

Reading of the exercise: "measure instability" is read as *two arms, two seeds
each, and three separate statistics* -- the converged mean, its spread, and the
distance of `max_a Q(0,0,a)` from the exact `V*` -- because a single paired run
cannot distinguish instability from seed noise. Disabling the target network means
passing `online` where the lesson's own `train_step` expects `target`; nothing else
changes, and the with-target arm also records how far the two nets had drifted at
each sync.

**ANSWER: neither.** Over the last 100 episodes the four runs mean -6.27, -6.34
(target) against -6.32, -6.30 (no target), with standard deviations 0.69, 0.78
against 0.89, 0.77. The arms are inside each other's seed-to-seed spread.

**FINDING: by value accuracy the no-target arm is the better of the two.**
`max_a Q(0,0,a)` lands at -5.850 and -5.851 without the target network, against
-5.873 and -5.748 with it, and `V*(0,0) = -5.8520`.

**MECHANISM: the target network was never stale.** It is refreshed every 200
steps, 21 times over the run, and the sup-norm gap between online and target just
before a refresh averages 0.64 on a `Q` scale of about 6. Removing a ~10%
perturbation of the bootstrap target does not destabilise anything.

**FINDING: the deadly triad has no bootstrap-interference to offer here.**
`state_features` is one-hot over 16 states, so each state owns its own column of
`W1`, and 676 parameters are being fitted to 64 `Q` values. The target network
exists to damp off-policy bootstrapping through a *shared* function approximator;
this one barely shares.

**FINDING: what looks like oscillation is `ε = 0.05` and nothing else.** A 6-step
optimal episode takes at least one random action 26.5% of the time, which is the
whole of the -6-to-about--10 band both arms sit in after convergence.

Structure: `train` is `main`'s own loop with the target swappable; `gap` records
the online-vs-target sup-norm at each sync.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "09-reinforcement-learning", "05-dqn"
EPISODES, CAP, BATCH, CAPACITY, SYNC = 400, 50, 32, 2000, 200
GAMMA, LR, EPSILON_FLOOR = 0.99, 0.05, 0.05
SEEDS, CELLS = (0, 1), [(r, c) for r in range(4) for c in range(4)]


def gap(ref, online, target):
    """Sup-norm between the two nets' Q over the whole board."""
    return max(abs(a - b) for state in CELLS
               for a, b in zip(ref.forward(online, ref.state_features(state))[0],
                               ref.forward(target, ref.state_features(state))[0]))


def train(ref, seed, use_target):
    """`main`'s loop; `use_target=False` passes the online net where train_step wants target."""
    rng = random.Random(seed)
    online = ref.init_net(16, 32, len(ref.ACTIONS), rng)
    target, buffer, log, drift, count = ref.clone(online), [], [], [], 0
    for episode in range(EPISODES):
        state, total = ref.reset(), 0.0
        for _ in range(CAP):
            action = ref.epsilon_greedy(online, state, rng, max(EPSILON_FLOOR, 1.0 - episode / 200))
            nxt, reward, done = ref.step(state, ref.ACTIONS[action])
            total += reward
            buffer.append((state, action, reward, nxt, done))
            del buffer[:-CAPACITY]
            if len(buffer) >= BATCH:
                ref.train_step(online, target if use_target else online,
                               rng.sample(buffer, BATCH), GAMMA, LR)
            count += 1
            if count % SYNC == 0:
                drift.append(gap(ref, online, target))
                target = ref.clone(online)
            if done:
                break
            state = nxt
        log.append(total)
    tail = log[-100:]
    return {"mean": statistics.fmean(tail), "sd": statistics.pstdev(tail), "drift": drift,
            "best": max(tail), "worst": min(tail), "log": log,
            "q0": max(ref.forward(online, ref.state_features((0, 0)))[0])}


def optimal(ref):
    """`V*(0,0)`, swept from the lesson's own `step`."""
    values = {s: 0.0 for s in CELLS}
    for _ in range(40):
        values = {s: (0.0 if s == ref.TERMINAL else
                      max((lambda n, r, d: r + (0.0 if d else GAMMA * values[n]))(*ref.step(s, a))
                          for a in ref.ACTIONS)) for s in CELLS}
    return values[(0, 0)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    net = ref.init_net(16, 32, len(ref.ACTIONS), random.Random(0))
    return {"runs": {(use, seed): train(ref, seed, use)
                     for use in (True, False) for seed in SEEDS},
            "star": optimal(ref), "slip": 1 - (1 - EPSILON_FLOOR) ** 6,
            "params": sum(len(r) for r in net["W1"]) + len(net["b1"])
            + sum(len(r) for r in net["W2"]) + len(net["b2"])}


def digest(result):
    """Every number the checks read, computed once."""
    runs, star = result["runs"], result["star"]
    out = {name: {k: [runs[(use, s)][k] for s in SEEDS]
                  for k in ("mean", "sd", "q0", "best", "worst")}
           for name, use in (("with", True), ("without", False))}
    for arm in ("with", "without"):
        out[arm]["err"] = max(abs(q - star) for q in out[arm]["q0"])
    out["drift"] = [d for s in SEEDS for d in runs[(True, s)]["drift"]]
    out["syncs"] = len(runs[(True, SEEDS[0])]["drift"])
    out["apart"] = abs(statistics.fmean(out["with"]["mean"])
                       - statistics.fmean(out["without"]["mean"]))
    out["within"] = abs(out["with"]["mean"][0] - out["with"]["mean"][1])
    out["band"] = (max(out["with"]["best"] + out["without"]["best"]),
                   min(out["with"]["worst"] + out["without"]["worst"]))
    return out


def verify(result):
    star, d = result["star"], digest(result)
    got, none = d["with"], d["without"]
    drift = statistics.fmean(d["drift"])
    return [
        practice.Check(
            "ANSWER: neither -- the arms sit inside each other's seed-to-seed spread",
            d["apart"] < max(got["sd"] + none["sd"]) / 2 and d["band"][1] > -20,
            "mean / sd of the last 100 episodes -- with target "
            + ", ".join(f"{m:.2f}/{v:.2f}" for m, v in zip(got["mean"], got["sd"]))
            + "; without target "
            + ", ".join(f"{m:.2f}/{v:.2f}" for m, v in zip(none["mean"], none["sd"]))
            + f". The arms differ by {d['apart']:.3f} while the two seeds inside one arm differ "
            f"by {d['within']:.3f}. Nothing diverged: the worst tail episode anywhere is "
            f"{d['band'][1]:.0f} against a floor of {-CAP}",
        ),
        practice.Check(
            "FINDING: by value accuracy the no-target arm is the better of the two",
            none["err"] < got["err"],
            f"max_a Q(0,0,a) against V*(0,0) = {star:.4f}: without the target network "
            + ", ".join(f"{q:.4f}" for q in none["q0"]) + "; with it "
            + ", ".join(f"{q:.4f}" for q in got["q0"])
            + f". Worst error {none['err']:.4f} against {got['err']:.4f}. The component the "
            "exercise asks you to remove was not buying accuracy either",
        ),
        practice.Check(
            "MECHANISM: the target network was never stale enough to matter",
            drift < 0.2 * abs(star),
            f"it is refreshed every {SYNC} steps, {d['syncs']} times over the run, and the "
            f"sup-norm gap between online and target just before a refresh averages {drift:.3f} "
            f"and peaks at {max(d['drift']):.3f} on a Q scale of {abs(star):.1f} -- about "
            f"{100 * drift / abs(star):.0f}%. Deleting a perturbation that size cannot "
            "destabilise a bootstrap it was only shifting by that much",
        ),
        practice.Check(
            "FINDING: the deadly triad has no shared approximator to interfere through",
            result["params"] > 8 * 64,
            f"state_features is one-hot over 16 states, so each state owns its own column of W1 "
            f"and the first layer generalises between states not at all. The net carries "
            f"{result['params']} parameters for the 64 Q values it fits, "
            f"{result['params'] / 64:.1f}x more. A target network damps off-policy bootstrapping "
            "through a shared approximator; this one barely shares",
        ),
        practice.Check(
            "FINDING: what looks like oscillation is epsilon = 0.05 and nothing else",
            0.2 < result["slip"] < 0.35,
            f"a 6-step optimal episode takes at least one random action "
            f"{100 * result['slip']:.1f}% of the time at eps={EPSILON_FLOOR}, which is the whole "
            f"band both arms occupy after convergence: best tail episode {d['band'][0]:.0f}, "
            f"worst {d['band'][1]:.0f}. That spread is the exploration the run left switched on",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
