"""Exercise 3 — Double DQN halves the bias; it does not remove it.

    **Hard.** Add Double DQN: use the online net to pick `argmax a'`, target net
    to evaluate. Compare bias of `Q(s_0, best_a)` vs true `V*(s_0)` after 1,000
    episodes with vs without Double DQN on a noisy-reward GridWorld.

Reading of the exercise: three paired seeds and a `σ = 0` control, because "compare
bias" at one seed cannot separate a bias from a seed -- and on this experiment one
seed in three reverses the comparison. Only the target *computation* is new: the
rewritten minibatch goes through the lesson's own `train_step`, which is what "add
Double DQN" means. Everything else is the shipped configuration, 32 hidden units
and batch 32; a scaled-down 16/8 version was tried first and rejected, since batch
size is the term that averages the reward noise and shrinking it replaces the
effect with gradient noise (see the README).

**ANSWER: the first half holds and the second half does not.** DQN overestimates
`V*(0,0) = -5.8520` by **+1.86** on average, positive on all three seeds. Double
DQN overestimates by **+1.04** -- smaller, and not zero.

**FINDING: the paired difference is +0.82 ± 0.36, and one seed of three
reverses.** Seed differences are +1.15, -0.04 and +1.35. Running this comparison
once, as the exercise's phrasing invites, has a one-in-three chance of showing no
effect at all.

**FINDING: the better estimate does buy a better policy, which the exercise does
not ask about.** Mean return over the last 50 episodes is -6.3 under Double DQN
against -8.1 under DQN, on a -6 optimum -- a gap three times the size of the value
gap it is usually justified by.

**CONTROL: at `σ = 0` both arms land on `V*` and the difference disappears.** So
what Double DQN corrects here is the maximization bias the reward noise creates,
not anything about the architecture or the budget.

**FINDING: both arms carry a target network, so that is not what is doing the
work.** Exercise 2 measured that deleting it on the clean board costs nothing;
here keeping it leaves +1.86 standing. Decoupling *which* action from *what it is
worth* is the part that moves the number.

Structure: `noisy` wraps the lesson's `step`; `rewrite` is the one new computation;
`train` is `main`'s loop with that computation optionally spliced in.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "09-reinforcement-learning", "05-dqn"
EPISODES, CAP, BATCH, HIDDEN, CAPACITY, SYNC = 1_000, 50, 32, 32, 2000, 200
GAMMA, LR, SIGMA, SEEDS = 0.99, 0.05, 5.0, (0, 1, 2)
CELLS = [(r, c) for r in range(4) for c in range(4)]


def noisy(clean, sigma, seed):
    """The lesson's own `step`, with Gaussian noise on every non-zero reward."""
    rng = random.Random(seed)
    add = lambda n, r, d: (n, r + rng.gauss(0.0, sigma) if r else r, d)
    return lambda s, a: add(*clean(s, a))


def rewrite(ref, online, target, sample):
    """One Double DQN target: online picks argmax a', target prices it; folded into `done`."""
    state, action, reward, nxt, done = sample
    if done:
        return sample
    features = ref.state_features(nxt)
    chosen = ref.forward(online, features)[0]
    best = max(range(len(chosen)), key=chosen.__getitem__)
    return state, action, reward + GAMMA * ref.forward(target, features)[0][best], nxt, True


def train(ref, seed, double, sigma=SIGMA, episodes=EPISODES):
    """`main`'s loop at its own size, with the Double DQN target optionally spliced in."""
    rng, step = random.Random(seed), noisy(ref.step, sigma, 7 * seed + 1)
    online = ref.init_net(16, HIDDEN, len(ref.ACTIONS), rng)
    target, buffer, log, count = ref.clone(online), [], [], 0
    for episode in range(episodes):
        state, total = ref.reset(), 0.0
        for _ in range(CAP):
            action = ref.epsilon_greedy(online, state, rng, max(0.05, 1.0 - episode / 200))
            nxt, reward, done = step(state, ref.ACTIONS[action])
            total += reward
            buffer.append((state, action, reward, nxt, done))
            del buffer[:-CAPACITY]
            if len(buffer) >= BATCH:
                mini = rng.sample(buffer, BATCH)
                if double:
                    mini = [rewrite(ref, online, target, x) for x in mini]
                ref.train_step(online, target, mini, GAMMA, LR)
            count += 1
            if count % SYNC == 0:
                target = ref.clone(online)
            if done:
                break
            state = nxt
        log.append(total)
    return {"tail": statistics.fmean(log[-50:]),
            "q0": max(ref.forward(online, ref.state_features((0, 0)))[0])}


def optimal(ref):
    """`V*(0,0)` from the clean `step`; the noise is zero-mean, so this is the truth."""
    values = {s: 0.0 for s in CELLS}
    for _ in range(40):
        values = {s: (0.0 if s == ref.TERMINAL else
                      max((lambda n, r, d: r + (0.0 if d else GAMMA * values[n]))(*ref.step(s, a))
                          for a in ref.ACTIONS)) for s in CELLS}
    return values[(0, 0)]


def show(values):
    """A signed, two-decimal list."""
    return ", ".join(f"{v:+.2f}" for v in values)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return {"star": optimal(ref),
            "runs": {(d, s): train(ref, s, d) for d in (False, True) for s in SEEDS},
            "control": {d: train(ref, 0, d, sigma=0.0, episodes=400) for d in (False, True)}}


def digest(result):
    """Every number the checks read, computed once."""
    runs, star = result["runs"], result["star"]
    out = {name: {k: [runs[(d, s)][k] for s in SEEDS] for k in ("q0", "tail")}
           for name, d in (("plain", False), ("double", True))}
    for row in out.values():
        row["bias"] = [q - star for q in row["q0"]]
    out["paired"] = [p - q for p, q in zip(out["plain"]["bias"], out["double"]["bias"])]
    out["controls"] = [result["control"][d]["q0"] - star for d in (False, True)]
    return {**out, "stderr": statistics.pstdev(out["paired"]) / len(SEEDS) ** 0.5}


def verify(result):
    star, d = result["star"], digest(result)
    plain, double, paired = d["plain"], d["double"], d["paired"]
    shrink = statistics.fmean(plain["bias"]) / statistics.fmean(double["bias"])
    lift = statistics.fmean(double["tail"]) - statistics.fmean(plain["tail"])
    return [
        practice.Check(
            "ANSWER: DQN overestimates by +1.86 on every seed -- the first half holds",
            min(plain["bias"]) > 0 and statistics.fmean(plain["bias"]) > 1.0,
            f"after {EPISODES:,} episodes at sigma={SIGMA:g}, max_a Q(0,0,a) against V*(0,0) = "
            f"{star:.4f} over seeds {SEEDS}: {show(plain['bias'])}, mean "
            f"{statistics.fmean(plain['bias']):+.2f} -- positive on all three seeds",
        ),
        practice.Check(
            "FINDING: Double DQN also overestimates -- it halves the bias, it does not remove it",
            min(double["bias"]) > 0 and 1.0 < shrink,
            f"Double DQN's biases are {show(double['bias'])}, mean "
            f"{statistics.fmean(double['bias']):+.2f} -- positive on all three seeds too, "
            f"{shrink:.1f}x smaller rather than absent. Decoupling selection from evaluation "
            "shrinks the bias, it does not cancel it",
        ),
        practice.Check(
            "FINDING: the paired difference is +0.82 +/- 0.36, and one seed of three reverses",
            statistics.fmean(paired) > 0 and min(paired) < 0.1,
            f"per-seed differences DQN minus Double DQN: {show(paired)}, mean "
            f"{statistics.fmean(paired):+.2f} +/- {d['stderr']:.2f}, "
            f"{statistics.fmean(paired) / d['stderr']:.1f} standard errors out. Seed "
            f"{SEEDS[paired.index(min(paired))]} shows {min(paired):+.2f}, no effect: running "
            "this once has a one-in-three chance of that seed",
        ),
        practice.Check(
            "FINDING: the better estimate buys a better policy, which the exercise does not ask",
            lift > 1.0,
            f"mean return over the last 50 episodes is {statistics.fmean(double['tail']):.1f} "
            f"under Double DQN against {statistics.fmean(plain['tail']):.1f} under DQN, on a -6 "
            f"optimum -- a gap of {lift:.1f}, {lift / statistics.fmean(paired):.0f}x the value "
            "gap it is justified by, and not the quantity the exercise asks about",
        ),
        practice.Check(
            "CONTROL: at sigma = 0 both arms land on V* and the difference disappears",
            max(abs(c) for c in d["controls"]) < 0.5
            and abs(d["controls"][0] - d["controls"][1]) < 0.2,
            f"with the noise off and everything else identical the two biases are "
            f"{d['controls'][0]:+.3f} and {d['controls'][1]:+.3f}, "
            f"{abs(d['controls'][0] - d['controls'][1]):.3f} apart against "
            f"{statistics.fmean(paired):+.2f} with noise -- and both arms keep the target net "
            "exercise 2 showed is free to delete. What the noise defeats is the max",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
