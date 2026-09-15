"""Exercise 3 — the penalty turns itself off, by nine orders of magnitude.

    **Hard.** Replace the clipped surrogate with an adaptive KL penalty (`β`
    doubled if `KL > 2·target`, halved if `KL < target/2`). Compare final return,
    stability, and clip-free-ness.

Reading of the exercise: only the surrogate is replaced. The rollout, GAE, the
per-minibatch advantage normalisation and the critic update are the lesson's own,
so the two arms differ in one term: `ratio·A` with a hard clip becomes
`ratio·A − β·KL`, whose logit gradient is `(ratio·A + β)·∇log π`. `β` starts at 1.0
with `target = 0.01` and the stated doubling rule. "Clip-free-ness" is measured by
counting how many samples the `ε = 0.2` rule *would* have clipped, since the
penalty arm has no clip to count.

**ANSWER: the penalty matches the clip and then removes itself.** Final return
-6.06 against clipped PPO's -6.18 over 12 seeds, mean KL 0.0051 against 0.0069,
and 3.5% of samples would have been clipped.

**FINDING: `β` decays to 3.5e-10.** It starts at 1.0 and the halving rule fires
whenever `KL < target/2`, which is most updates: 0.50, 0.25, 0.031, 7.6e-06,
2.3e-10 at updates 1, 5, 20, 40, 60. By the end of training the penalty term is
nine orders of magnitude below where it began -- the arm is running unconstrained
policy gradient with normalised advantages.

**FINDING: which is fine, because unconstrained also works here.** Exercise 2
measured `eps = 1e9` at `K = 4` reaching -6.08. Clipped, penalised and entirely
unconstrained all land within 0.12 of each other, so "compare stability" has
nothing to separate at this `K`.

**FINDING: the comparison only becomes real where exercise 2 found the clip
working.** At `K = 30` the unconstrained arm's mean KL is 0.30. The adaptive rule
is supposed to catch exactly that -- and it can only do so if `β` has not already
been halved into irrelevance, which on this task it has.

Structure: `one_sample` is the one replaced term; everything else is called
unchanged from the lesson's module.
"""

from __future__ import annotations

import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "09-reinforcement-learning", "08-ppo"
UPDATES, SEEDS, EPS, TARGET = 60, 12, 0.2, 0.01
LR_A, LR_V, EPOCHS, BATCH, BETA0 = 0.05, 0.1, 4, 32, 1.0
MARKS = (1, 5, 20, 40, 60)


def one_sample(ref, theta, w, rec, adv, beta):
    """One penalised sample: returns (its KL, whether eps=0.2 would have clipped it)."""
    x = rec["x"]
    probs = ref.softmax(ref.logits(theta, x))
    logp = math.log(max(probs[rec["a"]], 1e-12))
    ratio = math.exp(logp - rec["log_pi_old"])
    scale = ratio * adv + beta
    for action in range(ref.N_ACTIONS):
        grad = (1.0 if action == rec["a"] else 0.0) - probs[action]
        for j in range(ref.N_FEAT):
            theta[action][j] += LR_A * scale * grad * x[j]
    error = rec["ret"] - ref.value(w, x)
    for j in range(ref.N_FEAT):
        w[j] += LR_V * error * x[j]
    return rec["log_pi_old"] - logp, (adv > 0 and ratio > 1 + EPS) or (adv < 0 and ratio < 1 - EPS)


def penalised(ref, theta, w, buffer, advantages, returns, beta, rng):
    """The lesson's update with the clipped surrogate replaced by `ratio*A - beta*KL`."""
    for rec, adv, ret in zip(buffer, advantages, returns):
        rec["adv"], rec["ret"] = adv, ret
    kl_total = count = would = 0.0
    for _ in range(EPOCHS):
        shuffled = buffer[:]
        rng.shuffle(shuffled)
        for i in range(0, len(shuffled), BATCH):
            mb = shuffled[i:i + BATCH]
            for rec, adv in zip(mb, ref.normalize([r["adv"] for r in mb])):
                kl, clipped = one_sample(ref, theta, w, rec, adv, beta)
                kl_total, would, count = kl_total + kl, would + clipped, count + 1
    return kl_total / max(1.0, count), would / max(1.0, count)


def run_penalty(ref, seed):
    """One penalised run; records beta at the marked updates."""
    rng = random.Random(seed)
    theta, w = ref.init_theta(rng), ref.init_w(rng)
    beta, kls, betas, woulds = BETA0, [], [], []
    for _ in range(UPDATES):
        buffer = ref.collect_rollout(theta, w, rng)
        advantages, returns = ref.gae(buffer)
        mean_kl, would = penalised(ref, theta, w, buffer, advantages, returns, beta, rng)
        # the rule the exercise states: double above 2*target, halve below target/2
        beta = beta * 2 if mean_kl > 2 * TARGET else (
            beta / 2 if mean_kl < TARGET / 2 else beta)
        kls.append(mean_kl)
        betas.append(beta)
        woulds.append(would)
    return {"final": ref.evaluate(theta, random.Random(999), episodes=200),
            "kl": statistics.fmean(kls), "would": statistics.fmean(woulds), "betas": betas}


def run_clip(ref, seed, eps=EPS, epochs=EPOCHS):
    """The lesson's own clipped update, unchanged."""
    rng = random.Random(seed)
    theta, w = ref.init_theta(rng), ref.init_w(rng)
    kls = []
    for _ in range(UPDATES):
        buffer = ref.collect_rollout(theta, w, rng)
        advantages, returns = ref.gae(buffer)
        kls.append(ref.ppo_update(theta, w, buffer, advantages, returns,
                                  eps=eps, epochs=epochs, rng=rng)[0])
    return {"final": ref.evaluate(theta, random.Random(999), episodes=200),
            "kl": statistics.fmean(kls)}


def average(rows, key):
    return statistics.fmean(r[key] for r in rows)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    penalty = [run_penalty(ref, s) for s in range(SEEDS)]
    return {"penalty": penalty, "clip": [run_clip(ref, s) for s in range(SEEDS)],
            "open": [run_clip(ref, s, eps=1e9) for s in range(SEEDS)],
            "open30": [run_clip(ref, s, eps=1e9, epochs=30) for s in range(3)],
            "beta": [statistics.median(r["betas"][i - 1] for r in penalty) for i in MARKS],
            "final_beta": statistics.median(r["betas"][-1] for r in penalty)}


def verify(result):
    penalty, clip, loose = result["penalty"], result["clip"], result["open"]
    betas, final_beta = result["beta"], result["final_beta"]
    arms = [average(x, "final") for x in (penalty, clip, loose)]
    spread = max(arms) - min(arms)
    return [
        practice.Check(
            "ANSWER: the penalty matches the clip -- -6.06 against -6.18",
            average(penalty, "final") >= average(clip, "final") - 0.2,
            f"over {SEEDS} seeds the penalised arm evaluates at {average(penalty, 'final'):.2f} "
            f"against the lesson's clipped {average(clip, 'final'):.2f}, mean KL "
            f"{average(penalty, 'kl'):.4f} against {average(clip, 'kl'):.4f}. It is clip-free by "
            f"construction, and {100 * average(penalty, 'would'):.1f}% of its samples would have "
            f"been clipped by the eps={EPS} rule. Only the surrogate differs: the rollout, GAE, "
            "the per-minibatch normalize and the critic update are the lesson's own, so both "
            "arms inherit the boundary-bleeding `gae` exercise 1 measured at 15.81",
        ),
        practice.Check(
            "FINDING: beta decays to 3.5e-10 -- the penalty removes itself",
            final_beta < 1e-6,
            f"beta starts at {BETA0} and the halving rule fires whenever KL < target/2 = "
            f"{TARGET / 2}, which is most updates. Median beta at updates {MARKS}: "
            + ", ".join(f"{b:.2e}" for b in betas)
            + f". By the end it is {BETA0 / final_beta:.1e}x smaller than it began: the penalty "
            "has vanished and the arm is unconstrained policy gradient with normalised "
            "advantages",
        ),
        practice.Check(
            "FINDING: which is fine, because unconstrained also works at this K",
            spread < 0.3,
            f"the lesson's own update at eps=1e9 -- no clip at all -- evaluates at "
            f"{average(loose, 'final'):.2f}, mean KL {average(loose, 'kl'):.4f}. Clipped, "
            f"penalised and unconstrained land within {spread:.2f} of one another, so 'compare "
            f"final return and stability' has nothing to separate at K={EPOCHS}",
        ),
        practice.Check(
            "FINDING: the comparison only becomes real where the clip was doing work",
            average(result["open30"], "kl") > 10 * average(clip, "kl"),
            f"exercise 2 measured the clip inert at K={EPOCHS} and load-bearing at K=30. "
            f"Unconstrained at K=30 the mean KL is {average(result['open30'], 'kl'):.4f}, "
            f"{average(result['open30'], 'kl') / average(clip, 'kl'):.0f}x the clipped K=4 "
            "figure -- the regime the adaptive rule exists for, which it can only reach if beta "
            "has not already been halved into irrelevance",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
