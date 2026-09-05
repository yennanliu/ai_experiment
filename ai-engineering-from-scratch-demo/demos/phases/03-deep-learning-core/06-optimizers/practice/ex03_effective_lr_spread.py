"""Exercise 3 — the effective learning rate is a distribution, not a number.

    Track the effective learning rate for each parameter during Adam training.
    The effective rate is lr * m_hat / (sqrt(v_hat) + eps). Plot the distribution
    of effective rates after 10, 50, and 200 steps. Are all parameters being
    updated at the same speed?

Reading of the exercise: a "step" is one sample, so 10/50/200 all land inside the first
epoch of the lesson's online loop, and min / median / max stands in for the plot. Rates
come off the optimiser's own m, v and t (check 5 proves that is the parameter delta);
check 3 replaces the picture with the ratio's two exact endpoints.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "06-optimizers"
LR, SNAPS = 0.001, (10, 50, 200)


def effective(opt, count):
    """The exercise's formula, read off the lesson's Adam after `step` ran."""
    return [LR * (opt.m[i] / (1 - opt.beta1 ** opt.t))
            / (math.sqrt(opt.v[i] / (1 - opt.beta2 ** opt.t)) + opt.epsilon)
            for i in range(count)]


def summarise(rates, grads, moved):
    scaled = sorted(abs(rate) / LR for rate in rates)
    dead = [i for i, grad in enumerate(grads) if grad == 0.0]
    return {"lo": scaled[0], "mid": scaled[len(scaled) // 2], "hi": scaled[-1],
            "spread": scaled[-1] / scaled[0], "dead": len(dead),
            "walking": sum(1 for i in dead if abs(moved[i]) > 0.1 * LR)}


def track(ref, epochs=20):
    """The lesson's loop, reading the per-parameter update out of every step."""
    data, opt = ref.make_circle_data(), ref.Adam(lr=LR)
    net = ref.OptimizerTestNetwork(opt, hidden_size=8)
    step, peak, drift, snaps = 0, 0.0, 0.0, {}
    for _epoch in range(epochs):
        for point, label in data:
            net.forward(point)
            grads, params = net.compute_grads(label), net.get_params()
            before = list(params)
            opt.step(params, grads)
            net.set_params(params)
            step += 1
            moved = [before[i] - params[i] for i in range(len(params))]
            rates = effective(opt, len(params))
            drift = max(drift, max(abs(moved[i] - rates[i]) for i in range(len(params))))
            peak = max(peak, max(abs(rate) for rate in rates) / LR)
            if step in SNAPS:
                snaps[step] = summarise(rates, grads, moved)
    return {"snaps": snaps, "peak": peak, "drift": drift, "steps": step}


def probe(ref, grads):
    """One parameter fed a chosen gradient sequence; its rate in units of lr."""
    opt, param, trail = ref.Adam(lr=LR), [0.0], []
    for grad in grads:
        before = param[0]
        opt.step(param, [grad])
        trail.append((before - param[0]) / LR)
    return trail


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    ad = ref.Adam(lr=LR)
    tail = probe(ref, [1.0] * 50 + [0.0] * 350)[50:]
    return dict(track(ref), steady=probe(ref, [2.5] * 10), decay=tail[100] / tail[99],
                flip=abs(probe(ref, [1.0, -1.0] * 10)[-1]), eps_bias=ad.epsilon / 2.5,
                ratio=(1 - ad.beta1) / (1 + ad.beta1), limit=ad.beta1 / math.sqrt(ad.beta2),
                quiet=next(i for i, rate in enumerate(tail) if rate < 0.01),
                ceiling=(1 - ad.beta1) / math.sqrt(
                    (1 - ad.beta2) * (1 - ad.beta1 ** 2 / ad.beta2)))


def verify(result):
    snaps, flip = result["snaps"], result["flip"]
    shape = "; ".join(f"{s}: {v['lo']:.4f}/{v['mid']:.4f}/{v['hi']:.4f} ({v['spread']:.1f}x,"
                      f" {v['dead']} dead, {v['walking']} moving)"
                      for s, v in sorted(snaps.items()))
    off = max(abs(rate - 1.0) for rate in result["steady"])
    return [
        practice.Check("ANSWER: no — the fastest parameter outruns the slowest by 20x or more",
                       all(v["spread"] > 20 for v in snaps.values()),
                       f"|effective rate| / lr as min/median/max over the 33 parameters, step "
                       f"{shape}. The median also drops "
                       f"{snaps[10]['mid'] / snaps[200]['mid']:.1f}x from step 10 to step 200"),
        practice.Check("FINDING: most parameters have gradient exactly 0 and are updated anyway",
                       snaps[50]["dead"] == 24 and snaps[50]["walking"] == 18
                       and all(v["walking"] > 0 for v in snaps.values()),
                       "24 of 33 gradients are exactly zero at step 50 and 18 of those still "
                       "move by over 0.1 lr (counts above). ReLU sends a hard 0 back through a "
                       "unit with z <= 0; Adam's numerator is m, not today's gradient"),
        practice.Check("MECHANISM: the ratio measures agreement, not gradient size",
                       off < 2 * result["eps_bias"] and abs(flip - result["ratio"]) < 1e-9,
                       f"a constant gradient gives m_hat/sqrt(v_hat) = 1 exactly, so its rate is "
                       f"lr for 10 steps running (worst gap {off:.1e}, the epsilon term). A "
                       f"sign-flipping gradient of the same size gives (1-beta1)/(1+beta1) = "
                       f"1/19 = {result['ratio']:.6f} (measured {flip:.6f}); once dead, a rate "
                       f"decays {result['decay']:.4f} per step towards beta1/sqrt(beta2) = "
                       f"{result['limit']:.4f}, {result['quiet']} steps to reach 0.01 lr"),
        practice.Check("FINDING: lr is not a ceiling on Adam's step",
                       1.0 < result["peak"] < result["ceiling"],
                       f"largest |effective rate| over {result['steps']} steps is "
                       f"{result['peak']:.4f} lr; the Cauchy-Schwarz bound on the two averages, "
                       f"(1-beta1)/sqrt((1-beta2)(1-beta1^2/beta2)), is "
                       f"{result['ceiling']:.2f} lr"),
        practice.Check("CONTROL: the tracked quantity is the update, not a parallel model",
                       result["drift"] < 1e-15,
                       f"the formula recomputed from the optimiser's own m, v and t matches the "
                       f"parameter delta to {result['drift']:.1e} over {result['steps']} steps"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
