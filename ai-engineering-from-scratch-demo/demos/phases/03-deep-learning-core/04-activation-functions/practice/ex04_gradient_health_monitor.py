"""Exercise 4 — a gradient health monitor, and the two thresholds it is given.

    Build a "gradient health monitor" that runs during training: at each epoch,
    compute the average gradient magnitude at each layer. Print a warning when any
    layer's gradient drops below 0.001 or exceeds 100.

Reading of the exercise: a graded run cannot print, so the monitor records instead and the
checks report what it would have printed. Its per-layer means are read out of the state the
lesson's own `backward` uses, and check 1 proves that by reproducing `backward`'s own weight
deltas. Checks 2-4 then ask what the two given thresholds catch on the network they are given,
and check 5 finds where the low one does fire.
"""

from __future__ import annotations

import statistics

from harness import parity, practice

PHASE, LESSON = "03-deep-learning-core", "04-activation-functions"
LOW, HIGH, HIDDEN, EPOCHS = 0.001, 100.0, 8, 200
RATES, LESSON_LR = (0.1, 1.0, 2.0, 5.0), 0.1
NAMES = ("sigmoid", "tanh", "relu", "gelu", "swish")


def acts(ref) -> dict:
    return {n: (getattr(ref, "tanh_act" if n == "tanh" else n),
                getattr(ref, f"{'tanh' if n == 'tanh' else n}_derivative")) for n in NAMES}


def probe(net, target) -> tuple:
    """|grad| per layer for one sample, read out of the state `backward` is about to use."""
    d_out = (net.out - target) * net.out * (1 - net.out)
    layer1, layer2 = [], [abs(d_out * net.h[i]) for i in range(net.hidden_size)] + [abs(d_out)]
    for i in range(net.hidden_size):
        d_h = d_out * net.w2[i] * net.act_d(net.z1[i])
        layer1 += [abs(d_h * net.x[0]), abs(d_h * net.x[1]), abs(d_h)]
    return layer1, layer2


def monitor(ref, act, deriv, data, lr, epochs=EPOCHS) -> dict:
    """The lesson's own training loop with the monitor wrapped around each epoch."""
    net = ref.ActivationNetwork(act, deriv, HIDDEN, lr)
    means, biggest, checked = [], 0.0, None
    for _epoch in range(epochs):
        seen, loss, right = ([], []), 0.0, 0
        for x, target in data:
            pred = net.forward(x)
            one = probe(net, target)
            before = list(net.w2)
            net.backward(target)
            checked = checked if checked is not None else max(
                abs(abs(b - a) / lr - g) for a, b, g in zip(before, net.w2, one[1]))
            for side, values in zip(seen, one):
                side.extend(values)
                biggest = max(biggest, max(values))
            loss, right = loss + (pred - target) ** 2, right + ((pred >= 0.5) == (target >= 0.5))
        means.append(tuple(statistics.mean(side) for side in seen))
        end = (loss / len(data), 100.0 * right / len(data))
    return {"means": means, "biggest": biggest, "delta": checked, "end": end,
            **summarise(means)}


def summarise(means) -> dict:
    """What the monitor would have printed: how often each threshold tripped."""
    return {"warned": sum(min(m) < LOW or max(m) > HIGH for m in means),
            "first": next((i for i, m in enumerate(means) if min(m) < LOW), None),
            "over": sum(max(m) > HIGH for m in means)}


def trace(ref, act, layers=10) -> list:
    """The magnitudes the lesson's own vanishing_gradient_experiment prints."""
    seen = []
    with parity.quiet():
        ref.vanishing_gradient_experiment(lambda z: seen.append(abs(act(z))) or act(z),
                                          "spy", n_layers=layers)
    return seen


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    table, data = acts(ref), ref.make_circle_data()
    runs = {(n, lr): monitor(ref, *table[n], data, lr) for n in NAMES for lr in RATES}
    return {"runs": runs, "n": len(data),
            "deep": {n: trace(ref, table[n][0]) for n in ("sigmoid", "relu", "gelu")}}


def spread(runs) -> tuple:
    """The whole range the per-layer means ever take, over every run."""
    seen = [v for r in runs.values() for m in r["means"] for v in m]
    return min(seen), max(seen)


def deepest(deep) -> dict:
    """Where the same rule fires on the lesson's own 10-layer trace."""
    return {"dead": {n: next((i + 1 for i, v in enumerate(t) if v < LOW), None)
                     for n, t in deep.items()},
            "tail": ", ".join(f"{n} {min(t[-3:]):.6f}" for n, t in deep.items())}


def digest(result) -> dict:
    """Every summary `verify` quotes, so that stays a list of comparisons."""
    runs = result["runs"]
    base = [runs[(n, LESSON_LR)] for n in NAMES]
    return {"quiet": sum(r["warned"] for r in base), "checks": len(base) * EPOCHS * 2,
            "over": sum(r["over"] for r in runs.values()), "span": spread(runs),
            "biggest": max(r["biggest"] for r in runs.values()),
            "delta": max(r["delta"] for r in runs.values()), **deepest(result["deep"]),
            "fired": [n for n in NAMES if runs[(n, 1.0)]["first"] is not None],
            "succeeded": ", ".join(f"{n} at epoch {runs[(n, 1.0)]['first']} ending "
                                   f"{runs[(n, 1.0)]['end'][1]:.1f}%"
                                   for n in NAMES if runs[(n, 1.0)]["first"] is not None)}


def verify(result):
    d, runs = digest(result), result["runs"]
    return [
        practice.Check("the monitor reproduces the lesson's own update, so its numbers are "
                       "`backward`'s",
                       d["delta"] < 1e-12,
                       f"worst gap between the recorded output-layer gradient and the weight "
                       f"delta `backward` actually applied, over every run: {d['delta']:.2e}"),
        practice.Check("ANSWER: on the settings the lesson runs, the monitor never warns",
                       d["quiet"] == 0,
                       f"{len(NAMES)} activations x {EPOCHS} epochs x 2 layers at the lesson's own "
                       f"lr = {LESSON_LR} on its own {result['n']}-point circle data: "
                       f"{d['quiet']} of {d['checks']} layer-epoch readings fall outside "
                       f"[{LOW}, {HIGH:.0f}]"),
        practice.Check("MECHANISM: the upper threshold cannot fire on this network at all",
                       d["over"] == 0 and d["biggest"] < 5.0,
                       f"the largest single gradient over {len(NAMES)} activations x "
                       f"{len(RATES)} learning rates is {d['biggest']:.4f}, {HIGH / d['biggest']:.0f}x "
                       f"below the alarm. |d_out| = |p - t| * p(1-p) <= 0.25, so the output-layer "
                       f"gradient is at most 0.25|h| and the hidden one at most 0.25|w2| * act' * "
                       f"|x| — reaching {HIGH:.0f} needs |h| > 400 or |w2| > 200, which a sigmoid "
                       f"output cannot ask for"),
        practice.Check("FINDING: wherever the low threshold fires, training had *succeeded*",
                       d["fired"] and all(runs[(n, 1.0)]["end"][1] > 95 for n in d["fired"]),
                       f"at lr = 1.0 it fires for {d['succeeded']}. The per-layer mean falls below "
                       f"{LOW} because the loss is near zero, so on this network 'gradient too "
                       f"small' is a convergence signal, not an alarm — the whole live range is "
                       f"{d['span'][0]:.1e} to {d['span'][1]:.1e}"),
        practice.Check("CONTROL: the threshold is right, its location is not",
                       d["dead"]["relu"] is not None and d["dead"]["sigmoid"] is None,
                       f"run the same rule over the lesson's own 10-layer "
                       f"`vanishing_gradient_experiment` and it fires on relu at layer "
                       f"{d['dead']['relu']} and gelu at layer {d['dead']['gelu']}, never on "
                       f"sigmoid — last three layers: {d['tail']}. The alarm catches the "
                       f"activation the lesson credits with fixing vanishing gradients, and stays "
                       f"silent on the one it blames"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
