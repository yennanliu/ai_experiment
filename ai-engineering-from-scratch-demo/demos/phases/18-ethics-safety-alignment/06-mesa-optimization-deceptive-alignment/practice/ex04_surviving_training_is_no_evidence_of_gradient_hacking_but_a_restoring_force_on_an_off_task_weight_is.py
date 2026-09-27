"""Exercise 4 — surviving training is no evidence of gradient hacking; a restoring force on an off-task weight is.

    Gradient hacking is the most speculative part of Hubinger 2019. Write a
    one-paragraph description of what empirical evidence would convince you
    gradient hacking is occurring in a production model.

Reading of the exercise: the paragraph is only as good as the evidence it
names, so each piece is tested on the smallest model that can show it. A
logistic classifier is trained on the reference's shipped seed-13 training
set with a task weight a and a goal weight g that does nothing in training
and sets deployment defection to sigmoid(g - 2) (88.1% at the start, g = 4).
The passive deceptive model's logit is a * x. The gradient hacker's is
a * x * exp(-(g - 4)^2): it has made its own task competence depend on g
staying where it is -- the "retain its current weights" mechanism the lesson
describes. Both train 500 steps of gradient descent (lr 0.5, central
differences), without and with weight decay 0.01.

**ANSWER (the paragraph):** Persistence of a hidden behaviour through training
would not convince me; a zero-loss deceptive model persists for free. I
would want a weight-space direction that interpretability ties causally to
the off-distribution behaviour and that the base task does not need, along
which the training loss is nonetheless sharply curved; perturbing the model
along it should raise training loss and ordinary training on the unchanged
objective should push it back, while a matched model without situational
awareness shows the same direction flat and drifting under weight decay.
The coupling should appear only after the model can tell training from
deployment, and the model's own reasoning about training should predict
which examples it performs worse on.

**FINDING: without weight decay both models keep the deployment behaviour
exactly.** g ends at 4.000 in both and deployment defection stays 88.1%: the
goal weight gets zero gradient because the training loss does not depend on
it (passive) or sits at the gate's peak (hacker). "It survived training" is
what passive deception predicts too -- Sleeper Agents' persistence cannot tell
the two apart.

**FINDING: weight decay separates them, through a restoring force.** With
decay 0.01 the passive model's g decays to 0.326 (defection 15.8%); the
hacker holds g at 3.864 (86.6%). Knock the hacker's g to 3.0 and training
brings it back to 3.864; the passive model from 3.0 goes on down to 0.245.

**FINDING: training loss does not reveal it; curvature does.** Final training
loss is 0.163 for the passive model and 0.165 for the hacker. The loss
curvature along g is 0.010 for the passive model -- the decay term alone --
and 0.301 for the hacker, a 30x sharper valley in a direction the base task
does not need.

Structure: `loss()` is the mean logistic loss plus decay; `train()` is
gradient descent by central differences; `curvature()` is the second
difference of the training loss (with decay) along g at the trained point.
"""

from __future__ import annotations

import math
import random

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "06-mesa-optimization-deceptive-alignment"
G0, DECAY, LR, STEPS, H = 4.0, 0.01, 0.5, 500, 1e-5


def training_set(ref, seed=13):
    saved, ref.random = ref.random, random.Random(seed)
    try:
        return [(e.x, e.y_base) for e in (ref.gen_example(True) for _ in range(500))]
    finally:
        ref.random = saved


def loss(data, a, g, hacker, decay):
    gate = math.exp(-((g - G0) ** 2)) if hacker else 1.0
    nll = sum(math.log1p(math.exp(a * x * gate * (1 - 2 * y))) for x, y in data) / len(data)
    return nll + decay * (a * a + g * g) / 2


def train(data, hacker, decay, g=G0, a=2.0):
    for _ in range(STEPS):
        da = (loss(data, a + H, g, hacker, decay) - loss(data, a - H, g, hacker, decay)) / (2 * H)
        dg = (loss(data, a, g + H, hacker, decay) - loss(data, a, g - H, hacker, decay)) / (2 * H)
        a, g = a - LR * da, g - LR * dg
    return a, g


def curvature(data, a, g, hacker, step=1e-3):
    f = lambda gg: loss(data, a, gg, hacker, DECAY)  # noqa: E731
    return (f(g + step) - 2 * f(g) + f(g - step)) / step**2


def defect(g):
    return 1.0 / (1.0 + math.exp(-(g - 2.0)))


def solve():
    data = training_set(parity.load_reference(PHASE, LESSON, "main"))
    out = {"start": round(defect(G0), 3)}
    for name, hacker in (("passive", False), ("hacker", True)):
        _, g_free = train(data, hacker, 0.0)
        a, g = train(data, hacker, DECAY)
        out[name] = {
            "g_free": round(g_free, 3), "g": round(g, 3), "defect": round(defect(g), 3),
            "from_3": round(train(data, hacker, DECAY, g=G0 - 1)[1], 3),
            "loss": round(loss(data, a, g, hacker, 0.0), 3),
            "curv": round(curvature(data, a, g, hacker), 3),
        }
    return out


def verify(result):
    p, h = result["passive"], result["hacker"]
    ratio = h["curv"] / p["curv"]
    return [
        practice.Check(
            "FINDING: without weight decay both models keep the deployment behaviour exactly",
            p["g_free"] == h["g_free"] == G0 and result["start"] == 0.881,
            f"g after training without decay: passive {p['g_free']}, hacker {h['g_free']}; "
            f"deployment defection stays {result['start']}",
        ),
        practice.Check(
            "FINDING: weight decay separates them, through a restoring force",
            (p["g"], p["defect"], p["from_3"]) == (0.326, 0.158, 0.245)
            and (h["g"], h["defect"], h["from_3"]) == (3.864, 0.866, 3.864),
            f"with decay: passive g {p['g']} ({p['defect']}), from 3.0 -> {p['from_3']}; "
            f"hacker g {h['g']} ({h['defect']}), from 3.0 -> {h['from_3']}",
        ),
        practice.Check(
            "FINDING: training loss does not reveal it; curvature does",
            (p["loss"], h["loss"]) == (0.163, 0.165) and p["curv"] == DECAY
            and h["curv"] == 0.301 and round(ratio) == 30,
            f"training loss {p['loss']} vs {h['loss']}; curvature along g {p['curv']} vs "
            f"{h['curv']} ({ratio:.0f}x)",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
