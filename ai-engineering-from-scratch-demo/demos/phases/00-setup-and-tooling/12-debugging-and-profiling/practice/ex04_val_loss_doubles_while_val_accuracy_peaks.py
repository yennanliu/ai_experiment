"""Exercise 4 — validation loss doubles while validation accuracy reaches its best.

    Set up TensorBoard for a simple training run and identify whether the model is
    overfitting.

Reading of the exercise: neither tensorboard nor torch is installed (nor in CI), so
the run logs through a stand-in with `SummaryWriter.add_scalar`'s signature (tag,
value, step) into a dict, and the curves are read with the lesson's own Part 8
rules. The run is logistic regression by full-batch gradient descent: 100
training and 2000 validation points, 50 features of which 5 carry signal, labels
drawn from the true logistic model, 2000 steps at lr 0.1.

**ANSWER: yes, by the lesson's rule.** "Train loss decreasing, val loss
increasing" holds from step 57: val loss bottoms at 0.565 and ends at 1.191,
while train loss falls from 0.329 to 0.060 and train accuracy reaches 100%.

**FINDING: the lesson's own TensorBoard snippet cannot answer this.** It logs
`loss/train` and `lr` and nothing else -- no validation scalar -- so the one
rule it gives for overfitting has no curve to read.

**FINDING: the rising val loss is overconfidence, not worse predictions.** Val
accuracy is 69.8% at the val-loss minimum and 71.1% at the end, its best value
of the run. Stopping where val loss is lowest keeps the less accurate model.
On separable training data the weights grow without bound, so wrong
predictions get more confident and log loss rises even as more of them are right.

**CONTROL:** with 2000 training points instead of 100 the same run does not
overfit: val loss ends within 1% of its minimum.

Structure: `make_data` draws from the true model; `add_scalar` is the writer's
one call; `train` logs four scalars a step.
"""

from __future__ import annotations

import re

import numpy as np

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "12-debugging-and-profiling"
DIM, SIGNAL, LR, STEPS = 50, 5, 0.1, 2000


def make_data(rng, n, w_true):
    x = rng.normal(size=(n, DIM))
    return x, (rng.random(n) < 1 / (1 + np.exp(-x @ w_true))).astype(float)


def add_scalar(writer, tag, value, step):
    """SummaryWriter.add_scalar's signature, written to a dict instead of an event file."""
    writer.setdefault(tag, []).append((step, float(value)))


def scores(x, y, w):
    z = x @ w
    return np.mean(np.logaddexp(0, z) - y * z), np.mean((z > 0) == (y > 0.5))


def train(n_train, seed=0):
    rng = np.random.default_rng(seed)
    w_true = np.r_[np.ones(SIGNAL), np.zeros(DIM - SIGNAL)]
    (xt, yt), (xv, yv) = make_data(rng, n_train, w_true), make_data(rng, 2000, w_true)
    w, writer = np.zeros(DIM), {}
    for step in range(STEPS):
        w -= LR * xt.T @ (1 / (1 + np.exp(-xt @ w)) - yt) / n_train
        for split, (x, y) in (("train", (xt, yt)), ("val", (xv, yv))):
            loss, acc = scores(x, y, w)
            add_scalar(writer, f"loss/{split}", loss, step)
            add_scalar(writer, f"acc/{split}", acc, step)
    return {tag: np.array([v for _, v in rows]) for tag, rows in writer.items()}


def solve():
    doc = parity.doc_text(PHASE, LESSON)
    snippet = doc.split("SummaryWriter(")[1].split("```")[0]
    small, large = train(100), train(2000)
    best = int(small["loss/val"].argmin())
    return {
        "doc_tags": re.findall(r'add_scalar\("([^"]+)"', snippet),
        "curves": small,
        "best": best,
        "large": (float(large["loss/val"].min()), float(large["loss/val"][-1])),
    }


def verify(result):
    c, best = result["curves"], result["best"]
    val, acc, train_loss = c["loss/val"], c["acc/val"], c["loss/train"]
    rising = val[-1] > 1.5 * val[best] and train_loss[-1] < train_loss[best]
    return [
        practice.Check(
            "ANSWER: overfitting by the lesson's rule -- train loss down, val loss up",
            rising and best < STEPS // 10 and c["acc/train"][-1] == 1.0,
            f"val loss bottoms at step {best} ({val[best]:.3f}) and ends at {val[-1]:.3f}; "
            f"train loss {train_loss[best]:.3f} -> {train_loss[-1]:.3f}, train accuracy "
            f"{c['acc/train'][-1]:.0%}",
        ),
        practice.Check(
            "FINDING: the lesson's TensorBoard snippet logs no validation curve",
            result["doc_tags"] == ["loss/train", "lr"],
            f"Part 8's add_scalar tags: {result['doc_tags']}",
        ),
        practice.Check(
            "FINDING: val accuracy is at its best where val loss is worst",
            acc[-1] > acc[best] + 0.005 and acc[-1] >= acc.max() - 0.002,
            f"val accuracy {acc[best]:.1%} at the val-loss minimum, {acc[-1]:.1%} at the end "
            f"(run maximum {acc.max():.1%}); val loss {val[-1] / val[best]:.2f}x its minimum",
        ),
        practice.Check(
            "CONTROL: with 2000 training points val loss ends within 1% of its minimum",
            result["large"][1] < 1.01 * result["large"][0],
            f"n_train=2000: val loss minimum {result['large'][0]:.4f}, last step "
            f"{result['large'][1]:.4f}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
