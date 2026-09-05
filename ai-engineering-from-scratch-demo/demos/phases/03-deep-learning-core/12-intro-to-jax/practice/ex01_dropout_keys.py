"""Exercise 1 — dropout with a threaded PRNG key, and what the key buys you.

    Add dropout to the MLP. In JAX, dropout requires a PRNG key -- thread a key
    through the forward pass and split it for each dropout layer. Compare test
    accuracy with and without.

Reading of the exercise: `get_mnist_data` downloads from OpenML, so the fixture
here is a **seeded synthetic stand-in** built with `jax.random.PRNGKey` — MNIST's
shapes, a random linear teacher plus label noise, no network — which keeps the
run offline and bit-reproducible. Check 1 answers the literal comparison; checks
2-4 test the clause the exercise leans on, "split it for each dropout layer",
which is load-bearing in a way that is easy to miss; check 5 is the control.
"""

from __future__ import annotations

import functools

from harness import parity, practice

try:
    import jax
    import jax.numpy as jnp
    import optax
except ImportError as exc:                       # pragma: no cover - env guard
    raise practice.Skip(f"needs jax: uv sync --extra llm ({exc})")

PHASE, LESSON = "03-deep-learning-core", "12-intro-to-jax"
STEPS, BATCH, N_TRAIN, RATES = 600, 128, 2048, (0.0, 0.2, 0.5)


def fixture(seed, n, noise=4.0):
    """Seeded stand-in for MNIST — MNIST's shapes, none of MNIST's download."""
    kx, kn = jax.random.split(jax.random.PRNGKey(seed))
    x = jax.random.uniform(kx, (n, 784))
    logits = (x - 0.5) @ jax.random.normal(jax.random.PRNGKey(3), (784, 10))
    return x, jnp.argmax(logits + noise * jax.random.normal(kn, (n, 10)), -1)


def drop(x, key, rate):
    return jnp.where(jax.random.bernoulli(key, 1.0 - rate, x.shape), x / (1.0 - rate), 0.0)


def forward_dropout(params, x, key, rate):
    """The lesson's `forward`, with one dropout layer after each hidden ReLU."""
    k1, k2 = jax.random.split(key)
    x = jax.nn.relu(jnp.dot(x, params["layer1"]["w"]) + params["layer1"]["b"])
    x = jax.nn.relu(jnp.dot(drop(x, k1, rate), params["layer2"]["w"]) + params["layer2"]["b"])
    return jnp.dot(drop(x, k2, rate), params["layer3"]["w"]) + params["layer3"]["b"]


@functools.partial(jax.jit, static_argnums=(0,))
def step(optimizer, params, state, xb, yb, key, rate):
    loss = lambda p: -jnp.mean(jnp.sum(                                     # noqa: E731
        jax.nn.log_softmax(forward_dropout(p, xb, key, rate)) * jax.nn.one_hot(yb, 10), -1))
    updates, state = optimizer.update(jax.grad(loss)(params), state, params)
    return optax.apply_updates(params, updates), state


def train(ref, xs, ys, rate):
    params, key = ref.init_params(jax.random.PRNGKey(0)), jax.random.PRNGKey(1234)
    state = ref.optimizer.init(params)
    for i in range(STEPS):
        key, sub = jax.random.split(key)
        j = (i * BATCH) % (N_TRAIN - BATCH)
        params, state = step(ref.optimizer, params, state,
                             xs[j:j + BATCH], ys[j:j + BATCH], sub, rate)
    return params


def solve():
    ref = parity.load_reference(PHASE, LESSON, "jax_intro")
    (xs, ys), (xt, yt) = fixture(7, N_TRAIN), fixture(11, 1024)
    fit = {r: train(ref, xs, ys, r) for r in RATES}
    key = jax.random.PRNGKey(1234)                       # the key-hygiene probes
    (a, b), (a2, _) = jax.random.split(key), jax.random.split(key)
    fresh = jax.random.split(jax.random.PRNGKey(1235))[0]
    return {"acc": {r: (float(ref.accuracy(p, xs, ys)), float(ref.accuracy(p, xt, yt)))
                    for r, p in fit.items()},
            "leaky": float(jnp.mean(jnp.argmax(forward_dropout(
                fit[0.2], xt, jax.random.PRNGKey(9), 0.2), -1) == yt)),
            "split": (bool(jnp.all(a == a2)), bool(jnp.any(a != b)), bool(jnp.any(
                jax.random.normal(a, (8,)) != jax.random.normal(fresh, (8,))))),
            "prefix": float(jnp.mean(jax.random.bernoulli(key, 0.8, (256,))[:128]
                                     == jax.random.bernoulli(key, 0.8, (128,)))),
            "kept": float(jnp.mean(drop(jnp.ones((20000,)), jax.random.PRNGKey(5), 0.5))),
            "identity": bool(jnp.all(forward_dropout(fit[0.0], xt, jax.random.PRNGKey(9), 0.0)
                                     == ref.forward(fit[0.0], xt)))}


def verify(result):
    acc, on = result["acc"], result["leaky"]
    off, gap = acc[0.2][1], acc[0.0][0] - acc[0.0][1]
    table = "; ".join(f"p={r}: train {acc[r][0]:.4f} test {acc[r][1]:.4f}" for r in RATES)
    return [
        practice.Check("ANSWER: dropout does not buy test accuracy on this fixture",
                       abs(acc[0.2][1] - acc[0.0][1]) < 0.02 and acc[0.5][1] < acc[0.0][1],
                       f"{STEPS} steps each — {table}. p=0.2 moves held-out accuracy "
                       f"{100 * (acc[0.2][1] - acc[0.0][1]):+.2f} points; p=0.5 costs "
                       f"{100 * (acc[0.0][1] - acc[0.5][1]):.1f}, because it also stops the net "
                       f"fitting at all. The train/test gap it was meant to close survives: "
                       f"{gap:.4f} at p=0, {acc[0.2][0] - acc[0.2][1]:.4f} at p=0.2 — the labels "
                       f"carry noise the net memorises"),
        practice.Check("FINDING: leaving dropout on at eval costs more than dropout ever wins",
                       off - on > 0.05,
                       f"the same p=0.2 params score {off:.4f} with dropout off, {on:.4f} with it "
                       f"on — {100 * (off - on):.1f} points, 20x the effect the exercise asks "
                       f"about. The key is a required argument, so the eval path must pass one, "
                       f"and passing one keeps dropout live"),
        practice.Check("MECHANISM: one key reused is not two masks — it is a prefix of one",
                       result["prefix"] == 1.0,
                       f"`bernoulli(k, .8, (256,))[:128]` and `bernoulli(k, .8, (128,))` agree on "
                       f"{100 * result['prefix']:.0f}% of positions: the counter PRNG indexes by "
                       f"position, so a new shape does not decorrelate a key. Hence 'split it for "
                       f"each dropout layer'"),
        practice.Check("MECHANISM: split is a pure function of its key, so it repeats exactly",
                       all(result["split"]),
                       "splitting PRNGKey(1234) twice gives bit-identical subkeys; the two "
                       "siblings of one split differ; PRNGKey(1235) differs again. Reproducibility "
                       "is a property of the value passed in, not of when anything last called "
                       "seed() — no interleaved draw shifts it"),
        practice.Check("CONTROL: inverted dropout preserves the mean, and p=0 is the identity",
                       abs(result["kept"] - 1.0) < 0.02 and result["identity"],
                       f"20000 ones at p=0.5 return mean {result['kept']:.5f}; without the 1/(1-p) "
                       f"rescale, {result['kept'] / 2:.5f} — the (1-p) factor that shifts every "
                       f"eval activation. At p=0.0 this forward is bit-identical to the lesson's"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
