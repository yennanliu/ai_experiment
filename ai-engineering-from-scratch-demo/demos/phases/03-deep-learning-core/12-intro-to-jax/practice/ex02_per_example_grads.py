"""Exercise 2 — per-example gradients by vmap, and which examples dominate.

    Use `jax.vmap` to compute per-example gradients for a batch of 32 MNIST
    images. Compute the gradient norm for each example. Which examples have the
    largest gradients, and why?

Reading of the exercise: the lesson's `get_mnist_data` downloads MNIST from
OpenML, so the 32 "images" here are a **seeded synthetic batch** built with
`jax.random.PRNGKey` — 784 features, 10 classes, a random linear teacher plus
label noise — which keeps the run offline and bit-reproducible. "Which examples,
and why" only has an answer once the model has an opinion, so checks 1-3 measure
the batch at initialisation *and* after 200 training steps and pin the mechanism
to the exact identity dL/dz = softmax - onehot. Checks 4-5 are about `vmap`
itself: what it costs against the loop it replaces, and what it throws away.
"""

from __future__ import annotations

import time

from harness import parity, practice

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:                       # pragma: no cover - env guard
    raise practice.Skip(f"needs jax: uv sync --extra llm ({exc})")

PHASE, LESSON = "03-deep-learning-core", "12-intro-to-jax"
BATCH, WARM, N_TRAIN = 32, 200, 2048


def fixture(seed, n, noise=4.0):
    """Seeded stand-in for MNIST — MNIST's shapes, none of MNIST's download."""
    kx, kn = jax.random.split(jax.random.PRNGKey(seed))
    x, teacher = jax.random.uniform(kx, (n, 784)), jax.random.normal(jax.random.PRNGKey(3),
                                                                    (784, 10))
    return x, jnp.argmax((x - 0.5) @ teacher + noise * jax.random.normal(kn, (n, 10)), axis=-1)


def flat_norms(grads, n):
    """||g_i|| over the whole pytree, one row per example."""
    return jnp.sqrt(sum(jnp.sum(leaf.reshape(n, -1) ** 2, axis=1)
                        for leaf in jax.tree.leaves(grads)))


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float(jnp.sum(a * b) / jnp.sqrt(jnp.sum(a * a) * jnp.sum(b * b)))


def timed(fn, reps=5):
    """Best of `reps`, after one warm call — the first call of a jitted function compiles."""
    jax.block_until_ready(fn())
    best = float("inf")
    for _ in range(reps):
        start = time.perf_counter()
        jax.block_until_ready(fn())
        best = min(best, time.perf_counter() - start)
    return best


def profile(ref, params, xs, ys):
    """Norms, confidences and the split by correct/wrong for one parameter set."""
    grads = jax.jit(jax.vmap(jax.grad(ref.loss_fn), in_axes=(None, 0, 0)))(params, xs, ys)
    norms = flat_norms(grads, BATCH)
    probs = jax.nn.softmax(ref.forward(params, xs))
    hit, err = jnp.argmax(probs, axis=-1) == ys, probs - jax.nn.one_hot(ys, 10)
    return {"norms": norms, "spread": float(norms.max() / norms.min()),
            "r_loss": pearson(norms, -jnp.log(probs[jnp.arange(BATCH), ys])),
            "r_delta": pearson(norms, jnp.linalg.norm(err, axis=1)), "n_wrong": int((~hit).sum()),
            "wrong": float(norms[~hit].mean()), "right": float(norms[hit].mean()),
            "bias_gap": float(jnp.max(jnp.abs(grads["layer3"]["b"] - err))),
            "top_p": [float(probs[i, ys[i]]) for i in jnp.argsort(-norms)[:3]],
            "bot_p": [float(probs[i, ys[i]]) for i in jnp.argsort(norms)[:3]]}


def train(ref, xs, ys):
    """WARM steps of the lesson's own train_step — the model has to acquire an opinion."""
    params = ref.init_params(jax.random.PRNGKey(0))
    state = ref.optimizer.init(params)
    for i in range(WARM):
        j = (i * 128) % (N_TRAIN - 128)
        params, state, _ = ref.train_step(params, state, xs[j:j + 128], ys[j:j + 128])
    return params


def race(ref, params, xs, ys):
    """Three ways to get 32 per-example gradients: vmap, an eager loop, a compiled loop."""
    single = jax.grad(ref.loss_fn)
    batched, per_ex = jax.jit(jax.vmap(single, in_axes=(None, 0, 0))), jax.jit(single)
    return {"t_vmap": timed(lambda: batched(params, xs, ys)),
            "t_loop": timed(lambda: [single(params, xs[i], ys[i]) for i in range(BATCH)], 2),
            "t_jit_loop": timed(lambda: [per_ex(params, xs[i], ys[i]) for i in range(BATCH)]),
            "mean_gap": float(max(jnp.max(jnp.abs(jnp.mean(v, axis=0) - b)) for v, b
                                  in zip(jax.tree.leaves(batched(params, xs, ys)),
                                         jax.tree.leaves(single(params, xs, ys)))))}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "jax_intro")
    xs, ys = fixture(7, N_TRAIN)
    batch = (xs[:BATCH], ys[:BATCH])
    cold, hot = ref.init_params(jax.random.PRNGKey(0)), train(ref, xs, ys)
    return {"cold": profile(ref, cold, *batch), "hot": profile(ref, hot, *batch),
            **race(ref, hot, *batch)}


def verify(result):
    cold, hot = result["cold"], result["hot"]
    speed = result["t_loop"] / result["t_vmap"]
    return [
        practice.Check("ANSWER: the examples the model gets wrong carry the biggest gradients",
                       hot["wrong"] > 1.8 * hot["right"] and hot["r_delta"] > 0.9,
                       f"after {WARM} steps, mean ||g|| is {hot['wrong']:.2f} over the "
                       f"{hot['n_wrong']} misclassified examples against {hot['right']:.2f} over "
                       f"the {BATCH - hot['n_wrong']} correct ones — "
                       f"{hot['wrong'] / hot['right']:.2f}x. The three largest sit at p(true class) "
                       + ", ".join(f"{p:.3f}" for p in hot["top_p"]) + "; the three smallest at "
                       + ", ".join(f"{p:.3f}" for p in hot["bot_p"])),
        practice.Check("MECHANISM: the output-layer error is exactly softmax - onehot",
                       hot["bias_gap"] < 1e-6 and cold["bias_gap"] < 1e-6,
                       f"the per-example gradient w.r.t. `layer3.b` matches p - y elementwise to "
                       f"{hot['bias_gap']:.2e}. Every other gradient in the tree is that vector "
                       f"back-propagated, so ||g_i|| is a monotone read-out of how wrong the model "
                       f"is about example i — correlation with the loss {hot['r_loss']:.3f}"),
        practice.Check("FINDING: at initialisation the question has almost no answer",
                       cold["spread"] < 2.0 and hot["spread"] > 8.0 and cold["r_delta"] < 0.75,
                       f"untrained, the 32 norms span only {cold['spread']:.2f}x (correlation "
                       f"with ||p - y|| just {cold['r_delta']:.3f}); after {WARM} steps they span "
                       f"{hot['spread']:.2f}x at {hot['r_delta']:.3f}. A near-uniform softmax makes "
                       f"p - y nearly the same vector for every example, so 'which examples "
                       f"dominate' is a property the model acquires, not one the data has"),
        practice.Check(f"…and vmap is {speed:.0f}x faster than that loop",
                       speed > 8.0,
                       f"best of 5: vmap+jit {1e3 * result['t_vmap']:.2f} ms, plain Python loop "
                       f"{1e3 * result['t_loop']:.1f} ms, loop over a *jitted* per-example grad "
                       f"{1e3 * result['t_jit_loop']:.2f} ms. The lesson's '10-100x faster than a "
                       f"Python loop' holds against the eager loop ({speed:.1f}x); against a loop "
                       f"that is itself compiled the margin is only "
                       f"{result['t_jit_loop'] / result['t_vmap']:.1f}x, so most of the win is "
                       f"dispatch overhead, not vectorisation"),
        practice.Check("CONTROL: the per-example gradients average to the batch gradient",
                       result["mean_gap"] < 1e-6,
                       f"mean over the 32 rows matches `grad(loss_fn)` on the whole batch to "
                       f"{result['mean_gap']:.2e}. `loss_fn` ends in `jnp.mean`, so grad of the "
                       f"mean is the mean of the grads — the batch gradient throws away exactly "
                       f"the spread the first check measures"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
