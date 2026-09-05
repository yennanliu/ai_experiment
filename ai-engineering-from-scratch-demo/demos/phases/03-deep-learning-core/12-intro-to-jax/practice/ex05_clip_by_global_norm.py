"""Exercise 5 — clip_by_global_norm in front of Adam, and how little it changes.

    Implement gradient clipping by composing `optax.chain(optax.clip_by_global_norm(1.0),
    optax.adam(1e-3))`. Train with and without clipping. Plot the gradient norm over
    training to see the effect.

Reading of the exercise: a graded run cannot plot, so the gradient norm is recorded at
every step and reported by its marks; the README carries the curve and the full tables.
Data is a seeded synthetic MNIST stand-in — `get_mnist_data` downloads from OpenML.
Checks 1-2 are the literal train-with-and-without over three seeds; check 3 isolates the
transform by handing one gradient to two optimizers from identical state; check 4 is the
control that the same chain in front of `optax.sgd` is decisive, so the null result for
Adam is a fact about Adam and not about the clip.
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
STEPS, BATCH, N_TRAIN, SEEDS, MARKS = 300, 128, 2048, (0, 1, 2), (0, 10, 150, 299)
CLIP, SGD_LR = optax.clip_by_global_norm(1.0), 0.5
OPTS = {"adam": optax.adam(1e-3), "sgd": optax.sgd(SGD_LR)}
OPTS.update({f"clip+{n}": optax.chain(CLIP, o) for n, o in tuple(OPTS.items())})
beats = lambda a, b: all(x > y for x, y in zip(a, b))                   # noqa: E731
rescaled = lambda p: (abs(p["cut"] - 1) < 1e-5 and abs(p["cos"] - 1) < 1e-5   # noqa: E731
                      and abs(p["sgd"] - 1 / p["raw"]) < 1e-4)


def fixture(seed, n, noise=4.0):
    """Seeded stand-in for MNIST — MNIST's shapes, none of MNIST's download."""
    kx, kn = jax.random.split(jax.random.PRNGKey(seed))
    x = jax.random.uniform(kx, (n, 784))
    logits = (x - 0.5) @ jax.random.normal(jax.random.PRNGKey(3), (784, 10))
    return x, jnp.argmax(logits + noise * jax.random.normal(kn, (n, 10)), axis=-1)


def norm(tree):                                  # the global norm optax clips on
    return jnp.sqrt(sum(jnp.sum(leaf ** 2) for leaf in jax.tree.leaves(tree)))


def batch_at(data, i):
    j = (i * BATCH) % (N_TRAIN - BATCH)
    return data[0][j:j + BATCH], data[1][j:j + BATCH]


@functools.partial(jax.jit, static_argnums=(0, 1))
def step(loss_fn, opt, params, state, xb, yb):
    loss, grads = jax.value_and_grad(loss_fn)(params, xb, yb)
    updates, state = opt.update(grads, state, params)
    return optax.apply_updates(params, updates), state, loss, norm(grads)


def train(ref, name, data, held, seed):
    """Train under `OPTS[name]`, recording the raw gradient norm at every step."""
    params, norms, loss = ref.init_params(jax.random.PRNGKey(seed)), [], None
    state = OPTS[name].init(params)
    for i in range(STEPS):
        params, state, loss, g = step(ref.loss_fn, OPTS[name], params, state, *batch_at(data, i))
        norms.append(float(g))
    return {"loss": float(loss), "acc": float(ref.accuracy(params, *held)), "peak": max(norms),
            "over": sum(n > 1.0 for n in norms), "marks": [norms[m] for m in MARKS],
            "state": state, "params": params}


def probe(ref, params, adam_state, batch):
    """One gradient, two optimizers, identical state — what does the clip change?"""
    g = jax.grad(ref.loss_fn)(params, *batch)
    cut = CLIP.update(g, CLIP.init(params), params)[0]
    dot = sum(jnp.sum(a * b) for a, b in zip(jax.tree.leaves(g), jax.tree.leaves(cut)))
    out = {"raw": float(norm(g)), "cut": float(norm(cut)),
           "cos": float(dot / (norm(g) * norm(cut)))}
    for name, state in (("adam", adam_state), ("sgd", OPTS["sgd"].init(params))):
        both = OPTS[f"clip+{name}"].update(g, (CLIP.init(params), state), params)[0]
        out[name] = float(norm(both) / norm(OPTS[name].update(g, state, params)[0]))
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "jax_intro")
    data, held = fixture(7, N_TRAIN), fixture(11, 1024)
    runs = {name: [train(ref, name, data, held, s) for s in SEEDS] for name in OPTS}
    warm, cold = runs["adam"][0], ref.init_params(jax.random.PRNGKey(0))
    got = {f: {k: [r[f] for r in v] for k, v in runs.items()}
           for f in ("loss", "acc", "peak", "over")}
    return {**got, "marks": runs["clip+adam"][0]["marks"],
            "cold": probe(ref, cold, OPTS["adam"].init(cold), batch_at(data, 0)),
            "warm": probe(ref, warm["params"], warm["state"], batch_at(data, STEPS))}


def evidence(result) -> tuple:
    """The four evidence strings, built here so `verify` stays a list of comparisons."""
    loss, acc, cold, warm = result["loss"], result["acc"], result["cold"], result["warm"]
    both = zip(loss["adam"], acc["adam"], loss["clip+adam"], acc["clip+adam"])
    gain = 1e2 * (sum(acc["clip+adam"]) - sum(acc["adam"])) / len(SEEDS)
    e1 = (f"{STEPS} steps, seeds {SEEDS}, as final-loss/held-out-accuracy without -> with the "
          f"clip: " + ", ".join(f"{a:.3f}/{b:.4f} -> {c:.3f}/{d:.4f}" for a, b, c, d in both)
          + f". Mean {gain:+.2f} accuracy points, against a "
          f"{1e2 * (max(acc['adam']) - min(acc['adam'])):.2f}-point spread across the unclipped "
          f"seeds alone: it regularises, it does not stabilise")
    e2 = (f"the norms a loop can record are the raw ones, and in the clipped run they still "
          f"cross 1.0 on {result['over']['clip+adam'][0]} of {STEPS} steps — "
          + ", ".join(f"step {m}: {v:.2f}" for m, v in zip(MARKS, result["marks"]))
          + ". The clip sits between `grad` and `apply_updates` and never enters that curve")
    e3 = (f"||g|| is {cold['raw']:.3f} at init and {warm['raw']:.3f} after {STEPS} Adam steps; "
          f"both clip to {cold['cut']:.6f}/{warm['cut']:.6f} at cosine "
          f"{cold['cos']:.6f}/{warm['cos']:.6f} to the raw gradient. That costs SGD exactly "
          f"1/||g|| of its step ({cold['sgd']:.4f}, {warm['sgd']:.4f}) and costs Adam "
          f"{1e2 * abs(cold['adam'] - 1):.2f}% / {1e2 * abs(warm['adam'] - 1):.0f}% — at t=1 "
          f"Adam's update is g/(|g|+eps) elementwise, and later sqrt(v-hat) still carries most "
          f"of the same factor, so `chain(clip, adam)` largely undoes its own clip")
    e4 = (f"sgd({SGD_LR}) peaks at ||g|| {result['peak']['sgd'][0]:.0f} and ends at loss "
          + ", ".join(f"{v:.3f}" for v in loss["sgd"]) + f" — chance; `chain(clip, sgd)` peaks "
          f"at {result['peak']['clip+sgd'][0]:.1f} and ends at "
          + ", ".join(f"{v:.3f}" for v in loss["clip+sgd"])
          + ". The clip is no no-op; Adam is what makes it one")
    return e1, e2, e3, e4


def verify(result):
    loss, acc, cold, warm = result["loss"], result["acc"], result["cold"], result["warm"]
    e1, e2, e3, e4 = evidence(result)
    return [
        practice.Check("ANSWER: the chain costs training loss and buys a little held-out "
                       "accuracy, on all three seeds",
                       beats(loss["clip+adam"], loss["adam"])
                       and beats(acc["clip+adam"], acc["adam"]), e1),
        practice.Check("FINDING: plotting the gradient norm shows the clip doing nothing",
                       result["over"]["clip+adam"][0] > 0.95 * STEPS, e2),
        practice.Check("MECHANISM: a pure rescale — SGD feels all of it, Adam almost none",
                       all(map(rescaled, (cold, warm)))
                       and abs(cold["adam"] - 1) < 1e-3
                       and abs(warm["adam"] - 1) < 0.5 * (1 - 1 / warm["raw"]), e3),
        practice.Check("CONTROL: the same chain in front of optax.sgd is decisive",
                       beats([v - 0.2 for v in loss["sgd"]], loss["clip+sgd"])
                       and result["peak"]["sgd"][0] > 20, e4),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
