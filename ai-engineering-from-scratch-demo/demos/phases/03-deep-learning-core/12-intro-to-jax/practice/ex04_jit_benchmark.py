"""Exercise 4 — what @jax.jit buys on the lesson's own train_step, and what it costs.

    Benchmark the training step with and without `@jax.jit`. Time 100 steps of
    each. How large is the speedup on your hardware? What is the compilation
    overhead on the first call?

Reading of the exercise: "without `@jax.jit`" is taken literally — the eager path is
`train_step.__wrapped__`, the decorator peeled off the lesson's own function, so both
paths run identical source. Wall clock is hardware-specific, so no check asserts a
millisecond: checks 2-3 are structural cache counts, the rest are ratios taken inside
one run against margins far looser than what is measured. Data is a seeded synthetic
stand-in — `get_mnist_data` downloads from OpenML. The README carries the full table.
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
STEPS, BATCH, N_TRAIN, SWEEP = 100, 128, 4096, (16, 2048)
MNIST_TRAIN = 60000                              # the size the lesson's own loop batches


def fixture(seed, n):
    """Seeded stand-in for MNIST — MNIST's shapes, none of MNIST's download."""
    kx, kt = jax.random.split(jax.random.PRNGKey(seed))
    x = jax.random.uniform(kx, (n, 784))
    return x, jnp.argmax((x - 0.5) @ jax.random.normal(kt, (784, 10)), axis=-1)


def run(fn, params, state, data, n, batch):
    """`n` steps of the lesson's own train loop, blocked at the end so time is honest."""
    xs, ys, out = *data, None
    for i in range(n):
        j = (i * batch) % (N_TRAIN - batch)
        out = fn(params, state, xs[j:j + batch], ys[j:j + batch])
        params, state = out[0], out[1]
    jax.block_until_ready(out)
    return params


def timed(fn, start, data, n, batch, reps=3):
    """Best-of-`reps` seconds per step. The first rep absorbs any compile."""
    best = float("inf")
    for _ in range(reps):
        clock = time.perf_counter()
        run(fn, *start, data, n, batch)
        best = min(best, time.perf_counter() - clock)
    return best / n


def solve():
    ref = parity.load_reference(PHASE, LESSON, "jax_intro")
    data = fixture(7, N_TRAIN)
    p0, jitted = ref.init_params(jax.random.PRNGKey(0)), ref.train_step
    start, eager = (p0, ref.optimizer.init(p0)), ref.train_step.__wrapped__
    cache = [jitted._cache_size()]               # private API, but it is the ground truth
    cold = timed(jitted, start, data, 1, BATCH, 1)         # trace + XLA compile + execute
    cache.append(jitted._cache_size())
    warm = timed(jitted, start, data, 1, BATCH, 1)
    hot = timed(jitted, start, data, STEPS, BATCH)
    slow = timed(eager, start, data, STEPS, BATCH)
    cache.append(jitted._cache_size())
    again = timed(jitted, start, data, 1, BATCH // 2, 1)
    cache.append(jitted._cache_size())
    sweep = {b: (timed(jitted, start, data, 30, b, 2), timed(eager, start, data, 30, b, 2))
             for b in SWEEP}
    ends = [run(fn, *start, data, STEPS, BATCH) for fn in (jitted, eager)]
    return {"cold": cold, "warm": warm, "jit": hot, "eager": slow, "again": again,
            "cache": cache, "sweep": sweep, "tail": MNIST_TRAIN % BATCH,
            "agree": max(float(jnp.max(jnp.abs(a - b)))
                         for a, b in zip(*(jax.tree.leaves(p) for p in ends)))}


def verify(result):
    hot, slow, cold, warm = (result[k] for k in ("jit", "eager", "cold", "warm"))
    speed, cache, even = slow / hot, result["cache"], result["cold"] / (slow - hot)
    small, big = (result["sweep"][b][1] / result["sweep"][b][0] for b in SWEEP)
    grow = [result["sweep"][SWEEP[1]][i] / result["sweep"][SWEEP[0]][i] for i in (0, 1)]
    return [
        practice.Check("ANSWER: jit wins per step, and its first call costs many steps",
                       speed > 2.0 and cold / hot > 5.0 and even < STEPS,
                       f"{STEPS} steps at batch {BATCH}: {1e3 * hot:.3f} ms/step jitted vs "
                       f"{1e3 * slow:.3f} ms/step for the same function undecorated — "
                       f"{speed:.1f}x. The first call takes {1e3 * cold:.0f} ms, worth "
                       f"{cold / hot:.0f} compiled steps, so jit goes net ahead only at step "
                       f"{even:.0f} of the {STEPS}"),
        practice.Check("MECHANISM: compile once, then reuse — a one-off cost, not a per-call tax",
                       cache[:3] == [0, 1, 1] and cold / warm > 10.0,
                       f"the cache holds {cache[0]} entries before the first call, {cache[1]} "
                       f"after, still {cache[2]} once hundreds more calls at that shape have "
                       f"run; the second call costs {1e3 * warm:.3f} ms against "
                       f"{1e3 * cold:.0f} for the first, {cold / warm:.0f}x"),
        practice.Check("FINDING: a different batch size is a different program, and recompiles",
                       cache[3] == cache[2] + 1 and result["again"] > 5 * hot,
                       f"one call at batch {BATCH // 2} takes the cache from {cache[2]} to "
                       f"{cache[3]} and costs {1e3 * result['again']:.0f} ms, "
                       f"{result['again'] / hot:.0f} compiled steps. The lesson's loop dodges "
                       f"this only by dropping the ragged tail — `n_batches = len(X_train) // "
                       f"batch_size` discards {result['tail']} of {MNIST_TRAIN} rows an epoch"),
        practice.Check("FINDING: the speedup is Python dispatch, and it collapses with batch size",
                       small > 1.5 * big and grow[1] < grow[0],
                       f"batch {SWEEP[0]}: {small:.1f}x. Batch {SWEEP[1]}, "
                       f"{SWEEP[1] // SWEEP[0]}x the arithmetic: {big:.1f}x. Over that range "
                       f"the jitted step slows {grow[0]:.1f}x, the eager step only "
                       f"{grow[1]:.1f}x — eager time is Python dispatching a fixed number of "
                       f"primitives, so what jit removes is mostly the interpreter"),
        practice.Check("CONTROL: jit changes the clock and nothing else",
                       result["agree"] < 1e-6,
                       f"{STEPS} identical steps down each path from one init leave the "
                       f"parameters agreeing to {result['agree']:.1e}. XLA fuses and reorders "
                       f"float32 arithmetic, so this is not free a priori"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
