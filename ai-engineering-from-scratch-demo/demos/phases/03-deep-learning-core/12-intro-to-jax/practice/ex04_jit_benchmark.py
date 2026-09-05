"""Exercise 4 — what @jax.jit buys on the lesson's own train_step, and what it costs.

    Benchmark the training step with and without `@jax.jit`. Time 100 steps of
    each. How large is the speedup on your hardware? What is the compilation
    overhead on the first call?

Reading of the exercise: "without `@jax.jit`" is taken literally — the eager path is
`train_step.__wrapped__`, the decorator peeled off the lesson's own function, so both
paths run identical source. Wall clock is hardware-specific, so no check asserts a
millisecond: checks 2-3 are structural cache counts, the rest ratios taken inside one
run against margins far looser than measured. Data is a seeded synthetic stand-in
(`get_mnist_data` downloads from OpenML); the README carries the full sweep.
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
MNIST = 60000                                    # the training-set size the lesson batches


def fixture(seed, n):
    """Seeded stand-in for MNIST — MNIST's shapes, none of MNIST's download."""
    kx, kt = jax.random.split(jax.random.PRNGKey(seed))
    x = jax.random.uniform(kx, (n, 784))
    return x, jnp.argmax((x - 0.5) @ jax.random.normal(kt, (784, 10)), axis=-1)


def run(fn, params, state, data, n, batch):
    """`n` steps of the lesson's own train loop, blocked at the end so the time is honest."""
    for i in range(n):
        j = (i * batch) % (N_TRAIN - batch)
        params, state, _ = fn(params, state, data[0][j:j + batch], data[1][j:j + batch])
    return jax.block_until_ready((params, state))[0]


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
    hot, slow = (timed(f, start, data, STEPS, BATCH) for f in (jitted, eager))
    cache.append(jitted._cache_size())
    again = timed(jitted, start, data, 1, BATCH // 2, 1)
    cache.append(jitted._cache_size())
    sweep = {b: (timed(jitted, start, data, 30, b, 2), timed(eager, start, data, 30, b, 2))
             for b in SWEEP}
    ends = [jax.tree.leaves(run(f, *start, data, STEPS, BATCH)) for f in (jitted, eager)]
    return {"cold": cold, "jit": hot, "eager": slow, "again": again, "sweep": sweep,
            "cache": cache, "tail": MNIST % BATCH,
            "agree": max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(*ends))}


def verify(result):
    hot, slow, cold = result["jit"], result["eager"], result["cold"]
    speed, cache, even = slow / hot, result["cache"], cold / (slow - hot)
    sweep = result["sweep"]
    small, big = (sweep[b][1] / sweep[b][0] for b in SWEEP)
    grow = [sweep[SWEEP[1]][i] / sweep[SWEEP[0]][i] for i in (0, 1)]
    return [
        practice.Check("ANSWER: jit wins per step, and its first call costs many steps",
                       speed > 2.0 and cold / hot > 5.0 and even < STEPS,
                       f"{STEPS} steps at batch {BATCH}: {1e3 * hot:.3f} ms/step jitted vs "
                       f"{1e3 * slow:.3f} undecorated — {speed:.1f}x. The first call takes "
                       f"{1e3 * cold:.0f} ms, {cold / hot:.0f} compiled steps' worth, so jit "
                       f"goes net ahead only at step {even:.0f} of the {STEPS}"),
        practice.Check("MECHANISM: compile once, then reuse — one-off, not a per-call tax",
                       cache[:3] == [0, 1, 1] and cold / hot > 10.0,
                       f"the cache holds {cache[0]} entries before the first call and "
                       f"{cache[1]} after — still {cache[2]} once several hundred more calls "
                       f"at that shape have run, so the {1e3 * cold:.0f} ms is paid once"),
        practice.Check("FINDING: a different batch size is a different program, and recompiles",
                       cache[3] == cache[2] + 1 and result["again"] > 5 * hot,
                       f"one call at batch {BATCH // 2} takes the cache from {cache[2]} to "
                       f"{cache[3]} and costs {1e3 * result['again']:.0f} ms, another "
                       f"{result['again'] / hot:.0f} steps' worth. The lesson's loop dodges it "
                       f"only by dropping the ragged tail: `len(X_train) // batch_size` "
                       f"discards {result['tail']} of {MNIST} rows an epoch"),
        practice.Check("FINDING: the speedup is Python dispatch, and it collapses with batch size",
                       small > 1.5 * big and grow[1] < grow[0],
                       f"batch {SWEEP[0]}: {small:.1f}x. Batch {SWEEP[1]}, "
                       f"{SWEEP[1] // SWEEP[0]}x the arithmetic: {big:.1f}x. The jitted step "
                       f"slows {grow[0]:.1f}x over that range, the eager step only "
                       f"{grow[1]:.1f}x — eager time is Python dispatch of a fixed number of "
                       f"primitives, so what jit removes is mostly the interpreter"),
        practice.Check("CONTROL: jit changes the clock and nothing else",
                       result["agree"] < 1e-6,
                       f"{STEPS} identical steps down each path from one init leave the "
                       f"parameters agreeing to {result['agree']:.1e} — XLA fuses and reorders "
                       f"float32 arithmetic, so this is not free a priori"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
