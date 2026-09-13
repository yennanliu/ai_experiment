"""Exercise 2 — the speedup is 1.01x when the completions are longer than the prompt.

    **Medium.** Implement prefix caching: given a prompt P and several
    completions, run one forward pass over P to fill the KV cache, then branch
    per-completion. Measure speedup vs re-encoding P for each.

Reading of the exercise: both arms are built on the lesson's own `KVCache` and
`attention_full`, counted in attention ops, and the measured numbers are checked
against a closed form so the sweep can go to sizes the pure-Python version cannot
run.

**ANSWER: 3.24x at a 512-token prompt with 8 completions of 64 -- and the
ceiling is the number of branches.**

| prompt | completion | branches | re-encode | prefix-cached | speedup |
|---:|---:|---:|---:|---:|---:|
| 512 | 64 | 8 | 1,329,408 | 410,112 | **3.24x** |
| 2,000 | 64 | 8 | 17,048,640 | 3,041,640 | 5.61x |
| 512 | 512 | 8 | 4,198,400 | 3,279,104 | 1.28x |
| 64 | 512 | 8 | 1,329,408 | 1,314,848 | **1.01x** |
| 512 | 64 | 1 | 166,176 | 166,176 | **1.00x** |

**FINDING: prefix caching buys exactly the share of the work that is prompt.**
With one branch it is 1.00x by definition -- there is nothing to share. With
completions four times the prompt it is 1.01x, because the decode steps dominate
and every one of them is unshared. The lever is `P / L`, and the exercise's
phrasing ("measure speedup") invites a single number for a quantity that spans
1.00x to the branch count.

**FINDING: the lesson's `KVCache` cannot be branched.** It has no `copy`, and
`cache.K` is a plain list -- so two branches taken from one cache alias it, and
the second sees the first's appended token. Branching requires copying `K` and
`V` per completion, which costs `O(P)` per branch and is the reason real
implementations page the cache instead of copying it.

**CONTROL: the measured counts match the closed form exactly.** Running the
lesson's own code at P=200, L=20, C=6 gives 145,860 against 45,360, a 3.22x that
the formula reproduces to the token.

Structure: `count` is the closed form; `measure` runs the lesson's own KVCache;
`alias` shows what happens without a copy.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "12-kv-cache-flash-attention"
WIDTH = 8
GRID = ((512, 64, 8), (2_000, 64, 8), (512, 512, 8), (64, 512, 8), (512, 64, 1))
SMALL = (200, 20, 6)


def count(prompt, length, branches, shared):
    """Attention ops for one scheme: sum of cache length over every decode step."""
    if not shared:
        return branches * (prompt + length) * (prompt + length + 1) // 2
    return (prompt * (prompt + 1) // 2
            + branches * (length * prompt + length * (length + 1) // 2))


def prefill(ref, keys, values, prompt):
    """One forward pass over the prompt, filling a cache the branches will copy."""
    base = ref.KVCache()
    for i in range(prompt):
        base.append(keys[i], values[i])
    return base, prompt * (prompt + 1) // 2


def measure(ref, prompt, length, branches, shared):
    """The same two schemes, run through the lesson's own KVCache and attention_full."""
    rng = random.Random(1)
    keys, values, queries = ([[rng.gauss(0, 1) for _ in range(WIDTH)]
                              for _ in range(prompt + length)] for _ in range(3))
    base, total = prefill(ref, keys, values, prompt if shared else 0)
    for _ in range(branches):
        cache = ref.KVCache()
        cache.K, cache.V = list(base.K), list(base.V)
        for i in range(prompt if shared else 0, prompt + length):
            cache.append(keys[i], values[i])
            ref.attention_full(queries[i], cache.K, cache.V)
            total += len(cache)
    return total


def alias(ref, keys, values):
    """Two branches taken from one cache without copying it."""
    base = ref.KVCache()
    base.append(keys[0], values[0])
    first, second = base, base
    first.append(keys[1], values[1])
    return len(second), hasattr(ref.KVCache, "copy")


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rng = random.Random(9)
    vectors = [[rng.gauss(0, 1) for _ in range(WIDTH)] for _ in range(4)]
    grid = {case: (count(*case, False), count(*case, True)) for case in GRID}
    return {
        "grid": grid, "speedup": {case: a / b for case, (a, b) in grid.items()},
        "measured": (measure(ref, *SMALL, False), measure(ref, *SMALL, True)),
        "predicted": (count(*SMALL, False), count(*SMALL, True)),
        "alias": alias(ref, vectors, vectors),
    }


def verify(result):
    speed, grid = result["speedup"], result["grid"]
    measured, predicted = result["measured"], result["predicted"]
    return [
        practice.Check(
            "ANSWER: 3.24x at a 512-token prompt with 8 completions of 64",
            abs(speed[GRID[0]] - 3.24) < 0.02 and speed[GRID[1]] > speed[GRID[0]],
            "attention ops, re-encode against prefix-cached: " + ", ".join(
                f"P={p} L={l} C={c} {a:,} -> {b:,} ({a / b:.2f}x)"
                for (p, l, c), (a, b) in grid.items()),
        ),
        practice.Check(
            "FINDING: one branch is exactly 1.00x",
            speed[GRID[-1]] == 1.0,
            f"with C=1 there is nothing to share, and the two counts are the same integer: "
            f"{grid[GRID[-1]][0]:,}. The ceiling at the other end is the branch count, so the "
            "answer the exercise asks for is a range and not a number",
        ),
        practice.Check(
            "FINDING: 1.01x when the completions are four times the prompt",
            speed[GRID[3]] < 1.05 and speed[GRID[2]] < 1.4,
            f"P=64 L=512 C=8 gives {speed[GRID[3]]:.2f}x and P=512 L=512 gives "
            f"{speed[GRID[2]]:.2f}x. Prefix caching buys exactly the share of the work that is "
            f"prompt; decode steps are unshared by construction, so the lever is P/L and nothing "
            "else -- not the branch count, not the model, not the cache implementation",
        ),
        practice.Check(
            "FINDING: the lesson's KVCache cannot be branched",
            result["alias"] == (2, False),
            f"it has no copy() ({result['alias'][1]}), and cache.K is a plain list, so two "
            f"branches taken from one cache alias it: after the first appends a token the second "
            f"reports {result['alias'][0]} entries. Branching needs a per-completion copy of K "
            "and V, O(P) each, which is why real implementations page the cache instead",
        ),
        practice.Check(
            "CONTROL: the measured counts match the closed form exactly",
            measured == predicted,
            f"running the lesson's own KVCache and attention_full at P={SMALL[0]}, L={SMALL[1]}, "
            f"C={SMALL[2]} gives {measured[0]:,} against {measured[1]:,}, "
            f"{measured[0] / measured[1]:.2f}x, and the formula reproduces both to the token. "
            "That is what lets the sweep reach sizes pure Python could not run",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
