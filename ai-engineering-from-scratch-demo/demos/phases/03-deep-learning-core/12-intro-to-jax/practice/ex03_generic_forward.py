"""Exercise 3 — one mlp_forward for any depth, with the depth read off the pytree.

    Replace the manual forward function with a generic `mlp_forward(params, x)`
    that works for any number of layers. Use `jax.tree.leaves` to determine the
    depth automatically.

Reading of the exercise: fixtures here are seeded synthetic arrays from `jax.random.PRNGKey`,
never the lesson's `get_mnist_data`, which downloads MNIST from OpenML — the shapes are what
this exercise is about, and an offline run is reproducible. Check 1 is the literal
replacement, held to bit-identity with the lesson's `forward` on its own params. Checks 2-3
take the second sentence seriously: `jax.tree.leaves` gives the right *count* and the wrong
*order*, and past nine layers the dict keys it sorts by stop meaning what they look like.
Check 4 shows what the replacement is worth, by putting the same two nets through the
function it replaces.
"""

from __future__ import annotations

from harness import parity, practice

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:                       # pragma: no cover - env guard
    raise practice.Skip(f"needs jax: uv sync --extra llm ({exc})")

PHASE, LESSON = "03-deep-learning-core", "12-intro-to-jax"
DEPTHS = ((784, 10), (784, 32, 10), (784, 64, 32, 16, 10), (784,) + (24,) * 10 + (10,))


def order(params):
    """Layer names in *depth* order. `sorted` is not it once there are ten."""
    return sorted(params, key=lambda name: (len(name), name))


def mlp_forward(params, x):
    """The lesson's `forward`, generalised: ReLU between layers, none after the last."""
    names = order(params)
    for name in names[:-1]:
        x = jax.nn.relu(jnp.dot(x, params[name]["w"]) + params[name]["b"])
    return jnp.dot(x, params[names[-1]]["w"]) + params[names[-1]]["b"]


def build(sizes, key):
    keys = jax.random.split(key, len(sizes) - 1)
    return {f"layer{i + 1}": {"b": jnp.zeros(sizes[i + 1]), "w": jnp.sqrt(2.0 / sizes[i])
            * jax.random.normal(keys[i], (sizes[i], sizes[i + 1]))}
            for i in range(len(sizes) - 1)}


def key_order(params):
    """The names `jax.tree.leaves` yields, in flatten order — sorted by key, so bias first."""
    return [jax.tree_util.keystr(p, simple=True, separator=".")
            for p, _ in jax.tree_util.tree_flatten_with_path(params)[0]]


def depths_work(x):
    """Does the generic forward run, and does len(tree.leaves)//2 report the depth?"""
    out = {}
    for sizes in DEPTHS:
        params = build(sizes, jax.random.PRNGKey(5))
        out[len(sizes) - 1] = (mlp_forward(params, x).shape == (x.shape[0], 10),
                               len(jax.tree.leaves(params)) // 2,
                               sorted(params) == order(params))
    return out


def naive_forward(params, x):
    """The same generic forward, but driven by `sorted(params)` instead of `order`."""
    names = sorted(params)
    for name in names[:-1]:
        x = jax.nn.relu(jnp.dot(x, params[name]["w"]) + params[name]["b"])
    return jnp.dot(x, params[names[-1]]["w"]) + params[names[-1]]["b"]


def attempt(fn, params, x):
    """Run `fn(params, x)` and report either the output shape or the exception it raised."""
    try:
        return f"shape {tuple(fn(params, x).shape)}"
    except (TypeError, KeyError) as exc:
        return " ".join(str(exc).split()).strip("'.")[:96]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "jax_intro")
    xs = jax.random.uniform(jax.random.PRNGKey(7), (64, 784))
    params = ref.init_params(jax.random.PRNGKey(0))
    thin, deep = (build(sizes, jax.random.PRNGKey(5)) for sizes in (DEPTHS[0], DEPTHS[-1]))
    return {"same": bool(jnp.all(mlp_forward(params, xs) == ref.forward(params, xs))),
            "grad_same": max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(
                jax.tree.leaves(jax.grad(lambda p: jnp.sum(mlp_forward(p, xs) ** 2))(params)),
                jax.tree.leaves(jax.grad(lambda p: jnp.sum(ref.forward(p, xs) ** 2))(params)))),
            "leaves": key_order(params), "depths": depths_work(xs),
            "deep_keys": sorted(deep)[:4], "misordered": attempt(naive_forward, deep, xs),
            "ref_thin": attempt(ref.forward, thin, xs), "ref_deep": attempt(ref.forward, deep, xs),
            "mine_thin": attempt(mlp_forward, thin, xs)}


def digest(result) -> dict:
    """Every summary `verify` quotes, computed here so that stays a list of comparisons."""
    depths = result["depths"]
    return {"shapes_ok": all(ok for ok, _d, _s in depths.values()),
            "counts_ok": all(d == n for n, (_ok, d, _s) in depths.items()),
            "depth_list": ", ".join(str(d) for d in depths),
            "counted": ", ".join(f"{d} for a {n}-layer net" for n, (_o, d, _s) in depths.items())}


def verify(result):
    d, depths = digest(result), result["depths"]
    return [
        practice.Check("ANSWER: one mlp_forward reproduces the lesson's forward bit for bit",
                       result["same"] and result["grad_same"] == 0.0 and d["shapes_ok"],
                       "identical float32 output on the lesson's own 3-layer params (not close — "
                       f"equal), and its gradient is equal too (max |Δ| {result['grad_same']:.1e}); "
                       f"depths {d['depth_list']} all return (64, 10)"),
        practice.Check("MECHANISM: jax.tree.leaves gives the depth, two leaves per layer",
                       d["counts_ok"],
                       f"len(jax.tree.leaves(params)) // 2 recovers {d['counted']}. But the "
                       "flattening is by *sorted key*, so within a layer it yields "
                       + ", ".join(result["leaves"][:2]) + " — bias first. Pairing leaves as "
                       "(w, b) in flatten order silently transposes every layer"),
        practice.Check("FINDING: past nine layers the key order stops meaning depth order",
                       not depths[11][2] and depths[11][1] == 11
                       and "dot_general" in result["misordered"],
                       "`sorted(params)` on an 11-layer net gives "
                       + ", ".join(result["deep_keys"]) + ", ... — `layer10` sorts before `layer2` "
                       "as a string. jax.tree.leaves still counts 11 layers correctly, so the "
                       "depth is right and the *order* is wrong, and the same forward driven by "
                       f"`sorted` fails on shapes: {result['misordered']}. Any scheme that reads "
                       "structure out of dict keys inherits this"),
        practice.Check("CONTROL: what the replacement buys — the lesson's forward is wired to "
                       "exactly three layers, and only one of its two failures is loud",
                       "layer2" in result["ref_thin"] and result["ref_deep"] == "shape (64, 24)"
                       and result["mine_thin"] == "shape (64, 10)",
                       f"on the 1-layer net the lesson's `forward` raises {result['ref_thin']} "
                       f"where mlp_forward returns {result['mine_thin']}; on the 11-layer net it "
                       f"returns {result['ref_deep']} and raises nothing at all — it stops after "
                       "layer3 and the remaining eight layers are never applied, so the output is "
                       "a hidden activation wearing the shape of a prediction"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
