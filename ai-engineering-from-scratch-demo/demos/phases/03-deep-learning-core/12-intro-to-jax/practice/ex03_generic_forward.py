"""Exercise 3 — one mlp_forward for any depth, with the depth read off the pytree.

    Replace the manual forward function with a generic `mlp_forward(params, x)`
    that works for any number of layers. Use `jax.tree.leaves` to determine the
    depth automatically.

Reading of the exercise: fixtures here are seeded synthetic arrays from
`jax.random.PRNGKey`, never the lesson's `get_mnist_data`, which downloads MNIST
from OpenML — the shapes are what this exercise is about, and an offline run is
reproducible. Check 1 is the literal replacement, held to bit-identity with the
lesson's `forward`. Checks 2-3 take the second sentence seriously: `jax.tree.leaves`
gives the right *count* and the wrong *order*, and at ten layers the dict keys it
sorts by stop meaning what they look like. Checks 4-5 verify the gradient of the
replacement independently, by central finite difference against a float64
reference — which is also where JAX's float32 default becomes visible.
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
STEPS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)


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
    return {f"layer{i + 1}": {"w": jnp.sqrt(2.0 / sizes[i]) * jax.random.normal(
        keys[i], (sizes[i], sizes[i + 1])), "b": jnp.zeros(sizes[i + 1])}
        for i in range(len(sizes) - 1)}


def key_order(params):
    return [f"{a}.{b}" for a, b in
            (jax.tree_util.keystr(p, simple=True, separator=".").split(".")
             for p, _ in jax.tree_util.tree_flatten_with_path(params)[0])]


def depths_work(x):
    """Does the generic forward run, and does len(tree.leaves)//2 report the depth?"""
    out = {}
    for sizes in DEPTHS:
        params = build(sizes, jax.random.PRNGKey(5))
        leaves = len(jax.tree.leaves(params))
        ok = mlp_forward(params, x).shape == (x.shape[0], 10)
        out[len(sizes) - 1] = (ok, leaves // 2, sorted(params) == order(params))
    return out


def naive_sorted_forward(params, x):
    """What `sorted(params)` gives you — correct up to nine layers, then not."""
    names = sorted(params)
    for name in names[:-1]:
        x = jax.nn.relu(jnp.dot(x, params[name]["w"]) + params[name]["b"])
    return jnp.dot(x, params[names[-1]]["w"]) + params[names[-1]]["b"]


def broken(x):
    try:
        naive_sorted_forward(build(DEPTHS[-1], jax.random.PRNGKey(5)), x)
    except TypeError as exc:
        return " ".join(str(exc).split()).rstrip(".")[:96]
    return "no error"


def directional(params, key):
    """A unit-norm random direction through parameter space."""
    subkeys = jax.random.split(key, len(jax.tree.leaves(params)))
    tree = jax.tree.unflatten(jax.tree.structure(params), list(subkeys))
    v = jax.tree.map(lambda leaf, k: jax.random.normal(k, leaf.shape), params, tree)
    scale = jnp.sqrt(sum(jnp.sum(leaf ** 2) for leaf in jax.tree.leaves(v)))
    return jax.tree.map(lambda leaf: leaf / scale, v)


def fd64(numpy, params, v, xs, ys, h):
    """Central difference of the same loss, recomputed entirely in float64."""
    def loss(sign):
        act = numpy.asarray(xs, dtype=numpy.float64)
        names = order(params)
        for name in names:
            w = numpy.asarray(params[name]["w"], numpy.float64) + sign * h * numpy.asarray(
                v[name]["w"], numpy.float64)
            b = numpy.asarray(params[name]["b"], numpy.float64) + sign * h * numpy.asarray(
                v[name]["b"], numpy.float64)
            act = act @ w + b
            act = numpy.maximum(act, 0) if name != names[-1] else act
        act = act - act.max(axis=-1, keepdims=True)
        logp = act - numpy.log(numpy.exp(act).sum(axis=-1, keepdims=True))
        return -logp[numpy.arange(len(ys)), numpy.asarray(ys)].mean()
    return (loss(1.0) - loss(-1.0)) / (2 * h)


def solve():
    numpy = parity.try_numpy()
    ref = parity.load_reference(PHASE, LESSON, "jax_intro")
    kx, kt = jax.random.split(jax.random.PRNGKey(7))
    xs = jax.random.uniform(kx, (64, 784))
    ys = jnp.argmax((xs - 0.5) @ jax.random.normal(kt, (784, 10)), axis=-1)
    params = ref.init_params(jax.random.PRNGKey(0))
    v = directional(params, jax.random.PRNGKey(99))
    grads = jax.grad(lambda p: ref.loss_fn(p, xs[:16], ys[:16]))(params)
    slope = float(sum(jnp.sum(g * d) for g, d in zip(jax.tree.leaves(grads), jax.tree.leaves(v))))
    move = lambda s, h: jax.tree.map(lambda p, d: p + s * h * d, params, v)   # noqa: E731
    fd32 = {h: float((ref.loss_fn(move(1, h), xs[:16], ys[:16])
                      - ref.loss_fn(move(-1, h), xs[:16], ys[:16])) / (2 * h)) for h in STEPS}
    return {"same": bool(jnp.all(mlp_forward(params, xs) == ref.forward(params, xs))),
            "grad_same": max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(
                jax.tree.leaves(jax.grad(lambda p: jnp.sum(mlp_forward(p, xs) ** 2))(params)),
                jax.tree.leaves(jax.grad(lambda p: jnp.sum(ref.forward(p, xs) ** 2))(params)))),
            "leaves": key_order(params), "depths": depths_work(xs), "broken": broken(xs),
            "deep_keys": sorted(build(DEPTHS[-1], jax.random.PRNGKey(5)))[:4],
            "slope": slope, "fd32": fd32,
            "truth": fd64(numpy, params, v, xs[:16], ys[:16], 1e-5)}


def verify(result):
    fd32, truth, slope = result["fd32"], result["truth"], result["slope"]
    err = {h: abs(v - truth) / abs(truth) for h, v in fd32.items()}
    best = min(err, key=err.get)
    depths = result["depths"]
    return [
        practice.Check("ANSWER: one mlp_forward reproduces the lesson's forward bit for bit, "
                       "at every depth from 1 to 11 layers",
                       result["same"] and result["grad_same"] == 0.0
                       and all(ok for ok, _d, _s in depths.values()),
                       "identical float32 output on the lesson's own 3-layer params (not "
                       "close — equal), and its gradient is equal too (max |Δ| "
                       f"{result['grad_same']:.1e}); depths "
                       + ", ".join(str(d) for d in depths) + " all return (64, 10)"),
        practice.Check("MECHANISM: jax.tree.leaves gives the depth, two leaves per layer",
                       all(d == n for n, (_ok, d, _s) in depths.items()),
                       "len(jax.tree.leaves(params)) // 2 recovers "
                       + ", ".join(f"{d} for a {n}-layer net" for n, (_o, d, _s) in depths.items())
                       + ". But the flattening is by *sorted key*, so within a layer it "
                       "yields " + ", ".join(result["leaves"][:2]) + " — bias first. Pairing "
                       "leaves as (w, b) in flatten order silently transposes every layer"),
        practice.Check("FINDING: at ten layers the key order stops meaning depth order",
                       not depths[11][2] and depths[11][1] == 11
                       and "dot_general" in result["broken"],
                       "`sorted(params)` on an 11-layer net gives "
                       + ", ".join(result["deep_keys"]) + ", ... — `layer10` sorts before "
                       "`layer2` as a string. jax.tree.leaves still counts 11 layers "
                       "correctly, so the depth is right and the *order* is wrong, and the "
                       f"forward pass fails on shapes: got {result['broken']}. Any scheme "
                       "that reads structure out of dict keys inherits this"),
        practice.Check("CONTROL: jax.grad agrees with a float64 central difference to 2e-07",
                       abs(slope - truth) / abs(truth) < 1e-6,
                       f"directional derivative along a unit random direction: jax.grad says "
                       f"{slope:.10f}, a float64 recomputation of the same loss with h=1e-5 "
                       f"says {truth:.10f} — {abs(slope - truth) / abs(truth):.2e} relative. "
                       f"The autodiff answer is correct; float32 is what limits it"),
        practice.Check("FINDING: the same finite difference *in float32* cannot confirm it "
                       "past two digits, and shrinking h only makes it worse",
                       err[best] > 1e-3 and err[1e-5] > 1.0
                       and abs(fd32[1e-3] - fd32[1e-4]) < 1e-8,
                       "central difference of the lesson's own float32 `loss_fn`: "
                       + ", ".join(f"h={h:g} -> {err[h]:.1e} rel" for h in STEPS)
                       + f". The best step is the *largest* one, h={best:g}, and it is still "
                       f"{err[best]:.1e} off; h=1e-3 and h=1e-4 return the identical value "
                       f"{fd32[1e-3]:.9f} because the two losses differ by a fixed number of "
                       f"float32 ulps either way; h=1e-5 is {err[1e-5]:.1f}x wrong, pure "
                       f"quantisation noise. jax.grad beats the best float32 difference by "
                       f"{err[best] / (abs(slope - truth) / abs(truth)):.0f}x — a JAX "
                       f"gradient check needs jax_enable_x64, or it measures rounding"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
