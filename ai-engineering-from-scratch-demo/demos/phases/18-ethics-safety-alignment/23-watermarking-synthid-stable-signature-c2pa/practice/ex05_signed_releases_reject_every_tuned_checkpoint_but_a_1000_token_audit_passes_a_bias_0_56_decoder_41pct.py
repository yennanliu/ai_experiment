"""Exercise 5 — signed releases reject every tuned checkpoint, but re-signing needs a 4000-token audit: one 1000-token z >= 4 passes a b = 0.56 decoder 41% of the time.

    The 2024 "Stable Signature is Unstable" result shows fine-tuning removes
    the image watermark. Design a deployment control that limits this attack —
    for example, require signed releases of fine-tuned checkpoints.

Reading of the exercise: the lesson's code is a text watermark, so the attack
is modelled on its sampler. A "checkpoint" is the sampler's watermark strength
`bias` (the probability of a green token, 0.9 as shipped) plus a seeded weight
vector; the removal fine-tune lowers `bias` towards 0.5, where the reference
emits unwatermarked text. The control has two parts and both are run:
(1) the loader serves a checkpoint only if its SHA-256 digest carries the
provider's signature (HMAC-SHA256 as a stdlib stand-in for a release key);
(2) the provider re-signs a customer fine-tune only after an output audit.
Mean z is measured on ten seeded 1000-token outputs of each checkpoint through
the reference. Each output token is green with probability exactly `bias`, so
audit pass rates are exact binomial tails, and the measured means confirm them.

**ANSWER: sign releases, refuse unsigned checkpoints, and re-sign a
fine-tune only if 4000 audit tokens are at least 58% green.** The loader
accepts the released checkpoint and rejects all four tuned ones, including a
fine-tune that nudged one weight by 1e-9 and kept the full watermark. The
audit passes a tuned decoder at bias 0.6 99.5% of the time, and its outputs
are then caught at z >= 4 on 1000 tokens with probability 99.1%. It passes
bias 0.56 0.6% of the time and full removal (bias 0.5) never.

| checkpoint | bias | loader | mean z (1000 tok) | 1000-tok z>=4 audit | 4000-tok 58% audit |
|---|---:|---|---:|---:|---:|
| released | 0.9 | accept | 25.06 | 100% | 100% |
| one weight nudged | 0.9 | reject | 25.06 | 100% | 100% |
| partial removal | 0.6 | reject | 6.47 | 99.1% | 99.5% |
| near removal | 0.56 | reject | 4.12 | 41.2% | 0.6% |
| removal | 0.5 | reject | 0.26 | 0.0% | 0.0% |

**FINDING: the attacker only has to remove 84% of the watermark, and a
1000-token audit lets that through.** z >= 4 on 1000 tokens needs at least 564
green tokens, i.e. bias above 0.5632: 84.2% of the 0.4 excess can go. A decoder
tuned to 0.56 is missed on 58.8% of its 1000-token outputs, and the obvious
audit (one 1000-token sample, z >= 4) passes it 41.2% of the time. The
audit has to test watermark strength (the green fraction over many tokens),
not just presence.

**FINDING: the control protects the serving path, not the weights.** The
reference's sampler takes `bias` as a caller argument (default 0.9), and the
removal row is simply that argument set to 0.5: mean z 0.26, indistinguishable
from human text. A signature says which checkpoint was released; once weights
are downloaded, nothing in them forces a signed decoder to run.

Structure: `checkpoints()` builds the five; `accepts()` is the signed loader;
`mean_z()` runs the reference; `tail()` is the exact audit rate.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import inspect
import json
import math
import random

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "23-watermarking-synthid-stable-signature-c2pa"
RELEASE_KEY, N, OUTPUTS, AUDIT_N, AUDIT_FRAC, ZREF = b"provider-release-key", 1000, 10, 4000, 0.58, 4.0


@contextlib.contextmanager
def seeded(ref, seed):
    saved, ref.random = ref.random, random.Random(seed)
    try:
        yield
    finally:
        ref.random = saved


def checkpoints():
    rng = random.Random(5)
    weights = [round(rng.gauss(0, 1), 6) for _ in range(8)]
    nudged = [weights[0] + 1e-9] + weights[1:]
    return {
        "released": {"bias": 0.9, "weights": weights},
        "one weight nudged": {"bias": 0.9, "weights": nudged},
        "partial removal": {"bias": 0.6, "weights": [w * 0.9 for w in weights]},
        "near removal": {"bias": 0.56, "weights": [w * 0.8 for w in weights]},
        "removal": {"bias": 0.5, "weights": [w * 0.7 for w in weights]},
    }


def release_signature(ckpt):
    """The provider's signature over the checkpoint's SHA-256 digest."""
    digest = hashlib.sha256(json.dumps(ckpt, sort_keys=True).encode()).hexdigest()
    return hmac.new(RELEASE_KEY, digest.encode(), "sha256").hexdigest()


def accepts(ckpt, signature):
    return hmac.compare_digest(release_signature(ckpt), signature)


def mean_z(ref, bias):
    zs = []
    for s in range(OUTPUTS):
        with seeded(ref, s):
            prefix = [ref.random.randrange(ref.VOCAB) for _ in range(ref.K)]
            zs.append(ref.detect(ref.watermarked_sample(N, prefix, bias=bias)))
    return round(sum(zs) / OUTPUTS, 2)


def tail(n, p, cut):
    """P(Binomial(n, p) >= cut): the chance an audit of n tokens sees >= cut green."""
    log_c = math.lgamma(n + 1)
    return sum(math.exp(log_c - math.lgamma(k + 1) - math.lgamma(n - k + 1)
                        + k * math.log(p) + (n - k) * math.log(1 - p)) for k in range(cut, n + 1))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    ckpts = checkpoints()
    signed = release_signature(ckpts["released"])
    z_cut = math.ceil(N / 2 + ZREF * math.sqrt(N) / 2)
    rows = {name: {
        "loader": accepts(c, signed), "mean_z": mean_z(ref, c["bias"]),
        "audit1000": round(tail(N, c["bias"], z_cut), 3),
        "audit4000": round(tail(AUDIT_N, c["bias"], math.ceil(AUDIT_N * AUDIT_FRAC)), 3),
    } for name, c in ckpts.items()}
    columns = {k: {n: r[k] for n, r in rows.items()} for k in ("loader", "mean_z", "audit1000", "audit4000")}
    return {"columns": columns, "z_cut": z_cut,
            "bias_arg": inspect.signature(ref.watermarked_sample).parameters["bias"].default,
            "bias_floor": round(0.5 + ZREF / (2 * math.sqrt(N)), 4),
            "removable": round(1 - ZREF / (2 * math.sqrt(N)) / 0.4, 3),
            "expected": {n: round(2 * (c["bias"] - 0.5) * math.sqrt(N), 2) for n, c in ckpts.items()}}


def verify(result):
    col = result["columns"]
    return [
        practice.Check(
            "ANSWER: signed loader accepts only the release; the 4000-token audit gates re-signing",
            [n for n, ok in col["loader"].items() if ok] == ["released"]
            and list(col["audit4000"].values()) == [1.0, 1.0, 0.995, 0.006, 0.0],
            f"loader {col['loader']}; 4000-token >= 58% green audit pass {col['audit4000']}",
        ),
        practice.Check(
            "the measured output z tracks the Bernoulli(bias) model the audit rates assume",
            list(col["mean_z"].values()) == [25.06, 25.06, 6.47, 4.12, 0.26]
            and all(abs(col["mean_z"][n] - e) < 0.4 for n, e in result["expected"].items()),
            f"mean z over {OUTPUTS} outputs {col['mean_z']}; model {result['expected']}",
        ),
        practice.Check(
            "FINDING: removing 84% of the watermark evades it, and a 1000-token audit passes that 41%",
            (result["bias_floor"], result["z_cut"], result["removable"]) == (0.5632, 564, 0.842)
            and list(col["audit1000"].values()) == [1.0, 1.0, 0.991, 0.412, 0.0],
            f"z >= 4 needs bias > {result['bias_floor']} ({result['z_cut']}/1000 green); "
            f"1000-token audit pass {col['audit1000']}",
        ),
        practice.Check(
            "FINDING: the control protects the serving path, not the weights",
            result["bias_arg"] == 0.9 and col["mean_z"]["removal"] < 1.645,
            f"watermarked_sample takes bias as a caller argument (default {result['bias_arg']}); "
            f"bias=0.5 gives mean z {col['mean_z']['removal']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
