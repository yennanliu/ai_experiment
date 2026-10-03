"""Exercise 2 — the knee is where reconstruction stops improving, and past it the codes die.

    **Codebook size.** Train tokenizers with codebook sizes 512, 4096, 16384.
    Larger codebooks give better reconstruction but harder prediction. Find the
    knee.

Reading of the exercise: the lesson's own `train_codebooks` is fitted with
`CODEBOOK` set to each size, and two numbers are read per size -- held-out
reconstruction MSE (`reconstruction_mse`) and held-out next-scale prediction
difficulty, the mean negative log-likelihood per token under the lesson's own
`fit_predictor`. The three named sizes cannot be fitted on `main()`'s data, so
the sweep runs 16..512 on 512 training images, which is the largest set the
scale-1 codebook can be initialised from at 512 codes.

**ANSWER: the knee is at 256 codes, and it is a cliff rather than a bend.**
Val MSE falls from **2.5e-04** at 16 codes through **4.5e-06** (32) and
**3.4e-08** (128) to **2.5e-08** at 256, then rises to **2.1e-06** at 512.
Prediction NLL climbs the whole way, **1.46** to **2.56** nats per token. Past
256 the trade-off stops being a trade-off: both axes get worse. 512 is worse
than the best smaller codebook on all three data seeds tried.

**FINDING: the bigger codebook is mostly dead codes.** At 512 codes the four
scales hold only **111, 19, 17, 36** distinct values. The tokenizer is scalar,
and after a scale-1 codebook large enough to memorise every pooled mean the
residuals below it hold only a handful of values; k-means initialised on those
draws duplicate centres that never separate.

**FINDING: the exercise's sizes cannot be fitted on the lesson's data at all.**
`fit_codebook` needs at least as many samples as codes, and scale 1 has one
sample per image: on `main()`'s 64 images, 512, 4096 and 16384 each raise
`ValueError`. A 16384-code tokenizer needs 16384 training images just to start.

**CONTROL: part of "harder prediction" is the predictor's add-one prior.**
`fit_predictor` seeds every table with a count of 1 per code, so on 512 images
the scale-1 table puts **3%** of its mass on the prior at 16 codes and **50%**
at 512 -- at the exercise's 16384 codes on 16384 images it would also be half.
NLL still grows more slowly than `log(k)` (2.56 against 6.24 nats at 512), so
the effective vocabulary is far smaller than the nominal one.
"""

from __future__ import annotations

import numpy as np

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "19-visual-autoregressive-var"
SIZES, NAMED = (16, 32, 64, 128, 256, 512), (512, 4096, 16384)
IMAGES, SEEDS = 512, (0, 1, 2)


def nll(ref, predictors, streams):
    total, count = 0.0, 0
    uniform = np.ones(ref.CODEBOOK) / ref.CODEBOOK
    for stream in streams:
        for k, tokens in enumerate(stream):
            probs = predictors[k].get(ref.context_key(stream[:k]), uniform)
            total -= float(np.log(probs[tokens.reshape(-1)]).sum())
            count += tokens.size
    return total / count


def prior_mass(table, size, observed):
    # share of the scale-1 table that is the add-one prior; nan if the counts disagree
    counts = table * (size + observed)
    exact = np.allclose(counts, np.round(counts)) and counts.min() >= 1 - 1e-9
    return size / (size + observed) if exact else float("nan")


def fit(ref, size, train, val):
    ref.CODEBOOK = size
    books = ref.train_codebooks(train)
    predictors = ref.fit_predictor([ref.tokenize_multiscale(i, books) for i in train])
    return {
        "mse": ref.reconstruction_mse(val, books),
        "nll": nll(ref, predictors, [ref.tokenize_multiscale(i, books) for i in val]),
        "distinct": [len(np.unique(b)) for b in books],
        "prior": prior_mass(predictors[0][()], size, len(train)),
    }


def named_errors(ref):
    train = ref.make_patterns(np.random.default_rng(0), 64)
    errors = []
    for size in NAMED:
        ref.CODEBOOK = size
        try:
            ref.train_codebooks(train)
        except ValueError as err:
            errors.append(str(err))
    return errors


def seed_sweep(ref, seed):
    rng = np.random.default_rng(seed)
    train, val = ref.make_patterns(rng, IMAGES), ref.make_patterns(rng, IMAGES // 4)
    return {size: fit(ref, size, train, val) for size in SIZES}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    sweeps = [seed_sweep(ref, s) for s in SEEDS]
    errors = named_errors(ref)
    ref.CODEBOOK = 16
    mse = [sweeps[0][k]["mse"] for k in SIZES]
    nlls = [sweeps[0][k]["nll"] for k in SIZES]
    return {
        "sweep": sweeps[0],
        "errors": errors,
        "mse": mse,
        "nll": nlls,
        "knee": SIZES[int(np.argmin(mse))],
        "rising": bool(np.all(np.diff(nlls) > 0)),
        "worse": sum(
            s[512]["mse"] > min(s[k]["mse"] for k in SIZES[:-1]) for s in sweeps
        ),
    }


def verify(result):
    sw, mse, nlls = result["sweep"], result["mse"], result["nll"]
    return [
        practice.Check(
            "ANSWER: the knee is 256 codes, after which reconstruction gets worse",
            result["knee"] == 256 and mse[-1] > 10 * mse[-2],
            "val MSE by size "
            + ", ".join(f"{k}:{v:.1e}" for k, v in zip(SIZES, mse))
            + f"; 512 worse than the best smaller size on {result['worse']}/{len(SEEDS)} seeds",
        ),
        practice.Check(
            "ANSWER: prediction gets harder at every size",
            result["rising"],
            "held-out NLL nats/token " + ", ".join(f"{v:.2f}" for v in nlls),
        ),
        practice.Check(
            "FINDING: at 512 codes most of the codebook is duplicate values",
            max(sw[512]["distinct"][1:]) < 64,
            f"distinct code values per scale at 512 codes: {sw[512]['distinct']}",
        ),
        practice.Check(
            "FINDING: 512, 4096 and 16384 cannot be fitted on main()'s 64 images",
            len(result["errors"]) == 3,
            "; ".join(result["errors"]),
        ),
        practice.Check(
            "CONTROL: the add-one prior grows to half the scale-1 table",
            sw[16]["prior"] < 0.05 and abs(sw[512]["prior"] - 0.5) < 0.01,
            f"prior mass {sw[16]['prior']:.0%} at 16 codes, {sw[512]['prior']:.0%} at 512; "
            f"NLL {nlls[-1]:.2f} vs log(512) {np.log(512):.2f}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
