"""Exercise 2 — leave-one-out target encoding against the naive kind.

    Implement leave-one-out target encoding: for each row, compute the target mean
    excluding that row's own target value. Show how this reduces overfitting
    compared to naive target encoding.

Reading of the exercise: "show how this reduces overfitting" needs a held-out
split and a control — a category assigned *independently* of the target, where
apparent signal is leakage by construction. A second column with real signal
checks LOO does not destroy information with the leak. Numbers in the README.
"""

from __future__ import annotations

import random

from harness import parity, practice, stats

PHASE, LESSON = "02-ml-fundamentals", "08-feature-engineering"
N, N_CATEGORIES, TRAIN = 400, 80, 300  # 80 categories over 300 train rows: many are rare


def _category_stats(features, targets) -> tuple:
    total, count = {}, {}
    for feature, target in zip(features, targets):
        total[feature] = total.get(feature, 0.0) + target
        count[feature] = count.get(feature, 0) + 1
    return total, count, sum(targets) / len(targets)


def leave_one_out_encode(features, targets):
    """Each row gets its category's mean with its own target removed."""
    total, count, overall = _category_stats(features, targets)
    return [(total[f] - t) / (count[f] - 1) if count[f] > 1 else overall
            for f, t in zip(features, targets)]


def _apply(features, total, count, overall):
    """Naive encoding, and how both arms score held-out rows."""
    return [total[f] / count[f] if f in count else overall for f in features]


def _arm(ref, categories, targets) -> dict:
    """Both encodings of one column, fitted on the train half and scored on both."""
    train_c, train_y, test_y = categories[:TRAIN], targets[:TRAIN], targets[TRAIN:]
    total, count, overall = _category_stats(train_c, train_y)
    held_out = _apply(categories[TRAIN:], total, count, overall)
    out = {"baseline": stats.rmse(test_y, test_y, 0.0, overall),
           "singletons": sum(v == 1 for v in count.values())}
    for name, encoded in (("naive", _apply(train_c, total, count, overall)),
                          ("loo", leave_one_out_encode(train_c, train_y))):
        slope, intercept = stats.fit_line(encoded, train_y)
        out[name] = {"slope": slope, "corr": ref.correlation(encoded, train_y),
                     "train": stats.rmse(encoded, train_y, slope, intercept),
                     "test": stats.rmse(held_out, test_y, slope, intercept)}
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "features")
    rng = random.Random(11)
    noise_cats = [f"c{rng.randrange(N_CATEGORIES)}" for _ in range(N)]
    signal_cats = [f"s{rng.randrange(N_CATEGORIES)}" for _ in range(N)]
    noise_y = [rng.gauss(0, 1) for _ in range(N)]
    signal_y = [int(c[1:]) * 0.05 + rng.gauss(0, 1) for c in signal_cats]
    return {"noise": _arm(ref, noise_cats, noise_y),
            "signal": _arm(ref, signal_cats, signal_y),
            "smoothing": {s: ref.correlation(
                ref.target_encode(noise_cats[:TRAIN], noise_y[:TRAIN], smoothing=s)[0],
                noise_y[:TRAIN]) for s in (0, 10, 50)}}


def verify(result):
    noise, signal = result["noise"], result["signal"]
    naive, loo = noise["naive"], noise["loo"]
    return [
        practice.Check(f"{N} rows over {N_CATEGORIES} categories, {TRAIN} for training",
                       noise["singletons"] > 0, f"{noise['singletons']} training categories "
                       f"hold one row: there, naive encoding *is* that row's own target"),
        practice.Check("ANSWER: the naive encoding leaks — it finds signal that cannot exist",
                       naive["corr"] > 0.3 and loo["corr"] < 0,
                       f"training correlation on a category independent of y: "
                       f"{naive['corr']:+.4f} naive, {loo['corr']:+.4f} LOO. FINDING: LOO "
                       f"over-corrects, it does not land on zero"),
        practice.Check("…and the fitted slope says what naive encoding is doing",
                       abs(naive["slope"] - 1.0) < 0.02,
                       f"slope {naive['slope']:+.4f} — an identity map: it copies y"),
        practice.Check("ANSWER: which is overfitting — the train/test gap is the evidence",
                       naive["test"] - naive["train"] > 0.2
                       and abs(loo["test"] - loo["train"]) < 0.1,
                       f"naive {naive['train']:.4f}/{naive['test']:.4f} train/test RMSE, gap "
                       f"{naive['test'] - naive['train']:+.4f}; LOO "
                       f"{loo['train']:.4f}/{loo['test']:.4f}, gap "
                       f"{loo['test'] - loo['train']:+.4f}"),
        practice.Check("…and the leak costs real accuracy, not just honesty",
                       naive["test"] > noise["baseline"] > loo["test"] - 0.02,
                       f"held out: mean-only {noise['baseline']:.4f}, LOO {loo['test']:.4f} "
                       f"— a match — and naive *worse* at {naive['test']:.4f}"),
        practice.Check("CONTROL: where the category carries real signal, LOO keeps it",
                       signal["loo"]["test"] < signal["baseline"] * 0.9
                       and signal["loo"]["test"] < signal["naive"]["test"],
                       f"held-out RMSE {signal['loo']['test']:.4f} LOO, "
                       f"{signal['baseline']:.4f} mean-only, {signal['naive']['test']:.4f} "
                       f"naive"),
        practice.Check("FINDING: the lesson's smoothing does not address leakage",
                       min(result["smoothing"].values()) > 0.3,
                       "training correlation at smoothing "
                       + ", ".join(f"{s}: {v:+.4f}" for s, v in result["smoothing"].items())
                       + " — shrinkage is nearly monotone, so it rescales the leaked "
                         "values without unranking them"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
