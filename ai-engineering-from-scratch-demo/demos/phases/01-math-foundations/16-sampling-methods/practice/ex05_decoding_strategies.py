"""Exercise 5 — greedy / temperature / top-k / top-p over 5 runs, compared.

    Build a complete text generation demo: given a vocabulary of 10 words with
    logits, generate sequences of 20 tokens using (a) greedy, (b)
    temperature=0.7, (c) top-k=3, (d) top-p=0.9. Compare the diversity of
    outputs across 5 runs.

Reading of the exercise: "compare the diversity" needs a number. Two are used,
because they measure different things: **distinct tokens** within a sequence, and
**identical-sequence count** across the 5 runs. Greedy pins both to their floor
by construction, which makes it the control rather than a competitor. Check 4 is
the one worth having — it reports the support size each strategy actually samples
from, derived from the lesson's own distribution functions, so "more diverse" has
a cause rather than just a measurement.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "16-sampling-methods"
SEED, LENGTH, RUNS = 20260904, 20, 5
LOGITS = [4.0, 3.5, 3.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0]
VOCAB = ["the", "a", "cat", "dog", "runs", "sits", "fast", "slow", "red", "blue"]


def generate(ref, strategy, rng_seed):
    random.seed(rng_seed)
    picks = []
    for _ in range(LENGTH):
        if strategy == "greedy":
            picks.append(max(range(len(LOGITS)), key=lambda i: LOGITS[i]))
        elif strategy == "temperature=0.7":
            picks.append(ref.temperature_sample(LOGITS, 0.7))
        elif strategy == "top-k=3":
            picks.append(ref.top_k_sample(LOGITS, 3))
        else:
            picks.append(ref.top_p_sample(LOGITS, 0.9))
    return picks


def solve():
    ref = parity.load_reference(PHASE, LESSON, "sampling")
    strategies = ("greedy", "temperature=0.7", "top-k=3", "top-p=0.9")
    rows = {}
    for strategy in strategies:
        sequences = [generate(ref, strategy, SEED + run) for run in range(RUNS)]
        distinct = [len(set(s)) for s in sequences]
        rows[strategy] = {
            "distinct_mean": sum(distinct) / RUNS,
            "unique_sequences": len({tuple(s) for s in sequences}),
            "first": [VOCAB[i] for i in sequences[0][:6]],
        }
    def support(distribution):
        return sum(1 for p in distribution if p > 1e-9)

    supports = {
        "temperature=0.7": support(ref.temperature_distribution(LOGITS, 0.7)),
        "top-k=3": support(ref.top_k_distribution(LOGITS, 3)),
        "top-p=0.9": support(ref.top_p_distribution(LOGITS, 0.9)),
    }
    # a flatter logit vector, to separate top-p from top-k
    flat = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    flat_supports = {"top-k=3": support(ref.top_k_distribution(flat, 3)),
                     "top-p=0.9": support(ref.top_p_distribution(flat, 0.9))}
    return {"rows": rows, "supports": supports, "vocab": len(VOCAB),
            "flat_supports": flat_supports}


def _support_text(supports, vocab) -> str:
    return ", ".join(f"{k}: {v} of {vocab} tokens" for k, v in supports.items())


def verify(result):
    rows, supports = result["rows"], result["supports"]
    greedy, flat = rows["greedy"], result["flat_supports"]
    sampled = {k: v for k, v in rows.items() if k != "greedy"}
    return [
        practice.Check("greedy is deterministic: 1 distinct token, 1 unique sequence",
                       greedy["distinct_mean"] == 1.0 and greedy["unique_sequences"] == 1,
                       f"every run gives {greedy['first'][:3]}… — with fixed logits and no "
                       f"sampling there is nothing to vary, so greedy is the control"),
        practice.Check(f"all three sampling strategies vary across {RUNS} runs",
                       all(r["unique_sequences"] == RUNS for r in sampled.values()),
                       "; ".join(f"{k}: {v['unique_sequences']}/{RUNS} unique, "
                                 f"{v['distinct_mean']:.1f} distinct tokens"
                                 for k, v in sampled.items())),
        practice.Check("top-k=3 is the least diverse of the three, by construction",
                       rows["top-k=3"]["distinct_mean"] <= min(
                           rows[k]["distinct_mean"] for k in
                           ("temperature=0.7", "top-p=0.9")),
                       f"{rows['top-k=3']['distinct_mean']:.1f} distinct tokens per "
                       f"sequence against {rows['temperature=0.7']['distinct_mean']:.1f} "
                       f"for temperature and {rows['top-p=0.9']['distinct_mean']:.1f} "
                       f"for top-p"),
        practice.Check("CAUSE: diversity follows the support size each strategy samples from",
                       supports["top-k=3"] == 3
                       and supports["temperature=0.7"] == result["vocab"],
                       _support_text(supports, result["vocab"])
                       + " — temperature rescales but never truncates, top-p keeps the "
                         "smallest set reaching 0.9 mass, top-k keeps exactly 3"),
        practice.Check("…and on THESE logits top-p and top-k coincide, which is a coincidence",
                       supports["top-p=0.9"] == supports["top-k=3"] == 3
                       and flat["top-p=0.9"] > 3,
                       f"both admit 3 tokens here, because this vector happens to hold 0.9 "
                       f"mass in exactly 3 tokens — so the two strategies are "
                       f"indistinguishable on it. On a flat logit vector top-p admits "
                       f"{flat['top-p=0.9']} while top-k still admits {flat['top-k=3']}: "
                       f"top-p adapts to the distribution, top-k cannot"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
