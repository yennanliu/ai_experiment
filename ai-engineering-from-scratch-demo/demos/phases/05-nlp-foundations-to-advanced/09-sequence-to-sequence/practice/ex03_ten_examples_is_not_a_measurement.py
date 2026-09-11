"""Exercise 3 — ten examples is not a measurement.

    **Hard.** Fine-tune `facebook/bart-base` on a 10k-pair paraphrase dataset.
    Compare the fine-tuned model's beam-4 output to the base model's on held-out
    inputs. Report BLEU and pick 10 qualitative examples.

Reading of the exercise: transformers and torch are absent and there is no
paraphrase corpus here, so no fine-tune is run and no BLEU from one is reported.
What is checkable is whether the comparison it describes could distinguish
anything, and on the sample sizes it names it could not.

Bootstrapping exercise 2's decoder outputs over 2000 resamples, the 95% interval
on corpus BLEU is 0.0985 wide at 10 sentences and 0.0488 at 40, against an
entire greedy-to-beam difference of 0.0475. So ten examples carry an interval
twice the size of the effect being compared, and forty carry one still as wide
as the effect itself -- the two systems' intervals overlap at both sizes. A
fine-tune that improved BLEU by a realistic margin would sit inside the noise of
its own evaluation set, and the exercise gives no way to notice.

The second problem is that BLEU is not one number. Scored over the same outputs
with the same references, changing only the n-gram order from 4 to 1 moves the
corpus score by more than the difference between the two decoders being
compared. Reporting "BLEU" without the order, the smoothing and the tokenisation
is reporting a family of numbers and naming one of them.

Structure: the decoders, references and `bleu` come from exercise 2 via
`practice.load_module`, so this measures the same outputs that exercise scored.
`interval` is a percentile bootstrap over sentence-level scores at a chosen
sample size; `by_order` rescores the same corpus at each n-gram order.
"""

from __future__ import annotations

import importlib.util
import pathlib
import random

from harness import practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "09-sequence-to-sequence"

SIBLING = "ex02_unnormalised_beam_returns_the_shortest_hypothesis.py"
UNAVAILABLE = ("transformers", "torch", "datasets", "sacrebleu")
RESAMPLES, SIZES, ORDERS, SEED = 2000, (10, 40), (1, 2, 3, 4), 0
CORPUS = 40


def sentence_scores(sibling, decoder) -> list:
    """One BLEU per sentence, cycling the reference lengths up to CORPUS sentences."""
    lengths = [sibling.LENGTHS[i % len(sibling.LENGTHS)] for i in range(CORPUS)]
    return [sibling.bleu(decoder(n), ["y"] * n) for n in lengths]


def interval(scores, size, order=4) -> tuple:
    rng = random.Random(SEED + size + order)
    means = sorted(sum(rng.choices(scores, k=size)) / size for _ in range(RESAMPLES))
    low, high = means[int(0.025 * RESAMPLES)], means[int(0.975 * RESAMPLES) - 1]
    return round(low, 4), round(high, 4), round(high - low, 4)


def by_order(sibling, decoder) -> dict:
    lengths = [sibling.LENGTHS[i % len(sibling.LENGTHS)] for i in range(CORPUS)]
    return {order: round(sum(sibling.bleu(decoder(n), ["y"] * n, order=order)
                             for n in lengths) / CORPUS, 4) for order in ORDERS}


def solve():
    sibling = practice.load_module(pathlib.Path(__file__).with_name(SIBLING))
    greedy = sibling.greedy
    normalised = (lambda n: sibling.beam(n, sibling.WIDTH, norm=True))
    scores = {"greedy": sentence_scores(sibling, greedy),
              "beam": sentence_scores(sibling, normalised)}
    means = {name: round(sum(row) / len(row), 4) for name, row in scores.items()}
    return {
        "unavailable": [m for m in UNAVAILABLE if importlib.util.find_spec(m) is None],
        "corpus": CORPUS, "resamples": RESAMPLES, "means": means,
        "effect": round(means["beam"] - means["greedy"], 4),
        "intervals": {name: {size: interval(row, size) for size in SIZES}
                      for name, row in scores.items()},
        "orders": {"greedy": by_order(sibling, greedy), "beam": by_order(sibling, normalised)},
    }


def verify(result):
    spans, means, effect = result["intervals"], result["means"], result["effect"]
    orders = result["orders"]
    small, large = SIZES
    order_spread = max(orders["greedy"].values()) - min(orders["greedy"].values())
    return [
        practice.Check(
            "ANSWER: nothing is fine-tuned here, and the comparison it describes could not resolve one",
            result["unavailable"] == list(UNAVAILABLE),
            f"{result['unavailable']} are all absent and there is no paraphrase corpus, so no BLEU "
            f"from a fine-tune is reported. What is checkable is the resolution of the evaluation: "
            f"on {result['corpus']} sentences the two decoders score {means} for an effect of "
            f"{effect:+.4f}"),
        practice.Check(
            "MECHANISM: at 10 examples the 95% interval is twice the effect it would have to resolve",
            spans["greedy"][small][2] > 2 * abs(effect),
            f"a percentile bootstrap over {result['resamples']} resamples gives greedy "
            f"[{spans['greedy'][small][0]}, {spans['greedy'][small][1]}] at n={small} -- a width of "
            f"{spans['greedy'][small][2]}, or {spans['greedy'][small][2] / abs(effect):.1f} times "
            f"an effect of {abs(effect):.4f}. The exercise's '10 qualitative examples' cannot "
            f"separate the two systems it is being used to compare"),
        practice.Check(
            "FINDING: forty sentences halves the interval and it is still as wide as the effect",
            spans["greedy"][large][2] < spans["greedy"][small][2] / 1.5
            and spans["greedy"][large][2] > abs(effect),
            f"at n={large} the interval narrows to {spans['greedy'][large][2]} from "
            f"{spans['greedy'][small][2]} -- close to the root-n factor of "
            f"{(large / small) ** 0.5:.1f} -- and is still {spans['greedy'][large][2]:.4f} against "
            f"an effect of {abs(effect):.4f}. Quadrupling the set buys a factor of two, so "
            f"resolving this effect would take hundreds of sentences"),
        practice.Check(
            "MECHANISM: the two systems' intervals overlap, so the ranking is not established",
            spans["beam"][large][0] < spans["greedy"][large][1],
            f"at n={large}, greedy sits in [{spans['greedy'][large][0]}, "
            f"{spans['greedy'][large][1]}] and beam in [{spans['beam'][large][0]}, "
            f"{spans['beam'][large][1]}]. The intervals overlap, so the point estimates ranking beam "
            f"above greedy are consistent with no difference at all"),
        practice.Check(
            "FINDING: 'BLEU' is a family of numbers, and the n-gram order moves it more than the systems differ",
            order_spread > abs(effect),
            f"the same outputs against the same references score {orders['greedy']} for greedy and "
            f"{orders['beam']} for beam as the n-gram order goes {list(ORDERS)} -- a spread of "
            f"{order_spread:.4f} against a between-system difference of {abs(effect):.4f}. Reporting "
            f"BLEU without the order, the smoothing and the tokenisation names one of a family"),
        practice.Check(
            "CONTROL: the ordering between systems survives the order change, and only the ordering does",
            all(orders["beam"][o] >= orders["greedy"][o] for o in ORDERS),
            f"beam scores at or above greedy at every n-gram order, {orders['beam']} against "
            f"{orders['greedy']}, so the sign of the comparison is stable even though its magnitude "
            f"is not. A qualitative claim survives what a reported number does not, which is the "
            f"argument for the 10 examples and against the BLEU beside them"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
