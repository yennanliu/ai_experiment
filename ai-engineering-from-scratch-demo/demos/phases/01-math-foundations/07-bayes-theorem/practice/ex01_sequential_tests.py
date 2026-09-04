"""Exercise 1 — two positive tests, posterior chained into prior.

    **Multiple tests.** A patient tests positive twice on independent tests
    (both 99% accurate, disease prevalence 1 in 10,000). What is P(sick) after
    both tests? Use the posterior from the first test as the prior for the
    second.

Reading of the exercise: "99% accurate" is ambiguous — it is read as sensitivity
= specificity = 0.99, so the false-positive rate is 0.01, which is the reading
the lesson's `bayes(prior, likelihood, false_positive_rate)` signature assumes.
The chaining instruction is the real content: check 3 shows it is equivalent to
one update with the squared likelihood ratio, which is what "independent" buys
you and is why the answer jumps from 1% to 50%.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "01-math-foundations", "07-bayes-theorem"
PREVALENCE = 1 / 10_000
SENSITIVITY, FALSE_POSITIVE = 0.99, 0.01


def solve():
    ref = parity.load_reference(PHASE, LESSON, "bayes")
    first = ref.bayes(PREVALENCE, SENSITIVITY, FALSE_POSITIVE)
    second = ref.bayes(first, SENSITIVITY, FALSE_POSITIVE)
    chained = ref.sequential_bayes(PREVALENCE, SENSITIVITY, FALSE_POSITIVE, 2)
    # one update with the likelihood ratio squared
    ratio = (SENSITIVITY / FALSE_POSITIVE) ** 2
    odds = PREVALENCE / (1 - PREVALENCE) * ratio
    one_shot = odds / (1 + odds)
    tests_to_ninety, posterior, n = None, PREVALENCE, 0
    while posterior < 0.90 and n < 20:
        posterior = ref.bayes(posterior, SENSITIVITY, FALSE_POSITIVE)
        n += 1
        if posterior >= 0.90 and tests_to_ninety is None:
            tests_to_ninety = n
    return {"first": first, "second": second, "chained": chained,
            "one_shot": one_shot, "to_ninety": tests_to_ninety, "at_ninety": posterior}


def verify(result):
    chained = result["chained"]
    final = chained[-1] if isinstance(chained, (list, tuple)) else chained
    return [
        practice.Check("one positive test leaves P(sick) below 1%",
                       abs(result["first"] - 0.00980) < 1e-4,
                       f"P(sick | +) = {result['first']:.6f} — a 99% accurate test on a "
                       f"1-in-10,000 disease is still ~99% false alarms"),
        practice.Check("two positive tests reach roughly 50%",
                       0.49 < result["second"] < 0.51,
                       f"P(sick | ++) = {result['second']:.6f}"),
        practice.Check("chaining equals one update with the likelihood ratio squared",
                       abs(result["second"] - result["one_shot"]) < 1e-12,
                       f"(0.99/0.01)² = {(SENSITIVITY / FALSE_POSITIVE) ** 2:.0f} applied to "
                       f"the prior odds gives {result['one_shot']:.9f} — identical, which is "
                       f"what independence means in odds form"),
        practice.Check("the lesson's sequential_bayes agrees",
                       abs(final - result["second"]) < 1e-12,
                       f"sequential_bayes(..., 2) -> {final:.9f}"),
        practice.Check("it takes 3 positives to pass 90%",
                       result["to_ninety"] == 3,
                       f"{result['to_ninety']} tests reach {result['at_ninety']:.4f} — each "
                       f"positive multiplies the odds by 99, and the prior is that far down"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
