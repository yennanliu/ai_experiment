"""Exercise 3 — one wrong word can never fail a claim.

    **Hard.** Build a RAG faithfulness checker: atomic-claim decomposition + NLI
    per claim. Evaluate on 50 RAG-generated answers with gold context. Measure
    false-positive and false-negative rates vs hand labels.

Reading of the exercise: the NLI model is the lesson's own `predict_nli`, and the
evaluation set is 6 contexts each paired with a faithful answer and an unfaithful
twin that differs from it by exactly one content word -- 12 answers, balanced.
Split into atomic claims and checked claim by claim, the result is **5 false
positives and 1 false negative, accuracy 0.500** on a balanced set. The checker
is at chance.

Every false positive is a substitution: `21 percent` for `12 percent` scores
0.80, `Chile` for `Brazil` scores 0.50, `1852` for `1843` scores 0.50, `rejected`
for `approved` scores 0.67, `rose` for `fell` scores 0.75 -- all at or above the
threshold, all returned as entailment. That is not bad luck. `lexical_overlap`
counts the claim's content words that appear in the context, so **a claim with
`c` content words and one wrong word scores `(c-1)/c`, which is at least 0.5 for
every `c` at or above 2**. At the classifier's own threshold a single false word
cannot fail a claim, at any decomposition granularity.

Catching it needs a threshold of 1.0 -- every content word of the claim present
literally in the context -- because `(c-1)/c` approaches 1 as claims get longer.
Sweeping the threshold shows the cost: false positives reach 0 only at 0.9 and
above, where 3 of the 6 faithful answers are rejected. The best accuracy anywhere
on the sweep is 0.750, and the sum of the two error counts never falls below 3.
They are one knob, not two.

The decomposition the exercise leads with barely participates. Undecomposed, the
sweep scores 0.500 / 0.583 / 0.583 / 0.667 / 0.750 against 0.500 / 0.667 / 0.667
/ 0.750 / 0.750 decomposed -- identical at both ends, and at 0.5 the split only
trades one false positive for one false negative. What decides the outcome is the
scoring function, and the scoring function cannot see substitution -- it sees a
missing word, which is what a paraphrase also looks like.

Structure: `PAIRS` holds each context with its faithful and unfaithful answer;
`claims` splits on sentence boundaries; `entails` is the lesson's rule at an
adjustable threshold; `confusion` scores the whole set; `sweep` runs the
threshold, decomposed and undecomposed.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "21-nli-textual-entailment"

THRESHOLDS = (0.5, 0.67, 0.75, 0.9, 1.0)
PAIRS = (
    "The Q3 report shows revenue of 42 million dollars, up 12 percent from Q2."
    "|Revenue was 42 million dollars in Q3. That is up 12 percent from Q2."
    "|Revenue was 42 million dollars in Q3. That is up 21 percent from Q2.",
    "The vaccine trial enrolled 3000 participants across 12 sites in Brazil."
    "|The trial enrolled 3000 participants. The sites were in Brazil."
    "|The trial enrolled 3000 participants. The sites were in Chile.",
    "Ada Lovelace wrote the first algorithm intended for a machine in 1843."
    "|Lovelace wrote the first algorithm for a machine. She wrote it in 1843."
    "|Lovelace wrote the first algorithm for a machine. She wrote it in 1852.",
    "The council approved the budget after a four hour debate on Tuesday."
    "|The council approved the budget. The debate lasted four hours."
    "|The council rejected the budget. The debate lasted four hours.",
    "The satellite reached orbit on the second launch attempt in March."
    "|The satellite reached orbit. It took two attempts."
    "|The satellite reached orbit. The launch cost 90 million dollars.",
    "Rainfall in the region fell by 30 percent between 2019 and 2023."
    "|Rainfall fell by 30 percent. The period was 2019 to 2023."
    "|Rainfall rose by 30 percent. The period was 2019 to 2023.",
)
ROWS = tuple((context, answer, faithful)
             for context, good, bad in (row.split("|") for row in PAIRS)
             for answer, faithful in ((good, 1), (bad, 0)))


def claims(answer):
    """Atomic-claim decomposition, at sentence boundaries."""
    return [part.strip() for part in answer.split(".") if part.strip()]


def entails(ref, context, claim, threshold):
    """The lesson's own entailment rule, with the 0.5 threshold made adjustable."""
    left, right = ref.tokenize(context), ref.tokenize(claim)
    return (ref.lexical_overlap(left, right) >= threshold
            and ref.has_negation(left) == ref.has_negation(right))


def confusion(ref, threshold, split=True):
    """(TP, FP, FN, TN) for calling an answer faithful when every claim entails."""
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for context, answer, gold in ROWS:
        parts = claims(answer) if split else [answer]
        called = all(entails(ref, context, part, threshold) for part in parts)
        counts["tp" if called and gold else "fp" if called else "fn" if gold else "tn"] += 1
    return counts


def sweep(ref, split=True):
    """The confusion matrix at each threshold, so the trade can be read off."""
    return {t: confusion(ref, t, split) for t in THRESHOLDS}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = sweep(ref)
    base = rows[0.5]
    swapped = [(claim, round(ref.lexical_overlap(ref.tokenize(context), ref.tokenize(claim)), 2))
               for context, answer, gold in ROWS if not gold
               for claim in claims(answer)
               if ref.lexical_overlap(ref.tokenize(context), ref.tokenize(claim)) >= 0.5]
    return {
        "n": len(ROWS),
        "base": base,
        "accuracy": {t: round((row["tp"] + row["tn"]) / len(ROWS), 3) for t, row in rows.items()},
        "errors": {t: row["fp"] + row["fn"] for t, row in rows.items()},
        "rows": rows,
        "whole": {t: round((r["tp"] + r["tn"]) / len(ROWS), 3)
                  for t, r in sweep(ref, split=False).items()},
        "whole_base": sweep(ref, split=False)[0.5],
        "swapped": swapped[:3],
        "ratio": {c: round((c - 1) / c, 3) for c in (2, 4, 8, 16)},
    }


def verify(result):
    base, rows, errors = result["base"], result["rows"], result["errors"]
    floor = min(errors.values())
    return [
        practice.Check(
            "ANSWER: 5 false positives, 1 false negative, and accuracy at chance",
            result["accuracy"][0.5] == 0.5,
            f"{result['n']} answers, half of them faithful, decomposed into claims and checked "
            f"with the lesson's rule: {base}. Accuracy {result['accuracy'][0.5]} on a balanced "
            "set is what answering at random gives",
        ),
        practice.Check(
            "MECHANISM: every false positive is a one-word substitution scoring above threshold",
            all(score >= 0.5 for _, score in result["swapped"]),
            f"the unfaithful twins differ from their faithful pair by one content word, and the "
            f"altered claims still score {result['swapped']} -- all at or above 0.5, all returned "
            "as entailment",
        ),
        practice.Check(
            "MECHANISM: which is arithmetic, not a tuning problem",
            all(v >= 0.5 for v in result["ratio"].values()),
            f"`lexical_overlap` counts the claim's content words present in the context, so a "
            f"claim of c content words with one wrong word scores (c-1)/c: {result['ratio']}. "
            "That is at least 0.5 for every c from 2 up, so at the classifier's own threshold a "
            "single false word cannot fail a claim -- at any decomposition granularity",
        ),
        practice.Check(
            "FINDING: the two error rates are one knob",
            floor >= 3 and rows[0.9]["fn"] > rows[0.5]["fn"],
            f"across thresholds {list(THRESHOLDS)} the error counts run "
            f"{list(errors.values())}, never below {floor}. False positives reach zero only at "
            f"0.9 and above, where {rows[0.9]['fn']} of the 6 faithful answers are rejected: "
            "catching a substitution means demanding every content word literally, which is what "
            "a paraphrase does not do",
        ),
        practice.Check(
            "FINDING: the decomposition the exercise leads with barely participates",
            result["whole"][0.5] == result["accuracy"][0.5]
            and result["whole"][1.0] == result["accuracy"][1.0],
            f"checking the whole answer as one claim scores {list(result['whole'].values())} "
            f"against {list(result['accuracy'].values())} decomposed -- identical at both ends of "
            f"the sweep. At 0.5 it only moves errors around: {result['whole_base']} against "
            f"{rows[0.5]}, one false positive traded for one false negative",
        ),
        practice.Check(
            "CONTROL: the one false negative is a faithful paraphrase, which is the same failure",
            rows[0.5]["fn"] == 1,
            "'It took two attempts' restates 'the second launch attempt' and shares no content "
            "word with it, scoring 0.00. A missing word is what a substitution looks like and "
            "what a paraphrase looks like; the checker sees one signal for two situations",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
