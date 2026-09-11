"""Exercise 1 — faithfulness catches none of the ten.

    **Easy.** Use RAGAS on 10 RAG examples with known hallucinations. Verify the
    faithfulness metric catches each one.

Reading of the exercise: `ragas` is not installed, so the defensible substitute is
the lesson's own `faithfulness`, which `docs/en.md` presents as the stdlib
approximation of the RAGAS metric. Ten examples are built the way the exercise
asks -- a context sentence, a faithful answer, and the same answer with one
content token replaced by a false one, which is the smallest honest hallucination.

The metric catches **0 of 10**. Every hallucinated answer scores **1.00**, and so
does every faithful one; the two sets have identical means and there is no
threshold anywhere in [0, 1] that separates them. The doc's own CI gate value of
0.85 passes all twenty.

The mechanism is the 0.5 in `faithfulness`: a sentence is supported when at least
half its non-stopword tokens appear in the context. The hallucinated claims share
a mean of **0.8404** of their tokens with the context and a minimum of **0.8000**,
clearing the bar by **0.3000** in the worst case. One corrupted token moves the
ratio by the reciprocal of the sentence length -- a mean of **0.1596** here -- so
the nearest case would need two more corrupted tokens to fall below the threshold.
The metric is asking a question about sentence rewriting; a hallucination is a word.

The lesson's own transcript disagrees with the lesson's own code. `main()` prints
"case 1 = hallucinated date -> g-eval drops, faithfulness partial", and case 1's
faithfulness is exactly **1.0**, not partial.

The sign is inverted where it matters. "Marie Curie did not win the Nobel Prize in
Physics in 1903" against a context saying she won scores **1.00** -- `not` is one
token out of nine and is not in the stop list, so negation is invisible. A true
statement in different words, "Cupertino's handset debuted midway through that
summer", scores **0.00**. The metric rewards copying and punishes paraphrase,
which is the failure `docs/en.md` opens by promising to fix.

Structure: `CASES` packs ten context/gold-token/false-token triples; `swap` builds
the hallucinated variant; `score_pairs` runs the lesson's `faithfulness` over both
variants; `ratios` recovers the supported-token fraction the threshold is compared
against; `probes` holds the negation and paraphrase controls.
"""

from __future__ import annotations

import importlib.util
import inspect

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "27-llm-evaluation-frameworks"

CASES = """Apple released the first iPhone on June 29, 2007.|2007|2006
The Eiffel Tower stands 330 metres tall in Paris.|330|224
Marie Curie won the Nobel Prize in Physics in 1903.|won|lost
The Amazon river empties into the Atlantic Ocean.|Atlantic|Pacific
Guido van Rossum created Python in 1991.|1991|1989
Mount Everest rises 8849 metres above sea level.|8849|8611
The Berlin Wall fell in November 1989.|1989|1979
Water boils at 100 degrees celsius at sea level.|celsius|fahrenheit
Apollo 11 landed on the moon in July 1969.|1969|1968
The transformer paper appeared at NeurIPS in 2017.|2017|2014"""
ROWS = tuple(tuple(line.split("|")) for line in CASES.splitlines())
NEGATION = "Marie Curie did not win the Nobel Prize in Physics in 1903."
PARAPHRASE = "Cupertino's handset debuted midway through that summer."
GATE = 0.85


def swap(context, gold, wrong):
    """The hallucinated answer: the context sentence with one content token replaced."""
    return context.replace(gold, wrong)


def score_pairs(ref):
    """Faithfulness of the hallucinated and the faithful answer for each of the ten."""
    bad = [ref.faithfulness(swap(c, g, w), c) for c, g, w in ROWS]
    good = [ref.faithfulness(c, c) for c, _, _ in ROWS]
    return bad, good


def ratios(ref):
    """Supported-token fraction per hallucinated claim, and the per-token step size."""
    fractions, steps = [], []
    for context, gold, wrong in ROWS:
        seen = set(ref.tokenize(context))
        tokens = ref.tokenize(swap(context, gold, wrong))
        fractions.append(sum(1 for t in tokens if t in seen) / len(tokens))
        steps.append(1 / len(tokens))
    return fractions, steps


def lesson_case_one(ref):
    """Faithfulness of `main()`'s own hallucinated-date case, and what it claims."""
    context = "Apple released the first iPhone on June 29, 2007. The moon landing was in 1969."
    answer = "The first iPhone launched on June 29, 2006, shortly after the moon landing."
    return ref.faithfulness(answer, context), "faithfulness partial" in inspect.getsource(ref.main)


def signature_error(ref):
    """What the doc's Step 1 signature `faithfulness(answer, context, llm)` raises."""
    try:
        ref.faithfulness("a", "b", lambda prompt: prompt)
    except TypeError as exc:
        return f"TypeError: {exc}"
    return "no error"


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    bad, good = score_pairs(ref)
    case_one, doc_claim = lesson_case_one(ref)
    return {
        "ragas": importlib.util.find_spec("ragas") is not None,
        "bad": bad, "good": good, "n": len(ROWS),
        "caught": sum(1 for s in bad if s < GATE),
        "ratios": ratios(ref)[0], "steps": ratios(ref)[1],
        "case_one": case_one, "doc_claims_partial": doc_claim,
        "negation": ref.faithfulness(NEGATION, ROWS[2][0]),
        "paraphrase": ref.faithfulness(PARAPHRASE, ROWS[0][0]),
        "grid": sorted({round(s, 4) for s in bad + good}),
        "sig": signature_error(ref),
    }


def verify(result):
    bad, good, n = result["bad"], result["good"], result["n"]
    lo, hi = min(result["ratios"]), sum(result["ratios"]) / n
    return [
        practice.Check(
            "ANSWER: ragas is absent, and the shipped faithfulness catches 0 of 10",
            not result["ragas"] and result["caught"] == 0 and min(bad) == max(good) == 1.0,
            f"`importlib.util.find_spec('ragas')` is None, so the substitute is the lesson's own "
            f"`faithfulness`. All {n} hallucinated answers score {min(bad):.2f} and all {n} "
            f"faithful ones score {max(good):.2f}: {result['caught']} of {n} caught at the doc's "
            f"own CI threshold of {GATE}, and no threshold in [0, 1] separates the two sets",
        ),
        practice.Check(
            "MECHANISM: a hallucination is one token and the bar is half the sentence",
            lo > 0.5 and hi > 0.5,
            f"`faithfulness` supports a claim when >= 0.5 of its non-stopword tokens appear in "
            f"the context. The corrupted claims share a mean {hi:.4f} and a minimum {lo:.4f} of "
            f"their tokens; one swapped token in a six-to-ten token sentence moves that ratio by "
            f"{sum(result['steps']) / n:.4f} on average, so the closest case still clears 0.5 "
            f"by {lo - 0.5:.4f} -- two further corrupted tokens away from being flagged",
        ),
        practice.Check(
            "FINDING: main()'s own transcript contradicts main()'s own code",
            result["doc_claims_partial"] and result["case_one"] == 1.0,
            f"`main()` prints 'case 1 = hallucinated date -> g-eval drops, faithfulness partial'. "
            f"Case 1's faithfulness is {result['case_one']:.2f}. The interpretation the lesson "
            "ships beside its numbers is not the number it ships",
        ),
        practice.Check(
            "FINDING: negation scores 1.00 and a true paraphrase scores 0.00",
            result["negation"] > result["paraphrase"],
            f"'{NEGATION}' against a context saying she won scores {result['negation']:.2f} -- "
            f"`not` is one token of nine and is absent from the stop list. "
            f"'{PARAPHRASE}', which is true, scores {result['paraphrase']:.2f}. It rewards "
            "copying and punishes paraphrase -- the failure the lesson opens by promising to fix",
        ),
        practice.Check(
            "CONTROL: a constant 1.0 scorer is indistinguishable from the metric",
            result["grid"] == [1.0],
            f"over all {2 * n} answers the metric takes {result['grid']} as its only value, so a "
            "scorer that returns 1.0 without reading anything agrees with it on every case. With "
            "one-sentence answers the metric's range is the two-point grid {0.0, 1.0} anyway, so "
            "'verify it catches each one' is ten Bernoulli events that all came out the wrong way",
        ),
        practice.Check(
            "CONTROL: the doc's own Step 1 signature does not exist",
            result["sig"].startswith("TypeError"),
            f"`docs/en.md` Step 1 defines `faithfulness(answer, context, llm)` with an NLI "
            f"pipeline; the shipped one is `faithfulness(answer, context)`. The documented "
            f"form raises {result['sig']}. The NLI backend the write-up describes is not in the "
            "code, and `transformers` is not installed either",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
