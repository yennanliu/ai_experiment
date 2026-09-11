"""Exercise 1 — valid JSON is not compliance, and 100 reviews is an interval.

    **Easy.** Prompt a small open-weights model (e.g., Llama-3.2-3B) without
    constrained decoding for `Review(sentiment, confidence, evidence_span)`.
    Measure the fraction that parse as valid JSON on 100 reviews.

Reading of the exercise: no open-weights runtime is installed, so the rate itself
cannot be measured. What can be established is what that number would and would
not tell you, and both answers are unfavourable.

The metric over-reports. Over 14 outputs covering the failure modes small models
actually produce -- fenced blocks, prose preambles, single quotes, trailing
commas, missing fields, string confidences, arrays, `{}` -- **10 parse and only 2
satisfy the schema**. Eight outputs are valid JSON and not a `Review`:
`"positive"` is a complete JSON document, and so is `{}`. Parsing is necessary
and nowhere near sufficient, so a compliance rate reported as a parse rate is an
upper bound quoted as a measurement.

It is not even parser-independent. `json.loads` accepts `NaN` and `Infinity`,
which RFC 8259 does not, so one of the 14 counts as valid under Python's default
and invalid under a strict reader: 10 against 9 on the same corpus.

And 100 reviews does not resolve the question that motivates the exercise. By the
rule of three, a clean 100 for 100 bounds the true failure rate only at **3.0%**
-- three bad rows per hundred, from a run that looked perfect. The Wilson
interval for 80/100 spans 71.1% to 86.7%. Certifying 99% needs 300 clean samples;
99.9% needs 3,000. The lesson's own demonstration makes the same mistake in
miniature: `generate_unconstrained` has an exactly computable compliance rate of
0.0031863, and `main()` prints its result at n=20, where an all-invalid run is
the modal outcome at 93.8%.

One more thing the schema itself decides: `Review(sentiment, confidence,
evidence_span)` puts the answer before the evidence, which is exactly the field
order the lesson's own pitfall section warns against. Under constrained decoding
that ordering stops being a default and becomes enforced.

Structure: `CASES` is the labelled corpus; `reads` parses with Python's default
JSON or with RFC 8259's constant list, `complies` checks the declared schema;
`wilson` and the rule of three size the interval.
"""

from __future__ import annotations

import importlib.util
import json
import math

from harness import parity, practice

PHASE, LESSON = "05-nlp-foundations-to-advanced", "20-structured-outputs-constrained-decoding"

UNAVAILABLE = ("transformers", "torch", "llama_cpp", "vllm", "outlines")
SENTIMENTS = ("positive", "negative", "neutral")
ASKED = 100
CASES = (
    '{"sentiment": "positive", "confidence": 0.91, "evidence_span": "loved every minute"}',
    '```json\n{"sentiment": "negative", "confidence": 0.8, "evidence_span": "dull"}\n```',
    'Here is the JSON:\n{"sentiment": "neutral", "confidence": 0.5, "evidence_span": "fine"}',
    "{'sentiment': 'positive', 'confidence': 0.7, 'evidence_span': 'great'}",
    '{"sentiment": "positive", "confidence": 0.7, "evidence_span": "great",}',
    '{"sentiment": "positive"}',
    '{"sentiment": "Positive!", "confidence": "high", "evidence_span": "loved it"}',
    '{"sentiment": "positive", "confidence": 91, "evidence_span": "loved it"}',
    '[{"sentiment": "positive", "confidence": 0.9, "evidence_span": "loved it"}]',
    '"positive"',
    '{"sentiment": "positive", "confidence": 0.8, "evidence_span": null}',
    '{"sentiment": "positive", "confidence": NaN, "evidence_span": "loved it"}',
    "{}",
    '{"sentiment": "negative", "confidence": 0.95, "evidence_span": "wooden dialogue",'
    ' "reasoning": "the review complains about the script"}',
)


def reads(text, strict=False) -> bool:
    """Valid JSON -- to Python's own reader, or to one honouring RFC 8259's constant list."""
    def reject(name):
        raise ValueError(name)
    try:
        json.loads(text, parse_constant=reject if strict else float)
    except ValueError:
        return False
    return True


def complies(text) -> bool:
    """A `Review`: the three declared fields, with the declared types and range."""
    if not reads(text):
        return False
    obj = json.loads(text)
    if not isinstance(obj, dict) or obj.get("sentiment") not in SENTIMENTS:
        return False
    score, span = obj.get("confidence"), obj.get("evidence_span")
    return (isinstance(score, float | int) and not isinstance(score, bool)
            and score == score and 0.0 <= score <= 1.0 and isinstance(span, str))


LENIENT = tuple(i for i, case in enumerate(CASES) if reads(case))
STRICT = tuple(i for i, case in enumerate(CASES) if reads(case, True))
SCHEMA = tuple(i for i, case in enumerate(CASES) if complies(case))
DISPUTED = tuple(i for i in range(len(CASES)) if (i in LENIENT) != (i in STRICT))


def wilson(hits, total, z=1.96):
    """The 95% interval the exercise's fraction actually names."""
    rate, denom = hits / total, 1 + z * z / total
    centre = rate + z * z / (2 * total)
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total))
    return round((centre - half) / denom * 100, 1), round((centre + half) / denom * 100, 1)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    alphabet = list("0123456789-")
    exact = (10 / 11) ** 10 * (1 / 11) ** 2
    toy = sum(1 for seed in range(1000)
              if ref.re.fullmatch(ref.PHONE_REGEX, ref.generate_unconstrained(alphabet, 12, seed)))
    return {
        "absent": [n for n in UNAVAILABLE if importlib.util.find_spec(n) is None],
        "n": len(CASES),
        "parse": len(LENIENT),
        "strict": len(STRICT),
        "schema": len(SCHEMA),
        "parse_only": len(LENIENT) - len(SCHEMA),
        "disputed": DISPUTED,
        "scalar": reads(CASES[9]) and not complies(CASES[9]),
        "empty": reads(CASES[12]) and not complies(CASES[12]),
        "asked": ASKED, "clean": wilson(ASKED, ASKED), "eighty": wilson(80, ASKED),
        "rule_of_three": round(300 / ASKED, 1), "for_99": 300, "for_999": 3000,
        "fields": ("sentiment", "confidence", "evidence_span"),
        "exact": round(exact, 7), "toy": toy, "quiet": round((1 - exact) ** 20 * 100, 1),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the metric the exercise names over-reports compliance",
            result["schema"] < result["parse"],
            f"{result['absent']} are all absent, so no rate can be measured -- but over "
            f"{result['n']} outputs covering the failure modes small models produce, "
            f"{result['parse']} parse and {result['schema']} satisfy Review{result['fields']}: "
            f"{result['parse_only']} are valid JSON and not a Review, so a compliance rate "
            "reported as a parse rate is an upper bound quoted as a measurement",
        ),
        practice.Check(
            "MECHANISM: a bare scalar is a complete JSON document",
            result["scalar"] and result["empty"],
            "`\"positive\"` parses, and so does `{}`. Neither carries a sentiment, a confidence "
            "or a span. The gap is not exotic output -- JSON's grammar and the schema's differ",
        ),
        practice.Check(
            "FINDING: the parse rate is not even parser-independent",
            result["disputed"] and result["strict"] < result["parse"],
            f"`json.loads` accepts NaN and Infinity, which RFC 8259 omits, so case "
            f"{result['disputed'][0]} is valid under Python's default and invalid under a strict "
            f"reader: {result['parse']} against {result['strict']} on the same corpus",
        ),
        practice.Check(
            "FINDING: 100 reviews buys an interval, not a fraction",
            result["clean"][0] < 99.0,
            f"a clean {result['asked']} for {result['asked']} gives a 95% interval of "
            f"{result['clean']} -- the rule of three bounds the failure rate only at "
            f"{result['rule_of_three']}%, three bad rows per hundred from a perfect-looking run. "
            f"80 correct gives {result['eighty']}; 99% needs {result['for_99']} clean samples "
            f"and 99.9% needs {result['for_999']}",
        ),
        practice.Check(
            "MECHANISM: the lesson's own baseline reports a number its sample size cannot carry",
            result["quiet"] > 90.0,
            f"`generate_unconstrained` over an 11-character alphabet has an exactly computable "
            f"compliance rate of {result['exact']} -- {result['toy']} of 1000 measured -- and "
            f"`main()` prints at n=20, where an all-invalid run happens {result['quiet']}% of the "
            "time. The printed 0/20 is the modal outcome, not evidence",
        ),
        practice.Check(
            "CONTROL: the schema orders its fields the way the lesson warns against",
            result["fields"].index("sentiment") < result["fields"].index("evidence_span"),
            "the lesson's pitfall section says 'Put answer before reasoning, and the model "
            "commits to an answer before it thinks' -- then the exercise asks for "
            f"Review{result['fields']}, answer first. Constrained decoding makes that enforced",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
