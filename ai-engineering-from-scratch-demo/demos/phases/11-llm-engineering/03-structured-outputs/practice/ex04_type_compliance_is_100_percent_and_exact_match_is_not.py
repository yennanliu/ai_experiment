"""Exercise 4 — type compliance is 100%, and it is the fallback that guarantees it.

    Build an extraction eval suite. Create 50 product descriptions with
    hand-labeled JSON outputs. Run your extraction pipeline on all 50 and
    measure exact match, field-level accuracy, and type compliance. Identify
    which fields are hardest to extract correctly.

Reading of the exercise: the 50 descriptions are generated from a labelled
template table rather than written out, so the label is the generator's input
and cannot drift from the text. The pipeline run is the lesson's own
`extract_with_retry` against `PRODUCT_SCHEMA`, with its progress printing
swallowed. Exact match is dict equality, field accuracy is per key, and type
compliance is `validate_schema` returning no errors.

**ANSWER: type compliance 50/50, exact match 0/50.** Every output the pipeline
produces validates and every one of them is wrong. The two metrics the exercise
puts side by side are not merely uncorrelated -- one is at its floor while the
other is at its ceiling, and the ceiling is what the fallback guarantees.

**MECHANISM: the extractor has four outputs.** `simulate_llm_extraction`
branches on the substrings "headphones"/"sony", "laptop"/"macbook" and
"keyboard", and returns one fixed document per branch plus one fallback. Over
50 distinct descriptions it emits 4 distinct documents, so field accuracy is
bounded by how many descriptions happen to be about three products.

**FINDING: the fallback is schema-valid.** `{"product": "Unknown", "price": 0,
"in_stock": false}` has every required field and a non-negative price, so
`validate_schema` returns `[]`. The pipeline's failure mode is a valid
document, which is exactly the failure mode a schema cannot catch.

**FINDING: the hardest field is `price`, at 0 of 50, because no digit in the
description reaches the output.** The three branch documents carry hard-coded
prices and the fallback carries 0. `product` scores 15 -- exactly the five
descriptions each of the three names the branch table hard-codes -- and
`categories` 20, on the two families whose fixed category list happens to match.
`in_stock` leads at 25, entirely by accident: it is a coin flip with a constant.

**FINDING: the retry path is dead.** `extract_with_retry` returns on the first
parse that validates, and all four branch documents validate, so `attempt` is
never above 0 across all 50 runs -- and the `attempt >= 1` variant the Sony
branch carries, which drops `categories`, is unreachable.

Structure: `make_corpus` generates the 50 labelled pairs, `evaluate` runs the
lesson's pipeline over them, and `attempts` records what the retry loop asked.
"""

from __future__ import annotations

import contextlib
import io
import json

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "03-structured-outputs"
FIELDS = ["product", "price", "in_stock", "categories"]
# (name, category, matched by the lesson's branch table?)
CATALOGUE = [
    ("Sony WH-1000XM5", "audio", True), ("Sony WF-1000XM4", "audio", True),
    ("MacBook Pro 16", "computers", True), ("MacBook Air 13", "computers", True),
    ("Keychron Q1", "peripherals", True), ("Keychron K8", "peripherals", True),
    ("Dell U2724D", "monitors", False), ("Logitech MX Master", "peripherals", False),
    ("Anker 737", "power", False), ("Elgato Wave 3", "audio", False),
]
NOUN = {"audio": "headphones", "computers": "laptop", "peripherals": "keyboard",
        "monitors": "monitor", "power": "power bank"}


def make_corpus(n=50):
    """50 labelled (description, expected JSON) pairs, generated so they cannot drift."""
    corpus = []
    for i in range(n):
        name, category, _ = CATALOGUE[i % len(CATALOGUE)]
        price, stocked = 99.0 + 50 * (i % 7), i % 2 == 0
        noun = NOUN[category] if i % len(CATALOGUE) < 6 else category
        text = (f"The {name} {noun} is priced at ${price:.0f} and is "
                f"{'available now' if stocked else 'sold out'}.")
        corpus.append((text, {"product": name, "price": price, "in_stock": stocked,
                              "categories": [category]}))
    return corpus


def extract(ref, text):
    """The lesson's pipeline, with its progress printing swallowed."""
    with contextlib.redirect_stdout(io.StringIO()):
        return ref.extract_with_retry(text, ref.PRODUCT_SCHEMA)


def field_hits(got, want):
    return {key: got is not None and got.get(key) == want.get(key) for key in FIELDS}


def tally(outputs, corpus):
    hits = [field_hits(got, want) for got, (_, want) in zip(outputs, corpus)]
    return {"exact": sum(got == want for got, (_, want) in zip(outputs, corpus)),
            "by_field": {key: sum(h[key] for h in hits) for key in FIELDS},
            "distinct": len({json.dumps(got, sort_keys=True) for got in outputs}),
            "fallback": sum(got.get("product") == "Unknown" for got in outputs)}


def evaluate(ref, corpus):
    outputs = [extract(ref, text) for text, _ in corpus]
    valid = [got is not None and ref.validate_schema(got, ref.PRODUCT_SCHEMA) == []
             for got in outputs]
    return {**tally(outputs, corpus), "compliant": sum(valid), "total": len(corpus)}


def attempts(ref, corpus):
    """What the retry loop actually asked the extractor for."""
    seen, original = [], ref.simulate_llm_extraction

    def spy(text, schema, attempt=0):
        seen.append(attempt)
        return original(text, schema, attempt)

    try:
        ref.simulate_llm_extraction = spy
        for text, _ in corpus:
            extract(ref, text)
    finally:
        ref.simulate_llm_extraction = original
    return sorted(set(seen))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    corpus = make_corpus()
    fallback = json.loads(ref.simulate_llm_extraction("nothing here", ref.PRODUCT_SCHEMA))
    return {**evaluate(ref, corpus), "attempts": attempts(ref, corpus),
            "fallback_doc": fallback,
            "fallback_errors": ref.validate_schema(fallback, ref.PRODUCT_SCHEMA),
            "retry_variant": sorted(json.loads(
                ref.simulate_llm_extraction("sony", ref.PRODUCT_SCHEMA, 1)))}


def verify(result):
    by_field = result["by_field"]
    worst = min(by_field, key=by_field.get)
    return [
        practice.Check(
            "ANSWER: type compliance 50 of 50, exact match 0 of 50",
            all([result["compliant"] == result["total"], result["exact"] == 0]),
            f"every one of the {result['total']} outputs validates against PRODUCT_SCHEMA "
            f"and every one of them is wrong. One metric is at its ceiling and the other "
            "at its floor, and the ceiling is what the fallback guarantees",
        ),
        practice.Check(
            "MECHANISM: the extractor has four outputs for any input",
            result["distinct"] == 4,
            f"`simulate_llm_extraction` branches on 'headphones'/'sony', "
            f"'laptop'/'macbook' and 'keyboard', so {result['total']} distinct "
            f"descriptions produce {result['distinct']} distinct documents. Field accuracy "
            "is bounded by how many descriptions happen to be about three products",
        ),
        practice.Check(
            "FINDING: the fallback is schema-valid, so failure looks like success",
            all([result["fallback_errors"] == [], result["fallback"] > 0]),
            f"{result['fallback_doc']} has every required field and a non-negative price, "
            f"so validate_schema returns {result['fallback_errors']}. It is returned for "
            f"{result['fallback']} of the {result['total']} descriptions, each of which "
            "is a schema-valid wrong answer -- the failure a schema cannot catch",
        ),
        practice.Check(
            "FINDING: the hardest field is price, at 0 of 50",
            all([worst == "price", by_field["price"] == 0,
                 by_field["in_stock"] == max(by_field.values())]),
            f"per-field hits out of {result['total']}: {by_field}. No digit in any "
            f"description reaches the output -- the three branch documents carry hard-coded "
            f"prices and the fallback carries 0. `product` scores exactly the five "
            "descriptions each of the three hard-coded names, and `in_stock` leads by "
            "accident, being a coin flip scored against a constant",
        ),
        practice.Check(
            "FINDING: the retry path is dead across all fifty runs",
            all([result["attempts"] == [0], "categories" not in result["retry_variant"]]),
            f"`extract_with_retry` returns on the first parse that validates and all four "
            f"branch documents validate, so the attempt values it ever asked for are "
            f"{result['attempts']}. The attempt>=1 variant the Sony branch carries -- "
            f"{result['retry_variant']}, without categories -- is unreachable",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
