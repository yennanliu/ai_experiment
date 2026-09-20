"""Exercise 4 — none of the ten reaches the refusal branch.

    Measure refusal rates. Construct ten inputs that should not be extractable
    (a song lyric, a math proof, a blank email) and run them through a real
    provider with strict mode. Count refusals vs hallucinated outputs. This is
    your ground truth for refusal-aware retries.

Reading of the exercise: the ten inputs are constructed as asked and run through
the lesson's own `process_model_output`, which is the only thing here that
classifies an output. A provider is out of reach at T0 -- no network -- and the
exercise says why that matters: it calls the result "ground truth", and ground
truth is exactly the thing a simulator cannot supply. What the ten *can* measure
is what the harness does with them when no provider has spoken, and the answer
is that it never reaches the branch the exercise is about.

**ANSWER: 0 of 10 are classified as refusals.** Seven are `parse_error` and
three are `violation`. The refusal branch is reachable only through the literal
prefix `__REFUSAL__`, which is a sentinel the *provider* emits -- so a song
lyric arriving as a song lyric is not a refusal to this code, it is malformed
JSON.

**FINDING: three of the ten are worse than the seven.** `{}`, `null` and `[]`
are valid JSON, so they clear the parse branch and land in `violation` with
**4, 1 and 1** errors. Those three are indistinguishable, at this layer, from a
model that tried and got the shape wrong -- which is the case the retry logic
is for. A retry on `{}` is reasonable; a retry on a song lyric is a loop.

**FINDING: the sentinel is the whole mechanism, and it is a string prefix.**
`process_model_output` branches on `raw.startswith("__REFUSAL__")` before it
parses anything, so any model output that happens to begin with those eleven
characters is a refusal and any genuine refusal that does not is not. A real
provider signals this out of band -- OpenAI in a typed `refusal` field,
Anthropic in `stop_reason` -- precisely so it cannot collide with content.

**FINDING: so the exercise is right that it needs a provider, for a reason it
does not give.** The count it asks for is refusals against *hallucinated
outputs*, and a hallucination is a well-formed document with invented values.
Exercise 1 already showed this schema accepts a total of 999.0 against a 10.00
line item -- so a hallucinated invoice validates **clean** and is
indistinguishable from a correct one here. The measurement needs a provider not
because refusals are hard to simulate, but because the other arm of the ratio is
invisible to the validator.

Structure: `UNEXTRACTABLE` is the ten inputs, `classify` runs each through the
lesson's own handler, and `hallucinated` is a well-formed invoice with an
invented total, to show which branch it lands in.
"""

from __future__ import annotations

import json
from collections import Counter

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "04-structured-output"
SENTINEL = "__REFUSAL__"
UNEXTRACTABLE = (
    "Yesterday, all my troubles seemed so far away",          # song lyric
    "Let n be even. Then n = 2k for some integer k. QED.",     # math proof
    "",                                                        # blank email
    "   ",                                                     # whitespace only
    "Subject: (no subject)",                                   # header, no body
    "-----BEGIN PGP MESSAGE-----",                             # encrypted blob
    "def f(x): return x + 1",                                  # source code
    "{}",                                                      # valid JSON, empty
    "null",                                                    # valid JSON, null
    "[]",                                                      # valid JSON, wrong type
)
HALLUCINATED = {"customer": "Acme", "line_items": [{"sku": "ABC-1", "qty": 1,
                                                    "unit_usd": 10.0}],
                "total_usd": 999.0, "currency": "USD"}


def classify(ref, raw):
    result = ref.process_model_output(raw, ref.INVOICE_SCHEMA)
    return result.kind, len(result.errors)


def census(ref, inputs=UNEXTRACTABLE):
    return Counter(classify(ref, raw)[0] for raw in inputs)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    counts = census(ref)
    valid_json = {raw: classify(ref, raw) for raw in ("{}", "null", "[]")}
    hallucinated_kind, hallucinated_errors = classify(ref, json.dumps(HALLUCINATED))
    sentinel_kind, _ = classify(ref, f"{SENTINEL} that is a song lyric")
    collision_kind, _ = classify(ref, f"{SENTINEL}{{\"customer\": \"Acme\"}}")
    return {
        "total": len(UNEXTRACTABLE), "counts": dict(counts),
        "refusals": counts["refusal"],
        "parse_errors": counts["parse_error"], "violations": counts["violation"],
        "valid_json": {raw: kind for raw, (kind, _) in valid_json.items()},
        "valid_json_errors": {raw: errors for raw, (_, errors) in valid_json.items()},
        "sentinel_kind": sentinel_kind, "collision_kind": collision_kind,
        "sentinel": SENTINEL, "sentinel_len": len(SENTINEL),
        "hallucinated_kind": hallucinated_kind,
        "hallucinated_errors": hallucinated_errors,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 0 of 10 are classified as refusals",
            all([result["total"] == 10, result["refusals"] == 0,
                 result["parse_errors"] == 7, result["violations"] == 3,
                 result["sentinel_kind"] == "refusal"]),
            f"the ten unextractable inputs classify as {result['counts']} -- "
            f"{result['refusals']} refusals. The branch is reachable only through the literal "
            f"prefix {result['sentinel']}, which is a sentinel the provider emits, so a song "
            "lyric arriving as a song lyric is not a refusal to this code: it is malformed "
            "JSON",
        ),
        practice.Check(
            "FINDING: three of the ten are worse than the seven",
            all([result["valid_json"] == {"{}": "violation", "null": "violation",
                                          "[]": "violation"},
                 result["valid_json_errors"] == {"{}": 4, "null": 1, "[]": 1}]),
            f"{list(result['valid_json'])} are valid JSON, so they clear the parse branch and "
            f"land in violation with {list(result['valid_json_errors'].values())} errors. At "
            "this layer they are indistinguishable from a model that tried and got the shape "
            "wrong -- which is the case the retry logic is for. A retry on {} is reasonable; "
            "a retry on a song lyric is a loop",
        ),
        practice.Check(
            "FINDING: the sentinel is the whole mechanism, and it is a string prefix",
            all([result["sentinel_kind"] == "refusal",
                 result["collision_kind"] == "refusal",
                 result["sentinel_len"] == 11]),
            f"process_model_output branches on raw.startswith({result['sentinel']!r}) before "
            f"it parses anything, so any output beginning with those {result['sentinel_len']} "
            f"characters is a refusal -- even one carrying a JSON body, which classifies as "
            f"{result['collision_kind']}. A real provider signals this out of band, in a "
            "typed refusal field or a stop_reason, precisely so it cannot collide with "
            "content",
        ),
        practice.Check(
            "FINDING: the exercise needs a provider, for a reason it does not give",
            all([result["hallucinated_kind"] == "ok",
                 result["hallucinated_errors"] == 0]),
            f"the count it asks for is refusals against hallucinated outputs, and a "
            f"hallucination is a well-formed document with invented values. An invoice "
            f"claiming {HALLUCINATED['total_usd']} against a "
            f"{HALLUCINATED['line_items'][0]['unit_usd']} line item classifies as "
            f"{result['hallucinated_kind']} with {result['hallucinated_errors']} errors. The "
            "measurement needs a provider not because refusals are hard to simulate, but "
            "because the other arm of the ratio is invisible to the validator",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
