"""Exercise 3 — the decoder says stop where the validator says invalid.

    Implement a more realistic constrained decoding simulator. Given a JSON
    Schema and a vocabulary of 100 tokens (letters, digits, punctuation,
    keywords), walk through generation step by step, masking invalid tokens at
    each position. Measure what percentage of the vocabulary is valid at each
    step.

Reading of the exercise: "masking invalid tokens" needs a decidable notion of
invalid, so the mask is a prefix automaton over an enumerated language -- every
serialisation of PRODUCT_SCHEMA over a fixed value set and every key ordering,
30 documents. A token is valid after a prefix exactly when some document starts
with prefix + token, and `<EOS>` exactly when the prefix is a whole document.
That is exact for the language it enumerates, which is what makes a percentage
mean something.

**ANSWER: the mask is 1% wide almost everywhere.** Along the canonical path the
valid fraction of the 100-token vocabulary is exactly 1 token at 21 of 27
positions, peaking at 7% right after the opening quote -- the only place where
more than one property name could still begin -- and at 6% and 4% at the two
later key boundaries. Mean 1.63%. Constrained decoding against a closed schema
is not a soft preference; it is a single legal token nearly every step.

**FINDING: the lesson's `next_valid_tokens` never reads its `schema`
argument.** Its output is identical for `PRODUCT_SCHEMA` and `{}` at every
state on the path. It is a JSON-grammar masker, so it cannot know that
`"produkt"` is not a key, and it admits key names the schema has no property
for.

**FINDING: it stops where the validator fails.** `next_valid_tokens` returns
`['<EOS>']` for `{"product": "Sony"}` -- because the string parses -- while
`validate_schema` on the same text reports two required fields missing. The
decoder's stop condition and the schema disagree on the same document.

**FINDING: for some reachable states it returns `['any']`** -- no mask at all,
100% of the vocabulary -- and its class labels ("0-9", "a-z") expand to 10 and
26 tokens, so its mean permitted fraction along the path is 30.46%
against the schema-aware 1.63% -- nineteen times as wide.

**CONTROL: the schema-aware mask cannot emit `<EOS>` early.** It permits the
end token at 1 of the path's positions against the character masker's 2, and
the one it permits is the complete document.

Structure: `VOCAB` is the 100 tokens, `language()` enumerates the documents,
`mask` is the prefix automaton, and `lesson_mask` expands the lesson's classes.
"""

from __future__ import annotations

import itertools
import json
import statistics

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "03-structured-outputs"
STRUCT = list('{}[]":, ') + [".", "-", "_"]
WORDS = ["product", "price", "in_stock", "categories", "Sony", "WH", "1000", "XM5",
         "audio", "headphones", "true", "false", "null", "computers", "peripherals",
         "MacBook", "Pro", "Keychron", "Q1", "stock", "name", "sku", "tier", "rating",
         "value", "label", "count"]
VOCAB = STRUCT + list("0123456789") + list("abcdefghijklmnopqrstuvwxyz") + \
    list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + WORDS
VALUES = {"product": ['"Sony"'], "price": ["348"], "in_stock": ["true"],
          "categories": ['["audio"]']}
REQUIRED = ["product", "price", "in_stock"]
PATH = ['{', '"', 'product', '"', ':', ' ', '"', 'Sony', '"', ',', ' ', '"', 'price',
        '"', ':', ' ', '348', ',', ' ', '"', 'in_stock', '"', ':', ' ', 'true', '}']


def language():
    """Every serialisation the schema admits, over a fixed value set."""
    documents = set()
    for optional in ([], ["categories"]):
        for order in itertools.permutations(REQUIRED + optional):
            pairs = [f'"{key}": {VALUES[key][0]}' for key in order]
            documents.add("{" + ", ".join(pairs) + "}")
    return sorted(documents)


def mask(prefix, documents):
    """The schema-aware mask: tokens that keep the prefix completable."""
    allowed = [t for t in VOCAB if any(d.startswith(prefix + t) for d in documents)]
    return allowed + (["<EOS>"] if prefix in documents else [])


def lesson_mask(ref, prefix, schema):
    """The lesson's classes, expanded to token counts over the same vocabulary."""
    sizes = {"0-9": 10, "a-z": 26, "any": len(VOCAB)}
    tokens = ref.next_valid_tokens(prefix, schema)
    return sum(sizes.get(t, 1) for t in tokens), tokens


def walk(documents):
    prefixes = ["".join(PATH[:i]) for i in range(len(PATH) + 1)]
    return prefixes, [len(mask(p, documents)) for p in prefixes]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    documents = language()
    prefixes, sizes = walk(documents)
    partial = '{"product": "Sony"}'
    stops = prefixes + [partial]
    lesson = [lesson_mask(ref, p, ref.PRODUCT_SCHEMA) for p in stops]
    return {
        "vocab": len(VOCAB), "documents": len(documents),
        "percent": [round(100 * n / len(VOCAB), 1) for n in sizes],
        "mean": round(100 * statistics.mean(sizes) / len(VOCAB), 2),
        "lesson_mean": round(100 * statistics.mean(n for n, _ in lesson) / len(VOCAB), 2),
        "schema_blind": all(ref.next_valid_tokens(p, ref.PRODUCT_SCHEMA)
                            == ref.next_valid_tokens(p, {}) for p in stops),
        "any_states": [p for p, (_, t) in zip(stops, lesson) if t == ["any"]],
        "early_stop": ref.next_valid_tokens(partial, ref.PRODUCT_SCHEMA),
        "validator": ref.validate_schema(json.loads(partial), ref.PRODUCT_SCHEMA),
        "early_stop_allowed": "<EOS>" in mask(partial, documents),
        "lesson_eos": sum("<EOS>" in t for _, t in lesson),
        "schema_eos": sum("<EOS>" in mask(p, documents) for p in stops),
        "stops": len(stops),
    }


def verify(result):
    percent = result["percent"]
    return [
        practice.Check(
            "ANSWER: the mask is tight at structure and loose only inside strings",
            all([result["vocab"] == 100, result["documents"] == 30, result["mean"] < 3]),
            f"over {result['vocab']} tokens and the {result['documents']} serialisations "
            f"the schema admits, the valid fraction along the canonical path runs "
            f"{percent[:8]}...{percent[-3:]}, mean {result['mean']}%. Constrained decoding "
            "is a structural constraint almost everywhere",
        ),
        practice.Check(
            "FINDING: the lesson's next_valid_tokens never reads its schema argument",
            result["schema_blind"],
            "its output is identical for PRODUCT_SCHEMA and {} at every state on the path, "
            "so it is a JSON-grammar masker and not a schema one: it cannot know that "
            "'produkt' is not a property, and admits any key name the grammar allows",
        ),
        practice.Check(
            "FINDING: it stops where the validator fails",
            all([result["early_stop"] == ["<EOS>"], len(result["validator"]) == 2,
                 not result["early_stop_allowed"]]),
            f"for '{{\"product\": \"Sony\"}}' it returns {result['early_stop']} -- the "
            f"string parses -- while validate_schema on the same text reports "
            f"{result['validator']}. The schema-aware mask refuses the end token there, "
            "because no document in the language stops at that prefix",
        ),
        practice.Check(
            "FINDING: it returns ['any'] at reachable states, which is no mask at all",
            all([result["any_states"], result["lesson_mean"] > 2 * result["mean"]]),
            f"{len(result['any_states'])} states on the path return ['any'] -- 100% of the "
            f"vocabulary -- and its classes '0-9' and 'a-z' expand to 10 and 26 tokens, so "
            f"its mean permitted fraction is {result['lesson_mean']}% against the "
            f"schema-aware {result['mean']}%",
        ),
        practice.Check(
            "CONTROL: the schema-aware mask cannot end the document early",
            all([result["schema_eos"] == 1, result["lesson_eos"] == 2]),
            f"over the path's {result['stops']} positions plus the early-closing "
            f"'{{\"product\": \"Sony\"}}', the end token is permitted "
            f"{result['schema_eos']} time by the schema-aware mask and "
            f"{result['lesson_eos']} times by the character one. The one it permits is "
            "the complete document. "
            "Required-field completeness is a property of the language, not of the parser",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
