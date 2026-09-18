"""Exercise 5 — the fallback recovers none of the failure it is for.

    Parse three Qwen2.5-VL JSON tool-call outputs into Python dicts. What fails
    for malformed JSON and what recovery strategy does the Qwen cookbook
    recommend?

Reading of the exercise: the lesson's own `parse_tool_call` is run on its own
four examples, and then on two cases it does not ship -- prose containing a
stray brace, and a valid call truncated at every position -- because "what fails
for malformed JSON" is a question about coverage and four hand-picked strings
cannot answer it. The truncation sweep is the one that matters: hitting
`max_tokens` mid-object is the failure mode a production agent actually sees.

**ANSWER: three of the four parse; the truncated one does not.** The clean calls
parse directly, the prose-wrapped one is recovered by slicing from the first `{`
to the last `}`, and `{"tool": "type_text", "text": "hello"` has no closing brace
to slice to, so it returns `{"tool": "PARSE_ERROR", "raw": ...}`.

**FINDING: the brace slice fails on any prose that contains a brace.**
`Clicking {here} at {"tool": "mouse_click", "coords": [1, 2]} now` has a valid
object inside it and returns PARSE_ERROR, because the slice starts at `{here}`
and spans both. The fallback assumes the first `{` in the output belongs to the
tool call.

**FINDING: it recovers 0 of 63 truncations of a valid call, and 0 of 32 of a
nested one.** A prefix of a JSON object almost never ends at a closing brace, so
the slice has nothing to slice to. The recovery that exists is for a failure
mode (chatty preamble) that constrained decoding already solves; the failure
mode it does not address (truncation) is the one that survives constrained
decoding.

**FINDING: the sentinel lives in the tool namespace.** The failure value is
`{"tool": "PARSE_ERROR", "raw": ...}`, in the same field a dispatcher switches
on -- so a parse failure arrives looking like a tool call to a tool named
PARSE_ERROR, and is distinguishable from a model that legitimately emitted that
string only by the presence of the `raw` key.

**ANSWER: the recovery worth having is not string surgery.** Constrain the
decode to the tool schema so malformed output cannot be produced; on a
truncation, detect it (an unbalanced brace depth, not a failed parse) and retry
with a higher limit; on anything else, feed the offending text and the parser's
own error back for one repair turn; and validate against the schema before
dispatch, so a parse that succeeds and a call that is safe to run remain two
different questions.

Structure: `EXAMPLES` is the lesson's own list, `truncations` sweeps every
prefix of a valid call, and `depth` is the balance check the fallback does not
do.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "09-qwen-vl-family-dynamic-fps"
EXAMPLES = (
    '{"tool": "mouse_click", "coords": [1024, 512], "button": "left"}',
    'Sure, clicking at {"tool": "mouse_click", "coords": [800, 400]} now.',
    '{"tool": "type_text", "text": "hello"',
    '{"tool": "scroll", "direction": "down", "amount": 300}',
)
BRACED = 'Clicking {here} at {"tool": "mouse_click", "coords": [1, 2]} now'
NESTED = '{"tool":"a","args":{"x":1},"z":2}'
SENTINEL = "PARSE_ERROR"


def parsed(ref, raw):
    return ref.parse_tool_call(raw)


def failed(result):
    return result.get("tool") == SENTINEL


def truncations(ref, text):
    """How many prefixes of a valid call the fallback can still recover."""
    prefixes = [text[:k] for k in range(1, len(text))]
    return sum(1 for prefix in prefixes if not failed(parsed(ref, prefix))), len(prefixes)


def depth(text):
    """Brace balance -- what distinguishes a truncation from a malformed object."""
    return text.count("{") - text.count("}")


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    results = [parsed(ref, raw) for raw in EXAMPLES]
    recovered, total = truncations(ref, EXAMPLES[0])
    nested_recovered, nested_total = truncations(ref, NESTED)
    return {
        "tools": [row.get("tool") for row in results],
        "ok": sum(1 for row in results if not failed(row)),
        "examples": len(EXAMPLES),
        "failed_index": [i for i, row in enumerate(results) if failed(row)],
        "recovered_by_slice": [i for i, raw in enumerate(EXAMPLES)
                               if not raw.lstrip().startswith("{")
                               and not failed(results[i])],
        "braced": parsed(ref, BRACED).get("tool"),
        "braced_has_object": '{"tool"' in BRACED,
        "truncation": (recovered, total),
        "nested_truncation": (nested_recovered, nested_total),
        "truncation_depth": depth(EXAMPLES[2]),
        "valid_depth": depth(EXAMPLES[0]),
        "sentinel_field": "tool",
        "sentinel_collides": parsed(ref, '{"tool": "PARSE_ERROR"}').get("tool") == SENTINEL,
        "distinguished_by": "raw" in parsed(ref, EXAMPLES[2])
        and "raw" not in parsed(ref, '{"tool": "PARSE_ERROR"}'),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: three of the four parse; the truncated one does not",
            all([result["ok"] == 3, result["examples"] == 4,
                 result["failed_index"] == [2],
                 result["recovered_by_slice"] == [1],
                 result["tools"] == ["mouse_click", "mouse_click", SENTINEL, "scroll"]]),
            f"the parsed tools are {result['tools']}. Two parse directly, example "
            f"{result['recovered_by_slice'][0]} is recovered by slicing from the first "
            f"brace to the last, and example {result['failed_index'][0]} has no closing "
            "brace to slice to",
        ),
        practice.Check(
            "FINDING: the brace slice fails on any prose that contains a brace",
            all([result["braced"] == SENTINEL, result["braced_has_object"]]),
            f"{BRACED!r} holds a valid tool call and returns {result['braced']}, because the "
            "slice starts at the brace in the prose and spans both objects. The fallback "
            "assumes the first brace in the output belongs to the tool call",
        ),
        practice.Check(
            "FINDING: it recovers 0 of 63 truncations, and 0 of 32 of a nested call",
            all([result["truncation"] == (0, 63), result["nested_truncation"] == (0, 32)]),
            f"sweeping every prefix of a valid call recovers {result['truncation'][0]} of "
            f"{result['truncation'][1]}, and {result['nested_truncation'][0]} of "
            f"{result['nested_truncation'][1]} for a call with a nested object. A prefix "
            "almost never ends at a closing brace, so the slice has nothing to slice to -- "
            "and truncation is the failure that survives constrained decoding",
        ),
        practice.Check(
            "FINDING: the sentinel lives in the tool namespace",
            all([result["sentinel_field"] == "tool", result["sentinel_collides"],
                 result["distinguished_by"], result["truncation_depth"] == 1,
                 result["valid_depth"] == 0]),
            f"the failure value puts {SENTINEL!r} in the {result['sentinel_field']!r} field, "
            "the one a dispatcher switches on, and a model that legitimately emits that "
            "string is distinguishable only by the absence of the raw key. Brace depth would "
            f"separate the cases outright -- {result['truncation_depth']} for the truncated "
            f"example against {result['valid_depth']} for a complete one -- and the fallback "
            "never computes it",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
