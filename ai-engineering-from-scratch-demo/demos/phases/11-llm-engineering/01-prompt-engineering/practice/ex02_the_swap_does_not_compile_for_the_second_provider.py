"""Exercise 2 — the swap does not compile, and three of the four metrics misreport.

    Replace `simulate_llm_call` with real API calls to at least two providers
    (OpenAI and Anthropic free tiers work). Run the same prompt across both and
    measure: response length, format compliance, keyword coverage, and latency.
    Document which model follows instructions more precisely.

Reading of the exercise: no key is available here, so the half that can be run
is run. `live_call` is the drop-in replacement, signature-identical to
`simulate_llm_call`; whether the lesson's payloads are *sendable* is decided
against the installed SDKs, not against a guess; and the four metrics are
graded as instruments, on replies whose correct score is known by construction.

**ANSWER: the replacement is a drop-in for one provider and does not type-check
for the other.** `format_openai_request` emits 4 keys and `openai`'s
`chat.completions.create` accepts all 4. `format_anthropic_request` emits 5,
and `temperature` is not one of the 22 parameters of `anthropic`'s
`messages.create` -- so the "recommended temp" that every one of the 10 rows in
`PROMPT_PATTERNS` carries is unsendable to the second provider the exercise
names. `format_google_request` emits `model`, `contents`, `generationConfig`:
two belong to no SDK here, and `model` is a URL path segment of
`generateContent`, not a body field.

**FINDING: latency is the one metric the harness already measures and discards.**
`run_prompt_test` times the call into `wall_time_ms`; `compare_models` reports
`api_latency_ms`, which the callee supplies. Swap in real calls and the reported
number is still whatever the adapter chose to put there.

**FINDING: format compliance passes the empty answer and fails the real one.**
`json.loads` is run on the whole reply, so `null`, `42`, `"positive"` and `[]`
are valid JSON and score 1.0, while the object the few-shot test actually asked
for scores 0.0 as soon as a fence or a "Sure! Here is the JSON:" precedes it.
Six of twelve reply shapes are classified wrongly.

**FINDING: keyword coverage is a case-folded substring test.** Required keyword
"API" is found inside "therapies" and "key" inside "monkey", so a reply using
neither word scores coverage 1.0.

**FINDING: "more precisely" is not comparable across the suite.** Length is
counted in whitespace words, so a minified 1 KB JSON reply is one word and
clears `max_words: 200`; and the composite divides by however many criteria keys
a test happened to set -- 3, 2, 2, 1, 2 across the lesson's own five.

Structure: `live_call` is the replacement, `PARAMS` reads the two SDKs'
signatures, and `SHAPES` is the labelled reply set the metrics are graded on.
"""

from __future__ import annotations

import inspect
import json
import os

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "01-prompt-engineering"

OBJ = '{"sentiment": "positive"}'
# (reply, is it the JSON object the few-shot test asked for?)
SHAPES = [(OBJ, True), ('{"sentiment": "mixed", "food": null}', True), ("null", False),
          ("42", False), ('"positive"', False), ("[]", False), (f"```json\n{OBJ}\n```", True),
          (f"Sure! Here is the JSON:\n{OBJ}", True), ('{"sentiment": "positive",}', False),
          ("positive", False), ('{\n  "sentiment": "positive"\n}', True), ("", False)]
DECOYS = [("Therapies help the monkey", ["API", "key"]),
          ("Rapid capital turkey", ["API", "key"])]


def live_call(model_name, request):
    """The drop-in: same signature and return shape as `simulate_llm_call`."""
    import time
    ref = parity.load_reference(PHASE, LESSON, "prompt_engineering")
    start = time.time()
    if ref.MODEL_CONFIGS[model_name]["provider"] == "openai":
        import openai
        reply = openai.OpenAI().chat.completions.create(**request)
        text, used = reply.choices[0].message.content, reply.usage
        tokens = {"prompt": used.prompt_tokens, "completion": used.completion_tokens}
    else:
        import anthropic
        reply = anthropic.Anthropic().messages.create(
            **{k: v for k, v in request.items() if k != "temperature"})
        text, used = reply.content[0].text, reply.usage
        tokens = {"prompt": used.input_tokens, "completion": used.output_tokens}
    tokens["total"] = tokens["prompt"] + tokens["completion"]
    return {"response": text, "tokens_used": tokens, "finish_reason": "stop",
            "latency_ms": round((time.time() - start) * 1000, 1)}  # the real wall clock


def sdk_params(module_path, attribute):
    module = __import__(module_path, fromlist=["x"])
    return set(inspect.signature(getattr(module, attribute).create).parameters) - {"self"}


def sendability(ref):
    """Which of the lesson's three payloads the two installed SDKs would accept."""
    prompt = ref.build_prompt("persona", ref.TEST_SUITE[0]["variables"])
    keys = {name: set(fmt(prompt)) for name, fmt in ref.FORMATTERS.items()}
    accepted = {"openai": sdk_params("openai.resources.chat.completions", "Completions"),
                "anthropic": sdk_params("anthropic.resources.messages", "Messages")}
    return {
        "keys": {n: sorted(v) for n, v in keys.items()},
        "rejected": {n: sorted(keys[n] - accepted[n]) for n in accepted},
        "google_unknown": sorted(keys["google"] - accepted["openai"] - accepted["anthropic"]),
        "temperatures": len({p["temperature"] for p in ref.PROMPT_PATTERNS.values()}),
        "patterns": len(ref.PROMPT_PATTERNS),
        "signature": (str(inspect.signature(live_call))
                      == str(inspect.signature(ref.simulate_llm_call))),
    }


def format_metric(ref):
    """`expected_format: json` against twelve labelled reply shapes."""
    ok = [ref.score_response(t, {"expected_format": "json"})["format_valid"] for t, _ in SHAPES]
    return {"format_right": sum(a == b for a, (_, b) in zip(ok, SHAPES)),
            "format_shapes": len(SHAPES),
            "empty_passes": [t for (t, want), got in zip(SHAPES, ok) if got > want],
            "fenced_fails": [t for (t, want), got in zip(SHAPES, ok) if want > got]}


def coverage_and_length(ref):
    """The other two metrics, on inputs whose correct score is known already."""
    dense = json.dumps({f"k{i}": "v" * 8 for i in range(64)}, separators=(",", ":"))
    return {"decoy_texts": [t for t, _ in DECOYS], "minified_chars": len(dense),
            "decoys": [ref.score_response(t, {"required_keywords": k})["keyword_coverage"]
                       for t, k in DECOYS],
            "minified": ref.score_response(dense, {"max_words": 200}),
            "denominators": [len(t["criteria"]) for t in ref.TEST_SUITE]}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "prompt_engineering")
    case = ref.TEST_SUITE[0]
    reported = ref.compare_models(ref.run_prompt_test(
        ref.build_prompt("persona", case["variables"]), ["gpt-4o"]), case["criteria"])[0]["gpt-4o"]
    fields = sorted(reported)
    return {**sendability(ref), **format_metric(ref), **coverage_and_length(ref),
            "reported_latency": reported["latency_ms"],
            "canned_latency": ref.simulate_llm_call("gpt-4o", {})["latency_ms"],
            "reported_fields": fields, "wall_reported": any("wall" in f for f in fields),
            "live": bool(os.environ.get("OPENAI_API_KEY"))}   # the live path, if a key lands


def verify(result):
    keys, rejected = result["keys"], result["rejected"]
    return [
        practice.Check(
            "ANSWER: the drop-in fits OpenAI and does not type-check for Anthropic",
            all([result["signature"], rejected["openai"] == [],
                 rejected["anthropic"] == ["temperature"],
                 len(result["google_unknown"]) == 2]),
            f"`live_call` carries `simulate_llm_call`'s signature. OpenAI: {keys['openai']} "
            f"-> 0 rejected. Anthropic: {keys['anthropic']} -> {rejected['anthropic']} is not "
            f"one of the 22 parameters of messages.create, so the recommended temperature all "
            f"{result['patterns']} patterns carry cannot be sent there. Google: "
            f"{result['google_unknown']} belong to neither SDK, and 'model' is a URL segment",
        ),
        practice.Check(
            "FINDING: latency is measured by the harness and then discarded",
            all([result["reported_latency"] == result["canned_latency"],
                 not result["wall_reported"]]),
            f"`run_prompt_test` times the call into wall_time_ms; `compare_models` reports "
            f"{result['reported_fields']} -- latency_ms = {result['reported_latency']}, the "
            "number the callee handed back. Real calls move that constant; nothing measures",
        ),
        practice.Check(
            "FINDING: format compliance passes the empty answer and fails the real one",
            all([result["format_right"] < result["format_shapes"],
                 result["empty_passes"], result["fenced_fails"]]),
            f"json.loads() over the whole reply gets {result['format_right']} of "
            f"{result['format_shapes']} shapes right: 1.0 on {result['empty_passes']}, all "
            f"valid JSON and none of them the object asked for; 0.0 on "
            f"{len(result['fenced_fails'])} that do contain it -- fenced, and preambled",
        ),
        practice.Check(
            "FINDING: keyword coverage is a case-folded substring test",
            result["decoys"] == [1.0, 1.0],
            f"required_keywords ['API', 'key'] scores coverage {result['decoys']} on "
            f"{result['decoy_texts']} -- inside 'therapies', 'capital', 'monkey', 'turkey'. "
            "A reply using neither word is fully covered",
        ),
        practice.Check(
            "FINDING: 'more precisely' compares numbers on different scales",
            all([result["minified"]["word_count"] == 1,
                 result["minified"]["length_compliant"],
                 len(set(result["denominators"])) > 1]),
            f"length is str.split(), so a {result['minified_chars']}-character minified JSON "
            f"reply is one word and clears max_words: 200; and the composite divides by the "
            f"criteria keys a test happened to set, {result['denominators']} across the five",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
