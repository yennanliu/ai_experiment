"""Exercise 1 — the first line of the streaming model is not streaming.

    Your product accepts speech input and returns speech output. What's the
    end-to-end latency budget target? List the components that spend time.

Reading of the exercise: the budget and the component list both come from the
lesson's own `streaming_decode_latency`, which is a five-term model with its own
verdict thresholds, and the model is then swept over the two parameters it takes
-- prompt length and model size -- because a single 270 ms reading does not say
which terms a product can actually move.

**ANSWER: under 500 ms time-to-first-audio-byte, and the lesson's default lands
at 270.** Five components: tokenizing the mic audio (40 ms), prefilling the
prompt (80), the first output token (40), the residual-VQ levels (30) and the
speech decoder (80).

**FINDING: the tokenization term scales with the whole utterance, which is not
streaming.** It is `prompt_audio_seconds * 20`, so a 30-second turn costs
**600 ms** before the model starts and the total reaches **830** -- out of the
lesson's own conversational band and into "sluggish". A streaming tokenizer
contributes one frame, not one utterance, and the first line of a streaming
latency model charges for the latter.

**FINDING: only two of the five terms move with model size.** Prefill and first
token carry `model_size_b / 8`; the rest are constants. So a 70B model is
**1,200 ms** and a 1B model is **165** -- and the model-scaled share is only
**44.4%** of the 8B total, so halving the model buys less than a quarter of the
latency.

**FINDING: there is a 110 ms floor nothing can touch.** The residual-VQ levels
and the speech decoder are fixed at 30 and 80 ms, spent *after* the model has
decided what to say. At the 8B default that is **41%** of the budget, and at a
0-second prompt and an arbitrarily small model it is all that is left.

Structure: `total` sums the lesson's own trace, `sweep` runs it over prompt
lengths and model sizes, `fixed_floor` isolates the terms that take neither
parameter, and `VERDICT` is the lesson's own threshold table.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "16-mio-any-to-any-streaming"
DEFAULT_SECONDS, DEFAULT_SIZE = 2.0, 8
CONVERSATIONAL, ACCEPTABLE = 500, 800
SECONDS_SWEEP = (0.0, 2.0, 30.0)
SIZE_SWEEP = (1, 8, 70)
FIXED_TERMS = 2          # residual-VQ and the speech decoder


def trace(ref, seconds=DEFAULT_SECONDS, size=DEFAULT_SIZE):
    return ref.streaming_decode_latency(seconds, size)


def total(ref, seconds=DEFAULT_SECONDS, size=DEFAULT_SIZE):
    return round(sum(step.ms for step in trace(ref, seconds, size)))


def verdict(ms):
    if ms < CONVERSATIONAL:
        return "conversational"
    return "acceptable" if ms < ACCEPTABLE else "sluggish"


def fixed_floor(ref):
    rows = trace(ref)
    return round(sum(step.ms for step in rows[-FIXED_TERMS:]))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = trace(ref)
    default = total(ref)
    floor = fixed_floor(ref)
    scaled = default - floor - rows[0].ms
    return {
        "components": [(step.label, step.ms) for step in rows],
        "count": len(rows), "default": default, "verdict": verdict(default),
        "by_seconds": {seconds: total(ref, seconds) for seconds in SECONDS_SWEEP},
        "long_verdict": verdict(total(ref, 30.0)),
        "by_size": {size: total(ref, DEFAULT_SECONDS, size) for size in SIZE_SWEEP},
        "scaled_ms": scaled,
        "scaled_share": round(scaled / default * 100, 1),
        "floor": floor, "floor_share": round(floor / default * 100),
        "empty_prompt": total(ref, 0.0),
        "threshold": CONVERSATIONAL,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: under 500 ms, and the lesson's default lands at 270",
            all([result["count"] == 5, result["default"] == 270,
                 result["verdict"] == "conversational",
                 [ms for _, ms in result["components"]] == [40.0, 80.0, 40.0, 30, 80]]),
            f"the lesson's own model has {result['count']} components -- "
            f"{[label for label, _ in result['components']]} -- costing "
            f"{[ms for _, ms in result['components']]} ms for a total of "
            f"{result['default']}, which its own threshold table calls "
            f"{result['verdict']} at under {result['threshold']}",
        ),
        practice.Check(
            "FINDING: the tokenization term scales with the whole utterance",
            all([result["by_seconds"] == {0.0: 230, 2.0: 270, 30.0: 830},
                 result["long_verdict"] == "sluggish"]),
            f"the first term is prompt_audio_seconds * 20, so the total goes "
            f"{result['by_seconds']} ms across {list(SECONDS_SWEEP)}-second turns and a "
            f"30-second one is {result['long_verdict']} by the lesson's own thresholds. A "
            "streaming tokenizer contributes one frame, not one utterance",
        ),
        practice.Check(
            "FINDING: only two of the five terms move with model size",
            all([result["by_size"] == {1: 165, 8: 270, 70: 1200},
                 result["scaled_ms"] == 120, result["scaled_share"] == 44.4]),
            f"prefill and first token carry model_size_b / 8 and the rest are constants, so "
            f"the total is {result['by_size']} ms at {list(SIZE_SWEEP)}B. Only "
            f"{result['scaled_ms']} ms of the {result['default']} -- "
            f"{result['scaled_share']}% -- moves with the model, so halving it buys less "
            "than a quarter of the latency",
        ),
        practice.Check(
            "FINDING: there is a 110 ms floor nothing can touch",
            all([result["floor"] == 110, result["floor_share"] == 41,
                 result["empty_prompt"] == 230]),
            f"the residual-VQ levels and the speech decoder are fixed at 30 and 80 ms and "
            f"are spent after the model has decided what to say -- {result['floor']} ms, "
            f"{result['floor_share']}% of the default budget. With a zero-length prompt the "
            f"total is still {result['empty_prompt']}, and shrinking the model to nothing "
            f"leaves {result['floor']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
