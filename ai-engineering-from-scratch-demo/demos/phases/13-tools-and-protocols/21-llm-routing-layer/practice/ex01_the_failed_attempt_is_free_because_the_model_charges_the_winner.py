"""Exercise 1 — the failed attempt is free because the model charges the winner.

    Run `code/main.py`. Trigger the outage scenario; confirm fallback lands on
    the second provider and cost is attributed correctly.

Reading of the exercise: "attributed correctly" is the claim to test, so the
cost is checked against the model that answered *and* against the one that
did not. The failed attempt contributes nothing, which is right here only
because the simulated failure happens before any tokens exist -- a real 5xx
after generation would be charged by the provider and recorded as zero by
this gateway.

**ANSWER: the fallback lands on `anthropic/claude-sonnet` and the cost is the
second provider's.** With `openai/gpt-4o` in outage, `attempts` is
`['openai/gpt-4o', 'anthropic/claude-sonnet']`, `chosen_model` is the second,
and `cost_usd` is priced at **3.0/15.0** per million -- not gpt-4o's
**5.0/15.0**. The same call without the outage costs more and attempts one
model.

**FINDING: the failed attempt contributes zero, by construction.**
`provider_call` raises before returning usage, so there is nothing to price
and the gateway records **1** attempt it was not charged for. A provider that
streams tokens and then fails would bill for them; this model has no field in
which such a charge could appear -- `Invocation` prices exactly the response
it kept.

**FINDING: a partial outage is invisible outside `attempts`.**
`except RuntimeError: continue` discards the exception, and `inv.error` is
set only when the whole chain fails. So a degraded run and a clean run differ
in **1** list field and in nothing that looks like an error -- no count, no
per-attempt reason, no flag.

**FINDING: redaction happens once, before the chain, so every provider sees
the same text.** A prompt carrying an SSN is redacted once and the redacted
string is what both the failed and the successful provider receive --
`redacted` is `True` on the invocation and the raw value appears **0** times
in the request. The flag is a boolean, though, so which pattern matched is
not recorded.

Structure: `call` runs one alias under a chosen outage set and restores it, so
every row is the same request under different provider health.
"""

from __future__ import annotations

import contextlib
import io

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "21-llm-routing-layer"
PROMPT = "summarise the incident report"
SSN_PROMPT = "the customer is 123-45-6789, summarise the case"


def call(ref, alias, prompt, outage=()):
    previous = set(ref.OUTAGE)
    ref.OUTAGE.clear()
    ref.OUTAGE.update(outage)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return ref.route(alias, [{"role": "user", "content": prompt}])
    finally:
        ref.OUTAGE.clear()
        ref.OUTAGE.update(previous)


def priced(ref, inv, model):
    """What this invocation would have cost at another model's rates."""
    in_rate, out_rate = ref.PRICES[model]
    return (inv.input_tokens * in_rate + inv.output_tokens * out_rate) / 1_000_000


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    healthy = call(ref, "smart", PROMPT)
    degraded = call(ref, "smart", PROMPT, outage={"openai/gpt-4o"})
    dead = call(ref, "smart", PROMPT, outage=set(ref.ROUTES["smart"]))
    redacted = call(ref, "smart", SSN_PROMPT, outage={"openai/gpt-4o"})
    sent = redacted.response["choices"][0]["message"]["content"]
    return {
        "healthy_model": healthy.chosen_model, "healthy_attempts": healthy.attempts,
        "model": degraded.chosen_model, "attempts": degraded.attempts,
        "cost": round(degraded.cost_usd, 10),
        "as_chosen": round(priced(ref, degraded, degraded.chosen_model), 10),
        "as_failed": round(priced(ref, degraded, "openai/gpt-4o"), 10),
        "rates": ref.PRICES[degraded.chosen_model],
        "failed_rates": ref.PRICES["openai/gpt-4o"],
        "healthy_cost": round(healthy.cost_usd, 10),
        "dead_error": dead.error, "dead_cost": dead.cost_usd,
        "dead_attempts": len(dead.attempts),
        "degraded_error": degraded.error,
        "redacted": redacted.redacted, "raw_in_response": "123-45-6789" in sent,
        "redaction_marker": "[REDACTED]" in sent,
        "redacted_is_bool": isinstance(redacted.redacted, bool),
        "patterns": len(ref.PII_PATTERNS),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the fallback lands on the second provider and is priced as one",
            all([result["attempts"] == ["openai/gpt-4o", "anthropic/claude-sonnet"],
                 result["model"] == "anthropic/claude-sonnet",
                 result["cost"] == result["as_chosen"],
                 result["cost"] != result["as_failed"],
                 result["rates"] == (3.0, 15.0), result["failed_rates"] == (5.0, 15.0),
                 result["healthy_attempts"] == ["openai/gpt-4o"]]),
            f"with gpt-4o in outage the attempts are {result['attempts']}, the chosen model "
            f"is {result['model']} and the cost {result['cost']} matches that model's "
            f"{result['rates']} rather than the failed provider's {result['failed_rates']}, "
            f"which would have given {result['as_failed']}",
        ),
        practice.Check(
            "FINDING: the failed attempt contributes zero, by construction",
            all([len(result["attempts"]) == 2, result["dead_cost"] == 0.0,
                 result["dead_error"] == "all providers failed",
                 result["dead_attempts"] == 3]),
            f"provider_call raises before returning usage, so the gateway records "
            f"{len(result['attempts'])} attempts and prices one. A whole chain in outage "
            f"costs {result['dead_cost']} across {result['dead_attempts']} attempts -- a "
            "provider that streamed tokens and then failed would bill for them, and "
            "Invocation has no field in which that charge could appear",
        ),
        practice.Check(
            "FINDING: a partial outage is invisible outside attempts",
            all([result["degraded_error"] is None,
                 result["healthy_attempts"] != result["attempts"],
                 result["dead_error"] is not None]),
            f"the degraded run reports error={result['degraded_error']} -- the exception is "
            f"discarded by `except RuntimeError: continue` and error is set only when the "
            f"chain is exhausted. A degraded run and a clean one differ in the attempts list "
            "and in nothing that looks like a failure: no count, no reason, no flag",
        ),
        practice.Check(
            "FINDING: redaction happens once, before the chain",
            all([result["redacted"], not result["raw_in_response"],
                 result["redaction_marker"], result["redacted_is_bool"],
                 result["patterns"] == 2]),
            f"the SSN is redacted once and the redacted string is what every provider in the "
            f"chain receives -- the raw value appears {int(result['raw_in_response'])} times "
            f"in the response and the marker appears. The flag is a bool, though, so which "
            f"of the {result['patterns']} patterns matched is not recorded",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
