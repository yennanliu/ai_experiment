"""Exercise 3 — the lesson's attribute set leaves two of its five alerts blind and names a deprecated key.

    Design an OpenTelemetry GenAI attribute set your org's guideline should
    mandate on every LLM call.

Reading of the exercise: a mandated attribute is justified by the signal
that reads it, so the design starts from what the lesson says must be
computed on every call. That is the skill file's five alerts (error rate,
P99 TTFT, cost/request, prompt-cache hit rate, refusal rate), plus per-tenant
cost. Every name is checked against the current OpenTelemetry GenAI spans
spec (open-telemetry/semantic-conventions-genai, read 2026-09-26). The same
test runs on the skill file's list and on the tracer in `code/main.ts`.

**ANSWER: mandate 16 attributes, all metadata, and no content.**

- Spec Required: `gen_ai.operation.name`, `gen_ai.provider.name`.
- Model: `gen_ai.request.model`, and `gen_ai.response.model` so an alias is
  priced as the snapshot that actually answered.
- Usage: `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`,
  `gen_ai.usage.cache_read.input_tokens`.
- Response: `gen_ai.response.finish_reasons`, `gen_ai.request.stream`,
  `gen_ai.response.time_to_first_chunk`, `gen_ai.conversation.id`, and
  `error.type` on failure.
- Org namespace: `org.tenant_id`, `org.user_id`, `org.task`,
  `org.prompt_version`.

The spec's Opt-In content attributes (`gen_ai.input.messages`,
`gen_ai.output.messages`, `gen_ai.system_instructions`,
`gen_ai.tool.definitions`) stay off by default. With this set, all six
signals can be computed.

**FINDING: the lesson's own list leaves two of its five alerts blind and
omits both spec-Required attributes.** The skill file's nine attributes
carry no TTFT and no cached-token count, so P99 TTFT and prompt-cache hit
rate cannot be computed. It names `gen_ai.system`, which the spec deprecates
in favour of `gen_ai.provider.name`, and it has no `gen_ai.operation.name`.
`main.ts` has the same two blind spots, has no tenant field, and uses
`gen_ai.usage.cached_input_tokens`, which is not a spec name: the spec's is
`gen_ai.usage.cache_read.input_tokens`. Its sampler reads missing token
counts as `?? 0`, so a span without usage is never "high-cost", and its span
name `chat.completion` is not the spec's `{operation} {model}`. The whole
convention is still at Development status, not Stable.

**FINDING: the mandate costs 12.2% of a trace, so every call's metadata can
be kept for a fifth of full retention.** The example span encodes to 551
bytes of the reference's 4,500 per trace. Keeping that for all 1M calls a
day, plus full traces under the lesson's "5% success + errors + $$$" rule,
costs $12.90/month at the reference's ingest rate. That is 19% of 100%
retention ($67.50) and 2.4x the sampled-only $5.28. The lesson's "keep
aggregates always" becomes concrete: without it, 92% of calls leave no
record at all.

Structure: `SIGNALS` maps each alert to the attributes it reads; `blind()`
lists the alerts an attribute set cannot compute; the byte and cost figures
come from the reference `simulate_day()` and its constants.
"""

from __future__ import annotations

import json
import re

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "13-llm-observability"
# open-telemetry/semantic-conventions-genai, gen-ai-spans.md "Inference", read 2026-09-26.
SPEC_REQUIRED = {"gen_ai.operation.name", "gen_ai.provider.name"}
OPT_IN = "gen_ai.input.messages gen_ai.output.messages gen_ai.system_instructions"
SPEC_OPT_IN = set(OPT_IN.split()) | {"gen_ai.tool.definitions"}
SPEC_OTHER = """gen_ai.request.model gen_ai.response.model gen_ai.response.id
    gen_ai.usage.input_tokens gen_ai.usage.output_tokens gen_ai.usage.cache_read.input_tokens
    gen_ai.response.finish_reasons gen_ai.response.time_to_first_chunk
    gen_ai.request.temperature gen_ai.request.stream gen_ai.conversation.id error.type"""
SPEC_NAMES = SPEC_REQUIRED | SPEC_OPT_IN | set(SPEC_OTHER.split())
# The mandate: one span, every attribute a signal below reads, none of the content.
EXAMPLE = json.loads("""{"gen_ai.operation.name": "chat", "gen_ai.provider.name": "openai",
 "gen_ai.request.model": "gpt-4o-mini", "gen_ai.response.model": "gpt-4o-mini-2024-07-18",
 "gen_ai.usage.input_tokens": 1830, "gen_ai.usage.output_tokens": 212,
 "gen_ai.usage.cache_read.input_tokens": 1536, "gen_ai.response.finish_reasons": ["stop"],
 "gen_ai.request.stream": true, "gen_ai.response.time_to_first_chunk": 0.284,
 "gen_ai.conversation.id": "conv_5j66UpCpwteGg4YSxUnt7lPY", "org.tenant_id": "acme-eu",
 "org.user_id": "u_83f2a1", "org.task": "support.summarize",
 "org.prompt_version": "summarize@v14"}""")
MANDATE = set(EXAMPLE) | {"error.type"}  # error.type only when the call fails
# What the lesson's alerts read; span status is always present.
SIGNALS = {
    "error rate": "",
    "P99 TTFT": "gen_ai.response.time_to_first_chunk",
    "cost/request": "gen_ai.request.model gen_ai.usage.input_tokens gen_ai.usage.output_tokens",
    "prompt-cache hit rate": "gen_ai.usage.cache_read.input_tokens gen_ai.usage.input_tokens",
    "refusal rate": "gen_ai.response.finish_reasons",
    "per-tenant cost": "org.tenant_id gen_ai.usage.input_tokens",
}
BLIND = ["P99 TTFT", "prompt-cache hit rate"]
# outputs/skill-observability-stack.md, item 4, with its three org fields namespaced.
SKILL_TEXT = """gen_ai.system gen_ai.request.model gen_ai.usage.input_tokens
    gen_ai.usage.output_tokens gen_ai.request.temperature gen_ai.response.finish_reasons
    org.tenant_id org.user_id org.task"""
SKILL = set(SKILL_TEXT.split())


def audit(attrs):
    """(alerts it cannot compute, spec-Required keys it lacks, gen_ai names not in the spec)."""
    blind = [name for name, needs in SIGNALS.items() if not set(needs.split()) <= attrs]
    non_spec = sorted(n for n in attrs - SPEC_NAMES if n.startswith("gen_ai"))
    return blind, sorted(SPEC_REQUIRED - attrs), non_spec


def ts_attributes():
    source = (parity.lesson_dir(PHASE, LESSON) / "code" / "main.ts").read_text()
    block = source.split("type GenAIAttributes = {", 1)[1].split("};", 1)[0]
    return set(re.findall(r'"(gen_ai\.[a-z_.]+)"', block)), source


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    ts, source = ts_attributes()
    sets = {"mandate": MANDATE, "skill": SKILL, "main.ts": ts}
    meta = len(json.dumps(EXAMPLE, separators=(",", ":")).encode())
    kept = ref.simulate_day(ref.STRATEGIES[3])["retained"]
    per_gb_month = 30 * ref.OBSERVABILITY_INGEST_PER_GB / 1e9
    full = 1e6 * ref.BYTES_PER_TRACE * per_gb_month
    sampled = kept * ref.BYTES_PER_TRACE * per_gb_month
    hybrid = (1e6 * meta + kept * (ref.BYTES_PER_TRACE - meta)) * per_gb_month
    ratios = (round(hybrid / full, 2), round(hybrid / sampled, 1))
    return {
        "audit": {k: audit(v) for k, v in sets.items()},
        "content_free": not (MANDATE & SPEC_OPT_IN),
        "ts_quirks": (
            '["gen_ai.usage.input_tokens"] as number) ?? 0' in source,
            'startSpan("chat.completion"' in source,
        ),
        "share": meta / ref.BYTES_PER_TRACE,
        "dollars": (full, sampled, hybrid),
        "costs": (meta, round(hybrid, 2), *ratios),
    }


def verify(result):
    mandate, skill, ts = (result["audit"][k] for k in ("mandate", "skill", "main.ts"))
    full, sampled, hybrid = result["dollars"]
    return [
        practice.Check(
            "ANSWER: mandate 16 attributes, all metadata, and no content",
            len(MANDATE) == 16 and result["content_free"] and mandate == ([], [], []),
            f"{len(MANDATE)} attributes, both spec-Required ones included, no Opt-In "
            f"content; (blind alerts, missing Required, non-spec names) = {mandate}",
        ),
        practice.Check(
            "FINDING: the lesson's own list leaves two of its five alerts blind",
            skill == (BLIND, sorted(SPEC_REQUIRED), ["gen_ai.system"])
            and ts[0] == BLIND + ["per-tenant cost"]
            and ts[2] == ["gen_ai.system", "gen_ai.usage.cached_input_tokens"]
            and all(result["ts_quirks"]),
            f"skill file (blind, missing Required, non-spec) = {skill}; main.ts = {ts}; "
            "main.ts reads missing tokens as 0 and names its span chat.completion",
        ),
        practice.Check(
            "FINDING: the mandate costs 12.2% of a trace, so every call's metadata can be kept",
            result["costs"] == (551, 12.90, 0.19, 2.4),
            f"{result['costs'][0]} bytes = {result['share']:.1%} of a trace; 100% metadata + "
            f"sampled traces ${hybrid:.2f}/month against ${full:.2f} full, ${sampled:.2f} sampled",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
