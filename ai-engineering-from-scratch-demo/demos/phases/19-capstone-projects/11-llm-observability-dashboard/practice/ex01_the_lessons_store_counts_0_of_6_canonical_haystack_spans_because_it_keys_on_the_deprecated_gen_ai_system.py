"""Exercise 1 — the lesson's store counts 0 of 6 canonical Haystack spans because it keys on the deprecated gen_ai.system.

    Add custom instrumentation for the Haystack framework. Verify canonical spans land in ClickHouse with faithful `gen_ai.*` attributes.

Reading of the exercise: Haystack's own tracer emits one
`haystack.pipeline.run` span per run and one `haystack.component.run` span
per component, tagged `haystack.component.name` / `.type` and, with content
tracing on, `haystack.component.output`; a chat generator's output message
carries `meta` = model, finish_reason and usage (read in Haystack's
`core/pipeline/base.py`, `components/generators/chat/openai.py` and the
Anthropic integration, 2026-09-29). The instrumentation is `to_genai()`: it
turns those spans into the lesson's `Span`, adding the OpenTelemetry GenAI
attributes to generator spans. "ClickHouse" is the lesson's in-memory
`SpanStore`. "Canonical" and "faithful" are checked on the rows read back:
every `gen_ai.*` key is in the current registry, the two Required
attributes are present, the provider is a well-known value, and the model
and token counts equal the source meta. The registry is the
semantic-conventions-genai repo's `docs/registry/attributes/gen-ai.md`
(https://github.com/open-telemetry/semantic-conventions-genai, read
2026-09-29); its model-inference attributes are copied into `REGISTRY` so
nothing is fetched at run time.

**ANSWER: all 6 generator spans across 3 generator families land with
canonical, faithful `gen_ai.*` attributes.** 24 spans are stored (6 runs of
pipeline + prompt builder + retriever + generator). Every `gen_ai.*` key on
them is in the registry, `gen_ai.operation.name` and `gen_ai.provider.name`
are on 6/6, the providers are `openai`, `anthropic` and `gcp.gen_ai`, and
the model and all 2,388 input / 1,140 output tokens match the Haystack meta.
The two usage dialects (`prompt_tokens` for OpenAI and Google,
`input_tokens` for Anthropic) are both normalised.

**FINDING: the lesson's store counts 0 of the 6 canonical LLM spans.**
`Span.is_llm` tests for `gen_ai.system`, which the core semantic-conventions
registry marks deprecated and replaced by `gen_ai.provider.name`. So
`by_model` and `by_user` stay empty and cost is $0. Adding the deprecated
key back makes all 6 count.

**FINDING: the lesson's own spans break its skill's first hard reject**
("span schemas that invent attribute names not in the OpenTelemetry GenAI
semconv"). A `synth_trace` LLM span carries 5 non-semconv keys (`user_id`,
`prompt`, `response`, `context`, `cost_usd`) and the deprecated
`gen_ai.system`. It lacks the Required `gen_ai.operation.name`, and its
`gen_ai.system` values (`claude`, `gpt`, `gemini`) match 0 of the 16
well-known provider values. The doc's `llm.prompts` and `llm.completions`
are not `gen_ai.*` names either.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "11-llm-observability-dashboard"
# the registry's model-inference attributes; agent, tool, memory, retrieval, evaluation, prompt-template omitted
REGISTRY_TEXT = """input.messages operation.name output.messages output.type provider.name request.choice.count
request.encoding_formats request.frequency_penalty request.max_tokens request.model request.presence_penalty
request.previous_response.id request.reasoning.level request.seed request.stop_sequences request.stream
request.stream_cursor request.temperature request.top_k request.top_p response.finish_reasons response.id
response.model response.status response.time_to_first_chunk system_instructions usage.cache_read.input_tokens
usage.cache_write.input_tokens usage.input_tokens usage.output_tokens usage.reasoning.output_tokens"""
PROVIDERS_TEXT = """anthropic aws.bedrock azure.ai.inference azure.ai.openai cohere deepseek gcp.gemini gcp.gen_ai
gcp.vertex_ai groq ibm.watsonx.ai mistral_ai moonshot_ai openai perplexity x_ai"""
REGISTRY, PROVIDERS = set(REGISTRY_TEXT.split()), set(PROVIDERS_TEXT.split())
GENERATORS = {"OpenAIChatGenerator": ("openai", "gpt-5-mini", ("prompt_tokens", "completion_tokens")),
              "AnthropicChatGenerator": ("anthropic", "claude-sonnet-4-5", ("input_tokens", "output_tokens")),
              "GoogleGenAIChatGenerator": ("gcp.gen_ai", "gemini-2.5-flash", ("prompt_tokens", "completion_tokens"))}


def make_fixture(n=6, seed=11):
    """Haystack-shaped runs: a pipeline span plus three component spans, tagged as Haystack's tracer tags them."""
    rng, runs = random.Random(seed), []
    for i in range(n):
        kind = list(GENERATORS)[i % 3]
        _, model, (k_in, k_out) = GENERATORS[kind]
        meta = {"model": model, "index": 0, "finish_reason": "stop",
                "usage": {k_in: rng.randint(100, 600), k_out: rng.randint(20, 250)}}
        comps = [("prompt_builder", "ChatPromptBuilder", {}), ("retriever", "InMemoryBM25Retriever", {}),
                 ("llm", kind, {"replies": [{"role": "assistant", "meta": meta}]})]
        runs.append([("haystack.pipeline.run", f"h{i}", None, {})] + [
            ("haystack.component.run", f"h{i}.{j}", f"h{i}", {"haystack.component.name": c,
             "haystack.component.type": t, "haystack.component.output": out}) for j, (c, t, out) in enumerate(comps)])
    return runs


def genai_attrs(tags):
    """The instrumentation proper: Haystack generator tags -> GenAI semconv attributes."""
    kind = tags.get("haystack.component.type")
    if kind not in GENERATORS:
        return {}
    meta = tags["haystack.component.output"]["replies"][0]["meta"]
    usage = meta["usage"]
    return {"gen_ai.operation.name": "chat", "gen_ai.provider.name": GENERATORS[kind][0],
            "gen_ai.request.model": GENERATORS[kind][1], "gen_ai.response.model": meta["model"],
            "gen_ai.usage.input_tokens": usage.get("input_tokens", usage.get("prompt_tokens")),
            "gen_ai.usage.output_tokens": usage.get("output_tokens", usage.get("completion_tokens")),
            "gen_ai.response.finish_reasons": [meta["finish_reason"]]}


def to_genai(ref, run, i, extra=()):
    spans = []
    for name, span_id, parent, tags in run:
        attrs, tags = genai_attrs(tags), {k: v for k, v in tags.items() if k != "haystack.component.output"}
        name = f"chat {attrs['gen_ai.request.model']}" if attrs else name
        spans.append(ref.Span(f"trace{i}", span_id, parent, name, 0, 1, {**attrs, **dict(extra if attrs else ()), **tags}))
    return spans


def off_registry(keys):
    return sorted(k for k in keys if k.startswith("gen_ai.") and k[7:] not in REGISTRY)


def faithful(llm, runs):
    metas = [run[3][3]["haystack.component.output"]["replies"][0]["meta"] for run in runs]
    got = [[a["gen_ai.response.model"], a["gen_ai.usage.input_tokens"], a["gen_ai.usage.output_tokens"]] for a in llm]
    return sum(g == [m["model"], *m["usage"].values()] for g, m in zip(got, metas)), [
        sum(g[1] for g in got), sum(g[2] for g in got)]


def lesson_schema(ref):
    attrs = ref.synth_trace("t0", False, random.Random(5))[1].attributes
    systems = {ref.synth_trace("t", False, random.Random(s))[1].attributes["gen_ai.system"] for s in range(40)}
    return (sorted(k for k in attrs if not k.startswith("gen_ai.")), off_registry(attrs),
            "gen_ai.operation.name" in attrs, sorted(systems), len(systems & PROVIDERS), len(PROVIDERS))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    runs, store, legacy = make_fixture(), ref.SpanStore(), ref.SpanStore()
    for i, run in enumerate(runs):
        store.insert_trace(to_genai(ref, run, i))
        legacy.insert_trace(to_genai(ref, run, i, {"gen_ai.system": "x"}))
    llm = [s.attributes for s in store.spans if "gen_ai.provider.name" in s.attributes]
    return {"stored": len(store.spans), "llm": len(llm), "off_registry": off_registry({k for a in llm for k in a}),
            "required": sum({"gen_ai.operation.name", "gen_ai.provider.name"} <= set(a) for a in llm),
            "providers": sorted({a["gen_ai.provider.name"] for a in llm}), "faithful": faithful(llm, runs),
            "counted": [sum(st.by_model.values()) for st in (store, legacy)],
            "lesson": lesson_schema(ref)}


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: 6/6 Haystack generator spans land canonical and faithful across 3 generator families",
            (r["stored"], r["llm"], r["off_registry"], r["required"], r["faithful"]) == (24, 6, [], 6, (6, [2388, 1140]))
            and r["providers"] == ["anthropic", "gcp.gen_ai", "openai"],
            f"{r['stored']} spans stored, {r['llm']} LLM; off-registry gen_ai keys {r['off_registry']}; required on "
            f"{r['required']}/6; providers {r['providers']}; faithful, (input, output) tokens {r['faithful']}",
        ),
        practice.Check("FINDING: the lesson's store counts 0 of 6 canonical LLM spans; it keys on the deprecated "
                       "gen_ai.system", r["counted"] == [0, 6], f"by_model with / without gen_ai.system {r['counted'][::-1]}"),
        practice.Check(
            "FINDING: the lesson's own LLM span invents 5 attribute names and uses 0 well-known provider values",
            r["lesson"] == (["context", "cost_usd", "prompt", "response", "user_id"], ["gen_ai.system"], False,
                            ["claude", "gemini", "gpt"], 0, 16),
            "non-semconv keys, deprecated keys, has operation.name, system values, well-known hits, providers: "
            f"{r['lesson']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
