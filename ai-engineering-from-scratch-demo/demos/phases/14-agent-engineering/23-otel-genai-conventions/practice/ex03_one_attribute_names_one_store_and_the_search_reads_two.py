"""Exercise 3 — one attribute names one store and the search reads two.

    Read the spec for `gen_ai.data_source.id`. Wire it into your Lesson 09
    Mem0 search.

Reading of the exercise: the spec entry is one line -- "for RAG: which corpus
or store was consulted" -- and wiring it into Lesson 09 turns that line into
a question the attribute cannot answer. `Mem0.search` reads the vector store
and the KV store on every call and fuses the results, so "which store was
consulted" has two answers and the attribute holds one string.

**ANSWER: the span carries `corpus://mem0/default` and the search touched 2
stores.** Instrumenting `Mem0.search` produces **1** `tool_call` span whose
`gen_ai.data_source.id` is a single value, while counting the reads shows
`VectorStore.search` called **1** time and `KVStore.by_user` **1** time for
the same query. **1** of the **5** returned records came from the vector
index and **4** were reached only through KV -- so the one store the literal
names supplied the minority of the answer, and the attribute cannot say so.

**FINDING: the third store is never consulted, and the attribute would not
show that either.** `Mem0` owns a `GraphStore`, `add` writes edges to it, and
`search` reads it **0** times -- so a trace that names one data source is
accurate about the wrong thing. A per-store attribute would have made a
silently unused index visible; a single id makes it invisible.

**FINDING: the shipped id is a literal in `main()`, not a property of the
store.** `gen_ai.data_source.id` is written as `"corpus://mem0/default"` at
the call site, so it stays correct only while a human keeps it correct.
Deriving it from the store gives `mem0://vector` and `mem0://kv` -- **2**
values from the same search, from objects that know their own identity.

**FINDING: the attribute is on the tool span, which carries no model.** The
`tool_call` span holds **2** GenAI attributes and neither is
`gen_ai.request.model` or `gen_ai.provider.name`, so "which corpus did this
model consult" needs the parent -- and Lesson 23's `Span` has no parent id.
The RAG attribution the spec enables is blocked by the identity gap from
exercise 1.

Structure: `counted()` wraps Lesson 09's stores; `search_span()` emits the
span the exercise asks for, once per store rather than once per call.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "23-otel-genai-conventions"
MEM0 = "09-hybrid-memory-mem0"
SHIPPED_ID = "corpus://mem0/default"
FACTS = (
    ("prefers window seats on long flights", ("seat", "window")),
    ("allergic to shellfish", ("allergy", "shellfish")),
    ("lives in Taipei", ("city", "taipei")),
    ("works on agent engineering", ("topic", "agents")),
    ("flies to Tokyo every quarter", ("route", "tokyo")),
)


def counted(mem):
    """Record which of Mem0's three stores each search reads, and what each returned."""
    reads = {"vector": 0, "kv": 0, "graph": 0}
    seen = {"vector": set(), "kv": set()}
    originals = {"vector": mem.vector.search, "kv": mem.kv.by_user,
                 "graph": mem.graph.neighbors}

    def wrap(name):
        def inner(*args, **kwargs):
            reads[name] += 1
            out = originals[name](*args, **kwargs)
            if name == "vector":
                seen[name].update(record.rid for _, record in out)
            elif name == "kv":
                seen[name].update(record.rid for record in out)
            return out
        return inner

    mem.vector.search, mem.kv.by_user = wrap("vector"), wrap("kv")
    mem.graph.neighbors = wrap("graph")
    return reads, seen


def build(ref_mem0):
    mem = ref_mem0.Mem0()
    for text, (fact_type, entity) in FACTS:
        mem.add(text, user_id="u1", kv_triples=((fact_type, entity),),
                graph_triples=(("u1", fact_type, entity),))
    return mem


def search_span(tracer, mem, query, data_source=SHIPPED_ID):
    span = tracer.start_span("tool_call memory_search", attributes={
        "gen_ai.operation.name": "tool_call", "gen_ai.tool.name": "memory_search",
        "gen_ai.data_source.id": data_source})
    hits = mem.search(query, user_id="u1")
    tracer.end_span()
    return span, hits


def provenance(seen, hits):
    """Which store each returned record was actually read from."""
    return {"vector": sum(record.rid in seen["vector"] for _, record in hits),
            "kv_only": sum(record.rid not in seen["vector"] for _, record in hits)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    ref_mem0 = parity.load_reference(PHASE, MEM0, "main")
    mem = build(ref_mem0)
    reads, seen = counted(mem)
    tracer = ref.Tracer(capture_inline=False)
    span, hits = search_span(tracer, mem, "window seat")
    derived = sorted({f"mem0://{name}" for name, count in reads.items() if count})
    tool_attrs = sorted(span.attributes)
    return {
        "data_source": span.attributes["gen_ai.data_source.id"],
        "reads": dict(reads), "consulted": sum(1 for c in reads.values() if c),
        "returned": len(hits), "provenance": provenance(seen, hits),
        "graph_edges": len(mem.graph.all_edges()), "graph_reads": reads["graph"],
        "derived": derived, "literal": SHIPPED_ID,
        "tool_attrs": tool_attrs, "genai_attrs": len(tool_attrs) - 1,
        "model_attrs": [a for a in tool_attrs if "model" in a or "provider" in a],
        "span_fields": list(ref.Span.__dataclass_fields__),
        "parent_fields": [f for f in ref.Span.__dataclass_fields__ if "parent" in f],
    }


def verify(result):
    reads, prov = result["reads"], result["provenance"]
    return [
        practice.Check(
            "ANSWER: the span names one source and the search touched two stores",
            all([result["data_source"] == SHIPPED_ID, result["consulted"] == 2,
                 reads["vector"] == 1, reads["kv"] == 1, result["returned"] == 5,
                 prov["vector"] == 1, prov["kv_only"] == 4]),
            f"the span carries gen_ai.data_source.id={result['data_source']!r} while the "
            f"search read {result['consulted']} stores -- vector {reads['vector']} time, "
            f"kv {reads['kv']} time -- and returned {result['returned']} records, "
            f"{prov['vector']} of them from the vector index and {prov['kv_only']} "
            "reached only through KV. One string cannot say that",
        ),
        practice.Check(
            "FINDING: the third store is never consulted and the attribute hides it",
            all([result["graph_reads"] == 0, result["graph_edges"] == 5,
                 result["consulted"] == 2]),
            f"Mem0 owns a GraphStore, add wrote {result['graph_edges']} edges to it, and "
            f"search read it {result['graph_reads']} times. A trace naming one data "
            "source is accurate about the wrong thing: a per-store attribute would make a "
            "silently unused index visible, and a single id makes it invisible",
        ),
        practice.Check(
            "FINDING: the shipped id is a literal, not a property of the store",
            all([result["derived"] == ["mem0://kv", "mem0://vector"],
                 result["literal"] == SHIPPED_ID, len(result["derived"]) == 2]),
            f"gen_ai.data_source.id is written as {result['literal']!r} at the call site, "
            f"so it stays correct only while a human keeps it correct. Deriving it from "
            f"the stores that were read gives {result['derived']} -- "
            f"{len(result['derived'])} values from one search, from objects that know "
            "their own identity",
        ),
        practice.Check(
            "FINDING: the attribute is on a span that carries no model",
            all([result["model_attrs"] == [], result["genai_attrs"] == 2,
                 result["parent_fields"] == []]),
            f"the tool_call span holds {result['tool_attrs']} and "
            f"{len(result['model_attrs'])} of them names a model or a provider, so "
            "'which corpus did this model consult' needs the parent -- and Span carries "
            f"{len(result['parent_fields'])} parent fields. The attribution the spec "
            "enables is blocked by the identity gap",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
