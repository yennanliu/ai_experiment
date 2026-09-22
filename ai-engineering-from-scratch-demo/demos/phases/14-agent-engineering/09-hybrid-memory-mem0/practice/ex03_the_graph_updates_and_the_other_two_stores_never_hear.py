"""Exercise 3 — the graph updates and the other two stores never hear.

    Implement a conflict detector: if an incoming fact contradicts a graph
    edge, invalidate the old edge and log both. Test on "user lives in Berlin"
    -> "user lives in Lisbon."

Reading of the exercise: `GraphStore.add_edge` already invalidates every
valid edge with the same `(subject, relation)` before appending, so half of
this is shipped and the detector's real job is the *log* -- and the log is
what exposes that only one of the three stores ever hears about a conflict.
The test case is the demo's own: Berlin, then Lisbon.

**ANSWER: a detector that names both sides.** Writing `lives_in Lisbon` over
`lives_in Berlin` leaves **1** invalid edge and **1** valid one, and the
detector's log carries **1** entry naming both objects. The graph is correct
afterwards: `neighbors("ava")` returns exactly `Lisbon`.

**FINDING: the vector store still answers Berlin.** Asked
`where does ava live`, the shipped vector arm ranks the Berlin record **1**
of **2** -- the two texts tie on token overlap and ties break by insertion
order, so the *superseded* fact wins. The conflict was detected in the graph
and the retrieval path that actually feeds the model never saw it.

**FINDING: the KV store holds both cities.** `KVKey` includes `entity`, so
`(ava, city, Berlin)` and `(ava, city, Lisbon)` are different keys and `put`
overwrites neither. After the conflict the KV arm returns **2** city records,
and `Mem0.search` injects both at its fixed relevance floor.

**FINDING: the invalidation does not look at the object.** `add_edge`
invalidates on `(subject, relation)` alone, so re-asserting the *same* fact
invalidates and re-adds it: two writes of `lives_in Lisbon` leave **2**
edges, **1** of them invalid, both pointing at `Lisbon`. An idempotent write
produces a conflict record.

Structure: `detect_conflict()` wraps the lesson's own `add_edge` and reads
the edge list before and after; nothing here reimplements the graph.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "09-hybrid-memory-mem0"
QUERY = "where does ava live"


def detect_conflict(graph, subject, relation, obj):
    """Log what the shipped add_edge silently supersedes."""
    superseded = [edge for edge in graph.neighbors(subject)
                  if edge.relation == relation and edge.obj != obj]
    graph.add_edge(subject, relation, obj)
    return [{"subject": subject, "relation": relation,
             "was": edge.obj, "now": obj} for edge in superseded]


def build(ref):
    mem = ref.Mem0()
    mem.add("ava lives in Berlin", user_id="ava", importance=0.6,
            kv_triples=(("city", "Berlin"),),
            graph_triples=(("ava", "lives_in", "Berlin"),))
    mem.add("ava lives in Lisbon", user_id="ava", importance=0.8,
            kv_triples=(("city", "Lisbon"),))
    return mem


def repeat(ref):
    """The same fact written twice, through the same detector."""
    graph = ref.GraphStore()
    graph.add_edge("ava", "lives_in", "Lisbon")
    log = detect_conflict(graph, "ava", "lives_in", "Lisbon")
    edges = graph.all_edges()
    return {"repeat_edges": len(edges), "repeat_log": log,
            "repeat_invalid": sum(1 for edge in edges if not edge.valid),
            "repeat_objs": sorted({edge.obj for edge in edges})}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    mem = build(ref)
    log = detect_conflict(mem.graph, "ava", "lives_in", "Lisbon")
    edges = mem.graph.all_edges()
    hits = mem.vector.search(QUERY, top_k=3)
    return {
        "log": log, "edges": len(edges),
        "valid": [edge.obj for edge in edges if edge.valid],
        "invalid": [edge.obj for edge in edges if not edge.valid],
        "neighbors": [edge.obj for edge in mem.graph.neighbors("ava")],
        "vector_order": [record.text.split()[-1] for _, record in hits],
        "vector_scores": [round(score, 3) for score, _ in hits],
        "kv_cities": [record.text.split()[-1] for record in mem.kv.by_user("ava")],
        "kv_keys": sorted(key.entity for key in mem.kv._map if key.fact_type == "city"),
        **repeat(ref),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: one edge superseded, one entry logged, the graph left correct",
            all([len(result["log"]) == 1, result["log"][0]["was"] == "Berlin",
                 result["log"][0]["now"] == "Lisbon", result["edges"] == 2,
                 result["valid"] == ["Lisbon"], result["invalid"] == ["Berlin"],
                 result["neighbors"] == ["Lisbon"]]),
            f"writing lives_in Lisbon over lives_in Berlin leaves "
            f"{len(result['invalid'])} invalid edge {result['invalid']} and "
            f"{len(result['valid'])} valid one {result['valid']}, with a "
            f"{len(result['log'])}-entry log naming both. neighbors('ava') returns "
            f"{result['neighbors']}",
        ),
        practice.Check(
            "FINDING: the vector store still answers Berlin",
            all([result["vector_order"][0] == "Berlin",
                 len(result["vector_scores"]) == 2,
                 result["vector_scores"][0] == result["vector_scores"][1]]),
            f"asked {QUERY!r} the vector arm returns {result['vector_order']} at scores "
            f"{result['vector_scores']} -- the two texts tie on token overlap and ties "
            "break by insertion order, so the superseded fact wins. The conflict was "
            "detected in the graph and the retrieval path feeding the model never saw it",
        ),
        practice.Check(
            "FINDING: the KV store holds both cities",
            all([sorted(result["kv_cities"]) == ["Berlin", "Lisbon"],
                 result["kv_keys"] == ["Berlin", "Lisbon"]]),
            f"KVKey includes entity, so the two writes land on "
            f"{len(result['kv_keys'])} different keys {result['kv_keys']} and put "
            f"overwrites neither. The KV arm returns {sorted(result['kv_cities'])}, and "
            "Mem0.search injects both at its fixed relevance floor",
        ),
        practice.Check(
            "FINDING: the invalidation does not look at the object",
            all([result["repeat_edges"] == 2, result["repeat_invalid"] == 1,
                 result["repeat_objs"] == ["Lisbon"], result["repeat_log"] == []]),
            f"add_edge invalidates on (subject, relation) alone, so writing the same "
            f"fact twice leaves {result['repeat_edges']} edges, "
            f"{result['repeat_invalid']} of them invalid, both pointing at "
            f"{result['repeat_objs'][0]} -- while the detector correctly logs "
            f"{len(result['repeat_log'])} conflicts. An idempotent write still churns "
            "the graph",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
