"""Exercise 2 — the graph knows when a fact started, not when it stopped.

    Add a temporal query: `search(query, as_of=timestamp)`. Return only
    records valid at or before that time. Which store needs the most work?

Reading of the exercise: each of the three stores has to answer the same
question -- what was true at time T -- and each has a different amount of the
answer already. `Record` carries `ts`, so the vector arm is a filter. `Edge`
carries `ts` and a boolean `valid`, which says *whether* a fact is current and
never *when* it stopped being. The KV arm turns out to have lost something
else entirely.

**ANSWER: `as_of` over all three stores, and the vector arm is a one-line
filter.** Against a fixture where `Berlin` is written at T1 and `Lisbon` at
T3, the vector arm returns the right record for **3** of **3** probe times.
The graph arm returns the right edge for **1** of **3**: it is correct now
and cannot be asked about then.

**FINDING: the graph needs the most work.** `Edge` has **5** fields --
`subject`, `relation`, `obj`, `valid`, `ts` -- and **0** of them record when
an edge stopped being valid. `valid` is a snapshot, so the Berlin edge knows
it was asserted at T1 and not that it was superseded at T3. Asking
`as_of=T2` gives either both edges or neither, and neither is the answer.

**FINDING: the KV arm has lost the question, not the time.** `by_user`
returns `list[Record]`, and `Record` has **0** fields naming a fact type --
the type lives in `KVKey`, which the public read path does not return. So
"what was ava's city at T2" cannot be asked even at T = now: the two city
records come back as **2** untyped rows among **3**.

**FINDING: the two city facts never collide.** `KVKey` includes `entity`, so
`(ava, city, Berlin)` and `(ava, city, Lisbon)` are different keys and `put`
overwrites neither. The KV tier stores both cities forever, which is what
makes the missing fact type expensive rather than merely untidy.

Structure: `search_as_of()` adds the parameter to the lesson's own `Mem0`;
`graph_as_of()` is the best answer the shipped `Edge` allows.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "09-hybrid-memory-mem0"
T1, T2, T3 = 1_000.0, 2_000.0, 3_000.0
QUERY = "ava lives in"


def build(ref):
    mem = ref.Mem0()
    berlin = mem.add("ava lives in Berlin", user_id="ava", importance=0.6,
                     kv_triples=(("city", "Berlin"),),
                     graph_triples=(("ava", "lives_in", "Berlin"),))
    mem.vector._records[berlin].ts = T1
    mem.graph.all_edges()[0].ts = T1
    lisbon = mem.add("ava lives in Lisbon", user_id="ava", importance=0.8,
                     kv_triples=(("city", "Lisbon"),),
                     graph_triples=(("ava", "lives_in", "Lisbon"),))
    mem.vector._records[lisbon].ts = T3
    mem.graph.all_edges()[1].ts = T3
    mem.add("ava prefers terse writing", user_id="ava",
            kv_triples=(("writing_style", "terse"),))
    return mem, berlin, lisbon


def search_as_of(mem, query, *, user_id, as_of, top_k=3):
    """The vector arm's whole temporal implementation: filter, then newest first."""
    rows = [(score, record) for score, record in mem.vector.search(query, top_k=top_k * 3)
            if record.user_id == user_id and record.ts <= as_of]
    rows.sort(key=lambda pair: (-pair[0], -pair[1].ts))
    return [record for _, record in rows[:top_k]]


def graph_as_of(graph, subject, relation, as_of):
    """The best the shipped Edge allows: creation time, and a current boolean."""
    return [edge for edge in graph.all_edges()
            if edge.subject == subject and edge.relation == relation
            and edge.ts <= as_of and edge.valid]


def probe(mem, berlin, lisbon):
    """Both arms asked the same three questions."""
    want, truth = {T1: berlin, T2: berlin, T3: lisbon}, {
        T1: ["Berlin"], T2: ["Berlin"], T3: ["Lisbon"]}
    vector = {when: (search_as_of(mem, QUERY, user_id="ava", as_of=when) or [None])[0]
              for when in (T1, T2, T3)}
    graph = {when: [edge.obj for edge in graph_as_of(mem.graph, "ava", "lives_in", when)]
             for when in (T1, T2, T3)}
    return {
        "vector_right": sum(1 for when, row in vector.items()
                            if row is not None and row.rid == want[when]),
        "graph_right": sum(1 for when, objs in graph.items() if objs == truth[when]),
        "graph_answers": graph, "probes": 3,
    }


def shapes(ref, mem):
    edge, record = ref.Edge.__dataclass_fields__, ref.Record.__dataclass_fields__
    rows = mem.kv.by_user("ava")
    return {
        "edge_fields": list(edge),
        "edge_end_fields": [f for f in edge if "until" in f or "invalid" in f],
        "record_type_fields": [f for f in record if "fact" in f or "type" in f],
        "kv_rows": len(rows), "kv_keys": len(mem.kv._map),
        "kv_cities": sum(1 for row in rows if "lives in" in row.text),
        "distinct_city_keys": len({k for k in mem.kv._map if k.fact_type == "city"}),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    mem, berlin, lisbon = build(ref)
    return {**probe(mem, berlin, lisbon), **shapes(ref, mem)}


def verify(result):
    return [
        practice.Check(
            "ANSWER: the vector arm answers 3 of 3, the graph arm 1 of 3",
            all([result["vector_right"] == 3, result["graph_right"] == 1,
                 result["probes"] == 3,
                 result["graph_answers"][T2] == []]),
            f"against a fixture where Berlin is written at T1 and Lisbon at T3, an "
            f"as_of filter on Record.ts answers {result['vector_right']} of "
            f"{result['probes']} probe times correctly and the graph "
            f"{result['graph_right']} -- at T2 it returns "
            f"{result['graph_answers'][T2]}, which says ava lived nowhere",
        ),
        practice.Check(
            "FINDING: the graph needs the most work",
            all([result["edge_fields"] == ["subject", "relation", "obj", "valid", "ts"],
                 result["edge_end_fields"] == []]),
            f"Edge carries {result['edge_fields']} and "
            f"{len(result['edge_end_fields'])} of them record when an edge stopped being "
            "valid. `valid` is a snapshot, so the Berlin edge knows it was asserted at "
            "T1 and not that it was superseded at T3 -- an interval is the fix",
        ),
        practice.Check(
            "FINDING: the KV arm has lost the question, not the time",
            all([result["record_type_fields"] == [], result["kv_rows"] == 3,
                 result["kv_cities"] == 2]),
            f"by_user returns list[Record] and Record has "
            f"{len(result['record_type_fields'])} fields naming a fact type -- the type "
            f"lives in KVKey, which the read path does not return. The two city records "
            f"come back as {result['kv_cities']} untyped rows among {result['kv_rows']}",
        ),
        practice.Check(
            "FINDING: the two city facts never collide",
            all([result["distinct_city_keys"] == 2, result["kv_keys"] == 3]),
            f"KVKey includes entity, so (ava, city, Berlin) and (ava, city, Lisbon) are "
            f"{result['distinct_city_keys']} different keys and put overwrites neither. "
            f"The tier holds {result['kv_keys']} keys and both cities forever, which is "
            "what makes the missing fact type expensive rather than untidy",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
