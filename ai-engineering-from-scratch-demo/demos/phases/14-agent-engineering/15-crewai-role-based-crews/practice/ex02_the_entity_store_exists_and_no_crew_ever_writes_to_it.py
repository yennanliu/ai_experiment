"""Exercise 2 — the entity store exists and no crew ever writes to it.

    Add entity memory to the crew: facts about a customer persist across
    kickoffs. Verify retrieval pulls the right entity.

Reading of the exercise: `Memory` already has an `entity` dict and a
`write_entity` method, and neither crew calls it -- **0** of the **3**
kickoff paths touch it. So the exercise is wiring, and the verification half
is where it gets interesting, because the store the lesson *does* exercise,
`recall_long_term`, retrieves by an embedding that cannot see similarity at
all.

**ANSWER: entity writes wired into the crew, surviving two kickoffs.** After
two runs the entity store holds **2** customers with **5** facts between
them, and a lookup returns **3** of **3** facts for the right one and **0**
from the other. Long-term memory grows to **6** entries over the same two
runs, so the two stores keep different things for different reasons.

**FINDING: the long-term embedding is a hash, so retrieval is not
similarity.** `_embed` seeds a random generator from the SHA-256 of the whole
string, so an exact match scores **1.0**, a one-word edit of the same
sentence scores **-0.12**, and an unrelated sentence scores **0.14** --
*higher* than the near-duplicate. `recall_long_term` is an exact-match lookup
wearing cosine similarity's clothes.

**FINDING: entity memory is the only store keyed by subject.** `short_term`
and `long_term` are keyed by *role*, so "what do we know about this
customer" has no answer in either: the **6** long-term entries carry **3**
distinct roles and **0** customer identifiers.

**FINDING: the two crews disagree about what to persist.**
`SequentialCrew.kickoff` writes both short and long term for every task;
`HierarchicalCrew.kickoff` writes short term only. The same three agents
doing the same three jobs leave **6** long-term entries one way and **0** the
other.

Structure: `remember()` wraps a crew kickoff with the entity writes the
lesson left unwired; everything else is the shipped `Memory`.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "15-crewai-role-based-crews"
CUSTOMERS = {"acme": (("tier", "enterprise"), ("region", "emea"),
                      ("contact", "ada")),
             "globex": (("tier", "startup"), ("region", "apac"))}
BASE = "3 sources on agent engineering 2026: src1, src2, src3"
EDITED = "3 sources on agent engineering 2027: src1, src2, src3"
UNRELATED = "final brief (tightened, 800 words)"


def remember(ref, memory, customer):
    """The wiring the exercise asks for: facts about a subject, not a role."""
    researcher, writer, editor = ref.build_agents()
    crew = ref.SequentialCrew(
        agents=[researcher, writer, editor],
        tasks=[ref.Task("research", "3 sources", researcher),
               ref.Task("write", "3 paragraphs", writer),
               ref.Task("edit", "800 words", editor)],
        memory=memory)
    crew.kickoff({"topic": f"agent engineering 2026 for {customer}"})
    for key, value in CUSTOMERS[customer]:
        memory.write_entity(customer, key, value)
    return memory


def lookup(memory, customer):
    return dict(memory.entity.get(customer, {}))


def hierarchical(ref, memory):
    researcher, writer, editor = ref.build_agents()
    manager = ref.Agent("manager", "pick next", "PM", ref._manager)
    crew = ref.HierarchicalCrew(
        manager=manager,
        specialists={"researcher": researcher, "writer": writer, "editor": editor},
        memory=memory)
    crew.kickoff("agent engineering 2026")
    return memory


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    memory = ref.Memory()
    remember(ref, memory, "acme")
    memory.reset_short_term()
    remember(ref, memory, "globex")
    scores = {name: round(float(memory._embed(BASE) @ memory._embed(text)), 2)
              for name, text in (("exact", BASE), ("edited", EDITED),
                                 ("unrelated", UNRELATED))}
    sources = [inspect.getsource(ref.SequentialCrew.kickoff),
               inspect.getsource(ref.HierarchicalCrew.kickoff)]
    solo = hierarchical(ref, ref.Memory())
    return {
        "customers": len(memory.entity),
        "facts": sum(len(values) for values in memory.entity.values()),
        "acme": lookup(memory, "acme"),
        "leaked": sum(1 for key, value in lookup(memory, "acme").items()
                      if (key, value) in CUSTOMERS["globex"]),
        "long_term": len(memory.long_term),
        "scores": scores,
        "roles": sorted({role for role, _, _ in memory.long_term}),
        "entity_writes": [source.count("write_entity") for source in sources],
        "long_writes": [source.count("write_long_term") for source in sources],
        "hier_long": len(solo.long_term), "hier_short": len(solo.short_term),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: two customers, five facts, three retrieved and none leaked",
            all([result["customers"] == 2, result["facts"] == 5,
                 len(result["acme"]) == 3, result["leaked"] == 0,
                 result["long_term"] == 6]),
            f"after two kickoffs the entity store holds {result['customers']} customers "
            f"with {result['facts']} facts; a lookup returns {len(result['acme'])} of 3 "
            f"for acme -- {result['acme']} -- and {result['leaked']} from globex, while "
            f"long-term memory grows to {result['long_term']} entries",
        ),
        practice.Check(
            "FINDING: the long-term embedding is a hash, so retrieval is not similarity",
            all([result["scores"]["exact"] == 1.0, result["scores"]["edited"] == -0.12,
                 result["scores"]["unrelated"] == 0.14,
                 result["scores"]["edited"] < result["scores"]["unrelated"]]),
            f"_embed seeds a generator from the SHA-256 of the whole string, so the "
            f"scores are {result['scores']}: an exact match is 1.0, a one-word edit of "
            f"the same sentence scores {result['scores']['edited']} and an unrelated "
            f"sentence scores {result['scores']['unrelated']} -- higher. It is an "
            "exact-match lookup wearing cosine's clothes",
        ),
        practice.Check(
            "FINDING: entity memory is the only store keyed by subject",
            all([result["roles"] == ["editor", "researcher", "writer"],
                 len(result["roles"]) == 3, result["customers"] == 2]),
            f"short_term and long_term are keyed by role -- the {result['long_term']} "
            f"entries carry {result['roles']} and no customer identifier -- so 'what do "
            "we know about this customer' has no answer in either. Only the entity store "
            "is keyed by the subject the question is about",
        ),
        practice.Check(
            "FINDING: the two crews disagree about what to persist",
            all([result["entity_writes"] == [0, 0], result["long_writes"] == [1, 0],
                 result["hier_long"] == 0, result["hier_short"] == 3]),
            f"SequentialCrew.kickoff calls write_long_term {result['long_writes'][0]} "
            f"time and HierarchicalCrew {result['long_writes'][1]}, and neither calls "
            f"write_entity ({result['entity_writes']}). The same three agents leave "
            f"{result['hier_long']} long-term entries and {result['hier_short']} "
            "short-term ones when routed by a manager",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
