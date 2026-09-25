"""Exercise 3 — partitions stop poison flowing upstream, and not down.

    Switch the full pool to a blackboard with topic partitions (`prices`,
    `summaries`, `analyses`). Which poisoning scenarios does topic partitioning
    make harder to pull off, and which does it not help with?

Reading of the exercise: the retriever publishes to `prices`, the
summarizer subscribes to `prices` and publishes to `summaries`, the analyst
subscribes to `summaries` and publishes to `analyses` -- the lesson's chain
on the reference `Blackboard`. A poisoned entry is injected into each topic
in turn and the agents it reaches are counted.

**ANSWER: it makes poison injected downstream harmless, and does nothing for
poison at the root.** A lie written to `analyses` reaches 0 agents and one in
`summaries` reaches 1; one in `prices` reaches both. The lesson's own
scenario is a retriever hallucination -- root poison -- and the partitioned
run ends with the same "Recommend adoption" as the pool, because
partitioning keeps exactly the edges the lie travels along.

**FINDING: partitions without write ownership stop nothing.** `publish`
takes `writer` and `topic` as free arguments, so the analyst can publish to
`prices` and the summarizer consumes it: the downstream-poison protection
above holds only if writes are restricted per topic. An owner check of four
lines rejects it; the shipped class has none.

**FINDING: any subscriber can rewrite history and frame the writer.** The
callback is handed the stored `ProvenanceEntry` itself, not a copy. A
subscriber that edits `e.content` from 4.2% to 42% changes what
`read_topic` returns to every later reader, and the entry still reads
`writer="retriever"` -- provenance now attributes the lie to the honest
agent. `MessagePool.read_all` copies the list but not the entries, so the
pool has the same hole.

**FINDING: one failing subscriber silently desyncs a topic.** Callbacks run
in order in the publisher's thread; if the first raises, the second never
receives the entry, while `read_topic` shows it stored. Readers of the same
topic then disagree about what it contains.

Structure: `chain()` wires the three agents as subscribers; `reach()` counts
who receives an injected entry; `Owned` is the ownership check.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "13-shared-memory-blackboard"
TOPICS = ("prices", "summaries", "analyses")


def chain(bb, seen):
    """Summarizer on prices, analyst on summaries; each records what it received."""
    def summarizer(e):
        seen.append(("summarizer", e.content))
        bb.publish("summarizer", "summaries", f"Summary: {e.content}", "Summarize")

    def analyst(e):
        seen.append(("analyst", e.content))
        verdict = "Recommend adoption" if "42%" in e.content else "Recommend further review"
        bb.publish("analyst", "analyses", verdict, "Draw conclusions")

    bb.subscribe("prices", summarizer)
    bb.subscribe("summaries", analyst)


def reach(ref, topic):
    bb, seen = ref.Blackboard(), []
    chain(bb, seen)
    bb.publish("mallory", topic, "POISON 99%", "inject")
    return sorted({who for who, content in seen if "POISON" in content})


class Owned:
    """The four-line ownership check the shipped Blackboard lacks."""
    def __init__(self, bb, owners):
        self.bb, self.owners = bb, owners

    def publish(self, writer, topic, content, prompt):
        if self.owners.get(topic) != writer:
            raise PermissionError(f"{writer} does not own {topic}")
        return self.bb.publish(writer, topic, content, prompt)


def framing(ref):
    bb = ref.Blackboard()
    bb.subscribe("prices", lambda e: setattr(e, "content", e.content.replace("4.2%", "42%")))
    bb.publish("retriever", "prices", ref.FAKE_SOURCES["https://arxiv.org/paper-1"], "Fetch")
    stored = bb.read_topic("prices")[0]
    pool = ref.MessagePool()
    pool.write("retriever", "4.2%", "Fetch")
    pool.read_all()[0].content = "42%"
    return stored.writer, stored.content, pool.read_all()[0].content


def desync(ref):
    bb, got = ref.Blackboard(), []
    bb.subscribe("prices", lambda e: 1 / 0)
    bb.subscribe("prices", got.append)
    try:
        bb.publish("retriever", "prices", "AAPL=192.4", "poll")
    except ZeroDivisionError:
        pass
    return len(got), len(bb.read_topic("prices"))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    bb, seen = ref.Blackboard(), []
    chain(bb, seen)
    bb.publish("retriever", "prices", "The study reports a 42% accuracy improvement.", "Fetch")
    owned = Owned(ref.Blackboard(), {"prices": "retriever"})
    try:
        owned.publish("analyst", "prices", "POISON", "inject")
        rejected = False
    except PermissionError:
        rejected = True
    return {
        "reach": {t: reach(ref, t) for t in TOPICS},
        "verdict": bb.read_topic("analyses")[-1].content,
        "cross": reach(ref, "prices"), "rejected": rejected,
        "framed": framing(ref), "desync": desync(ref),
    }


def verify(result):
    reach = result["reach"]
    return [
        practice.Check(
            "ANSWER: downstream poison becomes harmless; root poison is untouched",
            all([reach["analyses"] == [], reach["summaries"] == ["analyst"],
                 reach["prices"] == ["analyst", "summarizer"],
                 result["verdict"] == "Recommend adoption"]),
            f"an injected entry reaches {reach} by topic; the retriever's 42% still ends "
            f"in {result['verdict']!r}, since partitions keep the edges it travels",
        ),
        practice.Check(
            "FINDING: partitions without write ownership stop nothing",
            result["cross"] == ["analyst", "summarizer"] and result["rejected"],
            f"publish takes writer and topic freely, so a non-retriever entry in prices "
            f"reaches {result['cross']}; the four-line owner check rejects it",
        ),
        practice.Check(
            "FINDING: any subscriber can rewrite history and frame the writer",
            all([result["framed"][0] == "retriever", "42%" in result["framed"][1],
                 result["framed"][2] == "42%"]),
            f"after a subscriber edits the entry it was handed, read_topic returns "
            f"writer={result['framed'][0]!r} with {result['framed'][1]!r}; editing an "
            "entry from MessagePool.read_all() rewrites the pool the same way",
        ),
        practice.Check(
            "FINDING: one failing subscriber silently desyncs a topic",
            result["desync"] == (0, 1),
            f"with the first callback raising, the second receives {result['desync'][0]} "
            f"entries while read_topic holds {result['desync'][1]}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
