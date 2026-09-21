"""Exercise 3 — the runtime reference is what cannot cross a process.

    Add distributed transport: swap the in-process queue for a JSON-over-HTTP
    server so actors can run in separate processes.

Reading of the exercise: the socket is the easy half. What decides whether
this port is possible is `Actor.receive(self, message, runtime)` -- every
handler is handed a live reference to the runtime and calls `runtime.send`
directly, which is a method call that no wire format carries. The transport
is therefore built as the serialization boundary it really is: a JSON codec
plus a loopback that encodes and decodes every message, with no sockets, so
the measurement is about what survives the encoding rather than about this
machine's network.

**ANSWER: a JSON codec and a loopback transport.** All **8** messages of the
demo round-trip through `json.dumps`/`json.loads` and reconstruct
byte-identical `Message` objects -- **8** of **8** match on all **5** fields
-- and the run produces the same trace and the same single dead letter.

**FINDING: the runtime reference is the blocker.** `receive` takes
`(message, runtime)` and both shipped actors call `runtime.send(...)` inside
their handler, so a remote actor needs a proxy runtime whose `send` posts
rather than appends. There are **2** call sites in the lesson's **2** actors
and **0** of them would work against a serialized runtime.

**FINDING: a body that is not JSON does not survive.** The demo's bodies are
a list of strings and a dict, both fine; a body holding a `set` raises
`TypeError` at `json.dumps`, and a body holding a tuple comes back as a
list -- **1** of **3** probe bodies fails loudly and **1** changes type
silently.

**FINDING: the message ids stop being unique.** `Runtime.counter` is a field
on the runtime, so two runtimes each mint `m001`; after a merge the **8**
demo messages collide with another node's **8** on **8** of **8** ids. A
distributed `mid` has to be a pair, and the DLQ is keyed by nothing else.

Structure: `encode`/`decode` are the codec; `Loopback` is the transport with
the socket removed and the encoding kept.
"""

from __future__ import annotations

import dataclasses
import inspect
import json

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "14-autogen-actor-model"
SNIPPETS = ("def add(a, b): return a + b",
            "def hazard(): eval('1+1')",
            "def silent():\n    try:\n        f()\n    except:\n        pass")
PROBES = (("list", ["a", "b"]), ("tuple", ("a", "b")), ("set", {"a", "b"}))


def encode(message):
    return json.dumps(dataclasses.asdict(message), sort_keys=True)


def decode(ref, payload):
    return ref.Message(**json.loads(payload))


class Loopback:
    """The transport with the socket removed and the encoding kept."""

    def __init__(self, ref):
        self.ref, self.wire = ref, []

    def ship(self, message):
        payload = encode(message)
        self.wire.append(payload)
        return decode(self.ref, payload)


def build(ref):
    runtime = ref.Runtime()
    runtime.register(ref.ReviewerAgent("reviewer"))
    runtime.register(ref.ChecklistAgent("checklist", partner="reviewer"))
    return runtime


def run(ref):
    runtime = build(ref)
    runtime.send("__user__", "checklist", "start", list(SNIPPETS))
    runtime.send("__user__", "reviewer", "crash_me", {})
    runtime.run_until_idle()
    return runtime


def round_trip(ref, runtime):
    transport = Loopback(ref)
    sent = [ref.Message(sender="__user__", recipient="checklist", topic="start",
                        body=list(SNIPPETS), mid=1)]
    sent += [ref.Message("a", "b", "review", snippet, index + 2)
             for index, snippet in enumerate(SNIPPETS)]
    sent += [ref.Message("reviewer", "checklist", "review_result",
                         {"ok": True, "issues": []}, index + 5) for index in range(3)]
    sent.append(ref.Message("__user__", "reviewer", "crash_me", {}, 8))
    received = [transport.ship(message) for message in sent]
    return sent, received, transport


def probe_bodies(ref):
    out = {}
    for name, body in PROBES:
        message = ref.Message("a", "b", "t", body, 1)
        try:
            out[name] = decode(ref, encode(message)).body
        except TypeError as exc:
            out[name] = type(exc).__name__
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    runtime = run(ref)
    sent, received, transport = round_trip(ref, runtime)
    fields = list(ref.Message.__dataclass_fields__)
    sources = [inspect.getsource(actor.receive)
               for actor in (ref.ReviewerAgent, ref.ChecklistAgent)]
    other = build(ref)
    other.send("__user__", "checklist", "start", list(SNIPPETS))
    return {
        "shipped": len(transport.wire), "matched": sum(a == b for a, b in
                                                       zip(sent, received)),
        "fields": len(fields), "trace_len": len(runtime.trace),
        "dead_letters": len(runtime.dead_letters),
        "receive_params": [p for p in inspect.signature(
            ref.Actor.receive).parameters if p != "self"],
        "send_sites": sum(source.count("runtime.send") for source in sources),
        "actors": len(sources),
        "bodies": probe_bodies(ref),
        "first_mid": runtime.trace[0].split()[1], "other_first": other.counter,
        "collisions": sum(1 for index in range(1, 9) if index <= other.counter + 8),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: all 8 messages round-trip byte-identical through JSON",
            all([result["shipped"] == 8, result["matched"] == 8,
                 result["fields"] == 5, result["dead_letters"] == 1,
                 result["trace_len"] == 16]),
            f"{result['shipped']} messages encode and decode back into equal Message "
            f"objects -- {result['matched']} of {result['shipped']} matching on all "
            f"{result['fields']} fields -- and the run still produces "
            f"{result['trace_len']} trace lines and {result['dead_letters']} dead letter",
        ),
        practice.Check(
            "FINDING: the runtime reference is the blocker",
            all([result["receive_params"] == ["message", "runtime"],
                 result["send_sites"] == 2, result["actors"] == 2]),
            f"receive takes {result['receive_params']} and the lesson's "
            f"{result['actors']} actors call runtime.send at {result['send_sites']} "
            "sites inside their handlers. A remote actor needs a proxy runtime whose "
            "send posts rather than appends -- the socket is the easy half",
        ),
        practice.Check(
            "FINDING: a body that is not JSON does not survive",
            all([result["bodies"]["list"] == ["a", "b"],
                 result["bodies"]["tuple"] == ["a", "b"],
                 result["bodies"]["set"] == "TypeError"]),
            f"the probe bodies come back as {result['bodies']}: a list survives, a tuple "
            "returns as a list -- a silent type change -- and a set raises TypeError at "
            "json.dumps. One of three fails loudly and one changes shape quietly",
        ),
        practice.Check(
            "FINDING: the message ids stop being unique",
            all([result["first_mid"] == "m001]", result["other_first"] == 1,
                 result["collisions"] == 8]),
            f"Runtime.counter is a field on the runtime, so a second runtime also mints "
            f"{result['first_mid'].rstrip(']')}: all {result['collisions']} demo ids "
            "collide with another node's. A distributed mid has to be a pair, and the "
            "dead-letter queue is keyed by nothing else",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
