"""Exercise 5 — the schema cannot key on the performative.

    Design a minimal JSON-Schema for the `content` field of a `request`
    performative in your own system. What does that schema give you that pure
    natural-language does not, and what does it cost?

Reading of the exercise: write the schema, then run it over the `request`
messages this lesson already builds -- because the first thing that happens is
that it rejects one of them, and that rejection is the answer to both halves
of the question.

**ANSWER: the schema has to key on `:ontology`, not on the performative, and
what it buys is the rejection the constructor will not make.** The lesson
builds **2** `request` messages. One carries `{'symbol': 'IBM'}`, an object;
the other carries `'def f(x): return x'`, a string. A schema written for
"request content" that says `{"type": "object"}` admits **1** of **2**; the
only schema keyed on the performative that admits both is `{}`, which has
**0** keywords and checks nothing. Keyed on the ontology instead -- one schema
for `lookup_stock`, one for `review-python` -- it admits **2** of **2** and
still rejects a tool call with its argument missing, which `ACLMessage`
accepts without complaint.

**FINDING: the cost is one schema per ontology, and there are five.** The
module constructs envelopes under **5** distinct ontologies and **3** distinct
languages. A performative-keyed schema is one artefact; an ontology-keyed one
is five, and each new tool adds a sixth. That is the price: the checkable
thing is the tool's argument list, which is exactly the thing the performative
abstracts away.

**FINDING: the constructor validates one field of nine.** `__post_init__`
checks `performative` against a **16**-name whitelist and nothing else.
`language`, `ontology` and `protocol` are free strings, so an envelope can
claim `:language JSON` over a Python repr -- and does -- and no assertion in
the module notices.

**FINDING: what it gives you is a rejection before dispatch.** A `tools/call`
with `arguments` empty builds a valid `ACLMessage`: **0** exceptions. The
ontology schema rejects it on the missing `symbol`. Natural language cannot
state "this call needs a symbol" in a form a receiver can check, which is the
whole of what the schema is worth here.

Structure: `matches()` is a three-keyword JSON-Schema subset; `admits()` runs
each candidate schema over the lesson's own request messages.
"""

from __future__ import annotations

import dataclasses
import inspect
import re

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "02-fipa-acl-heritage"
MCP_CALL = {"jsonrpc": "2.0", "method": "tools/call",
            "params": {"name": "lookup_stock", "arguments": {"symbol": "IBM"}}, "id": 42}
A2A_TASK = {"client": "research-host", "agent": "code-review-agent",
            "skill": "review-python", "input": "def f(x): return x", "task_id": "t-12"}
MCP_READ = {"params": {"uri": "file:///etc/hosts"}, "id": 43}
EMPTY_CALL = {"params": {"name": "lookup_stock", "arguments": {}}, "id": 44}
TYPES = {"object": dict, "string": str, "integer": int, "array": list}
BY_ONTOLOGY = {
    "lookup_stock": {"type": "object", "required": ["symbol"],
                     "properties": {"symbol": {"type": "string"}}},
    "review-python": {"type": "string"},
}


def matches(schema, value):
    """A JSON-Schema subset: type, required, properties. Enough to be wrong usefully."""
    expected = schema.get("type")
    if expected and not isinstance(value, TYPES[expected]):
        return False
    if not isinstance(value, dict):
        return True
    if any(key not in value for key in schema.get("required", ())):
        return False
    return all(matches(sub, value[key])
               for key, sub in schema.get("properties", {}).items() if key in value)


def envelopes(ref):
    """Every envelope the module builds, across all five constructors."""
    return [ref.mcp_tools_call_to_acl(MCP_CALL), ref.mcp_resources_read_to_acl(MCP_READ),
            ref.a2a_task_create_to_acl(A2A_TASK),
            ref.a2a_subscribe_to_acl("t-12", "c", "a"),
            ref.ACLMessage("cfp", "m", "w", "t", ontology="contract-net")]


def admits(messages, schema=None):
    """How many messages a schema admits -- one fixed, or one chosen per :ontology."""
    return sum(matches(schema if schema is not None else BY_ONTOLOGY[m.ontology],
                       m.content) for m in messages)


def keywords(schema):
    """How much a schema actually constrains, counted in keywords."""
    return len(schema) + sum(keywords(sub) for sub in schema.get("properties", {}).values())


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    src = inspect.getsource(ref)
    built = envelopes(ref)
    messages = [m for m in built if m.performative == "request"]
    empty = ref.mcp_tools_call_to_acl(EMPTY_CALL)
    return {
        "requests": len(messages),
        "types": sorted({type(m.content).__name__ for m in messages}),
        "as_object": admits(messages, {"type": "object"}),
        "as_empty": admits(messages, {}),
        "empty_keywords": keywords({}),
        "by_ontology": admits(messages),
        "ontology_keywords": sum(keywords(s) for s in BY_ONTOLOGY.values()),
        "ontologies": len({m.ontology for m in built}),
        "languages": len(set(re.findall(r'language="(\w+)"', src)) | {"SL0"}),
        "validated_fields": src.count("raise ValueError"),
        "fields": len(dataclasses.fields(ref.ACLMessage)),
        "empty_built": empty.content == {},
        "empty_rejected": not matches(BY_ONTOLOGY["lookup_stock"], empty.content),
        "whitelist": len(ref.PERFORMATIVES),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the schema has to key on :ontology, not on the performative",
            all([result["requests"] == 2, result["types"] == ["dict", "str"],
                 result["as_object"] == 1, result["as_empty"] == 2,
                 result["empty_keywords"] == 0, result["by_ontology"] == 2]),
            f"the {result['requests']} request messages carry a "
            f"{' and a '.join(result['types'])}; a performative-keyed "
            f"{{\"type\": \"object\"}} admits {result['as_object']} and the only one "
            f"admitting both has {result['empty_keywords']} keywords, while two "
            f"ontology-keyed schemas admit {result['by_ontology']} and constrain "
            f"{result['ontology_keywords']}",
        ),
        practice.Check(
            "FINDING: the cost is one schema per ontology, and there are five",
            all([result["ontologies"] == 5, result["languages"] == 3]),
            f"the module builds envelopes under {result['ontologies']} ontologies and "
            f"{result['languages']} languages, so the performative-keyed schema is one "
            f"artefact and the ontology-keyed one is {result['ontologies']} -- the "
            "checkable thing is the argument list the performative abstracts away",
        ),
        practice.Check(
            "FINDING: the constructor validates one field of nine",
            all([result["validated_fields"] == 1, result["fields"] == 9,
                 result["whitelist"] == 16]),
            f"__post_init__ raises once, checking performative against "
            f"{result['whitelist']} names and leaving the other {result['fields'] - 1} "
            "fields free -- which is how an envelope claims :language JSON over a "
            "Python repr with nothing noticing",
        ),
        practice.Check(
            "FINDING: what it gives you is a rejection before dispatch",
            all([result["empty_built"], result["empty_rejected"]]),
            "a tools/call with arguments empty builds a valid ACLMessage and raises "
            "nothing; the lookup_stock schema rejects it on the missing symbol, which "
            "is the assertion natural language cannot put in a form the receiver checks",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
