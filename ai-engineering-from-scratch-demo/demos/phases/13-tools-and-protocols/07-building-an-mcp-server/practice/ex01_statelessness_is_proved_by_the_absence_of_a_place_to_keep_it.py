"""Exercise 1 — statelessness is proved by the absence of a place to keep it.

    Remove capabilities from one request and prove the server does not reuse
    the previous request's declaration.

Reading of the exercise: "prove" is taken as more than one failing request. A
rejection shows the server did not reuse capabilities on that call; only the
absence of anywhere to have kept them shows it cannot. This server, unlike
Lesson 13.06's, *does* hold mutable state between calls -- so the proof has to
separate protocol state from application state, and the two behave differently.

**ANSWER: `-32602`, on every method, regardless of what came before.** A
capability-bearing `tools/list` succeeds, the next request without them fails,
and the one after with them succeeds again. All **6** methods reject the
omission identically.

**FINDING: this server does keep state between requests, and it is not protocol
state.** `NOTES` is a module dict that `notes_create` mutates: `resources/list`
returns **3** entries before a create and **4** after. So "stateless" here is a
claim about `_meta` only, and the exercise's proof strategy -- show nothing
carries over -- would be false if applied to the whole server.

**FINDING: the two kinds of state are distinguished by where they are read.**
`validate_request` reads `params._meta` and nothing else; every executor reads
`NOTES` and never touches `_meta`. Protocol context is re-established per
message and application data persists behind the handlers -- which is the split
the lesson's own "durable application state" paragraph describes, made
measurable.

**FINDING: and the capability that is required is still never consulted.** An
empty `{}` passes validation, and `SERVER_CAPABILITIES` advertises **3**
capability groups that no handler reads. The client must declare something; the
server does not care what.

Structure: `send` dispatches one request with a chosen `_meta` edit, `sequence`
runs the three-request proof, and `reads_meta` reports which functions touch
protocol context.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "07-building-an-mcp-server"
METHODS = ("server/discover", "tools/list", "resources/list", "resources/read",
           "prompts/list", "prompts/get")
PARAMS = {"resources/read": {"uri": "notes://note-1"},
          "prompts/get": {"name": "review_note", "arguments": {"note_id": "note-1"}}}


def send(ref, request_id, method="tools/list", drop=None, capabilities=None):
    message = ref.make_request(request_id, method, PARAMS.get(method))
    if capabilities is not None:
        message["params"]["_meta"][ref.CAPABILITIES_KEY] = capabilities
    if drop is not None:
        message["params"]["_meta"].pop(drop, None)
    return ref.dispatch(message)


def outcome(response):
    if "error" in response:
        return response["error"]["code"]
    return response["result"]["resultType"]


def sequence(ref):
    """With capabilities, without, with again -- the three-request proof."""
    return [outcome(send(ref, 1, capabilities={"sampling": {}})),
            outcome(send(ref, 2, drop=ref.CAPABILITIES_KEY)),
            outcome(send(ref, 3, capabilities={"sampling": {}}))]


def mentions(functions, needle):
    """Which of these functions name `needle` in their source."""
    return {name: needle in inspect.getsource(fn) for name, fn in functions.items()}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    ref.reset_notes()
    before = len(send(ref, 10, "resources/list")["result"]["resources"])
    created = ref.dispatch(ref.make_request(
        12, "tools/call", {"name": "notes_create",
                           "arguments": {"title": "T", "body": "B"}}))
    after = len(send(ref, 13, "resources/list")["result"]["resources"])
    ref.reset_notes()
    executors = dict(ref.TOOL_EXECUTORS)
    return {
        "sequence": sequence(ref),
        "by_method": {method: outcome(send(ref, 20, method, drop=ref.CAPABILITIES_KEY))
                      for method in METHODS},
        "methods": len(METHODS),
        "resources_before": before, "resources_after": after,
        "created_ok": created["result"]["isError"] is False,
        "validate_reads_meta": "_meta" in inspect.getsource(ref.validate_request),
        "executors_read_meta": mentions(executors, "_meta"),
        "executors_read_notes": mentions(executors, "NOTES"),
        "empty_caps": outcome(send(ref, 30, capabilities={})),
        "advertised": sorted(ref.SERVER_CAPABILITIES),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: -32602 on every method, regardless of what came before",
            all([result["sequence"] == ["complete", -32602, "complete"],
                 set(result["by_method"].values()) == {-32602},
                 len(result["by_method"]) == result["methods"]]),
            f"with capabilities, without, with again gives {result['sequence']}, and all "
            f"{result['methods']} methods reject the omission identically: "
            f"{result['by_method']}. A success never makes the next request cheaper",
        ),
        practice.Check(
            "FINDING: this server does keep state between requests, and it is not protocol state",
            all([result["resources_before"] == 3, result["resources_after"] == 4,
                 result["created_ok"]]),
            f"NOTES is a module dict that notes_create mutates: resources/list returns "
            f"{result['resources_before']} entries before a create and "
            f"{result['resources_after']} after. 'Stateless' here is a claim about _meta "
            "only, and the exercise's proof strategy would be false if applied to the whole "
            "server",
        ),
        practice.Check(
            "FINDING: the two kinds of state are distinguished by where they are read",
            all([result["validate_reads_meta"],
                 not any(result["executors_read_meta"].values()),
                 all(result["executors_read_notes"].values())]),
            f"validate_request reads params._meta; the executors read "
            f"{result['executors_read_notes']} and touch _meta in "
            f"{sum(result['executors_read_meta'].values())} of "
            f"{len(result['executors_read_meta'])} cases. Protocol context is re-established "
            "per message and application data persists behind the handlers -- the split the "
            "lesson's durable-application-state paragraph describes",
        ),
        practice.Check(
            "FINDING: and the required capability is still never consulted",
            all([result["empty_caps"] == "complete",
                 result["advertised"] == ["prompts", "resources", "tools"]]),
            f"an empty capability object returns {result['empty_caps']}, and the server "
            f"advertises {result['advertised']} -- {len(result['advertised'])} groups no "
            "handler reads. The client must declare something; the server does not care what",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
