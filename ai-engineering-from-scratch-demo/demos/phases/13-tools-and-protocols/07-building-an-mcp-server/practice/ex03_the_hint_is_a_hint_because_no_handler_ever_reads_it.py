"""Exercise 3 — the hint is a hint because no handler ever reads it.

    Add a destructive `notes_delete` tool and require an authorization check
    inside the executor. Keep `destructiveHint` as a UX hint only.

Reading of the exercise: the tool is added, the check is written into the
executor, and "keep the hint a hint" is verified rather than assumed -- the
lesson already ships three annotated tools, so whether annotations are advisory
is a fact about the existing code and not a decision this exercise makes. They
are. The interesting part is where the denial has to be raised, because the
server has two error channels and they mean different things to a model.

**ANSWER: `notes_delete` with the check inside the executor, denying by
default.** Authorised, it removes the note and `resources/list` drops from
**4** to **3**. Unauthorised, the note survives and the caller gets an error.
`destructiveHint: True` rides along in `tools/list` and changes nothing.

**FINDING: no handler reads an annotation, so the hint could not be anything
else.** Searching every `handle_*` function for `annotations` finds **0**, and
`handle_tools_call` looks up `TOOL_EXECUTORS` by name without consulting the
tool record at all. The shipped `readOnlyHint` and `idempotentHint` on three
tools are advisory for the same reason: the dispatcher never sees them.

**FINDING: the two error channels mean different things, and the exercise does
not say which to use.** `handle_tools_call` catches `ValueError` and returns
`isError: True` -- a *tool result* the model reads and may retry. `RpcProblem`
is **not** caught there, so it escapes to `dispatch` and becomes a JSON-RPC
`error` the model never sees as content. A denial raised the first way invites
a retry; raised the second way it terminates the call. Denying authorization is
the second kind.

**FINDING: and the destructive tool is the only one whose failure is
unrecoverable.** `notes_search` and `notes_list` can be retried freely, and
`notes_create` retried twice makes two notes. `notes_delete` retried after
success reports a missing note -- so the annotation that matters for retry is
`idempotentHint`, which the lesson sets on **2** of its **3** tools and which is
also read by nobody.

Structure: `delete_tool` is the tool record, `exec_notes_delete` is the executor
with the check, `call` dispatches through the lesson's own `tools/call`, and
`reads_annotations` searches the handlers.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "07-building-an-mcp-server"
ALLOWED = "owner"
DELETE_TOOL = {
    "name": "notes_delete",
    "description": ("Use when the user asks for a note to be removed permanently. "
                    "Do not use for archiving, which this does not do."),
    "inputSchema": {"type": "object",
                    "properties": {"note_id": {"type": "string"},
                                   "role": {"type": "string"}},
                    "required": ["note_id", "role"]},
    "annotations": {"destructiveHint": True, "idempotentHint": False},
}


def make_delete(ref):
    """The executor, with the authorization check inside it and denial by default."""
    def exec_notes_delete(arguments):
        if arguments.get("role") != ALLOWED:
            raise ref.RpcProblem(-32003, "Not authorized to delete notes",
                                 {"required_role": ALLOWED})
        note_id = arguments.get("note_id")
        if note_id not in ref.NOTES:
            raise ValueError(f"no such note: {note_id}")
        del ref.NOTES[note_id]
        return [{"type": "text", "text": f"Deleted {note_id}"}]
    return exec_notes_delete


def call(ref, name, arguments, request_id=1):
    return ref.dispatch(ref.make_request(request_id, "tools/call",
                                         {"name": name, "arguments": arguments}))


def outcome(response):
    if "error" in response:
        return ("rpc_error", response["error"]["code"])
    return ("tool_result", response["result"]["isError"])


def resources(ref):
    return len(ref.dispatch(ref.make_request(9, "resources/list"))["result"]["resources"])


def reads_annotations(ref):
    return sum("annotations" in inspect.getsource(getattr(ref, name))
               for name in dir(ref) if name.startswith("handle_"))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    ref.reset_notes()
    ref.TOOLS.append(DELETE_TOOL)
    ref.TOOL_EXECUTORS["notes_delete"] = make_delete(ref)
    try:
        before = resources(ref)
        denied = outcome(call(ref, "notes_delete", {"note_id": "note-1", "role": "guest"}))
        survived = resources(ref)
        allowed = outcome(call(ref, "notes_delete", {"note_id": "note-1",
                                                     "role": ALLOWED}))
        after = resources(ref)
        twice = outcome(call(ref, "notes_delete", {"note_id": "note-1",
                                                   "role": ALLOWED}))
        listed = ref.dispatch(ref.make_request(2, "tools/list"))["result"]["tools"]
        hints = {tool["name"]: tool.get("annotations", {}) for tool in listed}
        bad_args = outcome(call(ref, "notes_search", {"query": ""}))
        idempotent = sum(1 for tool in listed
                         if tool.get("annotations", {}).get("idempotentHint") is True)
        return {
            "before": before, "denied": denied, "survived": survived,
            "allowed": allowed, "after": after, "twice": twice,
            "hint": hints["notes_delete"],
            "tools": len(listed),
            "handlers_reading_annotations": reads_annotations(ref),
            "call_reads_tool_record": "annotations" in inspect.getsource(
                ref.handle_tools_call),
            "caught": inspect.getsource(ref.handle_tools_call).split("except ")[1]
                      .split(")")[0] + ")",
            "bad_args": bad_args,
            "idempotent_tools": idempotent,
        }
    finally:
        ref.TOOLS.remove(DELETE_TOOL)
        ref.TOOL_EXECUTORS.pop("notes_delete", None)
        ref.reset_notes()


def verify(result):
    return [
        practice.Check(
            "ANSWER: notes_delete with the check inside the executor, denying by default",
            all([result["before"] == 3, result["denied"] == ("rpc_error", -32003),
                 result["survived"] == 3, result["allowed"] == ("tool_result", False),
                 result["after"] == 2,
                 result["hint"] == {"destructiveHint": True, "idempotentHint": False}]),
            f"unauthorised, the call returns {result['denied']} and the note survives "
            f"({result['survived']} resources). Authorised, it returns {result['allowed']} "
            f"and resources drop to {result['after']}. The annotation "
            f"{result['hint']} rides along in tools/list and changes nothing",
        ),
        practice.Check(
            "FINDING: no handler reads an annotation, so the hint could not be anything else",
            all([result["handlers_reading_annotations"] == 0,
                 not result["call_reads_tool_record"], result["tools"] == 4]),
            f"searching every handle_* function for 'annotations' finds "
            f"{result['handlers_reading_annotations']}, and handle_tools_call looks up "
            "TOOL_EXECUTORS by name without consulting the tool record at all. The shipped "
            "readOnlyHint and idempotentHint are advisory for the same reason: the "
            "dispatcher never sees them",
        ),
        practice.Check(
            "FINDING: the two error channels mean different things",
            all([result["caught"] == "(KeyError, TypeError, ValueError)",
                 result["denied"][0] == "rpc_error",
                 result["bad_args"] == ("tool_result", True)]),
            f"handle_tools_call catches {result['caught']} and turns them into a tool result "
            f"the model reads -- a bad notes_search query gives {result['bad_args']}. "
            "RpcProblem is not in that list, so it escapes to dispatch and becomes a "
            "JSON-RPC error the model never sees as content. A denial raised the first way "
            "invites a retry; raised the second way it terminates the call",
        ),
        practice.Check(
            "FINDING: the destructive tool is the only one whose failure is unrecoverable",
            all([result["twice"] == ("tool_result", True),
                 result["idempotent_tools"] == 2]),
            f"retried after success, notes_delete returns {result['twice']} -- the note is "
            f"already gone. notes_search and notes_list retry freely and notes_create "
            f"retried twice makes two notes. The annotation that matters for retry is "
            f"idempotentHint, set on {result['idempotent_tools']} of the lesson's 3 original "
            "tools and read by nobody",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
