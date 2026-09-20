"""Exercise 4 — the ack drops silently where the notification raises.

    Implement a POST-response SSE adapter for `subscriptions/listen`. Do not
    add GET, `Last-Event-ID`, or a session header.

Reading of the exercise: the three prohibitions are the specification, so they
are checked as absences -- no GET route, no `id:` field on any frame, no
session header on the response -- rather than merely not written. Building the
adapter then puts the lesson's two subscription helpers next to each other for
the first time, and they disagree about what to do with a task id the caller
does not own.

**ANSWER: one POST, one SSE body, and nothing the prohibitions forbid.** The
adapter emits the acknowledgement, one `notifications/tasks` per accepted
task, and a terminal result -- **4** frames for **2** tasks, every one tagged
with the subscription id. The response carries **0** `id:` SSE fields, **0**
session headers, and the adapter exposes **1** verb.

**FINDING: the two helpers disagree about an unowned task id.**
`subscription_acknowledgement` filters by owner and returns the survivors, so
asking for someone else's task drops it from `taskIds` with no error;
`task_notification` calls `_owned_task` and raises `-32602 task not found` for
the same id. One silently narrows, the other refuses -- so the adapter has to
send notifications only for the ids the ack accepted, and the ack is the
authorization boundary.

**FINDING: the acknowledgement is the only place the narrowing is visible.**
A client asking for **2** ids where **1** is foreign gets an ack listing **1**
and then notifications for **1**. Nothing says the other was rejected; the
client has to compare the ack with what it sent, which is the same contract
the `notifications` filter has in the subscriptions lesson.

**FINDING: the prohibitions cost nothing because the tasks are the durable
thing.** Discarding the stream entirely and polling `tasks/get` recovers the
identical status and result -- the stream carries **0** information that the
store does not. That is why no resume mechanism is needed: there is nothing to
resume, only something to re-read.

Structure: `listen` is the adapter -- a generator of frames for one POST --
and `Response` is the minimal HTTP shape it is delivered in, so the absent
header and the absent verb are properties of an object rather than of prose.
"""

from __future__ import annotations

import pathlib
import tempfile

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "13-mcp-async-tasks"
SUBSCRIPTION = "sub-1"
OWNER, STRANGER = "user-42", "user-99"


class Response:
    """What the POST returns: a body, a content type, and nothing else."""

    verbs = ("POST",)

    def __init__(self, frames):
        self.status = 200
        self.headers = {"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}
        self.body = "".join(f"data: {frame}\n\n" for frame in frames)


def listen(ref, service, task_ids, principal=OWNER):
    """The adapter: ack, one notification per accepted task, then the result."""
    ack = service.subscription_acknowledgement(
        task_ids, subscription_id=SUBSCRIPTION, principal=principal)
    yield ack
    for task_id in ack["params"]["notifications"]["taskIds"]:
        yield service.task_notification(
            task_id, subscription_id=SUBSCRIPTION, principal=principal)
    terminal = ref.complete()
    # complete() writes _meta last, so the subscription id has to be merged after
    terminal["_meta"] = {**terminal["_meta"], ref_key(): SUBSCRIPTION}
    yield {"jsonrpc": "2.0", "id": SUBSCRIPTION, "result": terminal}


def tag(frame):
    meta = frame.get("params", frame.get("result", {})).get("_meta", {})
    return meta.get(ref_key())


def ref_key():
    return "io.modelcontextprotocol/subscriptionId"


def create(ref, service, principal):
    response = service.dispatch(ref.make_request(
        1, "tools/call", {"name": "generate_report", "arguments": {"size": "small"}}),
        principal=principal)
    return response["result"]["taskId"]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    with tempfile.TemporaryDirectory() as directory:
        service = ref.TaskService(pathlib.Path(directory) / "tasks")
        mine = [create(ref, service, OWNER), create(ref, service, OWNER)]
        theirs = create(ref, service, STRANGER)

        frames = list(listen(ref, service, mine))
        response = Response(frames)
        mixed = list(listen(ref, service, [mine[0], theirs]))
        ack = mixed[0]["params"]["notifications"]["taskIds"]

        raised = None
        try:
            service.task_notification(theirs, subscription_id=SUBSCRIPTION, principal=OWNER)
        except ref.McpError as exc:
            raised = exc.message

        streamed = frames[1]["params"]
        polled = service.dispatch(ref.make_request(
            2, "tasks/get", {"taskId": mine[0]}))["result"]
        return {
            "frames": len(frames),
            "methods": [f.get("method") for f in frames],
            "tags": [tag(f) for f in frames],
            "sse_ids": response.body.count("\nid:") + response.body.count("id: "),
            "headers": sorted(response.headers), "verbs": list(Response.verbs),
            "ack_of_mixed": ack, "mixed_frames": len(mixed),
            "requested": 2, "raised": raised,
            "recoverable": all(streamed[k] == polled[k]
                               for k in ("taskId", "status", "statusMessage")),
            "stream_only": sorted(set(streamed) - set(polled) - {"_meta"}),
        }


def verify(result):
    return [
        practice.Check(
            "ANSWER: one POST, one SSE body, and none of the three prohibited things",
            all([result["frames"] == 4,
                 result["methods"] == ["notifications/subscriptions/acknowledged",
                                       "notifications/tasks", "notifications/tasks", None],
                 result["tags"] == [SUBSCRIPTION] * 4, result["sse_ids"] == 0,
                 result["headers"] == ["Cache-Control", "Content-Type"],
                 result["verbs"] == ["POST"]]),
            f"the adapter emits {result['frames']} frames for two tasks, "
            f"{[m or 'result' for m in result['methods']]}, each tagged {SUBSCRIPTION!r}. "
            f"The response carries {result['sse_ids']} SSE id fields, headers "
            f"{result['headers']} with no session id, and exposes {result['verbs']}",
        ),
        practice.Check(
            "FINDING: the two helpers disagree about an unowned task id",
            all([len(result["ack_of_mixed"]) == 1, result["raised"] == "task not found",
                 result["mixed_frames"] == 3]),
            f"asking for two ids where one is foreign gives an ack listing "
            f"{len(result['ack_of_mixed'])}, because subscription_acknowledgement filters by "
            f"owner and returns survivors, while task_notification raises "
            f"{result['raised']!r} for the same id. One narrows silently and the other "
            "refuses, so the adapter must follow the ack",
        ),
        practice.Check(
            "FINDING: the acknowledgement is the only place the narrowing is visible",
            all([len(result["ack_of_mixed"]) < result["requested"],
                 result["mixed_frames"] == len(result["ack_of_mixed"]) + 2]),
            f"{result['requested']} ids requested, {len(result['ack_of_mixed'])} accepted, "
            f"{result['mixed_frames']} frames delivered. Nothing says the other was "
            "rejected -- the client has to compare the acknowledgement with what it sent",
        ),
        practice.Check(
            "FINDING: the prohibitions cost nothing, because the tasks are the durable thing",
            all([result["recoverable"], result["stream_only"] == []]),
            f"discarding the stream and polling tasks/get recovers the identical taskId, "
            f"status and statusMessage, and the notification carries "
            f"{result['stream_only']} that the polled result does not. There is nothing to "
            "resume, only something to re-read, which is why no resume mechanism is needed",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
