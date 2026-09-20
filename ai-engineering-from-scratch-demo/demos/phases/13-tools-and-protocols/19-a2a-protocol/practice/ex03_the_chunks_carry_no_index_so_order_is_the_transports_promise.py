"""Exercise 3 — the chunks carry no index, so order is the transport's promise.

    Implement task streaming: the writer agent emits three incremental
    artifact chunks over SSE and the caller accumulates them.

Reading of the exercise: accumulation is trivial if the chunks arrive in
order, so the solution asks what makes that true. Nothing in the lesson's
`Part` or `Artifact` carries a sequence number, so the caller's `+=` is
relying on the stream, and the test therefore shuffles the chunks and
accumulates them again -- which produces a different, equally plausible
document with no error.

**ANSWER: three chunks over SSE, accumulated into one artifact.** The writer
emits **3** `data:` frames plus a terminal one, the caller concatenates their
text, and the result equals the single-shot artifact the non-streaming path
produces -- byte for byte. The task ends `completed` with **1** artifact.

**FINDING: nothing in the payload orders the chunks.** `Part` is
`(kind, payload)` and `Artifact` is `(name, mimeType, parts)` -- **2** and
**3** fields, no index, no offset, no total. Replaying the same three frames
shuffled yields a different document and the accumulator cannot tell: the
ordering guarantee lives entirely in the transport the lesson does not have.

**FINDING: the terminal frame is the only thing that says "done".** Without
it the accumulator cannot distinguish a finished artifact from a stalled one,
because a chunk count is never announced. The stream carries **4** frames to
deliver **3** pieces of content, and the fourth exists only to end it.

**FINDING: the card advertises streaming and the module implements none of
it.** `capabilities.streaming` is `True`, and the writer's only entry points
are `writer_tasks_send` and `writer_tasks_reply`, both of which return a
finished `Task` -- **0** generators, **0** yields. Everything above is added
by this solution; the capability flag was already set.

Structure: `stream` is the writer's generator and `accumulate` is the
caller's loop, kept separate so the shuffle can be applied between them.
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import io

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "19-a2a-protocol"
CHUNKS = ["topic identified, ", "key points extracted, ", "conclusion drafted."]


def stream(ref, task, length):
    """The writer, emitting incremental chunks and then a terminal frame."""
    yield {"event": "artifact-chunk",
           "data": ref.Part("text", {"text": f"[writer agent] {length} summary "
                                             f"of provided source: "})}
    for chunk in CHUNKS:
        yield {"event": "artifact-chunk", "data": ref.Part("text", {"text": chunk})}
    yield {"event": "task-complete", "data": task.id}


def accumulate(ref, frames):
    """The caller, concatenating whatever arrives until the terminal frame."""
    text, done = "", False
    for frame in frames:
        if frame["event"] == "task-complete":
            done = True
            break
        text += frame["data"].payload["text"]
    return ref.Artifact(name="summary", mimeType="text/markdown",
                        parts=[ref.Part("text", {"text": text})]), done


def single_shot(ref):
    message = ref.Message(role="user", parts=[
        ref.Part("text", {"text": "source"}),
        ref.Part("data", {"targetLength": "short"})])
    return ref.writer_tasks_send("draft_report", message)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    with contextlib.redirect_stdout(io.StringIO()):
        reference_task = single_shot(ref)
    frames = list(stream(ref, reference_task, "short"))
    streamed, done = accumulate(ref, frames)
    shuffled = [frames[0], frames[3], frames[1], frames[2], frames[4]]
    out_of_order, _ = accumulate(ref, shuffled)
    truncated, unfinished = accumulate(ref, frames[:-1])
    writer_source = inspect.getsource(ref.writer_tasks_send)
    return {
        "frames": len(frames),
        "events": sorted({frame["event"] for frame in frames}),
        "streamed": streamed.parts[0].payload["text"],
        "reference": reference_task.artifact.parts[0].payload["text"],
        "identical": (streamed.parts[0].payload["text"]
                      == reference_task.artifact.parts[0].payload["text"]),
        "done": done, "unfinished": unfinished,
        "shuffled": out_of_order.parts[0].payload["text"],
        "shuffle_differs": (out_of_order.parts[0].payload["text"]
                            != streamed.parts[0].payload["text"]),
        "part_fields": [f.name for f in dataclasses.fields(ref.Part)],
        "artifact_fields": [f.name for f in dataclasses.fields(ref.Artifact)],
        "streaming_claimed": ref.WRITER_AGENT_CARD["capabilities"]["streaming"],
        "yields": writer_source.count("yield"),
        "state": reference_task.state,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: three chunks over SSE, accumulated into the single-shot artifact",
            all([result["frames"] == 5, result["identical"], result["done"],
                 result["events"] == ["artifact-chunk", "task-complete"],
                 result["state"] == "completed"]),
            f"the writer emits {result['frames']} frames, {result['events']}, and the caller "
            f"concatenates them into {result['streamed'][:44]!r}... -- byte for byte the "
            f"artifact the non-streaming path produces, on a task that ends "
            f"{result['state']!r}",
        ),
        practice.Check(
            "FINDING: nothing in the payload orders the chunks",
            all([result["part_fields"] == ["kind", "payload"],
                 result["artifact_fields"] == ["name", "mimeType", "parts"],
                 result["shuffle_differs"]]),
            f"Part is {result['part_fields']} and Artifact is {result['artifact_fields']} -- "
            f"no index, no offset, no total. Replaying the same frames shuffled yields "
            f"{result['shuffled'][:44]!r}... and the accumulator cannot tell: the ordering "
            "guarantee lives entirely in a transport the lesson does not have",
        ),
        practice.Check(
            "FINDING: the terminal frame is the only thing that says done",
            all([result["done"], not result["unfinished"], result["frames"] == 5]),
            f"dropping the last frame leaves the accumulator at done={result['unfinished']} "
            f"with no way to tell a finished artifact from a stalled one, because no chunk "
            f"count is announced. {result['frames']} frames deliver "
            f"{result['frames'] - 1} pieces of content; the last exists only to end the "
            "stream",
        ),
        practice.Check(
            "FINDING: the card advertises streaming and the module implements none of it",
            all([result["streaming_claimed"] is True, result["yields"] == 0]),
            f"capabilities.streaming is {result['streaming_claimed']} while "
            f"writer_tasks_send contains {result['yields']} yields and returns a finished "
            "Task. Everything above is added by this solution; the capability flag was "
            "already set",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
