"""Exercise 2 — the accumulator has no branch for anything going wrong.

    Extend the accumulator to handle a "call was cancelled mid-stream" case by
    dropping its buffer and emitting a `cancelled` event. What provider
    documents this case explicitly? Check Anthropic's `content_block_stop`
    semantics and OpenAI's `finish_reason: "length"` behavior.

Reading of the exercise: the extension is written, and the three failure modes
it is written against are measured on the lesson's own `StreamAccumulator`
first -- because "handle a cancelled call" implies there is a path for it to
take, and there is not. The provider half is answered from the two mechanisms
the exercise names, which turn out to describe different events.

**ANSWER: a `cancelled` event drops the buffer; the lesson's accumulator has
nowhere to put it.** `on_event` branches on three kinds -- `call_start`,
`args_delta`, `call_stop` -- and returns `[]` for everything else. Feeding it
`{"type": "cancelled"}` is accepted silently and changes nothing, so a
cancellation is indistinguishable from an event that never arrived.

**FINDING: an unfinished call is not an error, it is a leak.** Start a call,
send one `args_delta`, and stop: the buffer stays in `buffers` with
`done=False` forever, `try_parse()` returns `None`, and `replay_and_execute`
never submits it. Nothing raises, nothing logs, and the id is still in the dict
when the stream ends -- which is exactly the state a cancellation leaves behind,
so the "handle it" work is detection, not cleanup.

**FINDING: the one thing that does raise is the one that should not.** An
`args_delta` for an id that never had a `call_start` raises **`KeyError`**
straight out of `on_event` and kills the loop, taking the other two in-flight
calls with it. A dropped first frame is a recoverable stream error; an
unfinished call is a silent one. The accumulator has the two backwards.

**ANSWER: neither provider documents cancellation -- they document truncation,
which is a different event.** Anthropic's `content_block_stop` always arrives
for a block that started, so its absence is a transport failure rather than a
signalled cancel. OpenAI's `finish_reason: "length"` marks a stream the *model*
stopped filling, so the buffer holds valid-prefix JSON that `json.loads` will
reject: **`Unterminated string`** on the lesson's own fixture truncated at
`{"city":"Tok`. Both tell you a call will never complete; neither tells you a
caller asked for it to stop. Cancellation is a client-side concept, which is why
it is MCP's `notifications/cancelled` that names it.

Structure: `Extended` subclasses the lesson's accumulator with the `cancelled`
branch and a guard on unknown ids, `probe` runs one event sequence through
either accumulator, and `TRUNCATED` is the lesson's own fixture cut short.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "03-parallel-and-streaming-tool-calls"
TRUNCATED = '{"city":"Tok'
LESSON_KINDS = ("call_start", "args_delta", "call_stop")


def extended(ref):
    """The lesson's accumulator with the two branches the exercise asks for."""

    class Extended(ref.StreamAccumulator):
        def on_event(self, event):
            kind, idx = event["type"], event.get("id")
            if kind == "cancelled":
                dropped = self.buffers.pop(idx, None)
                self.cancelled = getattr(self, "cancelled", [])
                if dropped is not None:
                    self.cancelled.append(idx)
                return []
            if kind == "args_delta" and idx not in self.buffers:
                self.orphans = getattr(self, "orphans", [])
                self.orphans.append(idx)
                return []
            return super().on_event(event)

    return Extended()


def start(idx):
    return {"type": "call_start", "id": idx, "name": "get_weather"}


def delta(idx, chunk):
    return {"type": "args_delta", "id": idx, "chunk": chunk}


def probe(accumulator, events):
    """Run events through an accumulator, returning its state or the exception."""
    try:
        for event in events:
            accumulator.on_event(event)
        return {"error": None, "buffers": sorted(accumulator.buffers)}
    except Exception as error:  # noqa: BLE001 - the failure mode is the measurement
        return {"error": type(error).__name__, "buffers": sorted(accumulator.buffers)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    unfinished = [start("call_A"), delta("call_A", TRUNCATED)]
    orphan = [delta("ghost", "{}")]

    shipped_unfinished = probe(ref.StreamAccumulator(), unfinished)
    shipped_orphan = probe(ref.StreamAccumulator(), orphan)
    shipped_cancel = probe(ref.StreamAccumulator(), unfinished + [{"type": "cancelled",
                                                                  "id": "call_A"}])
    mine = extended(ref)
    mine_cancel = probe(mine, unfinished + [{"type": "cancelled", "id": "call_A"}])
    mine_orphan = probe(extended(ref), orphan)

    stalled = ref.StreamAccumulator()
    for event in unfinished:
        stalled.on_event(event)
    buffer = stalled.buffers["call_A"]
    truncated_error = None
    try:
        ref.CallBuffer(id="x", name="get_weather", args_buf=TRUNCATED, done=True).try_parse()
    except Exception as error:  # noqa: BLE001
        truncated_error = str(error).split(":")[0]

    return {
        "kinds": list(LESSON_KINDS),
        "shipped_cancel_kept": shipped_cancel["buffers"],
        "shipped_cancel_error": shipped_cancel["error"],
        "unfinished_buffers": shipped_unfinished["buffers"],
        "unfinished_done": buffer.done, "unfinished_parse": buffer.try_parse(),
        "orphan_error": shipped_orphan["error"],
        "mine_cancel_kept": mine_cancel["buffers"],
        "mine_cancelled": getattr(mine, "cancelled", []),
        "mine_orphan_error": mine_orphan["error"],
        "truncated_error": truncated_error,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: a cancelled event drops the buffer; the lesson has nowhere to put it",
            all([result["kinds"] == ["call_start", "args_delta", "call_stop"],
                 result["shipped_cancel_error"] is None,
                 result["shipped_cancel_kept"] == ["call_A"],
                 result["mine_cancel_kept"] == [], result["mine_cancelled"] == ["call_A"]]),
            f"on_event branches on {result['kinds']} and returns [] for anything else, so "
            f"feeding it a cancelled event is accepted silently and leaves "
            f"{result['shipped_cancel_kept']} still buffered. The subclass here drops the "
            f"buffer and records {result['mine_cancelled']}, leaving "
            f"{result['mine_cancel_kept']}",
        ),
        practice.Check(
            "FINDING: an unfinished call is not an error, it is a leak",
            all([result["unfinished_buffers"] == ["call_A"],
                 result["unfinished_done"] is False,
                 result["unfinished_parse"] is None]),
            f"start a call, send one args_delta, and stop: the buffer stays in buffers as "
            f"{result['unfinished_buffers']} with done={result['unfinished_done']}, "
            f"try_parse returns {result['unfinished_parse']}, and replay_and_execute never "
            "submits it. Nothing raises and nothing logs -- which is exactly the state a "
            "cancellation leaves behind, so the work is detection, not cleanup",
        ),
        practice.Check(
            "FINDING: the one thing that does raise is the one that should not",
            all([result["orphan_error"] == "KeyError",
                 result["mine_orphan_error"] is None]),
            f"an args_delta for an id that never had a call_start raises "
            f"{result['orphan_error']} straight out of on_event and kills the loop, taking "
            f"the other in-flight calls with it. A dropped first frame is a recoverable "
            f"stream error and an unfinished call is a silent one; the accumulator has the "
            f"two backwards. Guarding it turns the raise into {result['mine_orphan_error']}",
        ),
        practice.Check(
            "ANSWER: neither provider documents cancellation -- they document truncation",
            result["truncated_error"] == "Unterminated string starting at",
            f"Anthropic's content_block_stop always arrives for a block that started, so its "
            f"absence is a transport failure rather than a signalled cancel. OpenAI's "
            f"finish_reason 'length' marks a stream the model stopped filling, leaving "
            f"valid-prefix JSON that json.loads rejects -- {result['truncated_error']} on "
            f"{TRUNCATED}. Both say a call will never complete; neither says a caller asked "
            "it to stop, which is why MCP names it separately as notifications/cancelled",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
