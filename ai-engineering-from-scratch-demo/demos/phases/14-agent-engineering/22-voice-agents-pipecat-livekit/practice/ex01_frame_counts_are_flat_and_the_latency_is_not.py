"""Exercise 1 — frame counts are flat and the latency is not.

    Add a metrics observer to your toy pipeline: count frames per stage per
    second. Where does latency accumulate?

Reading of the exercise: counting frames and finding latency turn out to be
two different measurements, and the exercise is worth doing because it shows
they disagree. The observer is built as Pipecat's would be -- it sees every
frame at every processor, in both directions -- and the latency question is
answered from the lesson's own published bands rather than from a clock on
this machine, which would measure the laptop.

**ANSWER: every stage sees the same number of frames, and the LLM owns 40% of
the latency.** One utterance produces **1** frame at each of the **5**
stages. Against the lesson's bands the chain sums to **400-990ms**, of which
LLM first token is **150-400ms** -- **37.5%** of the floor and **40.4%** of
the ceiling, the largest single contributor by both measures.

**FINDING: frames per second is a throughput metric on a serial chain.**
Because each processor forwards exactly one frame, the per-stage counts are
`[1, 1, 1, 1, 1]` and carry **0** information about where the time went. The
stage with the most frames is not the slow one; on this pipeline there is no
such stage, and a dashboard of frame rates would show five flat lines while
end-to-end sat at **695ms**.

**FINDING: the trace cannot be used for this and never could.** The lesson's
module references a clock **0** times, `Processor.trace` holds plain strings,
and `Frame` carries **3** fields with no identity among them -- so the same
frame appears as **5** unrelated log lines and no pair of them can be
subtracted. The observer has to be an addition, not an extraction.

**FINDING: upstream frames double the count without adding latency.** A
barge-in cancel is seen at all **5** stages travelling back, taking the
observed total from **5** to **10** frames while the user-perceived latency
*falls*. An observer that does not read `frame.direction` reports a **100%**
traffic increase for an event that makes the turn shorter.

Structure: `Observer` wraps each processor's `process`; `budget()` folds the
lesson's published bands into a cumulative profile.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "22-voice-agents-pipecat-livekit"
# "Typical 2026 latencies", in milliseconds.
BANDS = (("vad", 20, 60), ("stt", 100, 250), ("llm", 150, 400),
         ("tts", 100, 200), ("transport", 30, 80))
PREMIUM = (450, 600)


class Observer:
    """Pipecat's observer: every frame at every processor, both directions."""

    def __init__(self, processors):
        self.counts = {p.name: 0 for p in processors}
        self.directions = {"downstream": 0, "upstream": 0}
        for proc in processors:
            self._wrap(proc)

    def _wrap(self, proc):
        inner = proc.process

        def wrapped(frame):
            self.counts[proc.name] += 1
            self.directions[frame.direction] += 1
            return inner(frame)
        proc.process = wrapped


def pipeline(ref):
    stages = (ref.VAD("vad"), ref.STT("stt"),
              ref.LLM("llm", replies={"hello": "hi there, how can I help today?"}),
              ref.TTS("tts"), ref.Transport("transport"))
    ref.link(*stages)
    return stages


def budget():
    """Cumulative arrival time at each stage, from the lesson's own bands."""
    low = high = 0
    rows = []
    for name, lo, hi in BANDS:
        low, high = low + lo, high + hi
        rows.append({"stage": name, "low": low, "high": high, "band": (lo, hi)})
    return rows


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    stages = pipeline(ref)
    observer = Observer(stages)
    stages[0].process(ref.Frame("audio_chunk", "hello"))
    downstream = dict(observer.counts)
    stages[-1].process(ref.Frame("cancel", None, direction="upstream"))
    rows = budget()
    total_low, total_high = rows[-1]["low"], rows[-1]["high"]
    llm = next(row for row in rows if row["stage"] == "llm")["band"]
    return {
        "stages": len(stages), "per_stage": list(downstream.values()),
        "downstream_total": sum(downstream.values()),
        "observed_total": sum(observer.counts.values()),
        "upstream_seen": observer.directions["upstream"],
        "inflation": round(100 * (sum(observer.counts.values())
                                  - sum(downstream.values()))
                           / sum(downstream.values())),
        "profile": rows, "total": (total_low, total_high),
        "midpoint": (total_low + total_high) // 2,
        "llm_share": (round(100 * llm[0] / total_low, 1),
                      round(100 * llm[1] / total_high, 1)),
        "biggest": max(BANDS, key=lambda row: row[2] - 0)[0],
        "premium": PREMIUM,
        "clock_refs": sum(name in inspect.getsource(ref)
                          for name in ("perf_counter", "monotonic", "time.time")),
        "trace_types": sorted({type(line).__name__ for line in stages[1].trace}),
        "frame_fields": list(ref.Frame.__dataclass_fields__),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: five stages see one frame each, the LLM owns 40% of the latency",
            all([result["stages"] == 5, result["per_stage"] == [1, 1, 1, 1, 1],
                 result["total"] == (400, 990), result["llm_share"] == (37.5, 40.4),
                 result["biggest"] == "llm"]),
            f"one utterance produces {result['per_stage']} frames across "
            f"{result['stages']} stages. Against the lesson's bands the chain sums to "
            f"{result['total'][0]}-{result['total'][1]}ms, of which LLM first token is "
            f"{result['llm_share'][0]}% of the floor and {result['llm_share'][1]}% of the "
            "ceiling -- the largest single contributor by both measures",
        ),
        practice.Check(
            "FINDING: frames per second is a throughput metric on a serial chain",
            all([len(set(result["per_stage"])) == 1, result["midpoint"] == 695,
                 result["downstream_total"] == 5]),
            f"each processor forwards exactly one frame, so the per-stage counts are "
            f"{result['per_stage']} and carry nothing about where the time went. A "
            f"dashboard of frame rates would show {result['stages']} flat lines while "
            f"end-to-end sat at {result['midpoint']}ms, outside the "
            f"{result['premium'][0]}-{result['premium'][1]}ms premium band",
        ),
        practice.Check(
            "FINDING: the trace cannot be used for this and never could",
            all([result["clock_refs"] == 0, result["trace_types"] == ["str"],
                 result["frame_fields"] == ["kind", "payload", "direction"]]),
            f"the module references a clock {result['clock_refs']} times, Processor.trace "
            f"holds {result['trace_types'][0]} entries, and Frame carries "
            f"{result['frame_fields']} with no identity -- so one frame appears as five "
            "unrelated log lines and no pair can be subtracted. The observer has to be an "
            "addition, not an extraction",
        ),
        practice.Check(
            "FINDING: upstream frames double the count without adding latency",
            all([result["upstream_seen"] == 5, result["observed_total"] == 10,
                 result["inflation"] == 100]),
            f"a barge-in cancel is seen at {result['upstream_seen']} stages travelling "
            f"back, taking the observed total from {result['downstream_total']} to "
            f"{result['observed_total']} frames while user-perceived latency falls. An "
            f"observer blind to frame.direction reports a {result['inflation']}% traffic "
            "increase for an event that makes the turn shorter",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
