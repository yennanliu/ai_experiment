"""Exercise 3 — the span already has the duration the histogram needs.

    Add the tool-execution metric `gen_ai.tool.execution.duration` and emit it
    as a histogram sample per call.

Reading of the exercise: a histogram sample needs a value and a set of
attributes, and both already exist on the span -- `end_ns - start_ns` and the
`gen_ai.*` keys. So the metric is derived from the spans rather than timed
separately, which is the only way the two can agree, and the interesting
question becomes *which* spans: there are two per tool call and they measure
different things.

**ANSWER: one sample per tool call, from `tool.execute`, in seconds.**
**3** calls give **3** samples bucketed into a histogram of **6** boundaries,
each carrying `gen_ai.operation.name` and `gen_ai.tool.name` as attributes.
The sum of the samples equals the sum of the `tool.execute` durations
exactly, because it *is* them.

**FINDING: timing it separately would disagree with the span.** Sampling
`mcp.call` instead gives **3** smaller values -- the child measures only the
crossing, where the parent measures the crossing plus the local execution.
Both spans carry `gen_ai.operation.name == "execute_tool"`, so a naive
"sample every execute_tool span" emits **6** samples and roughly doubles the
reported total.

**FINDING: the histogram needs a unit the span does not state.** Spans are
`start_ns`/`end_ns` and OTel's semconv gives this metric seconds, so the
conversion is a factor of **1e9** applied once, in one place. Emitting
nanoseconds into a bucket layout designed for seconds would put every sample
in the last bucket -- the layout is the unit's documentation.

**FINDING: the attributes have to be chosen, and cardinality is the choice.**
`tool.execute` carries `gen_ai.tool.call.id`, which is unique per call -- 
including it would make **3** samples into **3** distinct series. Keeping
`gen_ai.tool.name` and dropping the call id is what makes the histogram a
histogram rather than a log with buckets.

Structure: `Histogram` is the accumulator, and `samples_from` derives them
from whichever spans it is handed, so the two candidate sources can be
compared.
"""

from __future__ import annotations

import contextlib
import io

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "20-opentelemetry-genai"
METRIC = "gen_ai.tool.execution.duration"
BOUNDS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25)
KEYS = ("gen_ai.operation.name", "gen_ai.tool.name")
HIGH_CARDINALITY = "gen_ai.tool.call.id"


class Histogram:
    """The metric: bucket counts, a sum and a count, per attribute set."""

    def __init__(self, name=METRIC, unit="s", bounds=BOUNDS):
        self.name, self.unit, self.bounds, self.series = name, unit, bounds, {}

    def record(self, value, attrs):
        key = tuple(sorted(attrs.items()))
        entry = self.series.setdefault(key, {"buckets": [0] * (len(self.bounds) + 1),
                                             "sum": 0.0, "count": 0})
        entry["sum"] += value
        entry["count"] += 1
        index = next((i for i, bound in enumerate(self.bounds) if value <= bound),
                     len(self.bounds))
        entry["buckets"][index] += 1
        return entry


def samples_from(spans, name, keys=KEYS):
    """(seconds, attributes) for every span with this name."""
    return [((span.end_ns - span.start_ns) / 1e9,
             {key: span.attrs[key] for key in keys if key in span.attrs})
            for span in spans if span.name == name]


def emit(samples, **kwargs):
    histogram = Histogram(**kwargs)
    for value, attrs in samples:
        histogram.record(value, attrs)
    return histogram


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    ref.SPANS.clear()
    with contextlib.redirect_stdout(io.StringIO()):
        ref.agent_loop()
    spans = list(ref.SPANS)

    tool_samples = samples_from(spans, "tool.execute")
    mcp_samples = samples_from(spans, "mcp.call")
    histogram = emit(tool_samples)
    naive = emit(tool_samples + mcp_samples)
    per_call = emit(tool_samples, ).series
    with_id = emit([(value, {**attrs, HIGH_CARDINALITY: f"call_{i}"})
                    for i, (value, attrs) in enumerate(tool_samples)])
    series = next(iter(histogram.series.values()))
    return {
        "samples": len(tool_samples), "series": len(histogram.series),
        "count": series["count"], "sum": round(series["sum"], 6),
        "span_total": round(sum(v for v, _ in tool_samples), 6),
        "buckets": len(series["buckets"]), "bounds": len(BOUNDS),
        "unit": histogram.unit, "name": histogram.name,
        "attrs": sorted(next(iter(histogram.series))),
        "mcp_samples": len(mcp_samples),
        "mcp_smaller": all(m < t for (m, _), (t, _) in zip(mcp_samples, tool_samples)),
        "naive_count": sum(s["count"] for s in naive.series.values()),
        "naive_sum": round(sum(s["sum"] for s in naive.series.values()), 6),
        "id_series": len(with_id.series),
        "per_call_series": len(per_call),
        "ns_would_overflow": (tool_samples[0][0] * 1e9) > BOUNDS[-1],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: one sample per tool call, from tool.execute, in seconds",
            all([result["samples"] == 3, result["count"] == 3, result["series"] == 1,
                 result["sum"] == result["span_total"],
                 result["buckets"] == result["bounds"] + 1,
                 result["unit"] == "s", result["name"] == METRIC,
                 result["attrs"] == [("gen_ai.operation.name", "execute_tool"),
                                     ("gen_ai.tool.name", "get_weather")]]),
            f"{result['samples']} calls give {result['count']} samples in "
            f"{result['series']} series across {result['buckets']} buckets, carrying "
            f"{[k for k, _ in result['attrs']]}. The sum is {result['sum']} seconds, equal "
            "to the sum of the span durations exactly, because it is them",
        ),
        practice.Check(
            "FINDING: timing it separately would disagree with the span",
            all([result["mcp_samples"] == 3, result["mcp_smaller"],
                 result["naive_count"] == 6, result["naive_sum"] > result["sum"]]),
            f"sampling mcp.call instead gives {result['mcp_samples']} smaller values -- the "
            f"child measures the crossing, the parent the crossing plus local execution. "
            f"Both carry operation.name 'execute_tool', so sampling every execute_tool span "
            f"emits {result['naive_count']} samples summing {result['naive_sum']} against "
            f"{result['sum']}",
        ),
        practice.Check(
            "FINDING: the histogram needs a unit the span does not state",
            all([result["unit"] == "s", result["ns_would_overflow"]]),
            f"spans are start_ns/end_ns and the semconv gives this metric {result['unit']!r}, "
            f"so the conversion is one factor of 1e9 in one place. Emitting nanoseconds into "
            f"a layout topping out at {BOUNDS[-1]} would put every sample in the last "
            "bucket -- the layout is the unit's documentation",
        ),
        practice.Check(
            "FINDING: the attributes have to be chosen, and cardinality is the choice",
            all([result["id_series"] == 3, result["per_call_series"] == 1]),
            f"including gen_ai.tool.call.id, which tool.execute already carries and which is "
            f"unique per call, turns {result['per_call_series']} series into "
            f"{result['id_series']}. Keeping gen_ai.tool.name and dropping the call id is "
            "what makes this a histogram rather than a log with buckets",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
