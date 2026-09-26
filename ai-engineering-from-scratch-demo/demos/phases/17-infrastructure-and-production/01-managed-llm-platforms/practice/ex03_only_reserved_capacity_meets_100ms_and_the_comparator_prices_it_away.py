"""Exercise 3 — only reserved capacity meets 100 ms, and the comparator prices it away.

    A regulated healthcare customer requires BAAs, US-East data residency, and
    sub-100ms P99 TTFT. Pick a platform and justify with three specific
    features.

Reading of the exercise: the three requirements are a filter, then a cost
ranking among what survives. BAAs and US-East are compliance facts the
lesson's text supplies; the P99 is what the reference's numbers can test, so
each platform is run on the reference's second demo workload (30M in / 15M
out a day), once through the reference's `simulate()` and once with the
path chosen to meet the SLA first and minimise cost second.

**ANSWER: Azure OpenAI on a PTU in an East US region.** (1) *BAA* -- the
lesson lists HIPAA among Azure OpenAI's certifications and says all three
provide BAAs; (2) *US-East residency* -- a regional (not global) deployment
in East US; (3) *dedicated capacity* -- one PTU gives P99 57 ms at $240/day.
Bedrock Provisioned Throughput also passes, at 82.5 ms, but needs 2 units
for $1,008/day, 4.2x. Vertex has no reserved path in the reference and fails
at 160 ms. Nothing on shared capacity passes: the best on-demand P99 is 140 ms.

**FINDING: the reference's comparator fails the customer on every platform
because it picks the cheapest path before checking the SLA.** `simulate()`
with PTU enabled takes on-demand at $225 over the $240 PTU and reports all
three FAIL. Paying $15/day more, 6.7%, meets the SLA; the SLA is printed but
never allowed to change the choice.

**FINDING: the code cannot see two of the three requirements, and the third
is set by fiat.** `Platform` has no BAA, region or residency field, and the
lesson says all three platforms "meet the basic checkbox", so the choice
reduces to P99 -- which on a PTU is `ttft_p50 * 1.5`, not measured variance;
`random` and `statistics` are imported and never used.

**FINDING: the latency headline the choice leans on contradicts the lesson.**
It quotes Azure OpenAI at ~50 ms "on Llama 3.1 405B equivalents" while
saying the Azure OpenAI catalog has no non-OpenAI models, and it calls ~50 ms
the PTU figure in one section and the "shared on-demand" figure in another. The
code sides with the second: 50 ms is the shared median, 38 ms the PTU one.

Structure: `sla_first()` is the corrected selection over the reference's
`Platform` rows; `reference_verdicts()` parses `simulate()`'s printout.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import math

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "01-managed-llm-platforms"
T_IN, T_OUT, SLA = 30_000_000, 15_000_000, 100


def on_demand(p):
    return T_IN / 1e6 * p.per_mtok_input + T_OUT / 1e6 * p.per_mtok_output


def reserved(p):
    """(units, $/day, P99) on the PTU path, by the reference's own rules."""
    if p.ptu_hourly is None:
        return None
    units = max(1, math.ceil((T_IN + T_OUT) / (p.ptu_tokens_per_hour * 24)))
    return units, units * p.ptu_hourly * 24, p.ttft_median_ptu_ms * 1.5


def sla_first(p):
    """The cheapest path that meets the SLA, or None."""
    paths = [("on-demand", on_demand(p), p.ttft_p99_ms)]
    if reserved(p):
        units, cost, p99 = reserved(p)
        paths.append((f"{units} PTU", cost, p99))
    ok = [path for path in paths if path[2] < SLA]
    return min(ok, key=lambda path: path[1]) if ok else None


def reference_verdicts(ref):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ref.simulate(T_IN, T_OUT, SLA, use_ptu=True)
    rows = [line for line in out.getvalue().splitlines() if line.endswith("]")]
    return {r[:25].strip(): ("PASS" in r, r.rsplit("[", 1)[1][:-1]) for r in rows}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    doc = parity.doc_text(PHASE, LESSON)
    source = (parity.lesson_dir(PHASE, LESSON) / "code" / "main.py").read_text()
    fields = [f.name for f in dataclasses.fields(ref.Platform)]
    azure = ref.PLATFORMS[1]
    return {
        "choice": {p.name: sla_first(p) for p in ref.PLATFORMS},
        "shared_p99": min(p.ttft_p99_ms for p in ref.PLATFORMS),
        "reference": reference_verdicts(ref),
        "azure_od": on_demand(azure),
        "fields": fields,
        "compliance": [f for f in fields if f in ("baa", "region", "residency")],
        "unused": [m for m in ("random", "statistics") if f"{m}." not in source],
        "doc": (
            "No non-OpenAI models" in doc,
            "~50 ms (with PTUs)" in doc,
            "405B deployments (shared on-demand), Azure OpenAI" in doc,
        ),
        "azure_ms": (azure.ttft_median_ms, azure.ttft_median_ptu_ms),
    }


def verify(result):
    c, ref_ = result["choice"], result["reference"]
    azure, bedrock = c["Azure OpenAI (PTU)"], c["Bedrock on-demand"]
    return [
        practice.Check(
            "ANSWER: Azure OpenAI on a PTU in an East US region",
            all(
                [
                    azure == ("1 PTU", 240.0, 57.0),
                    bedrock == ("2 PTU", 1008.0, 82.5),
                    c["Vertex AI Gemini"] is None,
                    result["shared_p99"] > SLA,
                ]
            ),
            f"SLA-first: Azure {azure}, Bedrock {bedrock} ({bedrock[1] / azure[1]:.1f}x), "
            f"Vertex none; best shared P99 {result['shared_p99']} ms",
        ),
        practice.Check(
            "FINDING: the comparator picks the cheapest path before checking the SLA",
            all(
                [
                    not any(ok for ok, _ in ref_.values()),
                    all(path == "on-demand" for _, path in ref_.values()),
                    result["azure_od"] == 225.0,
                ]
            ),
            f"simulate() verdicts {ref_}; it takes Azure on-demand at ${result['azure_od']:.0f}, "
            f"and the PTU that passes costs {azure[1] / result['azure_od'] - 1:.1%} more",
        ),
        practice.Check(
            "FINDING: the code cannot see two of the three requirements, and the third is set by fiat",
            result["compliance"] == [] and result["unused"] == ["random", "statistics"],
            f"Platform fields {result['fields']}; PTU P99 = P50 x 1.5; imported and unused: "
            f"{result['unused']}",
        ),
        practice.Check(
            "FINDING: the latency headline the choice leans on contradicts the lesson",
            all(result["doc"]) and result["azure_ms"] == (50, 38),
            "the lesson puts Azure OpenAI at ~50 ms on Llama 3.1 405B, once 'shared "
            "on-demand' and once 'with PTUs', and says its catalog has no non-OpenAI models; "
            f"the code's Azure median is {result['azure_ms'][0]} ms shared and "
            f"{result['azure_ms'][1]} ms on a PTU",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
