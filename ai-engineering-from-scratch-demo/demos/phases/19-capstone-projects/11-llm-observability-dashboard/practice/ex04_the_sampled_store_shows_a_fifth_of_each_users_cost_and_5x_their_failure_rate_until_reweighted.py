"""Exercise 4 — the sampled store shows a fifth of each user's cost and 5x their failure rate until reweighted.

    Add a "user impact" page: cost-per-user and failure-rate-per-user with sparklines.

Reading of the exercise: the page is rendered from what the lesson's
dashboard actually holds, the tail-sampled `SpanStore`, fed by the same
loop as `main()`: `synth_trace`, then `enrich_with_evals`, then
`TailSampler(0.20).decide`. It is run at 10,000 traces rather than 200, so
each user has enough data. A failure is a trace with an error status or a
PII-leak eval over 0.8, the lesson's own alert line. Each user row carries
total cost and failure rate plus two 10-bucket SVG sparklines over trace
order. Because the store is a sample, every figure is read two ways: raw
from the store, and Horvitz-Thompson weighted (1 for a trace the sampler
must keep, 1 / 0.20 otherwise). Both are compared with the unsampled truth.

**ANSWER: `render()` returns the page: 4 user rows, 8 sparklines of 10
points each, and a weighted cost and failure rate per user.** On 10,000
traces the weighted cost is within 4.3% of the true cost for every user, and
the weighted failure rate is within 0.05 points of the true 0.75-1.15%. The
failure count itself is exact, because every failing trace is kept.

**FINDING: the raw store under-reports cost 5x and over-reports failures
about 5x.** Raw stored cost is 20-21% of true per user. Raw failure rate is
3.8-5.4% against a true 0.75-1.15%, because PII traces are kept at 100% and
everything else at 20%. On `main()`'s own run the printed "cost by user"
also reorders the users: u_01 is second by true cost ($1.26) and last in
the printout ($0.21).

**FINDING: the lesson's traffic cannot fill half the page.** Across 10,000
traces 0 spans have status "error", so every failure is a PII leak and the
error half of failure rate is always 0. `cost_usd` is drawn independently of
tokens and model: its correlation with total tokens is 0.01, so cost per
user measures nothing but the random draw.
"""

from __future__ import annotations

import itertools
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "11-llm-observability-dashboard"
N, RATE, BUCKETS = 10000, 0.20, 10


def run(ref, n, seed=5, rate=RATE):
    """main()'s loop, returning the sampled store and every trace (the truth)."""
    rng = random.Random(seed)
    sampler, store, traces = ref.TailSampler(sample_rate=rate, rng=rng), ref.SpanStore(), []
    for i in range(n):
        trace = ref.enrich_with_evals(ref.synth_trace(f"t{i:04d}", leak_pii=rng.random() < 0.01, rng=rng))
        traces.append(trace)
        if sampler.decide(trace):
            store.insert_trace(trace)
    return store, traces


def failed(spans):
    return any(s.status == "error" or (s.name == "eval" and s.attributes.get("pii_leak", 0) > 0.8) for s in spans)


def forced(spans):
    return failed(spans) or any(s.name == "eval" and s.attributes.get("toxicity", 0) > 0.5 for s in spans)


def per_user(traces, weighted=False, n=N):
    """{user: [cost, weighted traces, weighted failures, cost-by-bucket, failures-by-bucket]}."""
    rows = {}
    for t in traces:
        llm = next(s for s in t if s.is_llm())
        w = 1 / RATE if weighted and not forced(t) else 1
        r = rows.setdefault(llm.attributes["user_id"], [0.0, 0.0, 0.0, [0.0] * BUCKETS, [0] * BUCKETS])
        b = int(llm.trace_id[1:]) * BUCKETS // n
        r[0] += w * llm.attributes["cost_usd"]
        r[1] += w
        r[2] += w * failed(t)
        r[3][b] += w * llm.attributes["cost_usd"]
        r[4][b] += w * failed(t)
    return rows


def sparkline(values, w=100, h=20):
    top = max(values) or 1
    pts = " ".join(f"{i * w / (len(values) - 1):.0f},{h - v / top * h:.1f}" for i, v in enumerate(values))
    return f'<svg width="{w}" height="{h}"><polyline fill="none" stroke="currentColor" points="{pts}"/></svg>'


def render(rows):
    body = "".join(f"<tr><td>{u}</td><td>${r[0]:.2f}</td><td>{sparkline(r[3])}</td>"
                   f"<td>{r[2] / r[1]:.1%}</td><td>{sparkline(r[4])}</td></tr>" for u, r in sorted(rows.items()))
    return ("<html><body><h1>User impact</h1><table><tr><th>user</th><th>cost</th><th>trend</th>"
            f"<th>failure rate</th><th>trend</th></tr>{body}</table></body></html>")


def page_shape(page):
    svgs = page.split("<svg")[1:]
    return page.count("<tr><td>"), len(svgs), {len(x.split('points="')[1].split('"')[0].split()) for x in svgs}


def lesson_data(ref, truth):
    llm = [s for t in truth for s in t if s.is_llm()]
    tokens = [s.attributes["gen_ai.usage.input_tokens"] + s.attributes["gen_ai.usage.output_tokens"] for s in llm]
    small_store, small_truth = run(ref, 200)
    return {"main_printed": {u: round(v, 4) for u, v in small_store.cost_by_user.items()},
            "main_true": {u: round(r[0], 4) for u, r in per_user(small_truth, n=200).items()},
            "r_cost_tokens": round(statistics.correlation([s.attributes["cost_usd"] for s in llm], tokens), 2)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    store, truth = run(ref, N)
    kept = [list(g) for _, g in itertools.groupby(store.spans, key=lambda s: s.trace_id)]  # read back from the store
    true, raw, ht = per_user(truth), per_user(kept), per_user(kept, weighted=True)
    rate = lambda rows: {u: round(r[2] / r[1], 4) for u, r in sorted(rows.items())}
    return {"page": page_shape(render(ht)), "ht_cost_err": round(max(abs(ht[u][0] / true[u][0] - 1) for u in true), 3),
            "raw_cost": {u: round(raw[u][0] / true[u][0], 2) for u in sorted(true)},
            "rate_true": rate(true), "rate_raw": rate(raw), "rate_ht": rate(ht), **lesson_data(ref, truth),
            "errors": sum(s.status == "error" for t in truth for s in t)}


def verify(result):
    r = result
    rank = lambda d: sorted(d, key=d.get, reverse=True)
    return [
        practice.Check(
            "ANSWER: a 4-user page with 8 ten-point sparklines; weighted cost within 4.3%, failure rate within 0.05 points",
            r["page"] == (4, 8, {10}) and r["ht_cost_err"] == 0.043
            and max(abs(r["rate_ht"][u] - r["rate_true"][u]) for u in r["rate_true"]) <= 0.0005,
            f"rows, sparklines, points each {r['page']}; weighted cost off by at most "
            f"{r['ht_cost_err']:.1%}; failure rate weighted {r['rate_ht']} vs true {r['rate_true']}",
        ),
        practice.Check(
            "FINDING: the raw store shows 20-21% of true cost and over 3x the true failure rate",
            (min(r["raw_cost"].values()), max(r["raw_cost"].values())) == (0.2, 0.21)
            and all(r["rate_raw"][u] > 3 * r["rate_true"][u] for u in r["rate_true"])
            and rank(r["main_true"])[1] == rank(r["main_printed"])[-1] == "u_01",
            f"raw/true cost {r['raw_cost']}; failure rate raw {r['rate_raw']} vs true {r['rate_true']}; main() "
            f"printed {r['main_printed']} vs true {r['main_true']}",
        ),
        practice.Check(
            "FINDING: 0 error spans in 10,000 traces, and cost_usd is uncorrelated with tokens",
            (r["errors"], r["r_cost_tokens"]) == (0, 0.01),
            f"error-status spans {r['errors']}; r(cost_usd, tokens) {r['r_cost_tokens']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
