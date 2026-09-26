"""Exercise 2 — the gate fails realistic traffic on cache misses, and 40 iterations pass an at-SLA server 40% of the time.

    Write the k6 script for a CI gate: TTFT P95 < 800 ms at 100 concurrent,
    runtime 5 minutes.

Reading of the exercise: the script is `K6_SCRIPT` below, written for the
xk6-sse extension because stock k6 cannot see a first token. No k6 binary is
assumed, so the gate is exercised in Python: `gate()` applies k6's Trend
percentile (linear interpolation between sorted samples, as in k6's
`TrendSink.P`) and the script's own
threshold string to TTFT samples from the lesson's `simulate` at concurrency
100.

**ANSWER: constant-vus 100 for 5m, a first-SSE-event TTFT Trend, p(95)<800.**
100 VUs in a `constant-vus` scenario for `5m`; TTFT is a custom `Trend`
recorded at the first SSE event; thresholds `ttft: ['p(95)<800']` with
`abortOnFail`, and a custom `server_errors` Rate of `status >= 500` under
`rate<0.05` for the lesson's 5xx < 5% line (k6's `http_req_failed` also counts
4xx). k6 exits non-zero when a threshold fails, which breaks the build. Run it
as `k6 run -e BASE_URL=... -e MODEL=... gate.js` beside a `prompts.json`
sampled from real traffic. It is a custom Trend because k6's
built-in `http_req_waiting` is "time to first byte" -- the SSE response
headers, not the first token -- and k6's metrics reference lists no
time-to-first-token metric (checked 2026-09-26).

**FINDING: against the lesson's simulator the gate grades the cache, not the
server.** TTFT is 80 or 800 ms, and 800 is exactly the threshold, so
`p(95)<800` passes iff under about 5% of requests miss. Uniform prompts pass at
P95 80 ms; realistic prompts fail at P95 800 ms with 79 misses of 500 (15.8%).

**FINDING: 30-50 iterations cannot resolve a P95 gate.** The lesson's CI gate
is "30-50 iterations"; with k6's interpolation, 40 samples keep P95 under the
threshold only if at most 1 sample exceeds it. A server whose true P95 sits
exactly at 800 ms (5% slow) then passes 39.9% of runs, one at 3% slow 66.2%,
and one at 10% slow still 8.0%. The exercise's 100 VUs for 5 minutes, at the
lesson's 15 ms TPOT and the script's 256 max tokens, is about 7,400
iterations -- not 30-50.

**FINDING: "k6 v2026.1.0" is not a k6 version.** k6 moved to semantic
versioning at 1.0 (May 2025) and its release-notes index lists 0.47-0.57,
1.0-1.8 and 2.0-2.3; no calendar version (checked 2026-09-26). The lesson's
"k6 ... added streaming-aware metrics" has no built-in metric behind it; the
streaming here comes from the xk6-sse extension.

Structure: `K6_SCRIPT` is the deliverable; `thresholds()` parses it back so the
checks grade the shipped text; `pass_odds()` is an exact binomial.
"""

from __future__ import annotations

import math
import re

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "22-load-testing-llm-apis"
MAX_TOKENS, VUS, SECONDS = 256, 100, 300
K6_RELEASE_LINES = ("0.47-0.57", "1.0-1.8", "2.0-2.3")  # grafana.com/docs/k6/latest/release-notes

K6_SCRIPT = """\
import sse from 'k6/x/sse';
import { Rate, Trend } from 'k6/metrics';
const ttft = new Trend('ttft', true);
const serverErrors = new Rate('server_errors');
const URL = `${__ENV.BASE_URL}/v1/chat/completions`;
const PROMPTS = JSON.parse(open('./prompts.json'));  // sampled mean+stddev, never one prompt
export const options = {
  scenarios: { gate: { executor: 'constant-vus', vus: 100, duration: '5m' } },
  thresholds: {
    ttft: [{ threshold: 'p(95)<800', abortOnFail: true }],
    server_errors: ['rate<0.05'],
  },
};
export default function () {
  const body = JSON.stringify({ model: __ENV.MODEL, stream: true, max_tokens: 256,
    messages: [{ role: 'user', content: PROMPTS[Math.floor(Math.random() * PROMPTS.length)] }] });
  const start = Date.now(); let first = true;
  const res = sse.open(URL, { method: 'POST', body, headers: { 'Content-Type': 'application/json' } },
    (client) => {
      client.on('event', (e) => {
        if (first && e.data && e.data !== '[DONE]') { ttft.add(Date.now() - start); first = false; }
      });
    });
  serverErrors.add(!res || res.status >= 500);
}
"""


def thresholds(script):
    return dict(re.findall(r"(\w+): \[\{? ?(?:threshold: )?'([^']+)'", script))


def gate(expr, values):
    """k6's Trend percentile -- linear interpolation between sorted samples -- vs the limit."""
    pct, limit = (int(g) for g in re.fullmatch(r"p\((\d+)\)<(\d+)", expr).groups())
    s, i = sorted(values), pct / 100 * (len(values) - 1)
    lo, hi = s[math.floor(i)], s[math.ceil(i)]
    value = lo + (hi - lo) * (i - math.floor(i))
    return value < limit, value


def ttft_samples(ref, reqs):
    """`simulate` keeps its samples private; TTFT is two-valued, so its hit count rebuilds them."""
    out = ref.simulate(reqs, VUS)
    hit, miss = ref.PREFIX_CACHE_HIT_TTFT_MS, ref.PREFIX_CACHE_MISS_TTFT_MS
    return [hit] * out["cache_hits"] + [miss] * (out["n"] - out["cache_hits"])


def pass_odds(n, slow, expr="p(95)<800"):
    """P(gate passes) for n samples when a fraction `slow` exceed the limit."""
    allowed = max(k for k in range(n + 1) if gate(expr, [0] * (n - k) + [10**6] * k)[0])
    odds = sum(math.comb(n, k) * slow**k * (1 - slow) ** (n - k) for k in range(allowed + 1))
    return allowed, round(odds, 3)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    doc = parity.doc_text(PHASE, LESSON)
    rule = thresholds(K6_SCRIPT)
    real = ttft_samples(ref, ref.make_realistic_workload(500))
    e2e_s = (sum(real) / len(real) + ref.TPOT_MS * MAX_TOKENS) / 1000
    return {
        "rule": rule, "shape": all(s in K6_SCRIPT for s in ("vus: 100", "duration: '5m'")),
        "uniform": gate(rule["ttft"], ttft_samples(ref, ref.make_uniform_workload(500))),
        "real": gate(rule["ttft"], real), "misses": real.count(ref.PREFIX_CACHE_MISS_TTFT_MS),
        "odds": {s: pass_odds(40, s) for s in (0.03, 0.05, 0.10)},
        "iterations": round(VUS * SECONDS / e2e_s, -2),
        "doc": ["30-50 iterations" in doc, "k6 v2026.1.0" in doc],
    }


def verify(result):
    odds = {s: o for s, (_, o) in result["odds"].items()}
    allowed = {a for a, _ in result["odds"].values()}
    return [
        practice.Check(
            "ANSWER: constant-vus 100 for 5m, a first-SSE-event TTFT Trend, p(95)<800",
            result["shape"] and result["rule"] == {"ttft": "p(95)<800", "server_errors": "rate<0.05"},
            f"thresholds parsed from the shipped script: {result['rule']}",
        ),
        practice.Check(
            "FINDING: against the lesson's simulator the gate grades the cache, not the server",
            [result["uniform"], result["real"], result["misses"]] == [(True, 80), (False, 800), 79],
            f"uniform P95 {result['uniform'][1]} ms passes; realistic P95 "
            f"{result['real'][1]} ms fails with {result['misses']}/500 misses at "
            "exactly the 800 ms threshold",
        ),
        practice.Check(
            "FINDING: 30-50 iterations cannot resolve a P95 gate",
            all([result["doc"][0], allowed == {1}, result["iterations"] == 7400,
                 odds == {0.03: 0.662, 0.05: 0.399, 0.1: 0.08}]),
            f"40 samples tolerate {min(allowed)} slow one; pass probability by true slow "
            f"fraction {odds}; the exercise's run is ~{result['iterations']:.0f} iterations",
        ),
        practice.Check(
            "FINDING: 'k6 v2026.1.0' is not a k6 version",
            result["doc"][1] and not any(v.startswith("2026") for v in K6_RELEASE_LINES),
            f"the lesson names k6 v2026.1.0; k6's release notes list {K6_RELEASE_LINES} "
            "(checked 2026-09-26)",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
