"""Exercise 1 — static beats continuous 4.6x, because the toy bills a padded batch as one request.

    Run `code/main.py`. Compare `STATIC` to `CONTINUOUS` on a workload with
    mixed short and long requests. Where does the throughput gap come from —
    prefill efficiency, decode efficiency, or tail latency?

Reading of the exercise: the shipped 60-request workload is already mixed
(prompts of 128 to 8192 tokens, outputs of 50 to 300), so it is the workload
compared. "Where the gap comes from" is answered by splitting each mode's
virtual time into prefill, decode, and overhead-plus-idle, and then removing
each difference between the two cost models to see which one moves the gap.

**ANSWER: the gap runs the wrong way, and it is half prefill, half decode.**
STATIC does 4055 tok/s and CONTINUOUS 886: static wins 4.6x, although the
lesson says continuous "should dominate". Static finishes in 2.32 virtual
seconds (1.31 prefill, 0.53 decode, 0.48 overhead and idle), continuous in
10.63 (5.65 prefill, 4.71 decode, 0.27 overhead). So of the 8.31 s gap, 4.34 s
is prefill and 4.18 s is decode. Tail latency contributes nothing to
throughput. It shows up only in P99 ITL, 0.7 ms static against 219.3 ms
continuous.

**FINDING: the two modes use two different cost models.** `simulate_static`
charges a window's prefill at its *longest* prompt, as if 16 prompts cost the
same as one. It charges each decode step at `FORWARD_LATENCY_PER_TOKEN *
len(window) / 16`, so 16 tokens cost one token's time. `simulate_continuous`
charges every prefill token and every decode token at full price, so batching
buys it nothing. Its decode cost for n running sequences is n times one
sequence's cost. Continuous at 886 tok/s is only 15% above NAIVE's 768.

**FINDING: on one cost model, continuous wins.** Set prefill to free in both
modes and charge continuous decode at the same 1/16 per token that static
gets. Continuous then does 6767 tok/s against static's 6382. Its mean TTFT is
0.1 ms against 155.7 ms, and it finishes 0.08 s after the last arrival. The
lesson's conclusion holds only once the toy stops charging continuous
batching for the batching it does.

Structure: `run()` replays one mode on a fresh copy of the reference workload.
`split()` and `static_split()` compute each mode's time budget from the reference constants.
`patched()` sets reference constants for one run and restores them afterwards.
"""

from __future__ import annotations

import contextlib
import statistics

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "04-vllm-serving-internals"


def run(ref, mode):
    reqs = [ref.Request(r.req_id, r.prompt_len, r.output_len, r.arrived_at)
            for r in ref.make_workload()]
    end = (ref.simulate_static(reqs) if mode == "static"
           else ref.simulate_continuous(reqs, chunked=False))
    itls = sorted(d for r in reqs for d in r.itl_samples)
    return {"tps": round(sum(r.generated for r in reqs) / end), "end": end,
            "ttft_ms": round(statistics.mean(r.ttft - r.arrived_at for r in reqs) * 1e3, 1),
            "p99_ms": round(itls[int(0.99 * len(itls)) - 1] * 1e3, 1)}


def static_split(ref, reqs):
    """Static bills each 16-request window at its longest prompt and longest output."""
    windows = [reqs[i:i + 16] for i in range(0, len(reqs), 16)]
    pre = sum(max(r.prompt_len for r in w) for w in windows)
    dec = sum(max(r.output_len for r in w) * len(w) / 16 for w in windows)
    return pre * ref.PREFILL_LATENCY_PER_TOKEN, dec * ref.FORWARD_LATENCY_PER_TOKEN


def split(ref, mode, end):
    """(prefill, decode, overhead+idle) seconds, from the reference's own formulas."""
    reqs = ref.make_workload()
    if mode == "static":
        pre, dec = static_split(ref, reqs)
    else:
        pre = sum(r.prompt_len for r in reqs) * ref.PREFILL_LATENCY_PER_TOKEN
        dec = sum(r.output_len for r in reqs) * ref.FORWARD_LATENCY_PER_TOKEN
    return round(pre, 2), round(dec, 2), round(end - pre - dec, 2)


@contextlib.contextmanager
def patched(ref, **values):
    saved = {k: getattr(ref, k) for k in values}
    try:
        for k, v in values.items():
            setattr(ref, k, v)
        yield
    finally:
        for k, v in saved.items():
            setattr(ref, k, v)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    modes = {m: run(ref, m) for m in ("static", "continuous")}
    naive = [ref.Request(r.req_id, r.prompt_len, r.output_len, r.arrived_at)
             for r in ref.make_workload()]
    naive_tps = round(sum(r.output_len for r in naive) / ref.simulate_naive(naive))
    with patched(ref, PREFILL_LATENCY_PER_TOKEN=0.0):
        fair_static = run(ref, "static")
        with patched(ref, FORWARD_LATENCY_PER_TOKEN=ref.FORWARD_LATENCY_PER_TOKEN / 16):
            fair_cont = run(ref, "continuous")
    return {"modes": modes, "naive_tps": naive_tps,
            "split": {m: split(ref, m, modes[m]["end"]) for m in modes},
            "fair": {"static": fair_static, "continuous": fair_cont},
            "last_arrival": ref.make_workload()[-1].arrived_at}


def verify(result):
    st, co = result["modes"]["static"], result["modes"]["continuous"]
    ss, cs = result["split"]["static"], result["split"]["continuous"]
    fs, fc = result["fair"]["static"], result["fair"]["continuous"]
    return [
        practice.Check(
            "ANSWER: the gap runs the wrong way, and it is half prefill, half decode",
            all([st["tps"] == 4055, co["tps"] == 886, ss == (1.31, 0.53, 0.48),
                 cs == (5.65, 4.71, 0.27), st["p99_ms"] == 0.7, co["p99_ms"] == 219.3]),
            f"static {st['tps']} tok/s in {st['end']:.2f}s {ss}, continuous {co['tps']} "
            f"in {co['end']:.2f}s {cs} (prefill, decode, rest): prefill "
            f"{cs[0] - ss[0]:.2f}s and decode {cs[1] - ss[1]:.2f}s of the gap; P99 ITL "
            f"{st['p99_ms']} vs {co['p99_ms']} ms",
        ),
        practice.Check(
            "FINDING: the two modes use two different cost models",
            result["naive_tps"] == 768 and co["tps"] / result["naive_tps"] < 1.16,
            f"static bills a window's prefill at its longest prompt and decode at 1/16 per "
            f"token; continuous bills every token in full, so it does {co['tps']} tok/s "
            f"against NAIVE's {result['naive_tps']}",
        ),
        practice.Check(
            "FINDING: on one cost model, continuous wins",
            all([fc["tps"] == 6767, fs["tps"] == 6382, fc["ttft_ms"] == 0.1,
                 fs["ttft_ms"] == 155.7, fc["end"] - result["last_arrival"] < 0.1]),
            f"with prefill free and decode at 1/16 in both, continuous {fc['tps']} tok/s "
            f"(TTFT {fc['ttft_ms']} ms) beats static {fs['tps']} (TTFT {fs['ttft_ms']} ms), "
            f"ending {fc['end'] - result['last_arrival']:.2f}s after the last arrival",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
