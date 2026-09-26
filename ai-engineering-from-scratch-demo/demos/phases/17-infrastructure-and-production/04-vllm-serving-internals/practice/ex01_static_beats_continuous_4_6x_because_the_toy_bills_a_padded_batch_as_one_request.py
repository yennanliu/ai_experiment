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
10.63 (5.65 prefill, 4.71 decode, 0.27 overhead). So the 8.31 s gap is 4.34 s
of prefill and 4.18 s of decode, less the 0.21 s more that static spends on
overhead and idle. Tail latency contributes nothing to
throughput. It shows up only in P99 ITL, 0.7 ms static against 219.3 ms
continuous.

**FINDING: the two modes use two different cost models.** `simulate_static`
charges a window's prefill at its *longest* prompt, as if 16 prompts cost the
same as one. It charges each decode step at `FORWARD_LATENCY_PER_TOKEN *
len(window) / 16`, so 16 tokens cost one token's time. `simulate_continuous`
charges every prefill token and every decode token at full price, so batching
buys it nothing. Pricing one token kind at 1 s and everything else at 0 shows
it from the runs themselves: static bills 32,768 prefill and 1,056 decode
token-units, continuous bills all 141,184 prompt and 9,419 output tokens. Its decode cost for n running sequences is n times one
sequence's cost. Continuous at 886 tok/s is only 15% above NAIVE's 768.

**FINDING: on one cost model, continuous wins.** Set prefill to free in both
modes and charge continuous decode at the same 1/16 per token that static
gets. Continuous then does 6767 tok/s against static's 6382. Its mean TTFT is
0.1 ms against 155.7 ms, and it finishes 0.08 s after the last arrival. The
lesson's conclusion holds only once the toy stops charging continuous
batching for the batching it does.

Structure: `run()` replays one mode on a fresh copy of the reference workload.
`tokens()` and `split()` compute each mode's billed work and time budget.
`billed()` replays a mode with one token kind priced at 1 s to read what it bills.
`patched()` sets reference constants for one run and restores them afterwards.
"""

from __future__ import annotations

import contextlib
import statistics

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "04-vllm-serving-internals"


def run(ref, mode):
    reqs = ref.make_workload()  # fresh Requests on every call
    end = (ref.simulate_static(reqs) if mode == "static"
           else ref.simulate_continuous(reqs, chunked=False))
    itls = sorted(d for r in reqs for d in r.itl_samples)
    return {"tps": round(sum(r.generated for r in reqs) / end), "end": end,
            "ttft_ms": round(statistics.mean(r.ttft - r.arrived_at for r in reqs) * 1e3, 1),
            "p99_ms": round(itls[int(0.99 * len(itls)) - 1] * 1e3, 1)}


def tokens(reqs, mode):
    """(prefill, decode) token-units billed: static per 16-window at its maxima, continuous all."""
    windows = [reqs[i:i + 16] for i in range(0, len(reqs), 16)] if mode == "static" else [
        [r] * 16 for r in reqs]  # continuous: every request is its own full-price "window"
    return (sum(max(r.prompt_len for r in w) for w in windows),
            sum(max(r.output_len for r in w) * len(w) / 16 for w in windows))


def split(ref, mode, end):
    """(prefill, decode, overhead+idle) seconds, from the reference's own formulas."""
    pre, dec = tokens(ref.make_workload(), mode)
    pre, dec = pre * ref.PREFILL_LATENCY_PER_TOKEN, dec * ref.FORWARD_LATENCY_PER_TOKEN
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


def billed(ref, mode):
    """Measured (prefill, decode) token-units: one kind priced at 1 s, all else at 0."""
    kinds, ends = ("PREFILL_LATENCY_PER_TOKEN", "FORWARD_LATENCY_PER_TOKEN"), []
    for kind in kinds:
        with patched(ref, **{**dict.fromkeys(kinds + ("BATCH_OVERHEAD",), 0.0), kind: 1.0}):
            ends.append(run(ref, mode)["end"])
    return ends


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    modes = {m: run(ref, m) for m in ("static", "continuous")}
    naive = ref.make_workload()
    naive_tps = round(sum(r.output_len for r in naive) / ref.simulate_naive(naive))
    with patched(ref, PREFILL_LATENCY_PER_TOKEN=0.0):
        fair_static = run(ref, "static")
        with patched(ref, FORWARD_LATENCY_PER_TOKEN=ref.FORWARD_LATENCY_PER_TOKEN / 16):
            fair_cont = run(ref, "continuous")
    return {"modes": modes, "naive_tps": naive_tps,
            "formula": {m: [round(x) for x in tokens(ref.make_workload(), m)] for m in modes},
            "billed": {m: billed(ref, m) for m in modes},
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
            f"{cs[0] - ss[0]:.2f}s and decode {cs[1] - ss[1]:.2f}s of the {co['end'] - st['end']:.2f}s "
            f"gap, rest {cs[2] - ss[2]:+.2f}s; P99 ITL "
            f"{st['p99_ms']} vs {co['p99_ms']} ms",
        ),
        practice.Check(
            "FINDING: the two modes use two different cost models",
            all([result["naive_tps"] == 768, co["tps"] / result["naive_tps"] < 1.16,
                 result["formula"] == {"static": [32_768, 1056],
                                       "continuous": [141_184, 9419]},
                 # each run's unit-priced end is its billed tokens plus arrival idle only
                 all(0 <= b - f <= result["last_arrival"]
                     for m in ("static", "continuous")
                     for b, f in zip(result["billed"][m], result["formula"][m]))]),
            f"unit-priced runs bill (prefill, decode) token-units: static "
            f"{[round(b, 2) for b in result['billed']['static']]} vs longest-prompt/padded-1/16 "
            f"formula {result['formula']['static']}, continuous "
            f"{[round(b, 2) for b in result['billed']['continuous']]} vs every token "
            f"{result['formula']['continuous']}; continuous does {co['tps']} tok/s against "
            f"NAIVE's {result['naive_tps']}",
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
