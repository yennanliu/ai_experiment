"""Exercise 4 — for voice, TPOT and E2E drop out and time to first phrase is the SLO.

    Construct a consumer SLO for a voice assistant (first token is heard, not
    read). Which metric is most user-visible?

Reading of the exercise: "heard, not read" changes two things. A TTS engine
speaks phrases, not tokens, so the user hears nothing until the first
phrase exists; this takes 8 tokens, a short clause. And playback runs at
speech rate: 150 words a minute at 0.75 words per token is 300 ms per token.
So the SLO has two parts. First, the time until 8 tokens exist must be
<= 500 ms at P99. Second, the playback buffer must never underrun: every
token must arrive before playback reaches it. It is scored against the
lesson's own 2000-request workload, next to the lesson's text "target"
profile.

**ANSWER: the SLO is time to first phrase, <= 500 ms at P99, plus no
underrun; TTFT is the most user-visible metric, because it is 91% of that
time.** Human turn-taking leaves a gap of about 200 ms (Stivers et al.,
PNAS 2009), and ASR endpointing and TTS both spend some of it before the
LLM's part starts. On the lesson's workload, first-phrase P50/P90/P99 is
139/493/583 ms, so the deployment misses its P99 target. TTFT is 91% of it
at P99, and the 7 extra decode steps are the rest. Voice goodput is 91.35% at 500 ms, against 75.95% for the text
target profile on the same requests. The text profile's failures were almost
all E2E, and E2E is not heard.

**FINDING: TPOT and E2E drop out of a voice SLO.** Playback needs a token
every 300 ms, and the slowest single token in the workload takes 86.3 ms, so
no request in 2000 underruns. TPOT has 33x headroom at its P99 of 8.99 ms.
E2E is irrelevant: the mean answer takes 52.8 s to *speak*.

**FINDING: the SLO fails on long context, which conversations accumulate.**
Every prompt length up to 2048 tokens passes at 99.7% or more. The
8192-token prompts pass 57.3%, with a first-phrase P99 of 641 ms, because
prefill alone is 410 ms. A voice session appends every turn to its history,
so the same deployment drifts into failure as the conversation goes on.

Structure: `arrivals()` turns a reference trace into token arrival times
(token 1 at TTFT); `first_phrase()` and `underruns()` score the two SLO parts.
"""

from __future__ import annotations

import collections

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "08-inference-metrics-goodput"
PHRASE, SPEAK_MS, BUDGET_MS = 8, 300.0, 500.0  # 150 wpm at 0.75 words/token
PREFILL_MS_PER_TOKEN = 0.05


def arrivals(trace):
    out, clock = [trace.ttft_ms], trace.ttft_ms
    for gap in trace.decode_ms_per_token[:-1]:
        clock += gap
        out.append(clock)
    return out


def first_phrase(trace):
    return arrivals(trace)[PHRASE - 1]


def underruns(trace):
    """Playback starts at the first phrase and speaks one token per SPEAK_MS."""
    times = arrivals(trace)
    start = times[PHRASE - 1]
    return any(times[k] > start + (k - PHRASE + 1) * SPEAK_MS for k in range(PHRASE, len(times)))


def by_prompt(traces):
    groups = collections.defaultdict(list)
    for t in traces:
        groups[round(t.prefill_ms / PREFILL_MS_PER_TOKEN)].append(first_phrase(t) <= BUDGET_MS)
    return {k: round(sum(v) / len(v), 4) for k, v in sorted(groups.items())}


def playback(traces):
    return {
        "underruns": sum(underruns(t) for t in traces),
        "slowest_token": max(max(t.decode_ms_per_token) for t in traces),
        "speak_s": sum(t.output_tokens for t in traces) / len(traces) * SPEAK_MS / 1000,
    }


def nearest(traces, target):
    return min(traces, key=lambda t: abs(first_phrase(t) - target))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    traces = ref.synth_workload(n=2000)
    phrase = [first_phrase(t) for t in traces]
    at_p99 = nearest(traces, ref.percentiles(phrase, [0.99])[0])
    long = [first_phrase(t) for t in traces if t.prefill_ms > 400]
    return {
        "phrase": [round(x) for x in ref.percentiles(phrase, [0.5, 0.9, 0.99])],
        "ttft_share": at_p99.ttft_ms / first_phrase(at_p99),
        "voice": sum(x <= BUDGET_MS for x in phrase) / len(phrase),
        "text": ref.goodput(traces, 500, 15, 2000),
        **playback(traces),
        "tpot_p99": ref.percentiles([t.tpot_genaiperf() for t in traces], [0.99])[0],
        "by_prompt": by_prompt(traces), "long_p99": ref.percentiles(long, [0.99])[0],
    }


def verify(r):
    short = [v for k, v in r["by_prompt"].items() if k <= 2048]
    return [
        practice.Check(
            "ANSWER: time to first phrase <= 500 ms at P99 plus no underrun; TTFT is most visible",
            all([r["phrase"] == [139, 493, 583], r["ttft_share"] > 0.85,
                 r["voice"] == 0.9135, r["text"] == 0.7595]),
            f"first phrase P50/P90/P99 {r['phrase']} ms, TTFT {r['ttft_share']:.0%} of it at "
            f"P99; voice goodput {r['voice']:.2%} against the text profile's {r['text']:.2%}",
        ),
        practice.Check(
            "FINDING: TPOT and E2E drop out of a voice SLO",
            all([r["underruns"] == 0, r["slowest_token"] < SPEAK_MS, r["speak_s"] > 50]),
            f"{r['underruns']} underruns; slowest token {r['slowest_token']:.1f} ms against "
            f"{SPEAK_MS:.0f} ms of speech; TPOT P99 {r['tpot_p99']:.2f} ms is "
            f"{SPEAK_MS / r['tpot_p99']:.0f}x under; mean answer takes {r['speak_s']:.1f} s to speak",
        ),
        practice.Check(
            "FINDING: the SLO fails on long context, which conversations accumulate",
            all([min(short) >= 0.997, r["by_prompt"][8192] < 0.6, r["long_p99"] > BUDGET_MS]),
            f"pass rate by prompt length {r['by_prompt']}; 8192-token first-phrase P99 "
            f"{r['long_p99']:.0f} ms",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
