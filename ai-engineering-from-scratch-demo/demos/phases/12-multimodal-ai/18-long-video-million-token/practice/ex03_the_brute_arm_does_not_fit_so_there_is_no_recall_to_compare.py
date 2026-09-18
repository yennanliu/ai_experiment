"""Exercise 3 — the brute arm does not fit, so there is no recall to compare.

    Compare brute-context Qwen2.5-VL-72B (80k context) to VideoAgent (Claude 3.5
    + retrieval) on a 1-hour video. Which wins on recall? Which wins on latency?

Reading of the exercise: the first thing to check is whether the comparison can
be run at all, and it cannot -- an hour at the lesson's own rate is 3.65x an 80k
context. The brute arm is then costed at the sampling rate that *would* fit, and
the two are compared on that basis, because "which wins on recall" between a
model that runs and one that does not is not a measurement.

**ANSWER: VideoAgent wins recall by forfeit.** One hour at 1 FPS and 81 tokens a
frame is **291,600** tokens against an 80k context -- **3.65x** over. Fitting
requires dropping to **0.274 FPS**, one frame every **3.65 seconds**, at which
point a one-second event appears in no frame at all and the needle question the
exercise is really asking has no answer to find.

**ANSWER: brute context wins latency, and by more than the token count
suggests.** The agent spends **7,490** tokens -- **38.9x** fewer, **97.4%**
cheaper -- but pays three sequential tool round trips for them. Prefill is
parallel and a round trip is not, so the agent wins only if *each*
retrieval costs less than **24,170** tokens of prefill time; at a microsecond a
token that is 24 ms, and real retrieval is not that fast.

**FINDING: the agent's cost is flat and the brute arm's is linear, so they cross
at 92 seconds.** 7,490 tokens is 92 frames at 81 each. Above a minute and a half
of video the agent is cheaper in tokens for *any* duration, which is why the
lesson's own 2-hour claim of "99% cheaper" is really a claim about the crossover
being early rather than about two hours.

**FINDING: and that claim is 98.7%, not 99%.** 7,490 against 583,200 is 1.28%.
At the one hour the exercise names it is **97.4%**, because the agent's numerator
is constant and the denominator halves.

Structure: `brute_tokens` prices a duration at a sampling rate, `fitting_fps`
inverts a context into the rate that fits, `agent_tokens` reads the lesson's own
retrieval simulation, and `crossover` is where the flat cost meets the linear one.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "18-long-video-million-token"
HOUR, TWO_HOURS = 3600, 7200
FPS, PER_FRAME = 1, 81
BRUTE_CONTEXT = 80_000
CLIPS, CLIP_SECONDS, OVERHEAD = 3, 30, 200


def brute_tokens(ref, seconds, fps=FPS, per_frame=PER_FRAME):
    return ref.tokens(seconds, fps, per_frame)


def fitting_fps(seconds, context=BRUTE_CONTEXT, per_frame=PER_FRAME):
    return round(context / per_frame / seconds, 3)


def agent_tokens():
    return CLIPS * CLIP_SECONDS * PER_FRAME + OVERHEAD


def crossover(per_frame=PER_FRAME):
    return agent_tokens() // per_frame


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    hour = brute_tokens(ref, HOUR)
    two = brute_tokens(ref, TWO_HOURS)
    agent = agent_tokens()
    frames = BRUTE_CONTEXT // PER_FRAME
    return {
        "hour_tokens": hour, "context": BRUTE_CONTEXT,
        "overflow": round(hour / BRUTE_CONTEXT, 2),
        "fits": hour <= BRUTE_CONTEXT,
        "max_frames": frames, "max_minutes": round(frames / 60, 1),
        "fitting_fps": fitting_fps(HOUR),
        "seconds_per_frame": round(HOUR / frames, 2),
        "agent_tokens": agent,
        "ratio": round(hour / agent, 1),
        "cheaper_pct": round((1 - agent / hour) * 100, 1),
        "two_hour_pct": round((1 - agent / two) * 100, 1),
        "round_trips": CLIPS,
        "prefill_budget": (BRUTE_CONTEXT - agent) // CLIPS,
        "crossover_frames": crossover(),
        "crossover_seconds": crossover(),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: VideoAgent wins recall by forfeit -- the brute arm is 3.65x over",
            all([result["hour_tokens"] == 291_600, not result["fits"],
                 result["overflow"] == 3.65, result["fitting_fps"] == 0.274,
                 result["seconds_per_frame"] == 3.65,
                 result["max_minutes"] == 16.4]),
            f"an hour at {FPS} FPS and {PER_FRAME} tokens a frame is "
            f"{result['hour_tokens']:,} tokens against {result['context']:,} -- "
            f"{result['overflow']}x over, and the context holds only "
            f"{result['max_minutes']} minutes. Fitting an hour needs "
            f"{result['fitting_fps']} FPS, one frame every {result['seconds_per_frame']} "
            "seconds, at which point a one-second event is in no frame at all",
        ),
        practice.Check(
            "ANSWER: brute context wins latency, and by more than the token count suggests",
            all([result["agent_tokens"] == 7490, result["ratio"] == 38.9,
                 result["cheaper_pct"] == 97.4, result["round_trips"] == CLIPS,
                 result["prefill_budget"] == 24_170]),
            f"the agent spends {result['agent_tokens']:,} tokens -- {result['ratio']}x fewer, "
            f"{result['cheaper_pct']}% cheaper -- but pays {result['round_trips']} sequential "
            f"tool round trips. Prefill is parallel and a round trip is not, so the agent "
            f"wins only if each retrieval costs less than {result['prefill_budget']:,} tokens "
            "of prefill time -- 24 ms at a microsecond a token, and real retrieval is not "
            "that fast",
        ),
        practice.Check(
            "FINDING: flat against linear, so they cross at 92 seconds",
            all([result["crossover_frames"] == 92, result["crossover_seconds"] == 92,
                 result["agent_tokens"] // PER_FRAME == 92]),
            f"{result['agent_tokens']:,} tokens is {result['crossover_frames']} frames at "
            f"{PER_FRAME} each, so above {result['crossover_seconds']} seconds of video the "
            "agent is cheaper in tokens for any duration. The lesson's 2-hour claim is really "
            "a claim about the crossover being early",
        ),
        practice.Check(
            "FINDING: and that claim is 98.7%, not 99%",
            all([result["two_hour_pct"] == 98.7, result["cheaper_pct"] == 97.4,
                 result["two_hour_pct"] > result["cheaper_pct"]]),
            f"{result['agent_tokens']:,} against a 2-hour {TWO_HOURS * PER_FRAME:,} is "
            f"{result['two_hour_pct']}% cheaper, not 99%. At the one hour the exercise names "
            f"it is {result['cheaper_pct']}%, because the agent's numerator is constant and "
            "the denominator halves",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
