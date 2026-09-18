"""Exercise 3 — four screenshots buy 2,929 steps.

    Long-horizon memory compression: design a summary-chain with ≤4 screenshots
    kept live, any number logged.

Reading of the exercise: the design is written and then priced against a context
window, because "≤4 screenshots" is a constraint whose whole justification is a
number -- and the number turns out to be that keeping all of them exhausts a 128k
context at step **12**, which is shorter than either of the lesson's own
benchmark tasks would be in a real browser.

**ANSWER: four live screenshots, a full action log, and a rolling summary every
ten steps.** Per step the log costs ~30 tokens and a screenshot ~10,549 (Lesson
12.02's 1920x1080 at patch 14), so the live set is **42,196** tokens, the log
**1,500** over fifty steps and five summaries **1,000** -- **44,696** in total
against **527,450** for keeping every frame. **11.8x**.

**FINDING: keeping every screenshot exhausts a 128k context at step 12.**
131,072 / 10,549 = **12**. Neither of the lesson's two benchmark tasks is longer
than six steps, which is why its simulator never meets the constraint this
exercise exists to impose.

**FINDING: the compressed scheme is constant in step count and the log is what
grows.** After step 4 the screenshot term stops moving, so the remaining budget
buys **2,929** more steps at 30 tokens each. The design turns a linear cost into
a constant plus a slope 350x shallower.

**FINDING: and the four are not interchangeable.** The useful set is first,
previous, current and the last *state-changing* one -- not the last four. The
first anchors the goal, the current grounds the next action, the previous detects
a no-op, and the last state-changing frame is the only one that can answer "what
did my last real action do", which Exercise 4 needs and a sliding window of four
consecutive frames throws away during any run of failed clicks.

Structure: `screenshot_tokens` prices one frame, `budget` prices a policy over N
steps, and `POLICIES` compares keeping everything against the ≤4 design.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "25-multimodal-agents-computer-use"
SCREEN, PATCH = (1920, 1080), 14
LIVE, LOG_TOKENS, SUMMARY_TOKENS, SUMMARY_EVERY = 4, 30, 200, 10
STEPS = 50
CONTEXT = 131072
SLOTS = ("first", "previous", "current", "last state-changing")


def screenshot_tokens(screen=SCREEN, patch=PATCH):
    return (screen[0] // patch) * (screen[1] // patch)


def keep_all(steps=STEPS):
    return steps * screenshot_tokens()


def compressed(steps=STEPS, live=LIVE):
    summaries = steps // SUMMARY_EVERY
    return live * screenshot_tokens() + steps * LOG_TOKENS + summaries * SUMMARY_TOKENS


def exhausts_at(context=CONTEXT):
    return context // screenshot_tokens()


def headroom(context=CONTEXT, live=LIVE):
    fixed = live * screenshot_tokens() + (STEPS // SUMMARY_EVERY) * SUMMARY_TOKENS
    return (context - fixed) // LOG_TOKENS


def solve():
    parity.load_reference(PHASE, LESSON, "main")
    frame = screenshot_tokens()
    return {
        "frame": frame, "live": LIVE, "live_tokens": LIVE * frame,
        "log_total": STEPS * LOG_TOKENS,
        "summaries": STEPS // SUMMARY_EVERY,
        "summary_total": (STEPS // SUMMARY_EVERY) * SUMMARY_TOKENS,
        "compressed": compressed(), "keep_all": keep_all(),
        "ratio": round(keep_all() / compressed(), 1),
        "exhausts_at": exhausts_at(),
        "bench_steps": 6,
        "headroom_steps": headroom(),
        "slope_ratio": frame // LOG_TOKENS,
        "slots": SLOTS, "slot_count": len(SLOTS),
        "context": CONTEXT,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: four live frames, a full log, a summary every ten steps -- 44,696 tokens",
            all([result["frame"] == 10_549, result["live_tokens"] == 42_196,
                 result["log_total"] == 1_500, result["summary_total"] == 1_000,
                 result["compressed"] == 44_696, result["keep_all"] == 527_450,
                 result["ratio"] == 11.8]),
            f"a frame is {result['frame']:,} tokens at {SCREEN[0]}x{SCREEN[1]} and patch "
            f"{PATCH}, so {LIVE} live frames are {result['live_tokens']:,}, the log "
            f"{result['log_total']:,} over {STEPS} steps and {result['summaries']} summaries "
            f"{result['summary_total']:,} -- {result['compressed']:,} against "
            f"{result['keep_all']:,} for every frame, {result['ratio']}x",
        ),
        practice.Check(
            "FINDING: keeping every screenshot exhausts a 128k context at step 12",
            all([result["exhausts_at"] == 12, result["bench_steps"] < result["exhausts_at"]]),
            f"{result['context']:,} / {result['frame']:,} = {result['exhausts_at']}. Neither "
            f"of the lesson's benchmark tasks is longer than {result['bench_steps']} steps, "
            "which is why its simulator never meets the constraint this exercise imposes",
        ),
        practice.Check(
            "FINDING: the compressed scheme is constant and the log is what grows",
            all([result["headroom_steps"] == 2929, result["slope_ratio"] == 351]),
            f"after step {LIVE} the screenshot term stops moving, so the remaining budget "
            f"buys {result['headroom_steps']:,} more steps at {LOG_TOKENS} tokens each. The "
            f"design turns a linear cost into a constant plus a slope {result['slope_ratio']}x "
            "shallower",
        ),
        practice.Check(
            "FINDING: and the four are not interchangeable",
            all([result["slot_count"] == LIVE,
                 result["slots"][-1] == "last state-changing"]),
            f"the useful set is {list(result['slots'])} -- not the last four. The first "
            "anchors the goal, the current grounds the next action, the previous detects a "
            "no-op, and the last state-changing frame is the only one that answers what the "
            "last real action did. A sliding window throws that away during any run of failed "
            "clicks",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
