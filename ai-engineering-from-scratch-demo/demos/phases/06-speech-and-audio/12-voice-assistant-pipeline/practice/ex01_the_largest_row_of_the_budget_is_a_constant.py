"""Exercise 1 — the largest row of the budget is a constant.

    **Easy.** Run `code/main.py`. It simulates one full turn end-to-end with stub
    modules and prints per-stage latency.

Reading of the exercise: the file's own headline is Step 5's "TOTAL
user-perceived (to first audio): 798.1 ms (target: < 800 ms)", so the exercise is
read as "run it and check what that number is made of". It is made of four
numbers, none of which is a latency this program measured:

| row | source | nominal |
|---|---|---:|
| VAD + capture | `silent_ms`, the hang-over counter | **400.0 ms** |
| STT | `sleep(0.08 + samples/sr * 0.05)` | 166.0 ms |
| LLM + tool | `sleep(0.12)` | 120.0 ms |
| TTS TTFA | `sleep(0.10)` | 100.0 ms |
| | **total** | **786.0 ms** |

**The margin to the stated target is 14.0 ms, 1.8% of the budget** -- and
`time.sleep` overshoots, so the number printed is always larger than 786. On this
machine the four sleeps overshoot by enough to consume most of that margin, and
repeated runs straddle 800.

**The biggest row, 50.9% of the total, is read off the fixture.** `silent_ms`
counts 20 ms per non-speech chunk after the trigger and the loop breaks at 400.
`mic_generator`'s default mask ends in exactly **20** silent chunks, so it breaks
on the very last one. Shorten the tail to 19 and the generator runs out first:
`silent_ms` reaches only **380**, no break happens, and the same budget prints
**765.0 ms**. The row measures the length of the test tape.

**The capture double-counts one chunk.** When the trigger fires, the pre-roll ring
already holds the current chunk -- it was appended at the top of the same
iteration -- and the code then calls `buffered.extend(chunk)` again. **85 chunks
go in and 86 come out**: 1700 ms of audio becomes 1720 ms, and the duplicated
20 ms is the speech onset.

**The row labelled "LLM + tool" excludes the tool.** `t_llm` is stopped before the
`for call in response["tool_calls"]` loop begins, so `dispatch_tool`'s
`sleep(0.01)` is outside every stage. And two timings are measured and thrown
away: `t_capture` is printed in Step 1 and never enters the budget, `t_play` is
assigned and never referenced again.

Structure: `flush` returns the pre-roll's samples; `capture` re-runs the lesson's
Step 1 loop instrumented; `stages` times the four stubs as `main()` times them;
`nominal` is the same four rows computed from the sleep constants.
"""

from __future__ import annotations

import inspect
import statistics
import time

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "12-voice-assistant-pipeline"
SR, CHUNK_MS, PRE_ROLL, HANG_MS, TARGET_MS = 16000, 20, 15, 400, 800.0
STT_FIXED_MS, STT_RTF, LLM_MS, TOOL_MS, TTS_MS = 80.0, 0.05, 120.0, 10.0, 100.0
SHORT_MASK = [False] * 5 + [True] * 60 + [False] * 19
REPEATS = 5


def flush(pre_roll):
    """Every sample in the pre-roll ring -- which already holds the current chunk."""
    return [sample for chunk in pre_roll for sample in chunk]


def capture(ref, mask=None):
    """The lesson's Step 1 loop, instrumented: (samples, silent_ms, chunks yielded)."""
    buffered, pre_roll, triggered, silent, seen = [], [], False, 0, 0
    for chunk, _ in ref.mic_generator(speech_mask=mask):
        seen += 1
        pre_roll.append(chunk)
        if len(pre_roll) > PRE_ROLL:
            pre_roll.pop(0)
        if ref.vad(chunk):
            if not triggered:
                buffered.extend(flush(pre_roll))
            triggered, silent = True, 0
            buffered.extend(chunk)
        elif triggered:
            silent += CHUNK_MS
            buffered.extend(chunk)
            if silent >= HANG_MS:
                break
    return len(buffered), silent, seen


def nominal(samples, silent):
    """The four Step 5 rows, computed from the sleep constants rather than a clock."""
    stt = STT_FIXED_MS + samples / SR * 1000 * STT_RTF
    return {"VAD + capture": float(silent), "STT": stt, "LLM + tool": LLM_MS, "TTS TTFA": TTS_MS}


def timed(fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    return result, (time.perf_counter() - start) * 1000


def stages(ref, samples, silent):
    """One turn through the stubs, timed where `main()` times it."""
    text, stt = timed(ref.streaming_stt, [0.2] * samples)
    reply, llm = timed(ref.llm_with_tools, text)
    _, tool = timed(ref.dispatch_tool, "set_timer", {"seconds": 300})
    audio, tts = timed(ref.streaming_tts, reply["text"])
    return {"VAD + capture": float(silent), "STT": stt, "LLM + tool": llm,
            "TTS TTFA": tts}, tool, len(audio)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    samples, silent, seen = capture(ref)
    short_samples, short_silent, _ = capture(ref, SHORT_MASK)
    runs = [stages(ref, samples, silent) for _ in range(REPEATS)]
    totals = [sum(rows.values()) for rows, _, _ in runs]
    source = inspect.getsource(ref.main)
    budget = source.split("stages = [")[1].split("]")[0]
    return {
        "samples": samples, "silent": silent, "chunks_in": seen,
        "chunks_out": samples / (SR * CHUNK_MS / 1000),
        "nominal": nominal(samples, silent), "short": sum(nominal(short_samples, short_silent).values()),
        "short_silent": short_silent, "totals": totals,
        "tool": statistics.mean(t for _, t, _ in runs), "chunks": runs[0][2],
        "tool_in_budget": "t_tool" in budget or "dispatch" in budget,
        "capture_in_budget": "t_capture" in budget, "play_uses": source.count("t_play"),
    }


def verify(result):
    rows = result["nominal"]
    total = sum(rows.values())
    margin = TARGET_MS - total
    overshoot = statistics.mean(result["totals"]) - total
    return [
        practice.Check(
            "ANSWER: the budget is four constants summing to 786.0 ms, 14.0 ms under target",
            abs(total - 786.0) < 1e-9 and 0 < margin < 0.02 * total,
            f"{ {k: round(v, 1) for k, v in rows.items()} } sums to {total:.1f} ms against the "
            f"stated {TARGET_MS:.0f} ms target -- a margin of {margin:.1f} ms, "
            f"{margin / total:.1%} of the budget. None of the four is a measured latency",
        ),
        practice.Check(
            "MECHANISM: `time.sleep` overshoots, so the printed total is always larger",
            overshoot > 0,
            f"over {REPEATS} runs the same four rows time at "
            f"{statistics.mean(result['totals']):.1f} ms (min {min(result['totals']):.1f}, max "
            f"{max(result['totals']):.1f}), an overshoot of {overshoot:.1f} ms that eats "
            f"{overshoot / margin:.0%} of the {margin:.1f} ms margin. Whether the demo meets "
            "its own target is a scheduler question",
        ),
        practice.Check(
            "FINDING: the largest row is read off the fixture, not off the VAD",
            rows["VAD + capture"] / total > 0.5 and result["short_silent"] == 380,
            f"`silent_ms` is {rows['VAD + capture']:.0f} ms, {rows['VAD + capture'] / total:.1%} "
            f"of the total, and it reaches 400 only because the default mask ends in exactly 20 "
            f"silent chunks. Trim the tail to 19 and the generator runs out first: silent_ms "
            f"stops at {result['short_silent']}, no break fires, and the budget prints "
            f"{result['short']:.1f} ms",
        ),
        practice.Check(
            "FINDING: the capture writes one more chunk than the mic yielded",
            result["chunks_out"] == result["chunks_in"] + 1,
            f"{result['chunks_in']} chunks in, {result['chunks_out']:.0f} out. On the trigger "
            f"the pre-roll already holds the current chunk, and `buffered.extend(chunk)` adds it "
            f"again -- so {result['chunks_in'] * CHUNK_MS} ms of audio becomes "
            f"{result['chunks_out'] * CHUNK_MS:.0f} ms and the duplicate is the speech onset",
        ),
        practice.Check(
            "CONTROL: the row named 'LLM + tool' excludes the tool, and two timings are dead",
            not result["tool_in_budget"] and not result["capture_in_budget"]
            and result["play_uses"] == 1,
            f"`t_llm` stops before the dispatch loop, so the tool's {result['tool']:.1f} ms sits "
            f"outside every row. `t_capture` is printed in Step 1 and never enters the budget, "
            f"and `t_play` appears {result['play_uses']} time in `main()` -- its own assignment",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
