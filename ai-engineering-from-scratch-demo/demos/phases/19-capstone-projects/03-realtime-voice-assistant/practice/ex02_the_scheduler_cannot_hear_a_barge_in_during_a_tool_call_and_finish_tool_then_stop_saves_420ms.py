"""Exercise 2 — the scheduler cannot hear a barge-in during a tool call, and finish-tool-then-stop saves 420 ms.

    Add an interruption-arbitration policy: what does the agent do when the user barges in during a tool call? Compare three policies (hard cancel, finish-tool-then-stop, queue next turn).

Reading of the exercise: first measure what the reference `run_session` does
today. The lesson's weather call is run with the user speaking 20-400 ms
into the 420 ms tool call, for 300 ms ("no, wait") or 600 ms (a full
follow-up). Then the three policies are compared on the same 40 barge-ins.
The timings are the reference's own, measured from `run_session`: tool
420 ms, filler at 300 ms, 740 ms commit-to-audio with the tool and 320 ms
without, and a 480 ms commit window. Two things are assumed: filler audio
lasts 1 s and the old answer 3 s. Each policy is scored on three things.
How long does the agent talk over the user? How long from the new turn's
end to first audio, for a follow-up that reuses the weather result and for
a correction that needs a new call? Does the stale answer get spoken?

**ANSWER: finish-tool-then-stop.** It stops audio at once, so talk-over is
0 ms. It keeps the result, so a follow-up is answered in 800 ms from user
stop. Hard cancel also talks over 0 ms, but it throws the result away, so the
follow-up re-runs the tool: 1220 ms. On a correction both take 1220 ms.
Queue-next-turn talks over the user for 390 ms on average and speaks the
stale answer. The new turn then waits for that answer: 3880 ms for a
follow-up, 4300 ms for a correction. Hard cancel is right only when the tool
is cheap and idempotent.

**FINDING: the shipped scheduler cannot hear a barge-in during a tool call.**
Barge-in is checked only in SPEAKING and THINKING, and the tool runs in
State.TOOL. Of 40 barge-ins, 33 are heard, each exactly when the tool
returns, 440 ms after it fired, whatever the offset. The other 7 are short
"no, wait"s that end before the tool returns. For those the stale answer is
spoken anyway. The filler "one second, let me check" plays over the user in
30 of 40. The lesson's own TypeScript port runs the tool inside THINKING, so
there the same barge-in would be heard at once.

Structure: `reference_run()` injects the barge-in into `synth_call` frames
and reads the event log; `policy()` scores one barge-in under one policy
from `measured_constants()`.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "03-realtime-voice-assistant"
CALL = "what is the weather in tokyo tomorrow"
OFFSETS = range(20, 420, 20)         # barge-in start, ms after the tool fires (0 would mask the commit)
UTTERANCES = {"short 'no wait'": 300, "full follow-up": 600}
FILLER_MS, ANSWER_MS, WINDOW = 1000, 3000, 480   # assumed audio lengths; commit window measured in ex03


def reference_run(ref, offset, dur):
    """The shipped scheduler with the user speaking from tool start + offset for dur ms."""
    frames = ref.synth_call(CALL)
    fire = ref.run_session(frames, use_tool=True).turn_complete_ms
    start = fire + offset
    frames = [ref.Frame(f.t_ms, f.is_speech or start <= f.t_ms < start + dur, f.partial) for f in frames]
    m = ref.run_session(frames, use_tool=True, barge_in_at_ms=start)
    at = lambda word: [int(t) for t in re.findall(rf"(\d+)ms [^\n]*{word}", "\n".join(m.events))]
    barge, filler, audio = at("BARGE-IN"), at("filler"), at("first audio-out")
    return {"detect_ms": barge[0] - start if barge else None,
            "filler_over_user": bool(filler) and start <= filler[0] < start + dur,
            "stale_answer": bool(audio) and not barge}


def measured_constants(ref):
    m = ref.run_session(ref.synth_call(CALL), use_tool=True)
    filler = int(re.search(r"(\d+)ms filler", "\n".join(m.events)).group(1)) - m.turn_complete_ms
    plain = ref.run_session(ref.synth_call(CALL), use_tool=False).latency_ms()
    return {"tool": ref.WEATHER.latency_ms, "filler_at": filler, "with_tool": m.latency_ms(), "plain": plain}


def overlap(a, b, c, d):
    return max(0, min(b, d) - max(a, c))


def policy(name, k, offset, dur):
    """One barge-in under a policy: talk-over ms, next-turn latency for a follow-up and a correction."""
    stop = offset + dur
    if name == "queue next turn":   # agent plays filler and the old answer, then takes the new turn
        talk = overlap(k["filler_at"], k["filler_at"] + FILLER_MS, offset, stop)
        talk += overlap(k["with_tool"], k["with_tool"] + ANSWER_MS, offset, stop)
        done = max(stop, k["with_tool"] + ANSWER_MS)
        return {"talk_over": talk, "follow_up": done - stop + WINDOW + k["plain"],
                "correction": done - stop + WINDOW + k["with_tool"], "stale": True, "result_kept": True}
    kept = name == "finish-tool-then-stop"   # hard cancel throws the in-flight result away
    follow = WINDOW + (k["plain"] if kept else k["with_tool"])
    return {"talk_over": 0, "follow_up": follow, "correction": WINDOW + k["with_tool"],
            "stale": False, "result_kept": kept}


def averaged(name, k):
    rows = [policy(name, k, o, d) for o in OFFSETS for d in UTTERANCES.values()]
    mean = {key: round(sum(r[key] for r in rows) / len(rows)) for key in ("talk_over", "follow_up", "correction")}
    return {**rows[0], **mean}


def tally(rs):
    return {"detected": sum(r["detect_ms"] is not None for r in rs),
            "at_tool_return": sum(r["detect_ms"] == 440 - o for o, r in zip(OFFSETS, rs)),
            "filler_over_user": sum(r["filler_over_user"] for r in rs),
            "stale_answer": sum(r["stale_answer"] for r in rs)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    k = measured_constants(ref)
    ts = (parity.lesson_dir(PHASE, LESSON) / "code" / "ts" / "src" / "orchestrator.ts").read_text()
    return {"k": k, "table": {n: averaged(n, k) for n in ("hard cancel", "finish-tool-then-stop", "queue next turn")},
            "shipped": {u: tally([reference_run(ref, o, d) for o in OFFSETS]) for u, d in UTTERANCES.items()},
            "ts_tool_in_thinking": bool(re.search(r'toolPhase = "running";[\s\S]{0,120}?state = "THINKING"', ts))}


def verify(result):
    t, sh = result["table"], result["shipped"]
    short, full = sh["short 'no wait'"], sh["full follow-up"]
    view = {n: (v["talk_over"], v["follow_up"], v["correction"], v["stale"]) for n, v in t.items()}
    return [
        practice.Check(
            "ANSWER: finish-tool-then-stop -- 0 ms talk-over and an 800 ms follow-up, against 1220 and 3880",
            (result["k"], view) == ({"tool": 420, "filler_at": 300, "with_tool": 740, "plain": 320},
                                    {"hard cancel": (0, 1220, 1220, False), "finish-tool-then-stop": (0, 800, 1220, False),
                                     "queue next turn": (390, 3880, 4300, True)}),
            f"(talk-over ms, follow-up ms, correction ms, stale answer) per policy: {view}",
        ),
        practice.Check(
            "FINDING: the shipped scheduler cannot hear a barge-in during a tool call",
            (short["detected"], full["detected"], short["at_tool_return"], full["at_tool_return"],
             short["stale_answer"], full["stale_answer"], short["filler_over_user"] + full["filler_over_user"],
             result["ts_tool_in_thinking"]) == (13, 20, 13, 20, 7, 0, 30, True),
            f"heard {short['detected'] + full['detected']}/40, all at tool return; stale answers "
            f"{short['stale_answer']}; filler over the user {short['filler_over_user'] + full['filler_over_user']}/40; "
            f"TS runs the tool in THINKING: {result['ts_tool_in_thinking']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
