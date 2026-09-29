"""Exercise 1 — any real ASR lag drops the last word (14% WER), and 1 s whisper chunks never commit a turn.

    Swap Deepgram Nova-3 for faster-whisper v3 turbo on a g5.xlarge. Measure the latency and WER gap. Identify where CPU-vs-GPU decisions matter.

Reading of the exercise: neither service nor a g5.xlarge is reachable
offline, so the swap is made where the lesson's code can feel it. An ASR is
modelled by how late each word's partial appears after the word ends: 280 ms
for Nova-3 (the lesson's "partial @280ms"), 1 s for a chunked faster-whisper
loop, 0 for an ideal streamer. The lesson's `synth_call` shows each word from
its first frame, an oracle at -320 ms. Three scripts of 7, 6 and 9 words run
through the reference `run_session`. WER is scored on the transcript it
commits. Decode speed comes from the faster-whisper README (read 2026-09-29).

**ANSWER: at the lesson's scheduler, the latency gap is 520 ms and the WER
gap is total.** With a correct read of the transcript at commit, Nova-3's
words are all in by the 480 ms commit point: 800 ms from user stop to first
audio, WER 0. One-second whisper chunks finish 1000 ms after the stop, so a
turn cannot commit before then: 1320 ms. The shipped scheduler instead
commits nothing on whisper: 0/3 turns. CPU-vs-GPU matters in two places. The
first is the final decode, which must fit inside the 480 ms silence window.
Re-decoding the utterance hides up to 5.94 s of speech on the README's GPU
(large-v2 fp16), 2.38 s on its CPU (small fp32). The second is the load test:
real-time streams per device are 45.9 on the GPU with batch 8 and 5.0 on
8 CPU threads, both short of the lesson's 50 calls.

**FINDING: the scheduler drops the last word of any ASR that is not an
oracle.** `run_session` reads `f.partial` only on speech frames. A real ASR
reports a word after it is spoken, so the final word lands on a silence frame
and is never read. The WER is 14.3% on the 7-word call and 11.1% on the
9-word one, at 0 ms lag and at 280 ms alike. The 6-word call drops to 5 words,
scores 0.55 and never commits. Read at the commit frame, all 6 lag-0/280 runs
score WER 0.

**FINDING: the lesson's latency figure cannot see the ASR.** `latency_ms()`
is 320 ms for every run that commits, from the oracle to 280 ms lag, because
it starts the clock at the VAD commit.

Structure: `lagged_frames()` builds 20 ms frames with a lagged partial;
`run()` pushes them through `run_session` and scores WER both ways;
`solve()` adds the README's real-time factors.
"""

from __future__ import annotations

import re

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "03-realtime-voice-assistant"
SCRIPTS = ["what is the weather in tokyo tomorrow", "tell me a long story about",
           "please book a table for two at seven tonight"]
# ASR partial lag after a word ends: the lesson's own synth_call shows each word from its
# first frame (-320 ms, an oracle); 0 is an ideal streaming ASR; 280 ms is the lesson's
# "partial @280ms" figure for Nova-3; 1000 ms is a 1 s chunked faster-whisper loop.
LAGS = {"oracle (synth_call)": -320, "ideal streaming": 0, "Nova-3 (280 ms)": 280,
        "whisper 1 s chunks": 1000}
# faster-whisper README (github.com/SYSTRAN/faster-whisper, read 2026-09-29): seconds to
# transcribe 13 min (780 s) of audio, beam 5. GPU = RTX 3070 Ti, CPU = i7-12700K 8 threads.
README = {"GPU large-v2 fp16": 63, "GPU large-v2 fp16 batch 8": 17, "GPU large-v2 int8 batch 8": 16,
          "CPU small fp32": 157, "CPU small int8": 102}
WINDOW_MS = 480  # commit fires 480 ms after the last speech frame (silence_run_ms >= 500)


def lagged_frames(ref, script, lag):
    """20 ms frames of `script`, each word's partial visible `lag` ms after the word ends."""
    words, frames, t = script.split(), [], 0
    ends = [120 + 320 * (k + 1) for k in range(len(words))]
    stop = ends[-1]
    while t < stop + 2200:
        heard = " ".join(w for w, e in zip(words, ends) if e + lag <= t)
        frames.append(ref.Frame(t_ms=t, is_speech=120 <= t < stop, partial=heard))
        t += 20
    return frames, stop


def wer(ref_text, hyp):
    r, h = ref_text.split(), hyp.split()
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[-1] / len(r)


def run(ref, script, lag):
    frames, stop = lagged_frames(ref, script, lag)
    m = ref.run_session(frames, use_tool=False)
    said = re.findall(r"partial='(.*)'", "\n".join(m.events))
    at_commit = next(f.partial for f in frames if f.t_ms == stop + WINDOW_MS)
    return {"committed": bool(m.turn_complete_ms), "latency": m.latency_ms(),
            "wer": round(wer(script, said[0]), 3) if said else None,
            "wer_if_read_at_commit": round(wer(script, at_commit), 3)}


def readme_budget():
    """Real-time factors from the README: speech hidden in the commit window, streams per device."""
    rtf = {k: v / 780 for k, v in README.items()}
    return ({k: round(WINDOW_MS / 1000 / r, 2) for k, r in rtf.items()}, {k: round(1 / r, 1) for k, r in rtf.items()})


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    hidden, streams = readme_budget()
    runs = {name: [run(ref, s, lag) for s in SCRIPTS] for name, lag in LAGS.items()}
    pipeline = runs["oracle (synth_call)"][0]["latency"]  # commit -> first audio, measured
    return {
        "cols": {name: [(x["committed"], x["wer"]) for x in runs[name]] for name in LAGS},
        "wer_at_commit_280": [x["wer_if_read_at_commit"] for x in runs["Nova-3 (280 ms)"]],
        "latencies": sorted({x["latency"] for x in sum(runs.values(), []) if x["committed"]}),
        # a scheduler that waits for the whole transcript commits at max(window, lag)
        "from_stop_ms": {name: max(WINDOW_MS, lag) + pipeline for name, lag in LAGS.items()},
        "hidden_up_to_s": hidden, "streams": streams,
    }


def verify(result):
    cols, fair, hid, st = result["cols"], result["from_stop_ms"], result["hidden_up_to_s"], result["streams"]
    whisper = [c for c, _ in cols["whisper 1 s chunks"]]
    got = (fair["Nova-3 (280 ms)"], fair["whisper 1 s chunks"], result["wer_at_commit_280"], whisper,
           hid["GPU large-v2 fp16"], hid["CPU small fp32"], st["GPU large-v2 fp16 batch 8"], st["CPU small fp32"])
    lagged = [(True, 0.143), (False, None), (True, 0.111)]
    return [
        practice.Check(
            "ANSWER: a 520 ms latency gap, whisper commits 0/3 turns, and GPU vs CPU decides decode and load",
            got == (800, 1320, [0.0] * 3, [False] * 3, 5.94, 2.38, 45.9, 5.0),
            f"user stop to first audio {fair['Nova-3 (280 ms)']} vs {fair['whisper 1 s chunks']} ms; "
            f"whisper commits {sum(whisper)}/3; decode hidden up to {hid} s; real-time streams {st}",
        ),
        practice.Check(
            "FINDING: the scheduler drops the last word of any ASR that is not an oracle",
            (cols["oracle (synth_call)"], cols["ideal streaming"], cols["Nova-3 (280 ms)"])
            == ([(True, 0.0)] * 3, lagged, lagged),
            f"committed/WER per script: {cols}",
        ),
        practice.Check(
            "FINDING: the lesson's latency figure cannot see the ASR",
            result["latencies"] == [320],
            f"latency_ms of every committed run: {result['latencies']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
