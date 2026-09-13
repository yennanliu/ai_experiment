"""Exercise 1 — 98, not 100, and it is two short at every length.

    **Easy.** Run `code/main.py`. Confirm the frame count for a 1-second signal
    at 16 kHz with 10 ms hop is ~100 frames. For 30 seconds: ~3,000 frames.

Reading of the exercise: "~100" is read as a number to check rather than to
accept, so the lesson's own `frame_signal` is run at 1, 5 and 30 seconds and the
result compared with the closed form.

**ANSWER: 98 frames, and 2,998 at 30 seconds.** `frame_signal` steps
`range(0, len(x) - frame_size + 1, hop)`, so it emits
`floor((n - 400) / 160) + 1` frames -- exactly `n/160 - 2` whenever the hop
divides the length. The deficit is `(frame_size - hop) / hop = 1.5`, rounded up:
**2 frames at every duration**, not a percentage.

| duration | samples | frames | `n / hop` | short by |
|---|---:|---:|---:|---:|
| 1 s | 16,000 | **98** | 100 | 2 |
| 5 s | 80,000 | **498** | 500 | 2 |
| 30 s | 480,000 | **2,998** | 3,000 | 2 |

**FINDING: the lesson's own `TARGET_FRAMES` is unreachable from audio.** It is
3,000, and 30 seconds of real signal produces 2,998, so `pad_or_clip` appends
**2 zero frames** to every full-length Whisper window this pipeline builds. The
last 20 ms of every input is silence that was never recorded. Whisper's own
front end centre-pads the waveform instead, which is what gets it to 3,000
frames from 30 seconds.

**FINDING: every frame is 12.5 ms late.** Frame `k` covers samples
`[160k, 160k + 400)`, so its centre sits at `160k + 200` -- a constant **200
samples**, 12.5 ms, after the hop position it is named for. Whisper's front end
centres each frame on its hop instead, which is the same change that gets it to
3,000 frames. Here every timestamp the model could emit is offset by half a
frame, uniformly, and the offset is exactly `(frame - hop) / 2`.

**CONTROL: `pad_or_clip` truncates in the other direction without complaint.**
60 seconds of audio becomes 3,000 frames by dropping the second half, silently.

Structure: `frames_for` runs the lesson's framer at one duration; `closed_form`
is the arithmetic it should match.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "10-audio-transformers-whisper"
SECONDS, RATE, FRAME, HOP = (1.0, 5.0, 30.0), 16_000, 400, 160


def frames_for(ref, seconds):
    """(samples, frames from the lesson's framer) for a tone of this length."""
    signal = ref.sine_wave(440, seconds)
    return len(signal), len(ref.frame_signal(signal))


def closed_form(samples, frame=FRAME, hop=HOP):
    """floor((n - frame) / hop) + 1 -- what range(0, n - frame + 1, hop) emits."""
    return (samples - frame) // hop + 1


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    counts = {seconds: frames_for(ref, seconds) for seconds in SECONDS}
    full = ref.frame_signal(ref.sine_wave(440, 30.0))
    padded = ref.pad_or_clip(full, ref.TARGET_FRAMES)
    long = ref.pad_or_clip(ref.frame_signal(ref.sine_wave(440, 60.0)), ref.TARGET_FRAMES)
    return {
        "counts": counts, "target": ref.TARGET_FRAMES,
        "short": {s: samples // HOP - frames for s, (samples, frames) in counts.items()},
        "closed": all(closed_form(samples) == frames for samples, frames in counts.values()),
        "padded": len(padded), "silent": sum(1 for frame in padded if not any(frame)),
        "clipped": (len(long), len(ref.frame_signal(ref.sine_wave(440, 60.0)))),
        "centres": {k: HOP * k + FRAME // 2 for k in (0, 1, 97)},
        "offsets": {HOP * k + FRAME // 2 - HOP * k for k in range(98)},
    }


def verify(result):
    counts, short = result["counts"], result["short"]
    return [
        practice.Check(
            "ANSWER: 98 frames, 2,998 at 30 s, and 2 short at every length",
            counts[1.0][1] == 98 and counts[30.0][1] == 2_998 and set(short.values()) == {2},
            f"samples -> frames: "
            f"{ {s: counts[s] for s in SECONDS} }, against n/hop of "
            f"{ {s: counts[s][0] // HOP for s in SECONDS} }. The deficit is "
            f"(frame - hop)/hop = {(FRAME - HOP) / HOP} rounded up -- 2 frames at every duration, "
            "not a percentage",
        ),
        practice.Check(
            "ANSWER: it matches floor((n - 400) / 160) + 1 exactly",
            result["closed"],
            "range(0, len(x) - frame_size + 1, hop) emits exactly that many starts, so the count "
            "is arithmetic rather than an approximation. '~100' is 98 and '~3,000' is 2,998, at "
            "every length and every seed",
        ),
        practice.Check(
            "FINDING: TARGET_FRAMES is unreachable from audio",
            result["padded"] == result["target"] and result["silent"] == 2,
            f"TARGET_FRAMES is {result['target']:,} and 30 seconds of real signal gives "
            f"{counts[30.0][1]:,}, so pad_or_clip appends {result['silent']} zero frames to every "
            "full-length window this pipeline builds -- the last 20 ms of every Whisper input is "
            "silence that was never recorded. Whisper's own front end centre-pads the waveform",
        ),
        practice.Check(
            "FINDING: every frame is 12.5 ms late, by exactly (frame - hop) / 2",
            result["offsets"] == {FRAME // 2},
            f"frame k covers [{HOP}k, {HOP}k + {FRAME}), so its centre is at {HOP}k + "
            f"{FRAME // 2}: a constant {FRAME // 2} samples = "
            f"{FRAME // 2 / RATE * 1000:.1f} ms after the hop position it is named for, at every "
            f"one of the 98 frames. Whisper centres each frame on its hop instead -- the same "
            "change that gets it to 3,000 frames",
        ),
        practice.Check(
            "CONTROL: pad_or_clip truncates in the other direction without complaint",
            result["clipped"] == (result["target"], 5_998),
            f"60 seconds frames to {result['clipped'][1]:,} and comes back as "
            f"{result['clipped'][0]:,} -- the second half is dropped and nothing says so. The "
            "function is named for both behaviours and warns about neither",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
