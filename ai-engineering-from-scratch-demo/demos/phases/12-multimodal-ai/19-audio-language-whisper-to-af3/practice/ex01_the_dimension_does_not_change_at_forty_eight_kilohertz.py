"""Exercise 1 — the dimension does not change at 48 kHz.

    Compute the log-Mel spectrogram dimension for a 30-second clip at 16kHz,
    25ms window, 10ms hop, 80 Mel bins. How does this change at 48kHz?

Reading of the exercise: the frame count comes from the lesson's own
`window_frames`, run at both sample rates rather than reasoned about, because
the second half of the question invites an answer -- "three times as many
frames" -- that the arithmetic does not support. The window and hop are stated in
milliseconds, and a millisecond is not a sample.

**ANSWER: 2998 x 80 at 16 kHz, and 2998 x 80 at 48 kHz.** The shape does not
change. Tripling the sample rate triples the window from 400 to 1,200 samples and
the hop from 160 to 480, and the ratio that sets the frame count is untouched.

**FINDING: what changes is the bandwidth each Mel bin covers.** Nyquist goes
8 kHz to **24 kHz** over the same **80** bins, so each bin spans **100 Hz** at
16 kHz and **300 Hz** at 48 kHz. Same tensor, a third of the frequency
resolution -- which is the reason Whisper resamples everything to 16 kHz rather
than accepting whatever it is given.

**FINDING: the lesson's `window_frames` drops the tail, so it gives 2998 where
Whisper gives 3000.** Its loop runs `while i + win <= len(x)`, so the final
partial window is discarded and the last **20 ms** of every clip with it. Whisper
pads to a fixed 3,000 frames instead; the gap is 0.07% of the tensor and 100% of
the clip's last syllable.

**FINDING: and the lesson's own demo contradicts its own comment.** It prints
`frames : 98` beside the text "(should be ~99 at 1s)". One second at 16 kHz is
16,000 samples, which is `(16000 - 400) // 160 + 1 = 98` -- the comment rounds
1/hop up and the code rounds it down.

Structure: `frames_for` applies the lesson's own windowing arithmetic at a
sample rate, `bandwidth` divides Nyquist by the Mel bin count, and `RATES` is the
two rates compared.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "19-audio-language-whisper-to-af3"
DURATION, WIN_MS, HOP_MS, MELS = 30.0, 25, 10, 80
RATES = (16000, 48000)
WHISPER_FRAMES = 3000


def frames_for(rate, duration=DURATION, win_ms=WIN_MS, hop_ms=HOP_MS):
    samples = int(duration * rate)
    window, hop = int(rate * win_ms / 1000), int(rate * hop_ms / 1000)
    return (samples - window) // hop + 1


def bandwidth(rate, mels=MELS):
    return rate / 2 / mels


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    one_second = len(ref.window_frames(ref.synth_waveform(1.0, RATES[0]),
                                       RATES[0], WIN_MS, HOP_MS))
    shapes = {rate: (frames_for(rate), MELS) for rate in RATES}
    return {
        "shapes": shapes,
        "identical": shapes[RATES[0]] == shapes[RATES[1]],
        "windows": {rate: (int(rate * WIN_MS / 1000), int(rate * HOP_MS / 1000))
                    for rate in RATES},
        "nyquist": {rate: rate // 2 for rate in RATES},
        "bandwidth": {rate: bandwidth(rate) for rate in RATES},
        "bandwidth_ratio": round(bandwidth(RATES[1]) / bandwidth(RATES[0])),
        "whisper_frames": WHISPER_FRAMES,
        "dropped": WHISPER_FRAMES - frames_for(RATES[0]),
        "dropped_pct": round((WHISPER_FRAMES - frames_for(RATES[0]))
                             / WHISPER_FRAMES * 100, 2),
        "dropped_ms": (WHISPER_FRAMES - frames_for(RATES[0])) * HOP_MS,
        "one_second": one_second, "comment_says": 99,
    }


def verify(result):
    shapes, widths = result["shapes"], result["bandwidth"]
    return [
        practice.Check(
            "ANSWER: 2998 x 80 at 16 kHz, and 2998 x 80 at 48 kHz",
            all([shapes == {16000: (2998, 80), 48000: (2998, 80)},
                 result["identical"],
                 result["windows"] == {16000: (400, 160), 48000: (1200, 480)}]),
            f"the shape is {shapes[16000]} at both rates. Tripling the sample rate triples "
            f"the window and hop -- {result['windows']} samples -- and the ratio that sets "
            "the frame count is untouched, because both are stated in milliseconds",
        ),
        practice.Check(
            "FINDING: what changes is the bandwidth each Mel bin covers",
            all([result["nyquist"] == {16000: 8000, 48000: 24000},
                 widths == {16000: 100.0, 48000: 300.0},
                 result["bandwidth_ratio"] == 3]),
            f"Nyquist goes {result['nyquist'][16000]:,} to {result['nyquist'][48000]:,} Hz "
            f"over the same {MELS} bins, so a bin spans {widths[16000]:.0f} Hz at 16 kHz and "
            f"{widths[48000]:.0f} Hz at 48 kHz -- {result['bandwidth_ratio']}x. Same tensor, "
            "a third of the frequency resolution, which is why Whisper resamples to 16 kHz",
        ),
        practice.Check(
            "FINDING: window_frames drops the tail, so it gives 2998 where Whisper gives 3000",
            all([result["dropped"] == 2, result["dropped_pct"] == 0.07,
                 result["dropped_ms"] == 20]),
            f"the loop runs while i + win <= len(x), so the final partial window is discarded "
            f"and the last {result['dropped_ms']} ms of every clip with it: "
            f"{result['dropped']} frames of {result['whisper_frames']:,}, "
            f"{result['dropped_pct']}% of the tensor and all of the clip's last syllable",
        ),
        practice.Check(
            "FINDING: the lesson's own demo contradicts its own comment",
            all([result["one_second"] == 98, result["comment_says"] == 99,
                 result["one_second"] == frames_for(16000, 1.0)]),
            f"it prints {result['one_second']} frames beside the text 'should be ~"
            f"{result['comment_says']} at 1s'. One second at 16 kHz is "
            f"(16000 - 400) // 160 + 1 = {result['one_second']} -- the comment rounds up and "
            "the code rounds down",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
