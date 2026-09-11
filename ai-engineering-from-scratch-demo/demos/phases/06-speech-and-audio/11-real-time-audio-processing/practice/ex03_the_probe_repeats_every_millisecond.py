"""Exercise 3 — the probe repeats every millisecond.

    **Hard.** Build a full duplex echo test with `aiortc`: browser → WebRTC →
    Python → WebRTC → browser. Measure glass-to-glass latency with a 1 kHz pulse.

Reading of the exercise: `aiortc`, `av`, `sounddevice` and `torch` are all absent
and there is no browser, so the transport cannot be built. The *measurement* can
be, in full, and it is the half the exercise specifies in detail -- a 1 kHz pulse
recovered by cross-correlation. Injecting a known delay of **187 samples
(11.688 ms)** into a noisy return path and recovering it is a complete test of
the method, and the method has a defect the exercise's own choice of probe
creates.

**It works when the return is clean.** At 0 dB SNR both a 20 ms 1 kHz burst and a
20 ms 300-3000 Hz chirp recover 187 exactly, **20 of 20 trials each**, to a
resolution of one sample -- **0.0625 ms**.

**Every error the 1 kHz probe makes is a whole millisecond.** Drop the return to
-6 dB and the tone misses on **5 of 20**; at -10 dB, **11 of 20**. Across those
16 failures **every single one lands within one sample of a multiple of 16
samples** -- -16, -16, -1, +1, +16 at -6 dB; -48, -47, -16, -16, -16, +1, +1,
+16, +16, +16, +31 at -10 dB. The chirp fails **0 of 40** over the same trials.

**The cause is arithmetic, not noise.** 1000 Hz at 16 kHz is exactly **16 samples
per cycle**, so the probe's autocorrelation at a lag of one period is **0.9500**
of its own peak -- and the 5% it loses is the shortened overlap, not the
waveform. The argmax is choosing between candidates that are identical by
construction. The chirp's autocorrelation at the same lag is **-0.0598**.

**And the precision is 3000x finer than the thing being measured.** The lesson's
own figures put an echo path at `network in` 50-100 ms plus `network out`
50-100 ms (Step 4) plus a 60-80 ms jitter buffer (The Concept): **160-280 ms**,
which is **2560x to 4480x** the probe's 0.0625 ms resolution, on a transport
whose Opus frames quantise arrivals to 20 ms. A glass-to-glass figure quoted to
sub-millisecond precision is reporting the probe, not the path.

Structure: `tone` and `sweep` are the two probes; `receive` builds a delayed,
noisy return; `estimate` is the fixed-window cross-correlation; `lobe` measures
the autocorrelation one period out; `trials` runs one probe at one SNR.
"""

from __future__ import annotations

import importlib.util
import math
import random

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "11-real-time-audio-processing"
SR, PROBE_MS, DELAY, SLACK, TRIALS = 16000, 20, 187, 200, 20
PROBE_HZ, SWEEP = 1000.0, (300.0, 3000.0)
SNRS = (0, -6, -10)
NETWORK, JITTER = (50, 100), (60, 80)
ABSENT = ("aiortc", "av", "sounddevice", "torch")


def tone(count, hertz=PROBE_HZ):
    """The probe the exercise names: a 1 kHz burst."""
    return [math.sin(2 * math.pi * hertz * i / SR) for i in range(count)]


def sweep(count, low=SWEEP[0], high=SWEEP[1]):
    """The same energy spread over 300-3000 Hz, so no lag repeats."""
    span, rate = count / SR, (SWEEP[1] - SWEEP[0])
    return [math.sin(2 * math.pi * (low * (i / SR) + rate * (i / SR) ** 2 / (2 * span)))
            for i in range(count)]


def receive(probe, delay, snr_db, rnd):
    """A return path: silence, the probe at `delay`, and a noise floor at `snr_db`."""
    out = [0.0] * (delay + SLACK + len(probe) + 1)
    for index, value in enumerate(probe):
        out[delay + index] = value
    power = sum(x * x for x in probe) / len(probe)
    sigma = math.sqrt(power / 10 ** (snr_db / 10))
    return [x + rnd.gauss(0, sigma) for x in out]


def estimate(probe, received, maxlag):
    """Fixed-window cross-correlation: every lag sums the same number of products."""
    best_lag, best_score = 0, -1e18
    for lag in range(maxlag + 1):
        score = sum(value * received[lag + i] for i, value in enumerate(probe))
        if score > best_score:
            best_lag, best_score = lag, score
    return best_lag


def trials(make, snr_db, count=TRIALS):
    """Delay-estimation error, in samples, over `count` independent returns."""
    width = int(PROBE_MS * SR / 1000)
    probe = make(width)
    return [estimate(probe, receive(probe, DELAY, snr_db, random.Random(seed)),
                     DELAY + SLACK) - DELAY for seed in range(count)]


def lobe(signal, lag):
    """Autocorrelation at `lag`, normalised by the zero-lag peak."""
    return sum(signal[i] * signal[i + lag] for i in range(len(signal) - lag)) \
        / sum(x * x for x in signal)


def near_period(error, period):
    return abs(error - period * round(error / period)) <= 1


def survey(make):
    """(exact hits per SNR, every non-zero error at the noisy SNRs) for one probe."""
    table = {snr: trials(make, snr) for snr in SNRS}
    exact = {snr: sum(1 for error in errors if error == 0) for snr, errors in table.items()}
    return exact, sorted(error for snr in SNRS[1:] for error in table[snr] if error)


def solve():
    period, width = SR // int(PROBE_HZ), int(PROBE_MS * SR / 1000)
    tone_exact, missed = survey(tone)
    sweep_exact, sweep_missed = survey(sweep)
    return {
        "absent": [m for m in ABSENT if importlib.util.find_spec(m) is None],
        "period": period, "resolution_ms": 1000 / SR,
        "exact": {"tone": tone_exact, "sweep": sweep_exact},
        "missed": missed, "on_grid": sum(near_period(e, period) for e in missed),
        "sweep_missed": len(sweep_missed),
        "tone_lobe": lobe(tone(width), period), "sweep_lobe": lobe(sweep(width), period),
        "path": (2 * NETWORK[0] + JITTER[0], 2 * NETWORK[1] + JITTER[1]),
    }


def verify(result):
    exact, path = result["exact"], result["path"]
    ratio = path[0] / result["resolution_ms"]
    return [
        practice.Check(
            "ANSWER: a clean return recovers 187 samples exactly, with either probe",
            exact["tone"][0] == exact["sweep"][0] == TRIALS,
            f"find_spec is None for {result['absent']} and there is no browser, so the "
            f"transport cannot be built -- but the measurement can. At 0 dB SNR both a "
            f"{PROBE_MS} ms 1 kHz burst and a {PROBE_MS} ms sweep recover the "
            f"{DELAY}-sample ({DELAY / SR * 1000:.3f} ms) delay on {TRIALS}/{TRIALS} trials, to "
            f"a resolution of one sample -- {result['resolution_ms']:.4f} ms",
        ),
        practice.Check(
            "FINDING: every error the 1 kHz probe makes is a whole millisecond",
            result["on_grid"] == len(result["missed"]) > 10,
            f"at -6 dB the tone misses on {TRIALS - exact['tone'][-6]} of {TRIALS} and at "
            f"-10 dB on {TRIALS - exact['tone'][-10]}; all {len(result['missed'])} failures land "
            f"within one sample of a multiple of {result['period']} samples "
            f"({result['period'] / SR * 1000:.3f} ms): {result['missed']}",
        ),
        practice.Check(
            "CONTROL: the sweep makes no errors at all over the same returns",
            result["sweep_missed"] == 0,
            f"{result['sweep_missed']} failures across {2 * TRIALS} trials at -6 and -10 dB, "
            f"against the tone's {len(result['missed'])}. Same delay, same noise, same "
            "correlator -- the difference is the probe",
        ),
        practice.Check(
            "MECHANISM: 1 kHz at 16 kHz repeats every 16 samples, so the lobes are equal",
            result["tone_lobe"] > 0.9 > abs(result["sweep_lobe"]),
            f"1000 Hz at {SR} Hz is exactly {result['period']} samples per cycle, so the tone's "
            f"autocorrelation one period out is {result['tone_lobe']:.4f} of its own peak -- the "
            f"5% it loses is the shortened overlap, not the waveform. The sweep's is "
            f"{result['sweep_lobe']:.4f}. The argmax is choosing between identical candidates",
        ),
        practice.Check(
            "FINDING: the probe resolves 3000x finer than the path it is measuring",
            ratio > 1000,
            f"the lesson's own figures put an echo path at network in {NETWORK[0]}-{NETWORK[1]} "
            f"plus network out {NETWORK[0]}-{NETWORK[1]} plus a {JITTER[0]}-{JITTER[1]} ms "
            f"jitter buffer: {path[0]}-{path[1]} ms, which is {ratio:.0f}x to "
            f"{path[1] / result['resolution_ms']:.0f}x the probe's "
            f"{result['resolution_ms']:.4f} ms, on a transport whose Opus frames quantise "
            "arrivals to 20 ms. Sub-millisecond precision here reports the probe",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
