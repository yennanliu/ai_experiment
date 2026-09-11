"""Exercise 3 — the margin is the residue of two effects ten times its size.

    **Hard.** Fine-tune a RawNet2 or AASIST on ASVspoof 2019 LA. Measure EER.
    Test on a held-out set of F5-TTS-generated clips — see how OOD detection
    degrades.

Reading of the exercise: `torch`, `torchaudio`, `transformers` and `soundfile`
are absent, ASVspoof 2019 LA is not here and the reference tree holds no audio
file, so nothing can be fine-tuned. The question that survives is the one the
second half asks -- what happens off-distribution -- and it can be answered
exactly against the detector `code/main.py` ships, because that detector is one
scalar and its fixture has two moving parts.

**In distribution it is perfect, and the margin is an accident.** The 10 real
clips average **0.193926** and the 10 shipped fakes **0.210493**, disjoint, so
the EER is **0.0000**. But the gap is only **0.016567**, and it is what is left
after two much larger effects nearly cancel:

| family | mean high-band ratio | vs real |
|---|---:|---:|
| real (noise 0.02, no artifact) | 0.193926 | — |
| fake generator, artifact removed (noise 0.002) | **0.038557** | **-0.155369** |
| fake as shipped (artifact 6000 Hz, noise 0.002) | 0.210493 | +0.016567 |
| artifact 6000 Hz kept, real noise floor restored | **0.317614** | **+0.123688** |

The fake's noise floor is 10x lower than the real one, which pulls its high-band
ratio down by **0.155**; its 6 kHz artifact pushes it back up by **0.172**. The
detector lives on the **0.017** left over -- about a tenth of either term.

**Off distribution it does not degrade, it inverts.** `toy_detector_score` sums
`spec[len(spec)//2:]`, and at `n_fft=256` and 16 kHz that is bin 64 upward:
**4000 Hz**. Move the artifact 100 Hz below that edge, 6000 Hz to 3900 Hz, and
every other generator setting unchanged, and the EER goes **0.0000 to 1.0000** --
every spoof now ranks as more real than every real clip. A hundred hertz, on a
boundary that is a hard-coded `//2`.

And the class is decided as much by a parameter that is not an artifact at all.
Keeping the 6 kHz tone and restoring the real noise floor moves the score to
**0.317614**, which is **6.5x** the decision margin further from real than the
shipped fake is. Nothing about the spoof changed.

At 10 real and 10 fake the EER grid is multiples of **5.0 pp**, so neither the
0.42% the doc quotes for AASIST nor the 7.23% it quotes for ASVspoof 5 is a
number this fixture could produce.

Structure: `variant` builds one clip from a carrier, an optional artifact tone
and a noise floor; `family` scores ten of them with the lesson's own detector;
`eer` sweeps the threshold with fakes scoring high.
"""

from __future__ import annotations

import importlib.util
import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "16-anti-spoofing-audio-watermarking"
SR, SAMPLES, CLIPS, N_FFT = 16000, 16000, 10, 256
BASE_HZ, BASE_AMP = 220, 0.2
SHIPPED_TONE, TONE_AMP, FAKE_NOISE, REAL_NOISE = 6000, 0.05, 0.002, 0.02
BELOW_EDGE = 3900
QUOTED = (0.0042, 0.0723)
ABSENT = ("torch", "torchaudio", "transformers", "soundfile")


def variant(seed, tone_hz, tone_amp, noise):
    """One clip: the lesson's 220 Hz carrier, an optional artifact tone, a noise floor."""
    rng = random.Random(seed)
    return [BASE_AMP * math.sin(2 * math.pi * BASE_HZ * i / SR)
            + (tone_amp * math.sin(2 * math.pi * tone_hz * i / SR) if tone_amp else 0.0)
            + noise * rng.gauss(0, 1.0) for i in range(SAMPLES)]


def family(ref, first_seed, tone_hz, tone_amp, noise):
    return [ref.toy_detector_score(variant(first_seed + i, tone_hz, tone_amp, noise))
            for i in range(CLIPS)]


def eer(real, fake):
    """Equal error rate with the score rising toward fake -- the correct polarity."""
    best = (1.0, 0.0)
    for threshold in sorted(set(real + fake)):
        far = sum(1 for s in fake if s < threshold) / len(fake)
        frr = sum(1 for s in real if s >= threshold) / len(real)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]


def band_edge(ref):
    """The frequency `spec[len(spec)//2:]` starts at, for this n_fft and rate."""
    bins = len(ref.magnitude_spectrum(variant(0, 0, 0.0, REAL_NOISE), N_FFT))
    return bins // 2 * SR / N_FFT


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    real = [ref.toy_detector_score(ref.synth_real_speech(seed=i)) for i in range(CLIPS)]
    shipped = [ref.toy_detector_score(ref.synth_fake_speech(seed=100 + i)) for i in range(CLIPS)]
    families = {
        "artifact removed": family(ref, 300, 0, 0.0, FAKE_NOISE),
        "below the edge": family(ref, 200, BELOW_EDGE, TONE_AMP, FAKE_NOISE),
        "real noise floor": family(ref, 400, SHIPPED_TONE, TONE_AMP, REAL_NOISE),
    }
    means = {name: statistics.mean(scores) for name, scores in families.items()}
    means["real"], means["shipped"] = statistics.mean(real), statistics.mean(shipped)
    return {
        "absent": [m for m in ABSENT if importlib.util.find_spec(m) is None],
        "means": means, "margin": means["shipped"] - means["real"],
        "in_eer": eer(real, shipped),
        "eers": {name: eer(real, scores) for name, scores in families.items()},
        "edge": band_edge(ref), "step": 1 / CLIPS / 2,
        "reproduces": variant(100, SHIPPED_TONE, TONE_AMP, FAKE_NOISE)
        == ref.synth_fake_speech(seed=100),
    }


def verify(result):
    means, margin, eers = result["means"], result["margin"], result["eers"]
    floor = means["artifact removed"] - means["real"]
    tone = means["shipped"] - means["artifact removed"]
    return [
        practice.Check(
            "CONTROL: nothing can be fine-tuned, and the fixture generator is reproduced",
            len(result["absent"]) == len(ABSENT) and result["reproduces"],
            f"find_spec is None for {result['absent']} and ASVspoof 2019 LA is not here. The "
            "off-distribution families below are built from a generator that reproduces "
            "`synth_fake_speech` element for element at the shipped settings, so only the "
            "parameter named in each row differs",
        ),
        practice.Check(
            "ANSWER: the in-distribution EER is 0.0000 on a margin of 0.0166",
            result["in_eer"] == 0.0 and 0.01 < margin < 0.02,
            f"real averages {means['real']:.6f} and the shipped fakes {means['shipped']:.6f}, "
            f"disjoint, so the EER is {result['in_eer']:.4f}. The whole decision rests on "
            f"{margin:.6f} of high-band energy ratio",
        ),
        practice.Check(
            "MECHANISM: that margin is what is left after two effects ten times larger cancel",
            abs(floor) > 8 * margin and tone > 8 * margin,
            f"removing the artifact from the fake generator drops the ratio to "
            f"{means['artifact removed']:.6f}, {floor:+.6f} against real, because its noise "
            f"floor is 10x lower; the 6 kHz artifact then adds {tone:+.6f} back. The margin is "
            f"{margin:.6f} -- {abs(floor) / margin:.1f}x and {tone / margin:.1f}x smaller than "
            "the terms it is the difference of",
        ),
        practice.Check(
            "ANSWER: 100 Hz off distribution the detector inverts rather than degrades",
            eers["below the edge"] == 1.0,
            f"`toy_detector_score` sums `spec[len(spec)//2:]`, which at n_fft={N_FFT} and "
            f"{SR // 1000} kHz starts at {result['edge']:.0f} Hz. Moving the artifact from "
            f"{SHIPPED_TONE} Hz to {BELOW_EDGE} Hz and changing nothing else takes the EER from "
            f"{result['in_eer']:.4f} to {eers['below the edge']:.4f} -- every spoof now ranks as "
            "more real than every real clip",
        ),
        practice.Check(
            "FINDING: a parameter that is not an artifact moves the score further than the artifact",
            means["real noise floor"] - means["shipped"] > 5 * margin,
            f"keeping the 6 kHz tone and restoring the real noise floor gives "
            f"{means['real noise floor']:.6f}, "
            f"{(means['real noise floor'] - means['shipped']) / margin:.1f}x the decision margin "
            f"further from real than the shipped fake. At {CLIPS} and {CLIPS} the EER grid is "
            f"multiples of {result['step'] * 100:.1f} pp, so neither the {QUOTED[0] * 100:.2f}% "
            f"nor the {QUOTED[1] * 100:.2f}% the doc quotes is reachable here",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
