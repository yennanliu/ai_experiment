"""Exercise 2 — adding noise improves the bit accuracy.

    **Medium.** Install `audioseal`, embed a 16-bit payload in a TTS output,
    re-decode. Corrupt the audio with noise and measure Bit Recovery Accuracy.

Reading of the exercise: `audioseal`, `silentcipher`, `torch`, `torchaudio` and
`lameenc` are all absent and the reference tree ships no TTS output, so the
corruption sweep is run against the scheme `code/main.py` does ship. It produces
a flat line, and the direction it drifts in is the finding.

**The curve does not fall, it rises toward chance.**

| noise SNR | Bit Recovery Accuracy |
|---:|---:|
| 100 dB (effectively clean) | **0.3125** |
| 40 dB | 0.3219 |
| 20 dB | 0.4000 |
| 10 dB | 0.4297 |
| 0 dB | **0.4594** |

The readout is the sign of the carrier at 16 fixed indices and is independent of
the payload (Exercise 1), so the clean value is a fixed wrong pattern -- 5 of 16 --
and noise walks it toward the 0.5000 a coin would score. Corrupting the audio
makes the reported number better. There is nothing in the signal to degrade.

**What the detector would need.** Reading a sign blindly requires the embedded
step to outvote the carrier at every probe, so it must exceed **0.2335**, the
largest of the 16 carrier magnitudes -- **467x** the shipped `0.0005`, and
**0.79** of the clip's own peak of 0.2937. A watermark that loud is not a
watermark:

| step | BRA, no noise |
|---:|---:|
| 0.0005 (shipped) | 0.3125 |
| 0.05 | 0.6875 |
| **0.2335** | **1.0000** |

**At a step that works, the curve the exercise asks for finally exists**:
**1.0000** from clean down to 20 dB SNR, **0.9891** at 10 dB and **0.9234** at
0 dB.

And even repaired it is the wrong shape. The payload occupies **16 of 16,000
samples, 0.10%** of the clip, at indices fixed by `len(audio) // n_bits`, so
dropping one leading sample takes BRA to **0.5000**. AudioSeal adds a per-sample
field across the whole clip, which is why a crop or a resample leaves it
readable; this is a row of 16 pinpricks whose addresses are the message.

Structure: `embed_at` is the lesson's embed with the step as a parameter;
`corrupt` adds Gaussian noise at a target SNR; `bra` scores one recovery;
`curve` averages BRA over repeated draws at each SNR.
"""

from __future__ import annotations

import importlib.util
import math
import random

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "16-anti-spoofing-audio-watermarking"
SAMPLES, N_BITS, SHIPPED, SEED = 16000, 16, 0.0005, 42
PAYLOAD = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]
SNRS, TRIALS, WORKING = (100, 40, 20, 10, 0), 40, 0.30
STEPS = (0.0005, 0.05, 0.2335)
ABSENT = ("audioseal", "silentcipher", "torch", "torchaudio", "lameenc")


def embed_at(clip, payload, strength):
    """The lesson's `toy_watermark_embed` with its hard-coded step exposed."""
    out = list(clip)
    stride = max(1, len(clip) // len(payload))
    for index, bit in enumerate(payload):
        out[index * stride] += strength if bit else -strength
    return out


def corrupt(clip, snr_db, rng):
    power = sum(v * v for v in clip) / len(clip)
    sigma = math.sqrt(power / 10 ** (snr_db / 10))
    return [v + rng.gauss(0, sigma) for v in clip]


def bra(ref, payload, clip):
    bits = ref.toy_watermark_detect(clip, len(payload))
    return sum(1 for a, b in zip(payload, bits) if a == b) / len(payload)


def curve(ref, clip, strength, snrs=SNRS, trials=TRIALS):
    """Mean BRA at each SNR, averaged over repeated noise draws."""
    marked = embed_at(clip, PAYLOAD, strength)
    rng = random.Random(3)
    return {snr: sum(bra(ref, PAYLOAD, corrupt(marked, snr, rng))
                     for _ in range(trials)) / trials for snr in snrs}


def carrier_peak(clip, n_bits=N_BITS):
    stride = max(1, len(clip) // n_bits)
    return max(abs(clip[i * stride]) for i in range(n_bits))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    clip = ref.synth_real_speech(n_samples=SAMPLES, seed=SEED)
    needed = carrier_peak(clip)
    working = embed_at(clip, PAYLOAD, WORKING)
    return {
        "absent": [m for m in ABSENT if importlib.util.find_spec(m) is None],
        "shipped_curve": curve(ref, clip, SHIPPED),
        "repaired_curve": curve(ref, clip, WORKING),
        "steps": {s: bra(ref, PAYLOAD, embed_at(clip, PAYLOAD, s)) for s in STEPS},
        "needed": needed, "ratio": needed / SHIPPED,
        "peak": max(abs(v) for v in clip),
        "touched": N_BITS, "length": len(clip),
        "shifted": bra(ref, PAYLOAD, working[1:]),
    }


def verify(result):
    shipped, repaired = result["shipped_curve"], result["repaired_curve"]
    steps = result["steps"]
    clean, loud = shipped[SNRS[0]], shipped[SNRS[-1]]
    return [
        practice.Check(
            "CONTROL: no watermarking library and no TTS output to embed in",
            len(result["absent"]) == len(ABSENT),
            f"find_spec is None for {result['absent']} and the reference tree ships no audio "
            "file, so the sweep runs against the scheme `code/main.py` does ship -- which is "
            "what makes the shape of the curve readable at all",
        ),
        practice.Check(
            "ANSWER: the corruption curve rises toward chance instead of falling",
            loud > clean and clean < 0.4 < loud,
            f"BRA is {clean:.4f} at {SNRS[0]} dB SNR and {loud:.4f} at {SNRS[-1]} dB. The "
            f"readout is independent of the payload, so the clean value is a fixed wrong "
            f"pattern and noise walks it toward the 0.5000 a coin scores: "
            f"{ {s: round(v, 4) for s, v in shipped.items()} }",
        ),
        practice.Check(
            "MECHANISM: the step has to outvote the carrier, and 0.0005 does not",
            steps[STEPS[0]] < steps[STEPS[1]] < steps[STEPS[2]] == 1.0,
            f"a blind sign read needs the step above {result['needed']:.4f}, the largest of the "
            f"{N_BITS} carrier magnitudes -- {result['ratio']:.0f}x the shipped {SHIPPED} and "
            f"{result['needed'] / result['peak']:.2f} of the clip's own peak "
            f"{result['peak']:.4f}. BRA by step: { {s: round(v, 4) for s, v in steps.items()} }",
        ),
        practice.Check(
            "ANSWER: at a step that works, the curve the exercise asks for exists",
            repaired[SNRS[0]] == 1.0 and 0.8 < repaired[SNRS[-1]] < 1.0,
            f"at step {WORKING} BRA holds at {repaired[SNRS[2]]:.4f} down to {SNRS[2]} dB SNR, "
            f"{repaired[SNRS[3]]:.4f} at {SNRS[3]} dB and {repaired[SNRS[-1]]:.4f} at "
            f"{SNRS[-1]} dB. That is a robustness measurement; the shipped one is not",
        ),
        practice.Check(
            "FINDING: even repaired it is 16 pinpricks whose addresses are the message",
            abs(result["shifted"] - 0.5) < 0.2,
            f"the payload occupies {result['touched']} of {result['length']} samples "
            f"({result['touched'] / result['length'] * 100:.2f}%) at indices fixed by "
            f"`len(audio) // n_bits`, so dropping one leading sample gives "
            f"{result['shifted']:.4f}. AudioSeal adds a per-sample field over the whole clip, "
            "which is why a crop or a resample leaves it readable",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
