"""Exercise 3 — there is no audio to encode.

    **Hard.** Apply SilentCipher watermark to 20 clones, run them through 128
    kbps MP3 encode+decode, detect the payload. Report bit-accuracy.

Reading of the exercise: `silentcipher`, `audioseal`, `torch`, `lameenc`, `pydub`
and `soundfile` are all absent -- but that is the smaller obstacle. **A "clone"
in this module is a 64-element vector.** `fake_tts` mixes two 64-dimensional
embeddings; the result has no sample rate, no duration, and is never converted to
samples anywhere in the file. An MP3 encoder has nothing to consume. The
exercise's procedure -- embed, encode, decode, detect -- is undefined against the
object it is handed, and the 32-bit payload is being written into 64 numbers, one
bit per two "samples".

What the module does ship is measurable, and it measures 100% for a structural
reason. `detect_watermark(wave_original, wave_wm)` **takes the original as an
argument** and subtracts it. Real watermark detection is blind -- the original is
exactly what a distributor does not have -- so this is a difference, not a
detection, and it recovers the payload from an arbitrarily small perturbation:

| strength | bit accuracy over 20 clones |
|---:|---:|
| 0.003 (shipped) | 1.0000 |
| 1e-06 | 1.0000 |
| **1e-15** | **1.0000** |

Once the difference is taken away and real degradation is applied, the scheme's
actual margin appears. 16-bit PCM is the format the payload would ship in, and
its step is `1/32767` = 3.05e-05:

| degradation | bit accuracy |
|---|---:|
| 16-bit PCM, strength 0.003 | 1.0000 |
| 16-bit PCM, strength 1e-05 | 0.9406 |
| 16-bit PCM, strength 1e-06 | 0.5766 |
| additive noise at 40 dB SNR | 0.9969 |
| additive noise at 20 dB SNR | 0.5875 |
| **mean removed from each residue class** | **0.5203** |

That last row is the answer to the MP3 question, as closely as it can be asked
here. The payload is a **constant DC offset** applied to every index with the
same residue mod 32. Removing the mean of each residue class -- the minimum any
transform codec does to a constant -- takes the accuracy to **0.5203**, chance.
A per-sample DC shift is not a signal a perceptual codec preserves, so the scheme
could not survive 128 kbps MP3 even if there were audio to encode.

Structure: `clones` builds the 20 waves; `accuracy` embeds and detects one
payload; `quantise` is 16-bit PCM; `noisy` adds Gaussian noise at a target SNR;
`declass` removes the mean of each residue class.
"""

from __future__ import annotations

import importlib.util
import math
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "08-voice-cloning-conversion"
DIM, BITS, STRENGTH, CLONES = 64, 32, 0.003, 20
PAYLOAD = [int(b) for b in bin(0xDEADBEEF)[2:].zfill(BITS)]
STEP16 = 1 / 32767
ABSENT = ("silentcipher", "audioseal", "torch", "lameenc", "pydub", "soundfile")


def clones(ref):
    return [ref.fake_tts(ref.content_vector(f"clone {i} text", DIM),
                         ref.speaker_vector(f"s{i}", DIM), 0.5) for i in range(CLONES)]


def quantise(wave, step=STEP16):
    return [round(x / step) * step for x in wave]


def noisy(wave, snr_db, rnd):
    power = statistics.mean(x * x for x in wave)
    sigma = math.sqrt(power / 10 ** (snr_db / 10))
    return [x + rnd.gauss(0, sigma) for x in wave]


def declass(wave, bits=BITS):
    """Remove the mean of each residue class -- what a transform codec does to a constant."""
    out = list(wave)
    for start in range(bits):
        index = list(range(start, len(wave), bits))
        centre = statistics.mean(wave[i] for i in index)
        for i in index:
            out[i] = wave[i] - centre
    return out


def accuracy(ref, waves, strength, damage=None):
    """Mean bit accuracy over every clone, optionally after a degradation."""
    scores = []
    for wave in waves:
        marked = ref.watermark(wave, PAYLOAD, strength)
        scores.append(ref.bit_accuracy(PAYLOAD, ref.detect_watermark(
            wave, damage(marked) if damage else marked, BITS)))
    return statistics.mean(scores)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    waves = clones(ref)
    rnd = random.Random(5)
    return {
        "absent": [m for m in ABSENT if importlib.util.find_spec(m) is None],
        "length": len(waves[0]), "clones": len(waves),
        "clean": {s: accuracy(ref, waves, s) for s in (STRENGTH, 1e-6, 1e-15)},
        "pcm": {s: accuracy(ref, waves, s, quantise) for s in (STRENGTH, 1e-5, 1e-6)},
        "noise": {db: accuracy(ref, waves, STRENGTH, lambda w, d=db: noisy(w, d, rnd))
                  for db in (40, 20)},
        "declassed": accuracy(ref, waves, STRENGTH, declass),
        "step": STEP16,
    }


def verify(result):
    clean, pcm, noise = result["clean"], result["pcm"], result["noise"]
    return [
        practice.Check(
            "ANSWER: a 'clone' here is a 64-element vector, so there is nothing to encode",
            result["length"] == DIM,
            f"`fake_tts` mixes two {DIM}-dimensional embeddings; the result has no sample rate, "
            f"no duration, and is never converted to samples anywhere in the module, so an MP3 "
            f"encoder has nothing to consume. The {BITS}-bit payload is written into "
            f"{result['length']} numbers -- one bit per two 'samples'",
        ),
        practice.Check(
            "MECHANISM: the detector is given the original, so 100% is structural",
            min(clean.values()) == 1.0,
            f"`detect_watermark(wave_original, wave_wm)` subtracts the original, which is "
            f"exactly what a distributor does not have. It recovers the payload at strength "
            f"{STRENGTH} and equally at 1e-15: "
            f"{[round(v, 4) for v in clean.values()]} over {result['clones']} clones. That is a "
            "difference, not a detection",
        ),
        practice.Check(
            "FINDING: against 16-bit PCM the real margin is the quantisation step",
            pcm[STRENGTH] == 1.0 > pcm[1e-5] > pcm[1e-6],
            f"the 16-bit step is 1/32767 = {result['step']:.2e}: accuracy is "
            f"{pcm[STRENGTH]:.4f} at strength {STRENGTH}, {pcm[1e-5]:.4f} at 1e-05 and "
            f"{pcm[1e-6]:.4f} at 1e-06. Additive noise gives {noise[40]:.4f} at 40 dB SNR and "
            f"{noise[20]:.4f} at 20 dB",
        ),
        practice.Check(
            "ANSWER: the payload is a DC offset, so no transform codec could carry it",
            0.4 < result["declassed"] < 0.6,
            f"`watermark` adds a constant +/-{STRENGTH} to every index sharing a residue mod "
            f"{BITS}. Removing the mean of each residue class -- the minimum any codec does to "
            f"a constant -- gives {result['declassed']:.4f}, chance. 128 kbps MP3 would not "
            "preserve it even if there were audio to encode",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
