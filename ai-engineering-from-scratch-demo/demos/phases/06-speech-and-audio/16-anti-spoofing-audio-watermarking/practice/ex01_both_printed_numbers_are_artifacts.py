"""Exercise 1 — both numbers it prints as results are artifacts.

    **Easy.** Run `code/main.py`. Toy detector + toy watermark embed/detect on
    synthetic audio.

Reading of the exercise: running it is one command, so the exercise is read as
"run it and check what it printed". It prints two measurements -- an EER and a
bit accuracy -- and labels each with the production number it should be compared
against. Neither is a measurement of what its label says.

**`EER ~ 100.00%` is not a possible equal error rate.** A coin scores 50%, and
any detector above that can be improved by flipping its sign, so 100% means
perfect separation read backwards. It is: the 20 real clips score
**0.190020-0.197743** and the 20 fakes **0.210054-0.211017**, disjoint with a gap
of **0.012311**, and the correct polarity gives **EER 0.0000**. The sweep counts
`far` as the fakes scoring *above* the threshold, which for a score that rises
with fakeness is the detection rate; the printed figure is `1 - EER`. It sits one
line above "real AASIST on ASVspoof 2019 LA: 0.42% EER".

**The watermark detector never sees the payload.** `toy_watermark_detect` returns
`1 if audio[idx] > 0 else 0` -- the sign of the carrier at 16 fixed indices. Run
on the **unwatermarked** clip it returns the same 16 bits, byte for byte, and
inverting all 16 payload bits changes nothing. The embedded step is `0.0005`
against a carrier running to **0.2335** at those indices, **467x** too small, and
**0 of 16** probes sit close enough to zero for it to flip a sign. Over 500
random payloads the mean accuracy is **0.5000** exactly; the printed 31.25% is
5 of 16, one draw of a coin.

Two smaller things. At 20 real and 20 fake, false accepts and false rejects both
move in steps of `1/20`, so the reachable EERs are multiples of **2.5 pp** and
the 0.42% quoted beside it is not on the grid. And the doc's Build It Step 1
defines `spectral_rolloff` and `is_suspicious`; `code/main.py` contains neither,
scoring a high-band energy ratio instead.

Structure: `scores` runs the lesson's detector over one clip family;
`eer_at` sweeps the threshold under a chosen polarity; `readout` collects what
the watermark detector returns for a payload; `probe_carrier` reports the signal
the detector is actually reading.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "06-speech-and-audio", "16-anti-spoofing-audio-watermarking"
CLIPS, N_BITS, STRENGTH, SAMPLES = 20, 16, 0.0005, 16000
PAYLOAD = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]
DRAWS, QUOTED_EER = 500, 0.0042
DOC_STEP1 = ("spectral_rolloff", "is_suspicious")


def scores(ref, maker, seeds):
    return [ref.toy_detector_score(maker(seed=seed)) for seed in seeds]


def eer_at(real, fake, fake_below):
    """Equal error rate under one polarity; `fake_below` is the correct reading."""
    best = (1.0, 0.0)
    for threshold in sorted(set(real + fake)):
        far = sum(1 for s in fake if (s < threshold) == fake_below) / len(fake)
        frr = sum(1 for s in real if (s >= threshold) == fake_below) / len(real)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]


def readout(ref, clip, payload):
    return ref.toy_watermark_detect(ref.toy_watermark_embed(clip, payload))


def accuracy(payload, bits):
    return sum(1 for a, b in zip(payload, bits) if a == b) / len(payload)


def probe_carrier(clip, n_bits=N_BITS):
    """The carrier values at the indices the detector reads, unwatermarked."""
    step = max(1, len(clip) // n_bits)
    return [abs(clip[i * step]) for i in range(n_bits)]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    real = scores(ref, ref.synth_real_speech, range(CLIPS))
    fake = scores(ref, ref.synth_fake_speech, range(100, 100 + CLIPS))
    clip = ref.synth_real_speech(n_samples=SAMPLES, seed=42)
    rng = random.Random(0)
    random_payloads = [[rng.randint(0, 1) for _ in range(N_BITS)] for _ in range(DRAWS)]
    carrier = probe_carrier(clip)
    return {
        "real": (min(real), max(real)), "fake": (min(fake), max(fake)),
        "gap": min(fake) - max(real), "clips": CLIPS,
        "printed_eer": eer_at(real, fake, fake_below=False),
        "true_eer": eer_at(real, fake, fake_below=True),
        "shipped": accuracy(PAYLOAD, readout(ref, clip, PAYLOAD)),
        "inverted": readout(ref, clip, [1 - b for b in PAYLOAD]) == readout(ref, clip, PAYLOAD),
        "unmarked": ref.toy_watermark_detect(clip) == readout(ref, clip, PAYLOAD),
        "carrier": max(carrier), "flippable": sum(1 for v in carrier if v < STRENGTH),
        "coin": statistics.mean(accuracy(p, readout(ref, clip, p)) for p in random_payloads),
        "missing": [n for n in DOC_STEP1 if not hasattr(ref, n)],
    }


def verify(result):
    real, fake = result["real"], result["fake"]
    step = 1 / result["clips"] / 2
    return [
        practice.Check(
            "ANSWER: the printed 'EER ~ 100.00%' is 1 - EER, and the true EER is 0.0000",
            result["printed_eer"] == 1.0 and result["true_eer"] == 0.0,
            f"a coin scores 50% and anything above it improves by flipping sign, so 100% means "
            f"perfect separation read backwards -- real {real[0]:.6f}-{real[1]:.6f} against fake "
            f"{fake[0]:.6f}-{fake[1]:.6f}, disjoint with a gap of {result['gap']:.6f}. The sweep "
            "counts `far` as the fakes scoring above the threshold, which is the detection rate",
        ),
        practice.Check(
            "ANSWER: the watermark detector never sees the payload",
            result["unmarked"] and result["inverted"],
            f"`toy_watermark_detect` returns the sign of the carrier at {N_BITS} fixed indices. "
            f"Run on the unwatermarked clip it returns the same bits byte for byte, and "
            f"inverting all {N_BITS} payload bits changes nothing. The printed "
            f"{result['shipped'] * 100:.1f}% is a property of the carrier",
        ),
        practice.Check(
            "MECHANISM: the embedded step is 467x smaller than the signal it must outvote",
            result["carrier"] / STRENGTH > 100 and result["flippable"] == 0,
            f"`toy_watermark_embed` adds +/-{STRENGTH} where the carrier runs to "
            f"{result['carrier']:.4f} at the probe indices -- {result['carrier'] / STRENGTH:.0f}x "
            f"too small -- and {result['flippable']} of {N_BITS} probes sit close enough to zero "
            "for it to change a sign",
        ),
        practice.Check(
            "CONTROL: over 500 random payloads the mean accuracy is 0.5000",
            abs(result["coin"] - 0.5) < 0.02,
            f"{DRAWS} random payloads through the same embed and detect average "
            f"{result['coin']:.4f}. The shipped {result['shipped'] * 100:.2f}% is "
            f"{round(result['shipped'] * N_BITS)} of {N_BITS}, one draw of that coin, and it is "
            "printed beside 'real AudioSeal: > 99% pre-attack'",
        ),
        practice.Check(
            "CONTROL: at 20 and 20 the EER grid cannot express the 0.42% quoted beside it",
            abs(QUOTED_EER / step - round(QUOTED_EER / step)) > 1e-9,
            f"false accepts and false rejects both move in steps of 1/{result['clips']}, so the "
            f"reachable EERs are the multiples of {step * 100:.1f} pp. The doc's Build It Step 1 "
            f"also defines {list(DOC_STEP1)} and `code/main.py` defines neither -- "
            f"{result['missing']} are absent from the module, which scores a high-band energy "
            "ratio instead",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
