"""Exercise 1 — three bands, one schedule, so the axes are interchangeable.

    Compute M-RoPE rotations for a patch at (t=3, h=5, w=7) with hidden 48 (16
    per band, base theta 10000). Show the rotation angles for the first three
    pairs in each band.

Reading of the exercise: the angles come from the lesson's own `mrope_angles`,
and `mrope_rotate` is then used as an independent check on them -- rotating a
unit vector and recovering the angle with `atan2` is the only way to confirm the
two functions agree, since each recomputes theta from scratch. The theta
schedule is then read across all eight pairs, because three of eight is not
enough to see what the band is doing.

**ANSWER: (3.0, 0.948683, 0.3), (5.0, 1.581139, 0.5), (7.0, 2.213594, 0.7)** for
the temporal, height and width bands. Rotating e0 by the temporal band and
reading `atan2` back gives **3.0**, matching the first angle exactly.

**FINDING: all three bands share one theta schedule, so an angle cannot say
which axis it came from.** Every band is 16-dimensional, so theta_i is
`10000^(-i/8)` in all three and each angle is just `position x theta_i`. A patch
three frames later and a patch three rows down produce the **identical** triple
(3.0, 0.948683, 0.3). Only which slice of the hidden state they land in
distinguishes them.

**FINDING: the eight pairs span wavelengths from 6.28 to 19,869 positions.**
Pair 0 has theta = 1.0, so it completes a full turn every 2*pi positions -- a
27-wide patch grid wraps it **4.3** times -- while pair 7 has a wavelength
**3,162x** longer than the whole image. Three pairs is the fast end of a range
whose other end never turns at all.

**FINDING: `MRoPEConfig.hidden` is never read.** Neither `mrope_angles` nor
`mrope_rotate` touches it; the rotation is driven entirely by the three band
dimensions. `hidden=999` with the same three bands produces byte-identical
angles, so a configuration whose bands do not sum to its hidden size is
accepted silently.

Structure: `ANGLES` runs the lesson's own `mrope_angles`, `recovered` rotates a
basis vector and reads the angle back with `atan2`, and `schedule` is the theta
and wavelength table across all eight pairs.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "12-multimodal-ai", "09-qwen-vl-family-dynamic-fps"
HIDDEN, BAND, BASE = 48, 16, 10000.0
POSITION = (3, 5, 7)
SHOWN, GRID = 3, 27


def config(ref, hidden=HIDDEN):
    return ref.MRoPEConfig(hidden=hidden, temporal_dim=BAND, height_dim=BAND,
                           width_dim=BAND, base=BASE)


def schedule(band=BAND, base=BASE):
    """Theta and wavelength for every pair in one band."""
    thetas = [base ** (-2 * i / band) for i in range(band // 2)]
    return thetas, [2 * math.pi / theta for theta in thetas]


def recovered(ref, cfg, axis=0):
    """Rotate e0 by one band and read the angle back -- an independent check."""
    vector = [0.0] * HIDDEN
    vector[axis * BAND] = 1.0
    rotated = ref.mrope_rotate(cfg, vector, *POSITION)
    return math.atan2(rotated[axis * BAND + 1], rotated[axis * BAND])


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    cfg = config(ref)
    angles = ref.mrope_angles(cfg, *POSITION)
    thetas, waves = schedule()
    return {
        "bands": [[round(value, 6) for value in band[:SHOWN]] for band in angles],
        "pairs": len(angles[0]),
        "thetas": [round(theta, 6) for theta in thetas],
        "waves": [round(wave, 2) for wave in waves],
        "wave_range": round(waves[-1] / waves[0], 1),
        "grid_turns": round(GRID / waves[0], 1),
        "temporal_three": [round(v, 6) for v in ref.mrope_angles(cfg, 3, 0, 0)[0][:SHOWN]],
        "height_three": [round(v, 6) for v in ref.mrope_angles(cfg, 0, 3, 0)[1][:SHOWN]],
        "recovered": round(recovered(ref, cfg), 6),
        "hidden_ignored": ref.mrope_angles(config(ref, 999), *POSITION) == angles,
    }


def verify(result):
    bands = result["bands"]
    return [
        practice.Check(
            "ANSWER: (3.0, 0.948683, 0.3), (5.0, 1.581139, 0.5), (7.0, 2.213594, 0.7)",
            all([bands == [[3.0, 0.948683, 0.3], [5.0, 1.581139, 0.5],
                           [7.0, 2.213594, 0.7]],
                 result["recovered"] == float(POSITION[0])]),
            f"the first {SHOWN} pairs of the temporal, height and width bands at "
            f"(t, h, w) = {POSITION} are {bands}. Rotating e0 by the temporal band and "
            f"reading atan2 back gives {result['recovered']}, matching its first angle -- so "
            "mrope_angles and mrope_rotate, which recompute theta independently, agree",
        ),
        practice.Check(
            "FINDING: all three bands share one theta schedule",
            all([result["temporal_three"] == result["height_three"],
                 result["temporal_three"] == [3.0, 0.948683, 0.3],
                 len(set(result["thetas"])) == result["pairs"]]),
            f"every band is {BAND}-dimensional, so theta_i is the same in all three and an "
            f"angle is position x theta_i. A patch three frames later and a patch three rows "
            f"down give the identical triple {result['temporal_three']}; only which slice of "
            "the hidden state they land in tells them apart",
        ),
        practice.Check(
            "FINDING: the eight pairs span wavelengths from 6.28 to 19,869 positions",
            all([result["thetas"][0] == 1.0, result["waves"][0] == 6.28,
                 result["waves"][-1] == 19869.18, result["wave_range"] == 3162.3,
                 result["grid_turns"] == 4.3]),
            f"pair 0 has theta {result['thetas'][0]}, a wavelength of {result['waves'][0]} "
            f"positions -- a {GRID}-wide patch grid wraps it {result['grid_turns']} times -- "
            f"while pair {result['pairs'] - 1} has a wavelength {result['wave_range']}x "
            f"longer, {result['waves'][-1]:,.0f} positions. Three pairs is the fast end of a "
            "range whose other end never turns",
        ),
        practice.Check(
            "FINDING: MRoPEConfig.hidden is never read",
            result["hidden_ignored"],
            f"neither mrope_angles nor mrope_rotate touches the field; the rotation is driven "
            f"by the three band dimensions alone, so hidden=999 with the same "
            f"{BAND}+{BAND}+{BAND} bands produces byte-identical angles. A configuration "
            "whose bands do not sum to its hidden size is accepted silently",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
