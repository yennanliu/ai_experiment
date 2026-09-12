"""Exercise 1 — half the matrix has no stripes to get wider.

    **Easy.** Plot the sinusoidal `PE` matrix as a heatmap for `max_len=512,
    d=128`. Confirm the "stripes get wider as dimension index grows" pattern.

Reading of the exercise: "confirm" is read as a measurement rather than a glance,
so the stripes are counted -- sign changes per column of the lesson's own
`sinusoidal_pe(512, 128)` -- and the wavelengths they imply are compared against
`2*pi*base^(2i/d)` in closed form. `matplotlib` is absent, so `imshow` becomes a
text raster of the same matrix.

**ANSWER: confirmed, and the range is a factor of 8,664.** Band 0 has a
wavelength of 6.28 positions and band 63 has 54,410. Sign changes per column fall
monotonically from 163 to 0.

**FINDING: 33 of the 64 bands never complete a cycle.** A stripe needs a
wavelength of at most `max_len`, and `2*pi*base^(2i/d) <= 512` holds only up to
band 30. **Half the heatmap has no stripes at all** -- the top half is a smooth
ramp, and the "pattern" the exercise asks to confirm exists in the bottom half
only. At d=128 with max_len=512 the last band traverses **0.94%** of one cycle
across the entire matrix, so its two columns are very nearly constant.

**FINDING: every row has exactly the same norm.** `sin^2 + cos^2 = 1` per band,
so `|PE[pos]|` is `sqrt(d/2) = 8.000000000000` for all 512 positions, to the last
printed digit. A heatmap of this matrix redistributes brightness; it never adds
any. That is the invariant a picture cannot show.

**FINDING: the sinusoidal dot product is already purely relative.**
`PE[p] . PE[p+g]` depends only on `g`, to **3e-14** across base positions 0, 17,
100 and 300. The lesson's closing line says "RoPE encodes relative position in
the dot product itself" as if that distinguished it. It does not -- what
distinguishes RoPE is that it survives `Wq` and `Wk`, because sinusoidal PE is
*added to x* before those projections and the relative structure does not.

Structure: `stripes` counts sign changes; `wavelength` is the closed form;
`raster` is the requested heatmap, as text.
"""

from __future__ import annotations

import importlib.util
import math

from harness import parity, practice

PHASE, LESSON = "07-transformers-deep-dive", "04-positional-encoding"
MAX_LEN, DIM, BASE = 512, 128, 10000.0
SHADES, ROWS, COLS = " .:-=+*#%@", 16, 96


def wavelength(band, base=BASE, dim=DIM):
    """Positions per cycle for frequency band `i`: 2*pi*base^(2i/d)."""
    return 2 * math.pi * base ** (2 * band / dim)


def stripes(pe, column):
    """Sign changes down one column -- how many stripe edges the heatmap shows."""
    signs = [1 if pe[pos][column] >= 0 else -1 for pos in range(len(pe))]
    return sum(1 for a, b in zip(signs, signs[1:]) if a != b)


def spread(pe, gap, bases=(0, 17, 100, 300)):
    """How much PE[p].PE[p+gap] moves when only the base position p changes."""
    dots = [sum(a * b for a, b in zip(pe[p], pe[p + gap])) for p in bases]
    return max(dots) - min(dots), dots[0]


def raster(pe, rows=ROWS, cols=COLS):
    """`imshow`, as text: one row per sampled dimension, position running right."""
    step = DIM // rows
    lines = []
    for row in range(rows):
        column = row * step
        cells = [pe[p * MAX_LEN // cols][column] for p in range(cols)]
        lines.append(f"{column:>4} " + "".join(
            SHADES[min(len(SHADES) - 1, int((v + 1) / 2 * len(SHADES)))] for v in cells))
    return "\n".join(lines)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    pe = ref.sinusoidal_pe(MAX_LEN, DIM)
    bands = DIM // 2
    edges = {column: stripes(pe, column) for column in range(0, DIM, 2)}
    norms = [math.sqrt(sum(v * v for v in row)) for row in pe]
    return {
        "edges": edges, "rows": len(pe), "cols": len(pe[0]),
        "first": wavelength(0), "last": wavelength(bands - 1), "bands": bands,
        "striped": sum(1 for i in range(bands) if wavelength(i) <= MAX_LEN),
        "covered": MAX_LEN / wavelength(bands - 1),
        "norms": (min(norms), max(norms)), "target": math.sqrt(DIM / 2),
        "relative": {gap: spread(pe, gap) for gap in (1, 5, 50)},
        "raster": raster(pe), "plotting": importlib.util.find_spec("matplotlib") is None,
    }


def verify(result):
    edges, lo, hi = result["edges"], *result["norms"]
    counts = [edges[c] for c in sorted(edges)]
    worst = max(s for s, _ in result["relative"].values())
    return [
        practice.Check(
            "ANSWER: confirmed -- stripe edges fall from 162 to 0 across an 8,660x range",
            counts[0] > 150 and counts[-1] == 0 and all(a >= b for a, b in zip(counts, counts[1:])),
            f"the lesson's sinusoidal_pe({MAX_LEN}, {DIM}) gives {result['rows']}x{result['cols']}; "
            f"sign changes per sine column fall monotonically {counts[0]} -> {counts[1]} -> ... -> "
            f"{counts[-1]}, matching wavelengths of {result['first']:.2f} positions at band 0 and "
            f"{result['last']:.0f} at band {result['bands'] - 1}, a factor of "
            f"{result['last'] / result['first']:.0f}",
        ),
        practice.Check(
            "FINDING: 33 of the 64 bands never complete a cycle, so half has no stripes",
            result["striped"] == 31,
            f"a stripe needs 2*pi*base^(2i/d) <= {MAX_LEN}, true only up to band "
            f"{result['striped'] - 1}: {result['striped']} of {result['bands']} bands stripe and "
            f"{result['bands'] - result['striped']} do not. The last band traverses "
            f"{result['covered']:.2%} of one cycle across the whole matrix, so its two columns are "
            "nearly constant and the top half of the picture is a smooth ramp",
        ),
        practice.Check(
            "FINDING: every row has exactly the same norm, sqrt(d/2)",
            lo == hi == result["target"],
            f"sin^2 + cos^2 = 1 per band, so |PE[pos]| is {lo:.12f} = sqrt({DIM}/2) for all "
            f"{MAX_LEN} positions -- min and max agree to the last printed digit. The heatmap "
            "redistributes brightness across a row; it never adds any, which is the one thing a "
            "picture of this matrix cannot show",
        ),
        practice.Check(
            "FINDING: the sinusoidal dot product is already purely relative",
            worst < 1e-12,
            f"PE[p] . PE[p+g] moves by at most {worst:.1e} as p ranges over 0, 17, 100 and 300, at "
            f"gaps {sorted(result['relative'])}. The lesson's closing line credits RoPE with "
            "encoding relative position in the dot product; sinusoidal PE already does. What RoPE "
            "has is survival of Wq and Wk -- sinusoidal PE is added to x before those projections",
        ),
        practice.Check(
            "CONTROL: matplotlib is absent, so imshow is a text raster of the same matrix",
            result["plotting"] and len(result["raster"].splitlines()) == ROWS,
            f"find_spec('matplotlib') is None, so the picture is {ROWS} sampled dimensions "
            f"down and {COLS} sampled positions right: the top rows "
            f"alternate every few characters and the bottom rows do not alternate at all. Printed "
            "in the lesson README, where the widening is visible as well as counted",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
