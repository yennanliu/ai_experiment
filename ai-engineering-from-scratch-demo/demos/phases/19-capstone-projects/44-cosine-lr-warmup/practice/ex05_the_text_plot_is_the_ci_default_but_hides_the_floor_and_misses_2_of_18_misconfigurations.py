"""Exercise 5 -- the text plot is the better CI default, but it hides the floor and misses 2 of 18 misconfigurations.

    Add a `--plot-png` flag that writes a real plot via `matplotlib`. Defend
    whether the lesson's text plot or the PNG is the better default for CI
    runs.

Reading of the exercise: matplotlib is not a dependency of this repo, and
it is not importable here. The repo's earlier lessons faced the same gap
(03/13 ex03, 07/04 ex01). So `--plot-png PATH` writes a real PNG with a
stdlib encoder instead: 8-bit grayscale, one pixel column per step, 100
rows, zlib and CRC32 from the standard library. `main()` always returns the
lesson's text plot, at the demo's 40x10, and writes the PNG only when the
flag is given, into a temporary directory. Both plots are then scored as a
CI check would use them: rendered twice to test determinism, re-read to
test that the PNG is valid, and diffed against six misconfigured schedules
(warmup +1, warmup -1, total +1, lr_min 0, warmup doubled, lr_min x10) on
three runs: the 20-step demo, 200 steps and 2,000 steps.

**ANSWER: the text plot is the better default for CI, and the PNG belongs
behind the flag.** The text plot needs no dependency and lands in the log,
where a reviewer reads it without downloading an artifact. It is 1,007
bytes for a 200-step run and byte-identical across renders. The PNG is
valid (signature, IHDR 21x100 for the demo, every CRC checks, 21 ink
pixels), also byte-stable, and only 481 bytes at 200 steps (the exact size depends on the zlib
build, so the check asserts only that it is smaller than the text plot). But a PNG is
opaque in a log and in a diff, and producing one through matplotlib would
add a plotting stack to every CI image.

**FINDING: the text plot hides the floor and misses 2 of 18
misconfigurations.** Its y axis always bottoms out at 0.000000, so lr_min
never appears on it. At the demo's 40x10, lr_min = 0 renders identically
to lr_min = 1e-4. At 2,000 steps the default 60 columns hold 34 steps
each, so warmup 19 renders identically to warmup 20. That is the
off-by-one class; the lesson says the plot "catches the
misconfigured-schedule class of bugs at PR time". The PNG and
the lesson's own `write_schedule_csv` differ on all 18.

**FINDING: for a check, the CSV is the artifact, and the plot is for eyes.**
Diffing `write_schedule_csv` output catches every misconfiguration exactly,
with no rendering step. A CI gate should diff that file; the plot is the
human sanity check the doc describes.

Structure: `png_bytes()` is the encoder; `main()` is the flag; `decode()`
re-reads the PNG; `misses()` diffs one artifact against the six bugs on the
three runs.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import struct
import tempfile
import zlib

from harness import parity, practice

try:
    import torch  # noqa: F401  (main.py exits without it)
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "44-cosine-lr-warmup"
REF = parity.load_reference(PHASE, LESSON, "main")
SIG, HEIGHT = b"\x89PNG\r\n\x1a\n", 100
BUGS = {"warmup+1": (1, 0, 1), "warmup-1": (-1, 0, 1), "total+1": (0, 1, 1),
        "lr_min=0": (0, 0, 0), "warmup x2": (None, 0, 1), "lr_min x10": (0, 0, 10)}
RUNS = {"demo": (4, 20, (40, 10)), "200": (20, 200, (60, 12)), "2000": (20, 2000, (60, 12))}


def chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def png_bytes(schedule, height=HEIGHT):
    """One pixel column per step, 8-bit grayscale, black curve on white."""
    rates = [schedule.lr(k) for k in range(schedule.total_steps + 1)]
    rows = [bytearray(b"\xff" * len(rates)) for _ in range(height)]
    for col, rate in enumerate(rates):
        rows[round((height - 1) * (1 - rate / max(rates)))][col] = 0
    raw = b"".join(b"\x00" + bytes(r) for r in rows)
    head = struct.pack(">IIBBBBB", len(rates), height, 8, 0, 0, 0, 0)
    return SIG + chunk(b"IHDR", head) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def main(argv):
    (p := argparse.ArgumentParser()).add_argument("--plot-png", type=pathlib.Path)
    a, schedule = p.parse_args(argv), REF.CosineWithWarmup(4, 20, 1e-2, 1e-4)
    if a.plot_png:
        a.plot_png.write_bytes(png_bytes(schedule))
    return REF.plot_schedule_ascii(schedule, width=40, height=10)


def decode(data):
    """Re-read the PNG: every chunk's CRC, the dimensions, and the curve's pixel count."""
    pos, chunks, crc_ok = 8, {}, True
    while pos < len(data):
        (n,) = struct.unpack(">I", data[pos:pos + 4])
        kind, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + n]
        crc_ok &= struct.unpack(">I", data[pos + 8 + n:pos + 12 + n])[0] == zlib.crc32(kind + body)
        chunks[kind], pos = body, pos + 12 + n
    size = struct.unpack(">II", chunks[b"IHDR"][:8])
    return data[:8] == SIG, crc_ok, size, zlib.decompress(chunks[b"IDAT"]).count(0) - size[1]


def csv_text(schedule):
    with tempfile.TemporaryDirectory() as tmp:
        REF.write_schedule_csv(schedule, path := pathlib.Path(tmp) / "s.csv")
        return path.read_text()


def misses(render):
    """(run, bug) pairs whose schedule renders identically to the good one."""
    out = []
    for run, (w, total, size) in RUNS.items():
        good = REF.CosineWithWarmup(w, total, 1e-2, 1e-4)
        for bug, (dw, dt, k) in BUGS.items():
            bad = REF.CosineWithWarmup(2 * w if dw is None else w + dw, total + dt, 1e-2, 1e-4 * k)
            out += [(run, bug)] if render(good, size) == render(bad, size) else []
    return out


def solve():
    with tempfile.TemporaryDirectory() as tmp:
        out = pathlib.Path(tmp) / "lr.png"
        plot, data = main(["--plot-png", str(out)]), out.read_bytes()
        stable = (main([]), main(["--plot-png", str(out)]), out.read_bytes()) == (plot, plot, data)
    long = REF.CosineWithWarmup(20, 200, 1e-2, 1e-4)
    return {
        "matplotlib": importlib.util.find_spec("matplotlib") is not None, "png": decode(data), "stable": stable,
        "doc_claim": "catches the misconfigured-schedule class of bugs at PR time" in parity.doc_text(PHASE, LESSON),
        "bytes": (len(REF.plot_schedule_ascii(long).encode()), len(png_bytes(long))),
        "bottom_label": plot.splitlines()[9].split("|")[0].strip(),
        "text": misses(lambda s, size: REF.plot_schedule_ascii(s, *size)),
        "png_miss": misses(lambda s, size: png_bytes(s)), "csv_miss": misses(lambda s, size: csv_text(s)),
    }


def verify(r):
    (text_b, png_b), png = r["bytes"], r["png"]
    return [
        practice.Check(
            "ANSWER: the text plot is the better CI default; the PNG belongs behind the flag",
            (r["stable"], text_b, png_b < text_b, png) == (True, 1007, True, (True, True, (21, 100), 21)),
            f"text plot {text_b} B at 200 steps, PNG {png_b} B; demo PNG (signature, CRCs, size, ink) "
            f"{png}; stable {r['stable']}",
        ),
        practice.Check(
            "FINDING: the text plot hides the floor and misses 2 of 18 misconfigurations",
            (r["bottom_label"], r["text"], r["matplotlib"], r["doc_claim"])
            == ("0.000000", [("demo", "lr_min=0"), ("2000", "warmup-1")], False, True),
            f"bottom label {r['bottom_label']}; renders identically: {r['text']}; matplotlib {r['matplotlib']}",
        ),
        practice.Check(
            "FINDING: for a check, the CSV is the artifact, and the plot is for eyes",
            (r["png_miss"], r["csv_miss"]) == ([], []),
            f"misconfigurations missed -- PNG {r['png_miss']}, CSV {r['csv_miss']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
