"""Exercise 1 — token count memory wall.

    **(Easy)** Compute the token count for a 5-second 360p video at patch-t=2,
    patch-h=8, patch-w=8. Reason about memory for attention at this size.

Reading of the exercise: "5-second 360p" does not fix the numbers it is asked
for. It names no frame rate, and 360p is a height rather than a resolution --
the lesson's own `main()` silently answers for 480x360 at 30 fps, where the
standard 16:9 360p is 640x360. Both readings are computed here, because the
gap between them is the point: token count is linear in the pixel count but
attention memory is quadratic, so a 33% wider frame costs 78% more. "Reason
about memory" is then made concrete rather than argued -- one attention matrix,
one head, one layer, in bytes -- and checked against the divided attention the
lesson actually builds.

Structure: `pairs` and `gigabytes` convert a token count to attention entries
and to float32 bytes; `divided` is the lesson's own comment arithmetic,
S*T^2 + T*S^2; `grids` returns the token grid `TinyVideoDiT` really produces
for a clip, which is not the one `count_tokens` describes.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "28-world-models-video-diffusion"

FRAMES, FPS, SECONDS = 150, 30, 5
LESSON_WH, STANDARD_WH = (480, 360), (640, 360)      # main()'s reading, and 16:9 360p
PT, PH, PW = 2, 8, 8
CLIP = (1, 4, 8, 16, 16)

pairs = lambda tokens: tokens ** 2                                            # noqa: E731
gigabytes = lambda tokens: pairs(tokens) * 4 / 1e9                            # noqa: E731
divided = lambda time, space: space * time ** 2 + time * space ** 2           # noqa: E731
grid_of = lambda shape, patch: [side // step for side, step in zip(shape, patch)]   # noqa: E731


def count(ref, width, height) -> tuple:
    """Tokens, and the temporal and spatial factors the lesson's comments split them into."""
    tokens = ref.count_tokens(FRAMES, width, height, p_t=PT, p_h=PH, p_w=PW)
    return tokens, FRAMES // PT, (width // PH) * (height // PW)


def solve():
    try:
        import torch
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    lesson = count(ref, *LESSON_WH)
    standard = count(ref, *STANDARD_WH)
    half = count(ref, LESSON_WH[0] // 2, LESSON_WH[1] // 2)
    model = ref.TinyVideoDiT(in_channels=4, dim=64, depth=2, heads=2)
    with torch.no_grad():
        _out, grid = model(torch.randn(*CLIP))
    patch = model.patch.proj
    return {"lesson": lesson, "standard": standard, "half": half,
            "divided": divided(lesson[1], lesson[2]),
            "patch": tuple(patch.kernel_size), "stride": tuple(patch.stride),
            "grid": tuple(grid), "clip_tokens": int(grid[0] * grid[1] * grid[2]),
            "expected_grid": grid_of(CLIP[2:], tuple(patch.kernel_size)),
            "at_model_patch": ref.count_tokens(FRAMES, *LESSON_WH, p_t=2, p_h=2, p_w=2)}


def verify(result):
    tokens, time_tok, space_tok = result["lesson"]
    standard, half = result["standard"][0], result["half"][0]
    return [
        practice.Check(
            "ANSWER: 202,500 tokens, and one attention matrix is 164 GB",
            tokens == 202_500 and 160 < gigabytes(tokens) < 170,
            f"{FRAMES} frames at patch ({PT}, {PH}, {PW}) over {LESSON_WH[0]}x{LESSON_WH[1]} gives "
            f"{time_tok} temporal x {space_tok:,} spatial = {tokens:,} tokens. Joint attention over "
            f"them is {pairs(tokens):,} pairs, which at float32 is {gigabytes(tokens):.1f} GB for a "
            "single matrix in a single head of a single layer -- before any weights or activations"),
        practice.Check(
            "FINDING: '5-second 360p' does not fix the answer, and the gap is quadratic",
            standard > tokens and gigabytes(standard) / gigabytes(tokens) > 1.7,
            f"360p is a height, not a resolution, and the exercise names no frame rate. main() "
            f"answers for {LESSON_WH[0]}x{LESSON_WH[1]} at {FPS} fps; the standard 16:9 reading "
            f"{STANDARD_WH[0]}x{STANDARD_WH[1]} gives {standard:,} tokens, "
            f"{standard / tokens - 1:.0%} more -- but {gigabytes(standard):.1f} GB, "
            f"{gigabytes(standard) / gigabytes(tokens) - 1:.0%} more, because memory is quadratic "
            "where the token count is linear"),
        practice.Check(
            "MECHANISM: dividing the attention is what makes the number payable",
            pairs(tokens) / result["divided"] > 70,
            f"attending over time at every spatial site and over space at every frame costs "
            f"{space_tok:,}*{time_tok}^2 + {time_tok}*{space_tok:,}^2 = {result['divided']:,} pairs "
            f"against joint attention's {pairs(tokens):,} -- "
            f"{pairs(tokens) / result['divided']:.0f}x fewer, or "
            f"{result['divided'] * 4 / 1e9:.1f} GB against {gigabytes(tokens):.1f} GB"),
        practice.Check(
            "CONTROL: the model in the same file does not use these patch sizes",
            result["patch"] == (2, 2, 2) != (PT, PH, PW)
            and result["at_model_patch"] > 15 * tokens,
            f"count_tokens defaults to ({PT}, {PH}, {PW}), but TinyVideoDiT hardcodes "
            f"{result['patch']} in its own constructor. The same clip through the model's patching "
            f"is {result['at_model_patch']:,} tokens, {result['at_model_patch'] // tokens}x the "
            "number the lesson prints: the token arithmetic and the network in one file describe "
            "two different systems"),
        practice.Check(
            "CONTROL: the patching is exactly a strided conv, so the grid is division",
            list(result["grid"]) == result["expected_grid"] and result["clip_tokens"] == 256,
            f"VideoPatch3D is a Conv3d with kernel {result['patch']} and stride {result['stride']}, "
            f"so a {CLIP[2:]} clip yields grid {result['grid']} = {result['clip_tokens']} tokens, "
            f"exactly {CLIP[2:]} divided elementwise by the patch. Nothing overlaps and nothing is "
            "padded, which is why the count is a product of three integer divisions"),
        practice.Check(
            "CONTROL: halving each side divides memory by 17, not 16 -- the patch grid truncates",
            16 < gigabytes(tokens) / gigabytes(half) < 18 and half < tokens // 4,
            f"at {LESSON_WH[0] // 2}x{LESSON_WH[1] // 2} the clip is {half:,} tokens against an exact "
            f"quarter of {tokens // 4:,}, because {LESSON_WH[1] // 2} // {PH} truncates 22.5 to 22 and "
            f"loses half a patch row. Memory falls {gigabytes(tokens) / gigabytes(half):.1f}x rather "
            f"than 16x ({gigabytes(half):.2f} GB against {gigabytes(tokens):.1f}): resolution is a "
            "fourth-power decision about attention memory, and the flooring is not neutral"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
