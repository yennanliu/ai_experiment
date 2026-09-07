"""Exercise 1 — stage time budget.

    **(Easy)** Run the pipeline on 10 images from any open dataset. Report the
    average time per stage and the distribution of detection counts per image.

Reading of the exercise: the second half cannot be answered the way it is asked.
The lesson's `StubDetector.forward` never looks at its input -- it emits three
boxes at fixed fractions of `H` and `W` for any image -- so the "distribution of
detection counts per image" is a point mass at 3 on all ten images and carries
no information. What does vary is the *classification* count, and only through
`min_crop=16`: below a 64x64 frame the third box is too small to crop, and at
32x32 all three are. That is the distribution reported here, and it is a
property of the geometry rather than of the pictures. "10 images from any open
dataset" is taken as ten crops of the two photographs scikit-learn ships in
`sklearn.datasets.load_sample_images` -- real content, already on disk, nothing
downloaded. Timing then contradicts the lesson's own accounting. `benchmark()`
reports three buckets, and the largest true stage is not one of them: the
crop-and-resize loop is inlined in both `run()` and `benchmark()` with no method
to call, so it is billed to whichever bucket surrounds it, and the lesson puts
it inside `classify`. `crop_stage` below re-inlines those six lines so they can
be timed on their own; that it produces exactly the crops the lesson's `run()`
went on to classify, on all ten images, is checked rather than assumed. Medians
are reported rather than means, because a mean over 200 sub-millisecond samples
is dominated by whichever pass the OS descheduled.

Structure: `sample_images` cuts ten frames out of the two bundled photographs;
`crop_stage` is the lesson's own inlined crop-and-resize loop, isolated so it can
be timed; `walk_frames` sends every frame through `pipe.run()` once for the two
count distributions and through the four stages `REPEATS` times for the timings,
and also sweeps square frames across the `min_crop` floor; `resolution_probe`
scores one crop at seven resolutions to show how little of it the classifier
reads. At 139 code lines this sits above D14's 120-line target and 11 clear of the
ceiling: five checks over three probes -- 200 timed passes, a 6-point `min_crop`
sweep and a 7-point resolution sweep -- and the p95 arithmetic is derived from
the lesson's own expression rather than restated here.
"""

from __future__ import annotations

import collections
import statistics
import time

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "16-vision-pipeline-capstone"

CLASSES, CROP, REPEATS = 10, 64, 20
STAGES = ("preprocess", "detect", "crop", "classify")
SIZES, RESOLUTIONS = (16, 32, 50, 64, 96, 128), (4, 8, 16, 32, 64, 128, 256)
RUNS = (10, 20, 100)                              # num_runs values for benchmark()'s p95 index

row = lambda table: "  ".join(f"{k}={v:.4f}ms" for k, v in table.items())    # noqa: E731
p95_max = lambda table: all(i == last for i, last in table.values())        # noqa: E731
p95_row = lambda t: ", ".join(f"{n}->idx {i} of {e}" for n, (i, e) in t.items())  # noqa: E731


def sample_images(np):
    from sklearn.datasets import load_sample_images
    frames = []
    for base in load_sample_images().images:
        for step in range(5):
            box_h, box_w = 200 + 40 * step, 260 + 50 * step
            top, left = (step * 37) % (base.shape[0] - box_h), (step * 53) % (base.shape[1] - box_w)
            frames.append(np.ascontiguousarray(base[top:top + box_h, left:left + box_w]))
    return frames


def crop_stage(torch, pipe, tensor, detected):
    crops = []
    for box in detected["boxes"]:
        x1, y1, x2, y2 = [max(0, int(edge)) for edge in box.tolist()]
        x2, y2 = min(x2, tensor.shape[-1]), min(y2, tensor.shape[-2])
        if (x2 - x1) >= pipe.min_crop and (y2 - y1) >= pipe.min_crop:
            crops.append(torch.nn.functional.interpolate(
                tensor[:, y1:y2, x1:x2].unsqueeze(0), size=(CROP, CROP),
                mode="bilinear", align_corners=False)[0])
    return crops


def walk_frames(np, torch, pipe, frames) -> dict:
    samples = {name: [] for name in STAGES}
    boxes, labels, matched = collections.Counter(), collections.Counter(), 0
    for index, frame in enumerate(frames):
        result = pipe.run(frame, image_id=f"img{index:02d}")
        boxes[len(result.detections)] += 1
        labels[len(result.classifications)] += 1
        for _ in range(REPEATS):
            marks = [time.perf_counter()]
            tensor = pipe.preprocess(frame)
            marks.append(time.perf_counter())
            detected = pipe.detect(tensor)
            marks.append(time.perf_counter())
            crops = crop_stage(torch, pipe, tensor, detected)
            marks.append(time.perf_counter())
            pipe.classify(crops)
            marks.append(time.perf_counter())
            for name, start, stop in zip(STAGES, marks, marks[1:]):
                samples[name].append((stop - start) * 1000.0)
        matched += len(crops) == len(result.classifications)
    return {"stages": {n: statistics.median(v) for n, v in samples.items()}, "matched": matched,
            "boxes": dict(boxes), "labels": dict(labels), "frames": len(frames),
            "sweep": {s: len(pipe.run(np.ascontiguousarray(frames[0][:s, :s])).classifications) for s in SIZES}}


def resolution_probe(torch, pipe, tensor) -> dict:
    patch = tensor[:, 40:240, 60:240].unsqueeze(0)
    with torch.no_grad():
        logits = {size: pipe.classifier(torch.nn.functional.interpolate(
            patch, size=(size, size), mode="bilinear", align_corners=False))
            for size in RESOLUTIONS}
    return {"gap": max((logits[s] - logits[CROP]).abs().max().item() for s in RESOLUTIONS),
            "picks": sorted({int(value.argmax()) for value in logits.values()})}


def solve():
    try:
        import numpy as np
        import torch
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    pipe = ref.VisionPipeline(ref.StubDetector(), ref.StubClassifier(CLASSES),
                              [f"class_{i}" for i in range(CLASSES)])
    frames = sample_images(np)
    return {**walk_frames(np, torch, pipe, frames),
            "resolution": resolution_probe(torch, pipe, pipe.preprocess(frames[-1])),
            "p95": {n: (int(n * 0.95), n - 1) for n in RUNS}}


def verify(result):
    stage, sweep, res = result["stages"], result["sweep"], result["resolution"]
    total, folded = sum(stage.values()), stage["crop"] + stage["classify"]
    return [
        practice.Check(
            "ANSWER: four stage medians over 10 photographs x 20 passes, and a degenerate count",
            result["boxes"] == {3: result["frames"]} and result["frames"] == 10,
            f"ten crops, 200x260 to 360x460, of the two photographs `load_sample_images` keeps on disk: "
            f"{row(stage)}, total {total:.4f}ms. Detections {result['boxes']}; labels {result['labels']}"),
        practice.Check(
            "FINDING: the detector never reads the image, so the asked-for distribution is a point mass",
            len(result["boxes"]) == 1 and sweep[32] == 0 and sweep[64] == 3,
            f"`StubDetector.forward` builds its three boxes out of `H, W = img.shape[-2:]` alone, so the count "
            f"is 3 on all {result['frames']} images whatever they contain. What moves instead is the label "
            f"count, and only through `min_crop=16`: square frames {SIZES} give {sweep} labels"),
        practice.Check(
            "MECHANISM: the largest stage has no bucket — crop+resize is billed to `classify`",
            stage["crop"] > stage["classify"] * 1.5 and result["matched"] == result["frames"],
            f"crop+resize takes {stage['crop']:.4f}ms, {stage['crop'] / total:.0%} of the pipeline and "
            f"{stage['crop'] / stage['classify']:.1f}x the {stage['classify']:.4f}ms classifier forward; "
            f"`benchmark()` times it between `t2` and `t3`, so its `classify` row reads {folded:.4f}ms, "
            f"{folded / stage['classify']:.1f}x the model's cost. `crop_stage` re-inlines that loop and returns "
            f"exactly the crops `pipe.run()` classified on {result['matched']}/{result['frames']} frames"),
        practice.Check(
            "CONTROL: the largest stage builds detail the classifier discards one layer later",
            res["gap"] < 0.05 and len(res["picks"]) == 1,
            f"`StubClassifier` opens with `AdaptiveAvgPool2d(1)`, so a crop is three channel means by the time "
            f"`Linear(3, {CLASSES})` sees it: one crop scored at {RESOLUTIONS} moves the logits at most "
            f"{res['gap']:.2e} and picks class {res['picks'][0]} every time -- a 4x4 resize, 256x fewer pixels "
            f"than the pipeline's {CROP}x{CROP}, is indistinguishable at the output"),
        practice.Check(
            "CONTROL: the p95 column the lesson prints is the maximum at its own default `num_runs`",
            p95_max({n: v for n, v in result["p95"].items() if n <= 20}),
            f"`benchmark` sorts, then indexes at `int(len(times) * 0.95)`: num_runs {p95_row(result['p95'])}. "
            f"At the default {RUNS[0]} that names the last element, so the printed p95 is simply the slowest "
            f"sample -- 21 runs are needed before `int(n * 0.95)` stops meaning `n - 1`"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
