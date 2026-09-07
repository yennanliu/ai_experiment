"""Exercise 2 — mask rle payload size.

    **(Medium)** Add a mask output field to `Detection` and encode it as RLE.
    Verify the JSON stays under 1MB even for a 10-object image.

Reading of the exercise: the field is already there. `code/main.py` declares
`mask_rle: Optional[str] = None` on `Detection`, and nothing in the lesson ever
writes it -- `VisionPipeline.run()` constructs every `Detection` from box, score
and class only, so a real run emits three detections with `"mask_rle": null` in
a 612-byte payload. What the exercise actually asks for is the producer, so this
file supplies the COCO-style encoder (column-major run lengths, alternating and
starting from background) and populates the lesson's own model with it. The
verification clause then names the wrong variable. Object count barely moves the
payload; mask *boundary complexity* moves it by orders of magnitude, and the two
are independent. Ten solid boxes and ten dithered boxes cover identical pixels
and differ 24x in bytes. So "under 1MB even for a 10-object image" is true at
the lesson's own 400x600 demo size for any mask and false at 720p for fragmented
ones -- the same ten objects. The budget is read as 1,000,000 bytes; at 1 MiB
(1,048,576) no verdict below changes. Nothing is downloaded: the sole real image
comes from the two photographs scikit-learn keeps on disk.

Structure: `rle_encode` and `rle_decode` are the codec; `boxes_for` places ten
copies of the first box `StubDetector` emits (0.3W x 0.5H) at seeded offsets;
`make_mask` fills one box solid, as an inscribed ellipse, or with a Bernoulli(0.5)
dither; `payload` builds ten `Detection` records through the lesson's own pydantic
models, serialises, re-parses and reports bytes, run counts and whether the round
trip was lossless; `dead_field` runs the untouched pipeline to count the nulls
and to watch the box tuple change dtype across the JSON boundary. At 140 code
lines this sits above D14's 120-line target and 10 clear of the ceiling: five
checks over nine payloads (three mask kinds x three frame sizes, ten objects
each) plus the untouched-pipeline probe.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "16-vision-pipeline-capstone"

CLASSES, OBJECTS, BUDGET = 10, 10, 1_000_000
SHAPES = ((400, 600), (720, 1280), (1080, 1920))    # the lesson's demo size, then 720p and 1080p
KINDS = ("solid", "ellipse", "dither")

kib = lambda n: f"{n:,}B ({n / BUDGET:.1%})"                             # noqa: E731 - a formatter
sizes = lambda row: "  ".join(f"{k}={kib(v['bytes'])}" for k, v in row.items())      # noqa: E731
worst = lambda row: max(v["bytes"] for v in row.values())                            # noqa: E731
whole = lambda row: all(v["lossless"] for v in row.values())                         # noqa: E731
exact = lambda tab: all(t["solid"]["runs"] == 2 * t["solid"]["columns"] + 1  # noqa: E731
                        for t in tab.values())
runs_row = lambda tab: ", ".join(f"{s}: {t['solid']['columns']} cols, "              # noqa: E731
                                 f"{t['solid']['runs']} runs" for s, t in tab.items())


def rle_encode(np, mask) -> str:
    flat = mask.reshape(-1, order="F")
    edges = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    runs = np.diff(np.concatenate(([0], edges, [flat.size])))
    if flat[0]:                                     # COCO's first run is background, possibly empty
        runs = np.concatenate(([0], runs))
    return " ".join(map(str, runs.tolist()))


def rle_decode(np, text, shape):
    flat, position, value = np.zeros(shape[0] * shape[1], bool), 0, False
    for run in map(int, text.split()):
        flat[position:position + run] = value
        position, value = position + run, not value
    return flat.reshape(shape, order="F")


def boxes_for(np, shape):
    height, width = shape
    box_h, box_w = int(0.5 * height), int(0.3 * width)      # StubDetector's first box, ten times over
    rng = np.random.default_rng(7)
    corners = zip(rng.integers(0, width - box_w, OBJECTS), rng.integers(0, height - box_h, OBJECTS))
    return [(int(x), int(y), int(x) + box_w, int(y) + box_h) for x, y in corners]


def make_mask(np, kind, box, shape):
    x1, y1, x2, y2 = box
    mask = np.zeros(shape, bool)
    if kind == "ellipse":
        rows, cols = np.ogrid[:shape[0], :shape[1]]
        return (((rows - (y1 + y2) / 2) / ((y2 - y1) / 2)) ** 2
                + ((cols - (x1 + x2) / 2) / ((x2 - x1) / 2)) ** 2) <= 1.0
    mask[y1:y2, x1:x2] = (np.random.default_rng(x1).random((y2 - y1, x2 - x1)) < 0.5
                          if kind == "dither" else True)
    return mask


def payload(np, ref, shape, kind) -> dict:
    boxes = boxes_for(np, shape)
    masks = [make_mask(np, kind, box, shape) for box in boxes]
    detections = [ref.Detection(box=tuple(float(edge) for edge in box), score=0.9, class_id=1,
                                mask_rle=rle_encode(np, mask)) for box, mask in zip(boxes, masks)]
    text = ref.PipelineResult(image_id="ten", detections=detections, classifications=[],
                              inference_ms=1.0).model_dump_json()
    restored = ref.PipelineResult.model_validate_json(text).detections
    return {"bytes": len(text.encode()), "runs": len(detections[0].mask_rle.split()),
            "columns": boxes[0][2] - boxes[0][0],
            "lossless": all((rle_decode(np, d.mask_rle, shape) == m).all()
                            for d, m in zip(restored, masks)) if shape == SHAPES[0] else None}


def dead_field(np, ref, pipe) -> dict:
    from sklearn.datasets import load_sample_images
    result = pipe.run(np.ascontiguousarray(load_sample_images().images[0][:400, :600]), image_id="demo")
    text = result.model_dump_json()
    return {"fields": list(ref.Detection.model_fields), "bytes": len(text.encode()),
            "nulls": sum(d.mask_rle is None for d in result.detections), "given": (60, 40, 240, 240),
            "parsed": result.detections[0].box,
            "kinds": sorted({type(edge).__name__ for edge in result.detections[0].box})}


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
    return {"dead": dead_field(np, ref, pipe),
            "table": {s: {k: payload(np, ref, s, k) for k in KINDS} for s in SHAPES}}


def verify(result):
    table, dead = result["table"], result["dead"]
    demo, hd = table[SHAPES[0]], table[SHAPES[1]]
    return [
        practice.Check(
            "ANSWER: at the lesson's own 400x600 demo size a 10-object payload fits the budget 65x over",
            worst(demo) < BUDGET,
            f"ten masks in the lesson's own `Detection.mask_rle`, serialised by `model_dump_json()`: "
            f"{sizes(demo)} at {SHAPES[0]} -- {BUDGET / demo['solid']['bytes']:.0f}x headroom"),
        practice.Check(
            "FINDING: object count is the wrong variable — the same 10 objects break the budget at 720p",
            hd["dither"]["bytes"] > BUDGET and hd["ellipse"]["bytes"] < BUDGET,
            f"solid and dithered masks cover identical pixels yet differ "
            f"{demo['dither']['bytes'] / demo['solid']['bytes']:.0f}x in bytes at {SHAPES[0]}, and scaling "
            f"the frame lets boundary complexity win outright -- {SHAPES[1]}: {sizes(hd)}, {SHAPES[2]}: "
            f"{sizes(table[SHAPES[2]])}; ten objects throughout"),
        practice.Check(
            "MECHANISM: RLE length counts boundary crossings down the scan, not covered pixels",
            exact(table),
            f"the scan is column-major, so a solid box crosses the boundary twice per column it occupies "
            f"plus once to close the image -- {runs_row(table)}. The {SHAPES[0]} box covers "
            f"{int(0.3 * 600) * int(0.5 * 400):,} pixels, none of which enter the count; dithering that same "
            f"box costs {demo['dither']['runs']:,} runs"),
        practice.Check(
            "CONTROL: the codec is lossless through pydantic — decode(parse(encode(m))) == m",
            whole(demo),
            f"all {OBJECTS * len(KINDS)} masks at {SHAPES[0]} encode, serialise, re-parse through "
            f"`model_validate_json` and decode back to an equal array -- so the bytes buy no accuracy and "
            f"{demo['dither']['bytes']:,}B is the honest cost of that mask, not an encoder defect"),
        practice.Check(
            "CONTROL: the field the exercise asks for already exists — and the pipeline never fills it",
            "mask_rle" in dead["fields"] and dead["nulls"] == 3 and dead["kinds"] == ["float"],
            f"`Detection.model_fields` is {dead['fields']} straight out of `code/main.py`, yet the untouched "
            f"pipeline on a real photograph gives {dead['nulls']}/3 detections with `mask_rle` null in "
            f"{dead['bytes']}B. That round trip also retypes the geometry: box {dead['given']} was built "
            f"from ints and parses back as {dead['parsed']}, since the annotation is `Tuple[float, ...]`"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
