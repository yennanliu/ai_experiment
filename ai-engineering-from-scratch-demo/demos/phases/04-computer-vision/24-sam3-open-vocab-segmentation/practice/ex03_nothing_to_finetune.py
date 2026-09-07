"""Exercise 3 — nothing to finetune.

    **(Hard)** Fine-tune SAM 3 on a custom concept set (e.g. 5 types of
    electronic components) with 20 labelled images each. Compare to zero-shot
    SAM 3 on the same test set; measure mask IoU improvement.

Reading of the exercise: there is nothing here that can be fine-tuned. SAM 3 is
not installable, and the lesson's own `StubOpenVocabSeg` holds no parameters at
all -- it carries no instance state, and two separately constructed copies agree
bit for bit -- so "compare to zero-shot" has one arm and the improvement is
identically zero by construction rather than by measurement. What can be built is
everything around the missing model: the 5-concept, 20-image labelled set the
exercise specifies, a mask-IoU harness, and an oracle that returns the label
itself to establish the ceiling. Measured against those, the zero-shot number
turns out to be a property of where the labels were placed, which is the part of
the exercise worth keeping.

Structure: `dataset` plants one rectangular component per image at a seeded
position; `predict` decodes the stub's two masks for an image; `best_iou` scores
a prediction set against a label; `evaluate` averages that over the whole set,
and takes an `oracle` flag to score the label against itself.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "24-sam3-open-vocab-segmentation"

HEIGHT, WIDTH = 64, 80
COMPONENTS = ("resistor", "capacitor", "diode", "inductor", "transistor")
PER_CONCEPT = 20

iou = lambda a, b: float((a & b).sum()) / float((a | b).sum() or 1)          # noqa: E731
best_iou = lambda masks, truth: max(iou(m, truth) for m in masks)            # noqa: E731
signature = lambda dets: tuple(d.mask_rle for d in dets)                     # noqa: E731
mean = lambda values: sum(values) / len(values)                              # noqa: E731
rles = lambda labels: [len(r) for r in labels]                               # noqa: E731


def dataset(np) -> list:
    """One labelled component per image, placed anywhere the frame allows."""
    rng = np.random.default_rng(0)
    rows = []
    for concept in COMPONENTS:
        for _ in range(PER_CONCEPT):
            top, left = int(rng.integers(0, HEIGHT - 20)), int(rng.integers(0, WIDTH - 24))
            truth = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
            truth[top:top + 18, left:left + 22] = 1
            rows.append((concept, rng.integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8), truth))
    return rows


def predict(ref, np, frame, concept) -> tuple:
    dets = ref.StubOpenVocabSeg().detect(frame, concept)
    return dets, [ref.rle_decode(d.mask_rle, (HEIGHT, WIDTH)) for d in dets]


def evaluate(ref, np, rows, oracle=False) -> tuple:
    scored, seen = [], set()
    for concept, frame, truth in rows:
        dets, masks = predict(ref, np, frame, concept)
        seen.add(signature(dets))
        scored.append(1.0 if oracle else best_iou(masks, truth))
    return mean(scored), len(seen), max(scored), min(scored)


def solve():
    try:
        import numpy as np
    except ImportError as exc:                      # pragma: no cover - T0 needs numpy
        raise practice.Skip(f"needs numpy: uv sync --extra math ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = dataset(np)
    zero_shot, distinct, best, worst = evaluate(ref, np, rows)
    tuned = evaluate(ref, np, rows, oracle=True)[0]
    frame = rows[0][1]
    first, second = ref.StubOpenVocabSeg(), ref.StubOpenVocabSeg()
    exact = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    exact[int(HEIGHT * 0.3):int(HEIGHT * 0.8), int(WIDTH * 0.2):int(WIDTH * 0.5)] = 1
    return {"rows": len(rows), "concepts": len(COMPONENTS), "zero_shot": zero_shot,
            "tuned": tuned, "distinct": distinct, "best": best, "worst": worst,
            "state": [len(vars(first)), len(vars(second))],
            "identical": signature(first.detect(frame, "resistor"))
            == signature(second.detect(frame, "resistor")),
            "aligned": best_iou(predict(ref, np, frame, "resistor")[1], exact),
            "label_bytes": rles([ref.rle_encode(row[2]) for row in rows[:5]]),
            "dither_bytes": len(ref.rle_encode(
                np.random.default_rng(1).integers(0, 2, (HEIGHT, WIDTH)).astype(np.uint8))),
            "raw": HEIGHT * WIDTH}


def verify(result):
    zero_shot, tuned, rows = result["zero_shot"], result["tuned"], result["rows"]
    return [
        practice.Check(
            "ANSWER: there is nothing to fine-tune -- the model has no parameters",
            result["state"] == [0, 0] and result["identical"],
            f"StubOpenVocabSeg carries {result['state'][0]} attributes of instance state, and two "
            f"separately constructed copies return byte-identical masks for the same frame "
            f"({result['identical']}). SAM 3 itself is not installable, so 'fine-tuned against "
            "zero-shot' has one arm and an improvement of exactly 0.000 by construction"),
        practice.Check(
            "ANSWER: the labelled set the exercise specifies, and what it scores",
            rows == len(COMPONENTS) * PER_CONCEPT and 0.0 < zero_shot < 0.3,
            f"{len(COMPONENTS)} components x {PER_CONCEPT} images = {rows} labelled frames, one "
            f"18x22 component each. Zero-shot mean mask IoU is {zero_shot:.4f}, ranging "
            f"{result['worst']:.4f} to {result['best']:.4f}; an oracle returning the label itself "
            f"scores {tuned:.4f}, so the entire headroom is {tuned - zero_shot:.4f}"),
        practice.Check(
            "FINDING: all 100 predictions are the same two masks",
            result["distinct"] == 1,
            f"across all {rows} frames and {len(COMPONENTS)} concepts the stub emits "
            f"{result['distinct']} distinct mask pair. So the IoU spread of "
            f"{result['worst']:.4f} to {result['best']:.4f} is produced entirely by where the "
            "labels landed -- the prediction never varies, and the metric is measuring the dataset"),
        practice.Check(
            "MECHANISM: the score is a rectangle-overlap identity, not a capability",
            result["aligned"] == 1.0 > zero_shot,
            f"placing a label exactly on the rectangle detect() hardcodes -- rows 0.3-0.8 x columns "
            f"0.2-0.5 of the frame -- scores IoU {result['aligned']:.1f}. The same model scores "
            f"{zero_shot:.4f} on the exercise's own randomly placed components. Nothing about the "
            "model changed between those two numbers; only the label moved"),
        practice.Check(
            "CONTROL: an improvement measured this way would come from the labels",
            tuned - zero_shot > 0.7,
            f"the oracle's {tuned:.4f} against the stub's {zero_shot:.4f} is the largest 'improvement' "
            f"this test set can report, and it is bought by returning the answer. Any fine-tuning "
            "number quoted against a parameter-free baseline is measuring the evaluation set, which "
            "is why the exercise needs a model before it needs a metric"),
        practice.Check(
            "CONTROL: the encoding assumes the shape it is given",
            max(result["label_bytes"]) * 10 < result["raw"] < result["dither_bytes"],
            f"the first five labels encode to {result['label_bytes']} bytes through the lesson's own "
            f"rle_encode, against {result['raw']:,} raw pixels -- a 27x saving, because a rectangle "
            f"has one run per row boundary. The same encoder turns a dithered mask of identical size "
            f"into {result['dither_bytes']:,} bytes, {result['dither_bytes'] / result['raw']:.2f}x "
            "*larger* than raw. It is a compression scheme with a shape assumption inside it, and "
            "these labels happen to satisfy it"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
