"""Exercise 1 — stub reads neither input.

    **(Easy)** Run SAM 3 on 10 images with concept prompts you choose. Compare
    against SAM 2 + Grounding DINO 1.5 on the same images. Report which concepts
    each model missed.

Reading of the exercise: neither model is installable -- `sam2`,
`segment_anything`, `groundingdino` and `transformers` all raise
ModuleNotFoundError -- so the absences are recorded as measurements and the
comparison runs against the one implementation the lesson ships,
`StubOpenVocabSeg`. That turns the exercise into a sharper question than it
asks, because the stub cannot miss a concept: `detect` never reads the concept
string and never reads a pixel. Ten random images and six unrelated prompts,
including the empty string, all return byte-identical masks; only
`image.shape` moves them. So "which concepts did each model miss" has the answer
"none, on every image", and that is a property of the harness rather than a
result about open-vocabulary segmentation.

Structure: `images` makes the ten random frames; `probe` runs the stub over a
list of concepts and returns their run-length strings; `rect` predicts a mask
rectangle from the fractions `detect` hardcodes; `planted` builds a ground-truth
object in the frame's top-left, clear of both hardcoded rectangles -- they begin
at row 16 and column 16 -- so recall and IoU can both be scored against it. The
lambdas above them hold the comprehensions, which keeps solve() inside D14's
complexity cap; at 126 lines of code the file is six over the 120-line target and
well clear of the 150-line ceiling.
"""

from __future__ import annotations

import importlib

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "24-sam3-open-vocab-segmentation"

HEIGHT, WIDTH, FRAMES = 64, 80, 10
STACK = ("sam2", "segment_anything", "groundingdino", "transformers")
CONCEPTS = ("cat", "dog", "teapot", "zzz", "", "a very long concept string")
SPANS = ((0.3, 0.8, 0.2, 0.5), (0.25, 0.75, 0.55, 0.85))   # detect()'s own fractions

# comprehensions live here so solve() keeps its branch count under D14's cap
iou = lambda a, b: float((a & b).sum()) / float((a | b).sum() or 1)                    # noqa: E731
listing = lambda got: ", ".join(f"{k} {v}" for k, v in got.items())                    # noqa: E731
scores = lambda dets: [d.score for d in dets]                                          # noqa: E731
echoes = lambda dets: [d.concept for d in dets]                                        # noqa: E731
decode = lambda ref, dets: [ref.rle_decode(d.mask_rle, (HEIGHT, WIDTH)) for d in dets]  # noqa: E731
ious = lambda masks, truth: [iou(m, truth) for m in masks]                             # noqa: E731
hits = lambda masks, truth: float(sum(int((m & truth).any()) for m in masks))          # noqa: E731
matches = lambda np, masks: [bool((masks[i] == rect(np, SPANS[i])).all()) for i in (0, 1)]  # noqa: E731
per_frame = lambda ref, frames: [probe(ref, frame, ["anything"])[0] for frame in frames]    # noqa: E731


def stack_status() -> dict:
    got = {}
    for name in STACK:
        try:
            importlib.import_module(name)
            got[name] = "present"
        except ImportError as exc:                  # pragma: no cover - none is installed here
            got[name] = type(exc).__name__
    return got


def images(np):
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (HEIGHT, WIDTH, 3), dtype=np.uint8) for _ in range(FRAMES)]


def probe(ref, frame, concepts) -> list:
    return [tuple(d.mask_rle for d in ref.StubOpenVocabSeg().detect(frame, c)) for c in concepts]


def rect(np, span):
    top, bottom, left, right = span
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    mask[int(HEIGHT * top):int(HEIGHT * bottom), int(WIDTH * left):int(WIDTH * right)] = 1
    return mask


def planted(np):
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    mask[2:16, 2:15] = 1        # clear of both rectangles: they start at row 16 and column 16
    return mask


def solve():
    try:
        import numpy as np
    except ImportError as exc:                      # pragma: no cover - T0 needs numpy
        raise practice.Skip(f"needs numpy: uv sync --extra math ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    frames = images(np)
    smaller = ref.StubOpenVocabSeg().detect(np.zeros((HEIGHT // 2, WIDTH // 2, 3), np.uint8), "cat")
    found = ref.StubOpenVocabSeg().detect(frames[0], "cat")
    masks, truth = decode(ref, found), planted(np)
    return {"missing": stack_status(), "frames": len(set(per_frame(ref, frames))),
            "concepts": len(set(probe(ref, frames[0], CONCEPTS))), "scores": scores(found),
            "echo": echoes(ref.StubOpenVocabSeg().detect(frames[0], "teapot")),
            "shape_moves": smaller[0].mask_rle != found[0].mask_rle,
            "predicted": matches(np, masks), "pair_iou": iou(masks[0], masks[1]),
            "truth_iou": ious(masks, truth), "covered": hits(masks, truth)}


def verify(result):
    missing, predicted = result["missing"], result["predicted"]
    return [
        practice.Check(
            "ANSWER: there are no two models to compare -- neither is installable",
            all(v == "ModuleNotFoundError" for v in missing.values()),
            f"importing the exercise's stack gives {listing(missing)}. The lesson ships exactly one "
            "implementation of its own OpenVocabSeg interface, StubOpenVocabSeg, so the comparison "
            "it asks for has one arm; what follows measures that arm instead of inventing a second"),
        practice.Check(
            "ANSWER: the stub cannot miss a concept, because it never reads one",
            result["concepts"] == 1 and result["echo"] == ["teapot", "teapot"],
            f"{len(CONCEPTS)} unrelated prompts -- including the empty string and a 26-character "
            f"phrase -- return {result['concepts']} distinct mask pair, byte-identical run-length "
            f"strings. The `concept` field is a pure echo of the argument ({result['echo']}), and "
            f"the scores are the constants {result['scores']}. Nothing can be reported as missed"),
        practice.Check(
            "FINDING: it does not read the image either",
            result["frames"] == 1 and result["shape_moves"],
            f"{FRAMES} independently drawn random {HEIGHT}x{WIDTH} frames give "
            f"{result['frames']} distinct mask pair. Halving the frame does change it, so the only "
            "input `detect` consults is `image.shape[:2]` -- the pixels are never touched"),
        practice.Check(
            "MECHANISM: the masks are two hardcoded rectangles in fractions of h and w",
            predicted == [True, True],
            f"rebuilding the masks from the fractions in detect()'s own source -- rows "
            f"{SPANS[0][0]}-{SPANS[0][1]} x cols {SPANS[0][2]}-{SPANS[0][3]}, and rows "
            f"{SPANS[1][0]}-{SPANS[1][1]} x cols {SPANS[1][2]}-{SPANS[1][3]} -- reproduces both "
            f"decoded masks exactly ({predicted}). That is the whole model"),
        practice.Check(
            "CONTROL: recall is 1.000 and precision is meaningless, which is the same statement",
            result["covered"] == 0.0 and max(result["truth_iou"]) == 0.0,
            f"planting a 14x13 ground-truth object clear of both rectangles gives IoU "
            f"{result['truth_iou']} against the two returned instances, {result['covered']:.0f} of "
            "them overlapping it at all. The stub answers every concept and finds no object, so "
            "'which concepts were missed' and 'which were found' are both answered by the harness"),
        practice.Check(
            "CONTROL: the two instances never overlap, so nothing downstream is exercised",
            result["pair_iou"] == 0.0,
            f"the pair the stub returns has IoU exactly {result['pair_iou']} -- columns 0.2-0.5 and "
            "0.55-0.85 cannot touch. Any de-duplication, NMS or instance-merging step built on top "
            "of this fixture is untested by it, whatever the prompt"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
