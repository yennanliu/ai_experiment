"""Exercise 1 — clipping not crowding breaks iou.

    **(Easy)** Run the synthetic tracker above with 3, 10, and 30 objects. Report
    ID-switch count in each case. Identify where the simple IoU-only association
    starts to fail.

Reading of the exercise: "where it starts to fail" is asked as if the answer
were an object count, and on this fixture it is not. `synthetic_frames` clamps
every box with `max(0.0, cx - 10)` and `min(W - 1, cx + 10)`, so an object that
drifts off the left or top edge gets x2 < x1 -- a negative-width box. Those
score IoU exactly 0.0 against everything, so `SimpleTracker` opens a fresh track
for one every frame. Eight of the 30 objects do this, and that is the whole of
the 30-object blow-up: a control with twice as many objects that all stay inside
the frame tracks perfectly. What actually limits IoU-only association is
per-frame displacement against box width, and this fixture never approaches it
-- the tightest frame-to-frame self-overlap measured here is 0.4782 against a
0.30 gate. The one genuine identity error is a crossing, and it costs two
counted switches because MOT charges an ID switch per ground-truth track.

Structure: `drive` runs the lesson's tracker over a frame list and hands back
both the per-frame output and the tracker, so IDs minted and tracks surviving
can be read off it. `census` does that over the lesson's own fixture at one
object count and adds the degenerate-box tally. `steady` builds the control
world -- constant velocity, every box wholly inside a canvas big enough that
nothing clamps -- and exercises 2 and 3 import it from here rather than carrying
a second copy. `swaps` replays the assignment `count_id_switches` makes, so the
frame and the IoU of each switch can be named instead of just counted. `tie`
puts two tracks on one box, which is the only way the Hungarian solve here has
more than one optimum; the module-level `selfiou` sweeps a box against a shifted
copy of itself, which is where the displacement thresholds come from.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "27-multi-object-tracking"

FRAMES, COUNTS, GATE, BOX = 25, (3, 10, 30), 0.3, 20.0
CANVAS, CONTROL_N, SPEED = 4000, 60, 3.0
STACK = [[0.0, 0.0, BOX, BOX], [0.0, 0.0, BOX, BOX]]        # two tracks on one box
LEFT, RIGHT = [-6.0, 0.0, BOX - 6.0, BOX], [6.0, 0.0, BOX + 6.0, BOX]

thin = lambda box: box[2] - box[0] <= 0 or box[3] - box[1] <= 0                    # noqa: E731
flat = lambda frames: [b for frame in frames for b in frame]                       # noqa: E731
bent = lambda gt: sorted({i for frame in gt for i, box in frame if thin(box)})     # noqa: E731
selfiou = lambda ref, np, shift: float(ref.bbox_iou(                               # noqa: E731
    np.array([[0.0, 0.0, BOX, BOX]]), np.array([[shift, 0.0, BOX + shift, BOX]]))[0, 0])


def drive(ref, frames, **kwargs) -> tuple:
    tracker = ref.SimpleTracker(**kwargs)
    return [tracker.step(dets, f) for f, dets in enumerate(frames)], tracker


def census(ref, count) -> dict:
    frames, gt = ref.synthetic_frames(num_frames=FRAMES, num_objects=count, seed=0)
    per_frame, tracker = drive(ref, frames)
    boxes = flat(frames)
    return {"switches": ref.count_id_switches(per_frame, gt), "ids": tracker.next_id - 1,
            "active": len(tracker.tracks), "objects": bent(gt), "boxes": len(boxes),
            "degenerate": len([b for b in boxes if thin(b)])}


def steady(np, count, frames=FRAMES, speed=SPEED) -> tuple:
    rng = np.random.default_rng(0)
    starts = rng.uniform(CANVAS * 0.2, CANVAS * 0.6, size=(count, 2))
    velocity = rng.uniform(-speed, speed, size=(count, 2))
    boxes = [[[c[0], c[1], c[0] + BOX, c[1] + BOX] for c in starts + step * velocity]
             for step in range(frames)]
    return boxes, [list(enumerate(frame)) for frame in boxes]


def swaps(ref, np, per_frame, gt) -> dict:
    seen, found = {}, {"frames": [], "objects": [], "pairs": [], "ious": []}
    for index, (tracks, truth) in enumerate(zip(per_frame, gt)):
        iou = ref.bbox_iou(np.array([b for _, b in truth]), np.array([b for _, b in tracks]))
        for row, (gid, _) in enumerate(truth):
            column = int(iou[row].argmax())
            if iou[row, column] > 0.5:
                if seen.get(gid, tracks[column][0]) != tracks[column][0]:
                    found["frames"].append(index)
                    found["objects"].append(gid)
                    found["pairs"].append((seen[gid], tracks[column][0]))
                    found["ious"].append(float(iou[row, column]))
                seen[gid] = tracks[column][0]
    return found


def tie(ref, np) -> dict:
    cost = 1.0 - ref.bbox_iou(np.array(STACK), np.array([LEFT, RIGHT]))
    order = [drive(ref, [STACK, dets])[0][1][0][1][0] for dets in ([LEFT, RIGHT], [RIGHT, LEFT])]
    return {"cost": float(cost[0, 0]), "spread": float(cost.max() - cost.min()), "order": order,
            "diagonal": float(cost[0, 0] + cost[1, 1]), "anti": float(cost[0, 1] + cost[1, 0])}


def solve():
    try:
        import numpy as np
    except ImportError as exc:                      # pragma: no cover - T0 needs numpy
        raise practice.Skip(f"needs numpy: uv sync --extra math ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    frames, gt = ref.synthetic_frames(num_frames=FRAMES, num_objects=10, seed=0)
    control, control_gt = steady(np, CONTROL_N)
    per_frame, tracker = drive(ref, control)
    grid = np.arange(0.0, BOX + 0.005, 0.01)
    counts = {n: census(ref, n) for n in COUNTS}
    return {"thirty": counts[30], "tie": tie(ref, np),
            "switches": [counts[n]["switches"] for n in COUNTS],
            "minted": [counts[n]["ids"] for n in COUNTS],
            "alive": [counts[n]["active"] for n in COUNTS],
            "swaps": swaps(ref, np, drive(ref, frames)[0], gt),
            "control": {"ids": tracker.next_id - 1, "active": len(tracker.tracks),
                        "switches": ref.count_id_switches(per_frame, control_gt)},
            "slowest": min(ref.bbox_iou(np.array([a]), np.array([b]))[0, 0]
                           for a, b in zip(frames[0], frames[1])),
            "shift": {"gate": next(float(d) for d in grid if selfiou(ref, np, float(d)) < GATE),
                      "zero": selfiou(ref, np, BOX), "under": selfiou(ref, np, BOX - 0.01)}}


def verify(result):
    control, shift, ambiguous, swap = (result["control"], result["shift"], result["tie"],
                                       result["swaps"])
    switches, minted, thirty = result["switches"], result["minted"], result["thirty"]
    return [
        practice.Check(
            "ANSWER: 0, 2 and 18 ID switches at 3, 10 and 30 objects",
            switches == [0, 2, 18] and minted == [3, 10, 120],
            f"ID switches {switches} at object counts {list(COUNTS)}, from {minted} track IDs minted and "
            f"{result['alive']} tracks still alive at frame {FRAMES - 1}. At 30 objects the tracker mints "
            f"{minted[2]} IDs and ends holding {thirty['active']}: the switch count is the smaller half"),
        practice.Check(
            "MECHANISM: the 30-object failure is boundary clipping, not association",
            thirty["objects"] == [1, 13, 16, 18, 24, 25, 26, 29] and thirty["degenerate"] == 83,
            f"clamping with max(0.0, cx - 10) and min(W - 1, cx + 10) gives x2 < x1 to anything leaving the "
            f"left or top edge: {thirty['degenerate']} of {thirty['boxes']} boxes at 30 objects have "
            f"non-positive width or height, from objects {thirty['objects']} -- 8 of 30. `bbox_iou` clips "
            "the intersection to 0, so each scores 0.0 on everything and buys a new ID every frame"),
        practice.Check(
            "CONTROL: crowding is not the failure mode -- twice the objects, no errors",
            control["ids"] == CONTROL_N and control["switches"] == 0,
            f"{CONTROL_N} objects at up to {SPEED} px/frame on a {CANVAS}px canvas, every box wholly inside "
            f"it, give {control['ids']} IDs for {CONTROL_N} objects, {control['active']} live tracks and "
            f"{control['switches']} ID switches -- doubling the lesson's worst object count costs nothing "
            "once no box is clamped"),
        practice.Check(
            "FINDING: the 2 switches at 10 objects are one crossing, counted twice",
            swap["frames"] == [8, 8] and min(swap["ious"]) > 0.999,
            f"both switches land on frame {swap['frames'][0]}, on ground-truth objects {swap['objects']}, "
            f"exchanging track IDs {swap['pairs'][0][0]}<->{swap['pairs'][0][1]} at IoU "
            f"{swap['ious'][0]:.4f} and {swap['ious'][1]:.4f}: each track sits exactly on the other's "
            "object. MOT charges an IDSW per ground-truth track, so one mutual swap is two events"),
        practice.Check(
            "FINDING: motion never reaches the gate here -- IoU-only has not begun to fail",
            result["slowest"] > GATE and shift["gate"] > 10.0,
            f"the tightest frame-to-frame self-overlap in the fixture is {result['slowest']:.4f}, clear of "
            f"SimpleTracker's {GATE} gate. Sweeping a {BOX:.0f}px box against a shifted copy, IoU first "
            f"drops under {GATE} at {shift['gate']:.2f}px ({shift['gate'] / BOX:.3f} box widths) and hits "
            f"exactly {shift['zero']:.1f} at {BOX:.0f}px -- one box width -- from {shift['under']:.6f}"),
        practice.Check(
            "CONTROL: at full occlusion the Hungarian ties and the detector's list order decides",
            ambiguous["spread"] == 0.0 and ambiguous["diagonal"] == ambiguous["anti"] != 0.0
            and ambiguous["order"] == [-6.0, 6.0],
            f"two tracks on one box make every cost {ambiguous['cost']:.4f}, spread "
            f"{ambiguous['spread']:.1f}, and both pairings total {ambiguous['diagonal']:.4f}, so the "
            f"optimum is not unique. The same two detections in the two possible orders send track 1 to "
            f"x1={ambiguous['order'][0]:.0f} or x1={ambiguous['order'][1]:.0f}: list order decides"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
