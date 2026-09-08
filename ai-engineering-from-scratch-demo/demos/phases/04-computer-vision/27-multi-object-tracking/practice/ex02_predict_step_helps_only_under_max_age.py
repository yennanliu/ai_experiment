"""Exercise 2 — predict step helps only under max age.

    **(Medium)** Add a constant-velocity Kalman predict step before association.
    Show that short (2-3 frame) occlusions no longer cause ID switches.

Reading of the exercise: the demonstration works, and two things about it are
not what the wording implies. First, the fix is the extrapolation, not the
filter: initialised from its first two measurements, a constant-velocity Kalman
predicts the next box on constant-velocity input to 4.55e-13px, and inside the
lesson's tracker the innovation it sees is 0.000244px -- one float32 ULP,
because `SimpleTracker.step` downcasts every detection with `dtype=np.float32`.
The gain multiplies a rounding error; scaling the measurement noise 1000x leaves
every number in this file identical. The "Kalman" half is inert on noiseless
boxes and only the motion model earns the repair. Second, the repair has a hard
ceiling that has nothing to do with motion: `max_age` is 5, so at a 6-frame gap
the track is deleted before the object returns and both arms score identically.
A saturated comparison ranks nothing, so the sweep runs to that point and says
where it is. The lesson's own `synthetic_frames` cannot host the experiment --
it clamps boxes at the frame edge and manufactures ID churn of its own
(exercise 1) -- so the world comes from exercise 1's clean `steady`.

Structure: `KalmanCV` is a per-coordinate constant-velocity filter; all four box
coordinates share one 2x2 covariance because they share the dynamics, the noise
and the update times, and `update` two-point-initialises the velocity from the
first pair of measurements and returns the innovation it saw. `track` is the
tracker loop, optionally advancing every live track's box by its filter's
prediction before association and feeding matched boxes back afterwards --
`SimpleTracker` is imported from the lesson, never re-implemented -- and `sweep`
runs all three arms over every occlusion length. `filtered` reports one-step
prediction error against a known trajectory and `coasting` reports it with no
measurements at all; the module-level `occlude` deletes detections over a
window, `frozen` is the IoU between a track's stale box and its object after a
gap, and `ramp` is the constant-acceleration trajectory the control uses.
"""

from __future__ import annotations

import pathlib

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "27-multi-object-tracking"

OBJECTS, START, GAPS, MAX_AGE, GATE, LOUD, ACCEL = 5, 10, (1, 2, 3, 4, 5, 6), 5, 0.3, 1e3, 0.5
ARMS = (("plain", (False,)), ("kalman", (True,)), ("loud", (True, LOUD)))
SIBLING = pathlib.Path(__file__).with_name("ex01_clipping_not_crowding_breaks_iou.py")

occlude = lambda frames, gap: [[] if START <= f < START + gap else list(d)         # noqa: E731
                               for f, d in enumerate(frames)]
ramp = lambda np, n=12: [np.full(4, 100.0 + 0.5 * ACCEL * k * k) for k in range(n)]  # noqa: E731
tally = lambda arms, key: {n: [a[key] for a in arms[n]] for n in arms}             # noqa: E731
sunk = lambda late: [sum(1 for v in late[g] if v < GATE) for g in GAPS]            # noqa: E731
frozen = lambda np, ref, frames, gap: [round(float(v), 4) for v in np.diag(        # noqa: E731
    ref.bbox_iou(np.array(frames[START - 1]), np.array(frames[START + gap])))]
sweep = lambda np, ref, frames, gt: {name: [track(np, ref, occlude(frames, g), gt, *a)  # noqa: E731
                                           for g in GAPS] for name, a in ARMS}


class KalmanCV:
    def __init__(self, np, box, noise=1.0):
        self.np, self.noise, self.seen = np, noise, 1
        self.step = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.state = np.stack([np.asarray(box, float), np.zeros(4)], axis=1)
        self.cov = np.diag([noise, 4.0 * noise])

    def predict(self):
        self.state = self.state @ self.step.T
        self.cov = self.step @ self.cov @ self.step.T
        return self.state[:, 0].copy()

    def update(self, box) -> float:
        np, seen = self.np, self.np.asarray(box, float)
        innovation = float(np.abs(seen - self.state[:, 0]).max())
        if self.seen == 1:
            self.state = np.stack([seen, seen - self.state[:, 0] + self.state[:, 1]], axis=1)
        else:
            gain = self.cov[:, 0] / (self.cov[0, 0] + self.noise)
            self.state = self.state + np.outer(seen - self.state[:, 0], gain)
            self.cov = self.cov - np.outer(gain, self.cov[0])
        self.seen += 1
        return innovation


def track(np, ref, frames, gt, predict, noise=1.0) -> dict:
    tracker, filters, per_frame, gaps = ref.SimpleTracker(max_age=MAX_AGE), {}, [], []
    for index, dets in enumerate(frames):
        if predict:
            for live in tracker.tracks:
                live.bbox = filters[live.id].predict()
        per_frame.append(tracker.step(dets, index))
        for live in tracker.tracks:
            if live.id not in filters:
                filters[live.id] = KalmanCV(np, live.bbox, noise)
            elif live.last_frame == index:
                gaps.append(filters[live.id].update(live.bbox))
    return {"switches": ref.count_id_switches(per_frame, gt), "ids": tracker.next_id - 1,
            "innovation": max(gaps[OBJECTS:])}


def filtered(np, boxes, stop=None) -> tuple:
    filt, errors = KalmanCV(np, boxes[0]), []
    for step in range(1, stop or len(boxes)):
        error = filt.predict() - np.asarray(boxes[step], float)
        errors.append((float(np.abs(error).max()), float(error.mean())))
        filt.update(boxes[step])
    return errors, filt


def coasting(np, boxes) -> list:
    filt = filtered(np, boxes, 4)[1]
    return [float(np.abs(filt.predict() - np.asarray(boxes[3 + gap])).max()) for gap in GAPS]


def solve():
    try:
        import numpy as np
    except ImportError as exc:                      # pragma: no cover - T0 needs numpy
        raise practice.Skip(f"needs numpy: uv sync --extra math ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    frames, gt = practice.load_module(SIBLING).steady(np, OBJECTS)
    arms, boxes = sweep(np, ref, frames, gt), [f[0] for f in frames]
    late = dict(zip(GAPS, [frozen(np, ref, frames, gap) for gap in GAPS]))
    ids, straight = tally(arms, "ids"), filtered(np, boxes)[0]
    return {"switches": tally(arms, "switches"), "ids": ids, "stale": late, "sunk": sunk(late),
            "surplus": [n - OBJECTS for n in ids["plain"]], "coast": coasting(np, boxes),
            "innovation": arms["kalman"][2]["innovation"], "first": straight[0][0],
            "locked": max(e[0] for e in straight[1:]), "lag": [e[1] for e in filtered(np, ramp(np))[0]],
            "ulp": float(np.spacing(np.float32(np.max(frames)))), "span": float(np.max(frames))}


def verify(result):
    switch, ids, sank, lag = result["switches"], result["ids"], result["sunk"], result["lag"]
    plain, fixed, coast = switch["plain"], switch["kalman"], result["coast"]
    return [
        practice.Check(
            "ANSWER: the 2- and 3-frame occlusions stop costing ID switches -- 2 and 3 go to 0",
            plain[1:3] == [2, 3] and fixed[1:3] == [0, 0],
            f"over gaps {list(GAPS)} the lesson's tracker scores {plain} ID switches on {ids['plain']} "
            f"IDs for {OBJECTS} objects; with the constant-velocity predict step, {fixed} on "
            f"{ids['kalman']} -- 2 and 3 switches go to none, and no surplus track is ever opened"),
        practice.Check(
            "MECHANISM: the plain tracker fails exactly where the frozen box falls through the gate",
            result["surplus"][:5] == sank[:5],
            f"a gap of g freezes the box for g+1 frames; its IoU with its own detection on return is "
            f"{result['stale'][2]} at g=2 and {result['stale'][3]} at g=3. The count below the {GATE} "
            f"gate, {sank}, equals surplus IDs {result['surplus']} up to g={GAPS[-2]}"),
        practice.Check(
            "FINDING: on constant-velocity input the prediction is exact to float precision",
            result["locked"] < 1e-9 and max(coast) < 1e-9,
            f"two-point initialisation costs one frame ({result['first']:.4f}px); after it the largest "
            f"one-step error over the run is {result['locked']:.3g}px, and coasting {GAPS[-1]} frames "
            f"with no measurement at all stays at {max(coast):.3g}px -- nothing left to learn"),
        practice.Check(
            "CONTROL: the gain never fires -- the motion model is doing all the work",
            result["innovation"] <= result["ulp"] and fixed == switch["loud"],
            f"the innovation after initialisation is {result['innovation']:.3g}px, exactly one float32 "
            f"ULP here ({result['ulp']:.3g} at {result['span']:.0f}px) -- `SimpleTracker.step` downcasts "
            f"detections with dtype=np.float32, and that rounding is the whole residual. Scaling the "
            f"noise {LOUD:.0f}x leaves switches {switch['loud']} and IDs {ids['loud']}, identical"),
        practice.Check(
            "CONTROL: the repair stops at max_age, not at any property of the motion",
            fixed[-1] == plain[-1] != 0 and ids["kalman"][-1] == ids["plain"][-1],
            f"at a {GAPS[-1]}-frame gap both arms score {fixed[-1]} switches on {ids['kalman'][-1]} IDs, "
            f"because SimpleTracker drops a track after max_age={MAX_AGE} unmatched frames. The predict "
            f"step buys gaps 1 to {MAX_AGE} and not one more; past that the comparison ranks nothing"),
        practice.Check(
            "FINDING: the exactness belongs to the fixture, not to the filter",
            lag[0] < 0 and lag[-1] < 5 * lag[0] and abs(lag[1] + ACCEL) < 1e-9,
            f"on constant acceleration {ACCEL} px/frame^2 the same filter lags by "
            f"{' '.join(f'{v:.4f}' for v in lag[:6])} ... {lag[-1]:.4f}px, growing every step; the second "
            f"step's lag is exactly -a = {lag[1]:.4f}. `synthetic_frames` moves objects at a fixed "
            "velocity, so this lesson's data is the one case a CV model cannot miss"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
