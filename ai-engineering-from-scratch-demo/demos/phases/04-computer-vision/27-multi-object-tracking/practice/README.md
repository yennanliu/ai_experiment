<!-- generated:start -->
# 04-computer-vision / 27-multi-object-tracking

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/04-computer-vision/27-multi-object-tracking/) · upstream spec
`phases/04-computer-vision/27-multi-object-tracking/docs/en.md`

```bash
uv run demo practice run 27-multi-object-tracking --ex 1
uv run demo explain 27-multi-object-tracking --ex 1
uv run pytest demos/phases/04-computer-vision/27-multi-object-tracking
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | (Easy) Run the synthetic tracker above with 3, 10, and 30 objects. Report ID-switch count in… | code | T0 | `ex01_clipping_not_crowding_breaks_iou.py` |
| 2 | (Medium) Add a constant-velocity Kalman predict step before association. Show that short (2-3… | code | T0 | `ex02_predict_step_helps_only_under_max_age.py` |
| 3 | (Hard) Integrate SAM 2's memory-based tracker (via `transformers`) as an alternative tracker… | code | T0 | `ex03_mota_unbounded_below_idsw_blind.py` |
<!-- generated:end -->

## Answers

All three exercises are about *counting* tracking errors, and all three turn on
the same thing: the count you pick decides what you can see. ID switches at 3,
10 and 30 objects are 0, 2 and 18 — but 120 track IDs get minted for those 30
objects, so the switch count is the smaller half of the failure and the larger
half is invisible to it. The lesson's fixture manufactures that failure at the
frame edge rather than in the association, a constant-velocity predict step
repairs short occlusions exactly but stops dead at `max_age`, and MOTA — the
metric that *would* see the false positives — is unbounded below and reaches
-7.000 on output whose ID-switch count is 0.

These three run at **T0** on numpy and scipy, so unlike most of phase 04 they
execute in CI rather than skipping.

### 1 — Clipping, not crowding, breaks IoU association

**ANSWER: 0, 2 and 18 ID switches at 3, 10 and 30 objects.**

| objects | ID switches | track IDs minted | tracks alive at frame 24 |
|---|---:|---:|---:|
| 3 | **0** | 3 | 3 |
| 10 | **2** | 10 | 10 |
| 30 | **18** | **120** | **70** |

**MECHANISM: the 30-object failure is boundary clipping, not association.**
`synthetic_frames` clamps every box with `max(0.0, cx - 10)` and
`min(W - 1, cx + 10)`, so an object drifting off the left or top edge gets
x2 < x1. **83 of 750** boxes at 30 objects have non-positive width or height,
contributed by objects `[1, 13, 16, 18, 24, 25, 26, 29]` — **8 of 30**.
`bbox_iou` clips the intersection to 0, so each of those scores **0.0** against
every track and buys a brand-new ID every frame.

**CONTROL: crowding is not the failure mode.** Sixty objects at up to 3.0
px/frame on a 4000px canvas, every box wholly inside it, give **60 IDs for 60
objects, 60 live tracks and 0 ID switches**. Doubling the lesson's worst object
count costs exactly nothing once no box is clamped.

**FINDING: the 2 switches at 10 objects are one crossing, counted twice.** Both
land on frame **8**, on ground-truth objects `[6, 7]`, exchanging track IDs
7↔8 at IoU **1.0000** and **1.0000** — each track ends up sitting exactly on the
other's object. MOT charges an IDSW per ground-truth track, so a single mutual
swap is two events.

**FINDING: motion never reaches the gate here.** The tightest frame-to-frame
self-overlap in the fixture is **0.4782**, against SimpleTracker's 0.30 gate.
Sweeping a 20px box against a shifted copy of itself:

| shift | self-IoU |
|---|---:|
| **10.77px** = 0.538 box widths | first value below the **0.30** gate |
| 19.99px | **0.000250** |
| **20.00px** = one box width | **0.0** |

So IoU-only association has not *begun* to fail on this data; what fails is the
fixture's own geometry.

**CONTROL: at full occlusion the Hungarian ties, and the detector's list order
decides identity.** Two tracks on one box make every cost **0.4615**, spread
**0.0**, and both pairings total **0.9231** — the optimum is not unique. Feeding
the same two detections in the two possible orders sends track 1 to x1=**-6** or
x1=**6**.

### 2 — The predict step helps, and only under `max_age`

**ANSWER: the 2- and 3-frame occlusions stop costing ID switches.**

| occlusion gap | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| plain — ID switches | 0 | **2** | **3** | 4 | 4 | 5 |
| plain — IDs minted (5 objects) | 5 | 7 | 8 | 9 | 9 | 10 |
| + CV predict — ID switches | 0 | **0** | **0** | 0 | 0 | 5 |
| + CV predict — IDs minted | 5 | 5 | 5 | 5 | 5 | 10 |

**MECHANISM: the plain tracker fails exactly where the frozen box falls through
the gate.** A gap of *g* freezes the track box for *g+1* frames; per object its
IoU with its own detection on return is `[0.2464, 0.245, 0.3906, 0.4793,
0.6164]` at g=2 and `[0.1431, 0.1438, 0.2841, 0.3662, 0.5258]` at g=3. The count
below the 0.30 gate, `[0, 2, 3, 4, 4, 4]`, equals the surplus IDs
`[0, 2, 3, 4, 4, 5]` for every gap up to 5.

**FINDING: on constant-velocity input the prediction is exact to float
precision.** Two-point initialisation costs one frame (**2.9836px** error on the
first step); after it the largest one-step error over the run is
**4.55e-13px**, and coasting 6 frames with no measurement at all stays at
**2.27e-13px**.

**CONTROL: the Kalman gain never fires — the motion model does all the work.**
The innovation after initialisation is **0.000244px**, which is *exactly one
float32 ULP* at these coordinates (0.000244 at 2316px): `SimpleTracker.step`
downcasts detections with `dtype=np.float32`, and that rounding is the whole
residual. Scaling the measurement noise **1000×** leaves the switch counts and
the ID counts identical.

**CONTROL: the repair stops at `max_age`, not at any property of the motion.** At
a 6-frame gap both arms score **5 switches on 10 IDs** — identical — because
`SimpleTracker` drops a track after `max_age=5` unmatched frames. The predict
step buys gaps 1 to 5 and not one more; past that the comparison saturates and
ranks nothing.

**FINDING: the exactness belongs to the fixture, not to the filter.** On constant
acceleration 0.5 px/frame² the same filter lags by -0.2500, -0.5000, -0.8056,
-1.3051, -1.8403, -2.4169 … **-6.2851px**, growing every step; the second step's
lag is exactly −a = **-0.5000**. `synthetic_frames` moves objects at a fixed
velocity, so this lesson's data is the one case a CV model cannot miss.

### 3 — MOTA is unbounded below; ID-switch count is blind

**ANSWER: the second backend does not exist, so the comparison has one arm.**
`transformers` and `sam2` both raise `ModuleNotFoundError`, and the lesson ships
no 30-second crowd clip and no hand-labelled identities either. The one backend
that runs scores ID switches `[0, 2, 18]` at `[3, 10, 30]` objects, for MOTA
`[1.0, 0.992, 0.26]`.

**FINDING: ID-switch count — the quantity the exercise says to compare on —
ranks nothing.** Emitting *k* identically-placed copies of every hypothesis:

| k | `count_id_switches` | FP | MOTA |
|---|---:|---:|---:|
| 1 | 0 | 0 | **1.000** |
| 2 | 0 | 75 | **0.000** |
| 3 | 0 | 150 | **-1.000** |
| 5 | 0 | 300 | **-3.000** |
| 9 | **0** | 600 | **-7.000** |

A tracker outputting nine boxes per object scores the same 0 as a flawless run.

**MECHANISM: MOTA = 1 − (FN + FP + IDSW)/GT, and FP has no ceiling.** On the
3-object fixture (75 ground-truth boxes) FN and IDSW stay at 0, so MOTA is
**exactly 2 − k** for every k tried. At k=9 that is an "accuracy" of **-700%** —
which is what a sum of unbounded error terms over a fixed count does.

**CONTROL: the lesson's own worst case stays positive.** At 30 objects the
tracker scores FN **83** + FP **454** + IDSW **18** over **750** ground-truth
boxes, for MOTA **0.260** — degraded but positive. Going below zero needs the
three error terms to outnumber GT, which the fixture alone never reaches.

**FINDING: an ID switch and a fragmentation are different events.**

| run | IDSW | FRAG | FN | MOTA | track IDs |
|---|---:|---:|---:|---:|---:|
| crossing at 10 objects | **2** | 0 | 0 | 0.992 | 10 |
| object steps 12px aside for 2 unseen frames | **0** | **1** | 2 | 0.800 | **1** |

MOTA charges the misses and carries no fragmentation term at all; the two
failure modes are invisible to each other's counter.

**CONTROL: on the exercise's metric the fragmented run is indistinguishable from
perfect.** `count_id_switches` returns **0** on it — what the clean 3-object run
(MOTA 1.000) gets — because the track keeps its ID across the hole. Two backends
compared on ID switches would tie while one of them lost its object for 2 of 20
frames.

### A note on file lengths

The three files run 143 / 145 / 145 lines of code, over D14's 120-line target
and under its 150-line ceiling. The overrun is measurement, not machinery:
exercise 1 carries a 60-object control world plus a switch-replay that names the
frame and IoU of each swap, exercise 2 carries a Kalman filter and a three-arm
sweep over six gap lengths, and exercise 3 carries a CLEAR-MOT evaluator
(exclusive Hungarian matching, FN/FP/IDSW/FRAG) that the lesson does not ship.
Exercises 2 and 3 import exercise 1's `steady` and `drive` via
`practice.load_module` rather than carrying second copies.
