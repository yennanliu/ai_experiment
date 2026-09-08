"""Exercise 2 — scale invisible shift bends.

    **(Medium)** Given RGB + depth from Depth Anything V2, lift to a point cloud
    and render with `open3d`. Compare two scenes (indoor / outdoor) and note which
    looks more believable.

Reading of the exercise: `open3d` raises ModuleNotFoundError, so there is no
render, and "looks more believable" is not a criterion anyone can grade. Both
gaps close the same way, because what a render of a pinhole lift can and cannot
show is exact. A relative-depth model is ambiguous up to `a * d + b`, and those
two parameters are not alike under `depth_to_point_cloud`: a pure scale is a
similarity of the cloud -- every one of 62,250 sampled pairwise distances scales
by the same factor, so the render is the old render with the camera
moved -- while a shift is a non-rigid warp that bends straight surfaces. So the
only believability a viewer can judge is the shift, and the shift's damage is
b/z: the same 0.5 m error bends a 1-5 m indoor plane 8.9x harder than a 10-50 m
outdoor one. That is the answer to "which looks more believable", and it is
arithmetic rather than taste. The fixture the lesson ships would not survive
either render: `synthetic_depth`'s floor is linear in image row, and a real
ground plane is linear in *disparity*, so its floor is curved before any depth
model is involved.

Structure: `plane` builds a genuine ground plane by making 1/z linear in the row
index -- a plane through a pinhole is linear in disparity, never in depth;
`flatness` is the SVD condition number sigma3/sigma1 of a point set, zero for a
plane and larger the more it bows; `pairwise` samples the distance matrix on a
stride-37 subsample; `spread` is the max-over-min ratio of those distances
against a reference cloud, which is 1 exactly for a similarity; `lift` applies
one (a, b) to a depth map and pushes it through the lesson's own
`depth_to_point_cloud`; `bending` reports flatness for a scene untouched,
shifted and scaled. `status` is imported from exercise 1 rather than repeated.
`INTR` and the 96-pixel size are `main()`'s own (96, 96, 48, 48) pinhole.
"""

from __future__ import annotations

import pathlib

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "26-monocular-depth"
SIZE, STRIDE, SHIFT = 96, 37, 0.5
INTR = (96.0, 96.0, 48.0, 48.0)
SIMILAR, WARPED = ((2.0, 0.0), (1.7, 0.0)), ((1.0, 1.0), (2.0, 0.7))
SCENES = {"indoor": (1.0, 5.0), "outdoor": (10.0, 50.0)}
VARIANTS = (("true", (1.0, 0.0)), ("shifted", (1.0, SHIFT)), ("scaled", (2.0, 0.0)))

status = practice.load_module(
    pathlib.Path(__file__).with_name("ex01_grayscale_png_erases_metric_scale.py")).status
rows = lambda np: np.tile(np.arange(SIZE)[:, None], (1, SIZE))                    # noqa: E731
affine = lambda np, depth, ab: (ab[0] * depth + ab[1]).astype(np.float32)         # noqa: E731
label = lambda ab: f"a={ab[0]} b={ab[1]}"                                         # noqa: E731
lift = lambda np, ref, depth, ab: ref.depth_to_point_cloud(affine(np, depth, ab), INTR)  # noqa: E731


def plane(np, near, far):
    disparity = (1.0 / near) + (rows(np) / (SIZE - 1)) * ((1.0 / far) - (1.0 / near))
    return (1.0 / disparity).astype(np.float32)


def flatness(np, points) -> float:
    centred = points.reshape(-1, 3) - points.reshape(-1, 3).mean(0)
    values = np.linalg.svd(centred, compute_uv=False)
    return float(values[2] / values[0])


def pairwise(np, points):
    sample = points.reshape(-1, 3)[::STRIDE]
    return np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=-1)


def spread(np, reference, points) -> float:
    base, other = pairwise(np, reference), pairwise(np, points)
    ratio = other[base > 0] / base[base > 0]
    return float(ratio.max() / ratio.min())


def bending(np, ref, depth) -> dict:
    return {name: flatness(np, lift(np, ref, depth, ab)) for name, ab in VARIANTS}


def solve():
    try:
        import numpy as np
        import torch                            # noqa: F401 - the reference imports it
    except ImportError as exc:                  # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    ramp = ref.synthetic_depth(SIZE)
    cloud = ref.depth_to_point_cloud(ramp, INTR)
    return {"renderer": status("open3d"), "pairs": int((pairwise(np, cloud) > 0).sum()),
            "reproject": max(float(np.abs(cloud[..., 0] * INTR[0] / cloud[..., 2] + INTR[2]
                                          - rows(np).T).max()),
                             float(np.abs(cloud[..., 2] - ramp).max())),
            "similar": {label(ab): spread(np, cloud, lift(np, ref, ramp, ab)) for ab in SIMILAR},
            "warped": {label(ab): spread(np, cloud, lift(np, ref, ramp, ab)) for ab in WARPED},
            "ramp": flatness(np, cloud),
            "scenes": {n: bending(np, ref, plane(np, *e)) for n, e in SCENES.items()}}


def verify(result):
    indoor, outdoor = result["scenes"]["indoor"], result["scenes"]["outdoor"]
    similar, warped = result["similar"], result["warped"]
    return [
        practice.Check(
            "ANSWER: there is no renderer, so believability is scored as cloud geometry",
            result["renderer"] == "ModuleNotFoundError" and result["reproject"] == 0.0,
            f"importing open3d gives {result['renderer']}, so no viewer exists to look at. The lift is "
            f"exact: reprojecting the cloud through main()'s own {tuple(int(v) for v in INTR)} pinhole "
            f"returns the pixel grid and the depths to {result['reproject']:.1e}, so all of the below "
            "is a property of the depth map and never of the lift"),
        practice.Check(
            "ANSWER: a scale error is invisible in any render -- it is an exact similarity",
            similar["a=2.0 b=0.0"] == 1.0 and all(abs(v - 1.0) < 1e-5 for v in similar.values()),
            f"re-lifting a rescaled depth map keeps all {result['pairs']:,} sampled pairwise distances "
            "in proportion: max/min " + ", ".join(f"{k} -> {v:.12f}" for k, v in similar.items())
            + f". a=2.0 is binary-exact so it holds to the last bit; a=1.7 drifts "
            f"{similar['a=1.7 b=0.0'] - 1.0:.1e}, the float32 cast and not geometry. The render is the "
            "old render with the camera moved, so no viewer can grade the `a` in a*d + b"),
        practice.Check(
            "MECHANISM: a shift is not a similarity, because x and y carry z as a factor",
            all(v > 1.3 for v in warped.values()),
            "x = (u - cx) * z / fx scales with z, so adding a constant to z moves near points further "
            "than far ones: the same distances now spread "
            + ", ".join(f"{k} -> {v:.4f}x" for k, v in warped.items())
            + ". `align_scale_shift` fits both parameters, so the metric removes the shift a render "
            "could have caught and keeps the scale it could not"),
        practice.Check(
            "ANSWER: the outdoor scene looks more believable, by a factor of 8.9",
            outdoor["shifted"] < indoor["shifted"] / 5,
            f"the same {SHIFT} m shift bends a {SCENES['indoor']} m plane to flatness "
            f"{indoor['shifted']:.5f} and a {SCENES['outdoor']} m plane to {outdoor['shifted']:.5f} -- "
            f"{indoor['shifted'] / outdoor['shifted']:.1f}x. No indoor *content* is harder; the damage "
            "is b/z, so the farther scene hides the identical error"),
        practice.Check(
            "CONTROL: the scaled clouds stay planar to 1e-8, so the ranking is the shift alone",
            max(indoor["scaled"], outdoor["scaled"]) < 1e-6,
            f"both planes lift flat -- indoor {indoor['true']:.3e}, outdoor {outdoor['true']:.3e} on "
            f"sigma3/sigma1 -- and doubling the depth leaves them at {indoor['scaled']:.3e} and "
            f"{outdoor['scaled']:.3e}. The gap above is neither a lifting artefact nor a scale effect"),
        practice.Check(
            "FINDING: the lesson's own fixture is not a scene a pinhole camera could see",
            result["ramp"] > 0.15,
            f"`synthetic_depth` makes depth linear in image row, which lifts to a curved surface: "
            f"flatness {result['ramp']:.6f}, five orders off the {indoor['true']:.1e} of a real plane, "
            "whose *disparity* is what is linear. Its floor would bow in open3d whatever produced it"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
