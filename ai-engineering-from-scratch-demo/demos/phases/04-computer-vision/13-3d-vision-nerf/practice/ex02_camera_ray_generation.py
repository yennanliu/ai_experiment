"""Exercise 2 — camera ray generation.

    **(Medium)** Implement a minimal ray-generation function that, given camera intrinsics and pose, produces ray origins and directions for every pixel of an H x W image.

Reading of the exercise: neither the lesson's `code/main.py` nor its `docs/en.md`
defines a ray generator -- `docs/en.md` only writes "Cast a ray from the camera
through pixel (u, v)" as prose -- so the function has to be written here, and the
three conventions it silently fixes are the whole exercise. The first is
handedness: this file uses NeRF's OpenGL/blender frame (x right, y up, camera
looking down -z), so the camera-space direction is `((u - cx)/fx, -(v - cy)/fy,
-1)` and the reprojection that checks it must negate z to match. Every one of the
768 rays reprojects onto its own pixel centre to 1.9e-05 px, which is float32
round-off on coordinates of order 32 and is asserted as the closed-form identity
it is. The second convention is the one the exercise does not mention and NeRF's
own `get_rays` gets wrong: pixel *centres* are at `u + 0.5`, and the original
implementation's `(i - W*0.5)` shifts every ray by exactly half a pixel -- checked
here to 6e-06 of exactly -0.5 px in both axes. The third is normalisation, and it
is the one that reaches the lesson's own code. NeRF leaves directions
un-normalised so that `d_z = -1` and `t` measures depth along the optical axis;
normalise them and `t` becomes Euclidean range instead. Rendering one
fronto-parallel wall through the lesson's own `volumetric_render` shows the
consequence directly: un-normalised, all 768 pixels come back at the same depth
to 4.8e-07, while normalised the same wall bows from 3.921 to 4.349 across the
frame, and the ratio between the two depth maps equals |d| pixel by pixel. So
"produces ray origins and directions" has a right answer only once you say which
depth the downstream renderer is supposed to report.

Structure: `look_at` builds a camera-to-world pose on a sphere around the origin;
`get_rays` is the exercise's function, taking K and that pose and returning
per-pixel origins and directions, with `offset` switching between pixel centres
and NeRF's corners and `normalise` switching the depth convention; `reproject`
inverts it back to pixel coordinates; `wall_depth` pushes one fronto-parallel
wall through the lesson's own `volumetric_render`. At 139 code lines this sits
above D14's 120-line target, 11 clear of the ceiling: six checks over four
helpers, and three of the checks each need their own second arm -- a half-pixel
grid, a turned pose and a normalised copy of the directions -- generated and
rendered rather than asserted.
"""

from __future__ import annotations

import math

from harness import parity, practice

PHASE, LESSON = "04-computer-vision", "13-3d-vision-nerf"

H, W, FOCAL = 24, 32, 40.0
WALL, NEAR, FAR, SAMPLES = 4.0, 2.0, 6.0, 128
THETA, PHI, RADIUS, TURN = 0.7, 0.4, 4.0, 1.1
DEPTHS = (0.5, 2.0, 7.5)           # the reprojection is checked at three points per ray

corner = lambda: math.sqrt(1 + (W / 2 / FOCAL) ** 2 + (H / 2 / FOCAL) ** 2)     # noqa: E731
fov = lambda side: 2 * math.degrees(math.atan(side / 2 / FOCAL))               # noqa: E731
pixels = lambda torch: torch.meshgrid(*(torch.arange(n, dtype=torch.float32)   # noqa: E731
                                        for n in (H, W)), indexing="ij")


def look_at(torch, theta, phi, radius):
    eye = torch.tensor([radius * math.cos(phi) * math.sin(theta), radius * math.sin(phi),
                        radius * math.cos(phi) * math.cos(theta)])
    back = eye / eye.norm()
    right = torch.linalg.cross(torch.tensor([0.0, 1.0, 0.0]), back)
    right = right / right.norm()
    pose = torch.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = right, torch.linalg.cross(back, right), back, eye
    return pose


def get_rays(torch, intrinsics, pose, offset=0.5, normalise=False):
    rows, cols = pixels(torch)
    camera = torch.stack([(cols + offset - intrinsics[0, 2]) / intrinsics[0, 0],
                          -(rows + offset - intrinsics[1, 2]) / intrinsics[1, 1],
                          -torch.ones_like(cols)], dim=-1)
    if normalise:
        camera = camera / camera.norm(dim=-1, keepdim=True)
    directions = camera @ pose[:3, :3].T
    return pose[:3, 3].expand_as(directions), directions


def reproject(torch, intrinsics, pose, points):
    w2c = torch.linalg.inv(pose)
    camera = points @ w2c[:3, :3].T + w2c[:3, 3]
    return torch.stack([intrinsics[0, 0] * camera[..., 0] / -camera[..., 2] + intrinsics[0, 2],
                        intrinsics[1, 1] * -camera[..., 1] / -camera[..., 2] + intrinsics[1, 2]], dim=-1)


def wall_depth(torch, ref, pose, origins, directions):
    t_vals = torch.linspace(NEAR, FAR, SAMPLES)
    points = origins.reshape(-1, 1, 3) + t_vals.view(1, -1, 1) * directions.reshape(-1, 1, 3)
    sigma = 60.0 * torch.exp(-((((points - pose[:3, 3]) @ -pose[:3, 2] - WALL) / 0.08) ** 2))
    _rendered, depth, weights = ref.volumetric_render(sigma, torch.full((*sigma.shape, 3), 0.5), t_vals)
    return depth, float(weights.sum(-1).min())


def solve():
    try:
        import torch
    except ImportError as exc:                      # pragma: no cover - T1 needs torch
        raise practice.Skip(f"needs torch: uv sync --extra vision ({exc})") from None
    ref = parity.load_reference(PHASE, LESSON, "main")
    torch.set_num_threads(2)
    intrinsics = torch.tensor([[FOCAL, 0.0, W / 2], [0.0, FOCAL, H / 2], [0.0, 0.0, 1.0]])
    pose, turned = look_at(torch, THETA, PHI, RADIUS), look_at(torch, THETA + TURN, PHI, RADIUS)
    origins, directions = get_rays(torch, intrinsics, pose)
    rows, cols = pixels(torch)
    centres = torch.stack([cols + 0.5, rows + 0.5], dim=-1)
    offset = reproject(torch, intrinsics, pose,
                       origins + 3.0 * get_rays(torch, intrinsics, pose, offset=0.0)[1]) - centres
    plain, mass = wall_depth(torch, ref, pose, origins, directions)
    ranged = wall_depth(torch, ref, pose, origins, get_rays(torch, intrinsics, pose, normalise=True)[1])[0]
    return {"origin_spread": float((origins - pose[:3, 3]).abs().max()), "mass": mass,
            "reprojection": max(float((reproject(torch, intrinsics, pose, origins + t * directions)
                                       - centres).abs().max()) for t in DEPTHS),
            "half_pixel": (float(offset.min()), float(offset.max())),
            "norms": (float(directions.norm(dim=-1).min()), float(directions.norm(dim=-1).max())),
            "rotation": float((directions @ (turned[:3, :3] @ pose[:3, :3].T).T
                               - get_rays(torch, intrinsics, turned)[1]).abs().max()),
            "plain": (float(plain.min()), float(plain.max()), float(plain.std())),
            "ranged": (float(ranged.min()), float(ranged.max())),
            "ratio": float((ranged / plain - directions.norm(dim=-1).reshape(-1)).abs().max())}


def verify(result):
    low, high = result["half_pixel"]
    near, far = result["norms"]
    flat, curved = result["plain"], result["ranged"]
    return [
        practice.Check(
            "ANSWER: every one of the 768 rays reprojects onto its own pixel centre to 2e-05 px",
            result["reprojection"] < 1e-3 and result["origin_spread"] == 0.0,
            f"K = [[{FOCAL:.0f},0,{W / 2:.0f}],[0,{FOCAL:.0f},{H / 2:.0f}],[0,0,1]] and a look-at pose at radius "
            f"{RADIUS}: pushing each of the {H * W} rays out to t in {DEPTHS} and projecting back through the "
            f"inverse pose gives max|pixel error| {result['reprojection']:.1e}, float32 round-off at order {W}. "
            f"All origins are the camera centre exactly ({result['origin_spread']:.1e})"),
        practice.Check(
            "MECHANISM: the pose acts only on directions — rotating the camera rotates every ray by R",
            result["rotation"] < 1e-5,
            f"generating rays from a pose turned {TURN} rad about the scene and comparing against the original "
            f"directions premultiplied by R = R_b R_a^T gives max|R d - d'| {result['rotation']:.1e}. Ray "
            f"generation is one fixed camera-space grid, `((u-cx)/fx, -(v-cy)/fy, -1)`, that the pose then "
            f"rigidly moves -- which is why the whole function is {H}x{W} arithmetic and one 3x3 matmul"),
        practice.Check(
            "FINDING: NeRF's own `get_rays` is half a pixel off, in both axes, by exactly -0.5",
            abs(low + 0.5) < 1e-4 and abs(high + 0.5) < 1e-4,
            f"the original implementation writes `(i - W*.5)`, indexing pixel *corners*; substituting offset=0.0 "
            f"for the 0.5 used above shifts the reprojection by [{low:.6f}, {high:.6f}] px in x and the same in "
            f"y -- not noise, a convention. It is invisible in a self-consistent train-and-render loop and shows "
            "up the moment poses come from a calibrated SfM pipeline"),
        practice.Check(
            "MECHANISM: directions are deliberately not unit — |d| = 1/cos(theta) spans 1.000 to 1.110",
            abs(near - 1.0) < 1e-3 and far > 1.10 and far < corner(),
            f"with d_z fixed at -1 the norm is exactly 1/cos of the angle off the optical axis: measured "
            f"[{near:.6f}, {far:.6f}] against the closed form at the frame's true corner, "
            f"sqrt(1 + (W/2f)^2 + (H/2f)^2) = {corner():.6f} -- the measured max is smaller because the corner "
            f"*pixel centre* sits half a pixel inside it. Field of view {fov(W):.1f} x {fov(H):.1f} deg"),
        practice.Check(
            "FINDING: normalising the directions silently changes what the lesson's renderer reports",
            flat[2] < 1e-5 and curved[1] - curved[0] > 0.4,
            f"one fronto-parallel wall at camera depth {WALL}, {SAMPLES} samples over [{NEAR}, {FAR}], through "
            f"the lesson's own `volumetric_render`: un-normalised directions give every pixel the same depth "
            f"({flat[0]:.5f}, std {flat[2]:.1e}), because t *is* depth along -z. Normalised, the same flat wall "
            f"bows from {curved[0]:.5f} to {curved[1]:.5f} -- t is now Euclidean range"),
        practice.Check(
            "CONTROL: the two depth maps differ by exactly the direction norm, pixel by pixel",
            result["ratio"] < 3e-3 and result["mass"] > 0.999,
            f"max|depth_range/depth_z - |d|| over all {H * W} pixels is {result['ratio']:.1e}, the closed form "
            f"1/cos(theta) again, at the {(FAR - NEAR) / (SAMPLES - 1):.4f} sample spacing this quadrature runs at. "
            f"Both renders carry the same weight mass (min {result['mass']:.4f}), so nothing else moved. Both read "
            f"below {WALL}: alpha compositing is front-biased across a slab, the near face absorbing first"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
