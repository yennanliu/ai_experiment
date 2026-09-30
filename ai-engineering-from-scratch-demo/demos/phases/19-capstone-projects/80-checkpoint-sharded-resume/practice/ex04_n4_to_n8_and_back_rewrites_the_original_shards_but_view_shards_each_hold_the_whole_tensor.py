"""Exercise 4 — N=4 to N=8 and back rewrites the original shards byte for byte, but re-sharding with views writes the whole tensor into every shard.

    Add a cross-world-size load: shard rebalance from N=4 to N=8 by reading the manifest, concatenating, and re-sharding.

Reading of the exercise: `reshard` parses the lesson's `manifest.json` with
its own `ShardManifest`, checks that `param_shard_offset` /
`param_shard_numel` tile the flat tensor in rank order, loads every shard
through the lesson's `load_sharded` (so each sha256 is still checked),
concatenates `param_shard`, `m_shard` and `v_shard` in offset order, splits
them into N pieces with `torch.tensor_split`, and writes the result with the
lesson's `save_sharded`. The state is the lesson's 4 demo ranks
(`make_demo_state`) after 3 elementwise Adam-style steps, so the optimiser
moments are not the demo's constant zeros and 1e-6.

**ANSWER: the rebalance is exact.** The lesson's loader refuses the N=4
checkpoint at N=8 ("world_size mismatch: manifest=4, expected=8"), which is
why the rebalance has to go through the manifest. The N=8 checkpoint has
offsets 0, 512, ..., 3584 with 512 values each, and its concatenated param,
m and v are equal to the N=4 ones. One more optimiser step at N=8 gives the
same flat state as the same step at N=4, because the Adam update is
elementwise. Re-sharding N=8 back to N=4 writes files with the same sha256
as the original four.

**FINDING: re-shard with views and every shard holds the whole model.**
`tensor_split` returns views, and `torch.save` writes a view's whole
underlying storage. Without `.clone()` the 8 shards take 408,840 bytes
instead of 64,776, 6.3x, and a loaded shard has 512 values over a storage
of 4,096. `load_sharded` accepts them: the manifest's numel is still 512, and
the sha256 covers whatever bytes were written.

**FINDING: the offsets are never checked.** Set rank 1's
`param_shard_offset` to 0, so two shards claim [0, 1024), and the lesson's
`load_sharded` still accepts the checkpoint. The manifest has no checksum of
its own, and the loader never reads the offsets. A rebalance that trusts
them puts shards in the wrong place. The tiling check in `reshard` rejects
this one ("rank 1 offset 0 != 1024").

Expected output: three PASS checks.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "80-checkpoint-sharded-resume"
KEYS = ("param_shard", "m_shard", "v_shard")


def adam_step(per_rank):
    """Elementwise Adam-style update, so any sharding of the flat state gives the same result."""
    for s in per_rank:
        g = torch.sin(s["param_shard"])
        s["m_shard"].mul_(0.9).add_(g, alpha=0.1)
        s["v_shard"].mul_(0.999).addcmul_(g, g, value=0.001)
        s["param_shard"].sub_(0.01 * s["m_shard"] / (s["v_shard"].sqrt() + 1e-8))
        s["step"] += 1


def flat(per_rank, key):
    return torch.cat([s[key] for s in per_rank])


def check_tiling(manifest):
    """Offsets must tile [0, total) in rank order; load_sharded never looks at them."""
    at = 0
    for entry in sorted(manifest.shards, key=lambda e: e.rank):
        if entry.param_shard_offset != at:
            raise ValueError(f"rank {entry.rank} offset {entry.param_shard_offset} != {at}")
        at += entry.param_shard_numel


def reshard(ref, src, dst, new_world, clone=True):
    """Read the manifest, verify and concatenate every shard, split into new_world, save atomically."""
    manifest = ref.ShardManifest.from_json(Path(src, ref.MANIFEST_NAME).read_text())
    check_tiling(manifest)
    _, states = ref.load_sharded(src, manifest.world_size)
    pieces = {k: torch.tensor_split(flat(states, k), new_world) for k in KEYS}
    new = [{"rank": r, "world_size": new_world, **{k: pieces[k][r].clone() if clone else pieces[k][r] for k in KEYS},
            "step": states[0]["step"]} for r in range(new_world)]  # the lesson's key order, so bytes can match
    return ref.save_sharded(new, dst, manifest.step, manifest.wall_clock_seconds)


def refused(fn, *args):
    try:
        fn(*args)
    except Exception as exc:  # noqa: BLE001 - the message is the measurement
        return str(exc)
    return "accepted"


def rebalance(ref, tmp, states):
    m8 = reshard(ref, f"{tmp}/n4", f"{tmp}/n8", 8)
    _, s8 = ref.load_sharded(f"{tmp}/n8", 8)
    m4 = reshard(ref, f"{tmp}/n8", f"{tmp}/back4", 4)
    same_files = [e.sha256 for e in m4.shards] == [e.sha256 for e in ref.load_sharded(f"{tmp}/n4", 4)[0].shards]
    out = {"direct": refused(ref.load_sharded, f"{tmp}/n4", 8), "same_files": same_files,
           "layout": [(e.param_shard_offset, e.param_shard_numel) for e in m8.shards],
           "equal": all(torch.equal(flat(s8, k), flat(states, k)) for k in KEYS)}
    for per_rank in (states, s8):
        adam_step(per_rank)
    out["step_equal"] = all(torch.equal(flat(s8, k), flat(states, k)) for k in KEYS)
    return out


def hazards(ref, tmp):
    reshard(ref, f"{tmp}/n4", f"{tmp}/views", 8, clone=False)
    shard = ref.load_sharded(f"{tmp}/views", 8)[1][0]["param_shard"]
    out = {"bytes": [sum(p.stat().st_size for p in Path(tmp, d).glob("rank*.bin")) for d in ("n4", "n8", "views")],
           "view": (shard.numel(), shard.untyped_storage().nbytes() // shard.element_size())}
    mpath = Path(f"{tmp}/n4", "manifest.json")
    doc = json.loads(mpath.read_text())
    doc["shards"][1]["param_shard_offset"] = 0
    mpath.write_text(json.dumps(doc))
    out["overlap"] = (refused(ref.load_sharded, f"{tmp}/n4", 4), refused(reshard, ref, f"{tmp}/n4", f"{tmp}/bad", 8))
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    states = [ref.make_demo_state(r, 4) for r in range(4)]
    for _ in range(3):
        adam_step(states)
    with tempfile.TemporaryDirectory(prefix="ex04-") as tmp:
        ref.save_sharded(states, f"{tmp}/n4", step=103)
        return {**rebalance(ref, tmp, states), **hazards(ref, tmp)}


def verify(result):
    r, (b4, b8, bv) = result, result["bytes"]
    return [
        practice.Check(
            "ANSWER: N=4 -> N=8 rebalances exactly, trains identically, and N=8 -> N=4 rewrites the original bytes",
            r["direct"] == "world_size mismatch: manifest=4, expected=8" and r["equal"] and r["step_equal"]
            and r["same_files"] and r["layout"] == [(512 * i, 512) for i in range(8)],
            f"lesson loader at N=8: {r['direct']!r}; resharded layout {r['layout'][:2]}...; flat param/m/v equal "
            f"{r['equal']}; equal after one more step {r['step_equal']}; 8->4 sha256 identical {r['same_files']}",
        ),
        practice.Check(
            "FINDING: re-sharding with views writes the whole flat tensor into every one of the 8 shards",
            r["view"] == (512, 4096) and bv > 5 * b8 and abs(b8 - b4) < 0.2 * b4,
            f"shard bytes N=4 {b4:,}, N=8 cloned {b8:,}, N=8 views {bv:,} ({bv / b8:.1f}x); "
            f"a loaded view shard has numel {r['view'][0]} over a storage of {r['view'][1]}",
        ),
        practice.Check(
            "FINDING: load_sharded never reads param_shard_offset, so two shards claiming offset 0 load",
            r["overlap"] == ("accepted", "rank 1 offset 0 != 1024"),
            f"lesson loader: {r['overlap'][0]}; tiling check in reshard: {r['overlap'][1]}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
