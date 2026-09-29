"""Exercise 1 — sharding by .weight/.bias puts 95% of the parameters in one shard, and meta.pt still outweighs every shard together 3 to 1.

    Replace round-robin sharding with sharding by parameter group (layers ending in `.weight` vs `.bias`). When is each layout preferable?

Reading of the exercise: the layout function the lesson's
`save_sharded_checkpoint` calls, `shard_keys_by_prefix`, is swapped for one
that sends every key ending in `.weight` to shard 0 and every `.bias` to
shard 1, and the lesson's own `run_resume_demo(sharded=True)` is run with
each layout on its default model (16 -> 24 -> 24 -> 4 MLP, 24 steps,
interrupted at 10). Both layouts are then measured on what a layout is
for: how the parameters and the bytes split across files.

**ANSWER: both layouts resume exactly (max loss diff 0.0), so the choice is
about who reads the files.** By group, the three `.weight` tensors (1,056
parameters, 95.3%) land in one shard and the three `.bias` tensors (52) in
the other. Any reader that wants one group -- bias-only fine-tuning, a
weights-only quantiser, per-group weight decay -- opens one file; under
round robin both groups are spread over all 3 shards, so every such reader
opens every file. Round robin is the layout for parallel reads of the whole
model, but it deals out keys, not bytes: its shards hold 600 / 388 / 120
parameters (5.0x largest to smallest), against 1,056 / 52 (20.3x) by group.
With few, very unequal tensors neither layout balances; that needs a
byte-aware split.

**FINDING: sharding the model splits the smaller part of the checkpoint.**
The lesson says "the meta is small and reads first". Its `meta.pt` holds the
AdamW moments (two copies of every parameter) and the RNG state, and it is
3.1x the size of all model shards together under round robin (33,733 vs
10,895 bytes) and 3.6x under the group layout. The RNG state alone pickles
to about 19 KB, 2.7x the 7 KB model state_dict, because the torch CPU state
is stored as a Python list of 5,056 ints instead of a byte tensor.

Expected output: two PASS checks.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

from harness import parity, practice

try:
    import torch  # noqa: F401  -- the lesson's code needs it
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "47-checkpoint-save-resume"


def by_group(state_dict, num_shards):
    """Shard 0 holds every `.weight`, shard 1 every `.bias` (and anything else)."""
    keys = sorted(state_dict)
    weights = [k for k in keys if k.endswith(".weight")]
    return {0: weights, 1: [k for k in keys if k not in weights]}


def measure(ref, layout, num_shards):
    original = ref.shard_keys_by_prefix
    ref.shard_keys_by_prefix = layout
    try:
        with tempfile.TemporaryDirectory(prefix="ex01-") as tmp:
            out = ref.run_resume_demo(ckpt_dir=Path(tmp), total_steps=24, interrupt_at=10,
                                      sharded=True, num_shards=num_shards)
            shards = [torch.load(p, weights_only=False) for p in sorted(Path(tmp).glob("model.shard-*.pt"))]
            sizes = {p.name: p.stat().st_size for p in Path(tmp).iterdir()}
    finally:
        ref.shard_keys_by_prefix = original
    shard_bytes = sum(v for k, v in sizes.items() if k.startswith("model.shard"))
    return {"diff": out["max_loss_diff_after_resume"], **describe(shards),
            "shard_bytes": shard_bytes, "meta_bytes": sizes["meta.pt"]}


def describe(shards):
    keys = [s["keys"] for s in shards]
    return {
        "keys": keys,
        "files_per_group": [sum(any(k.endswith(g) for k in ks) for ks in keys) for g in (".weight", ".bias")],
        "numel": [sum(t.numel() for t in s["tensors"].values()) for s in shards],
    }


def saved_size(obj):
    buf = io.BytesIO()
    torch.save(obj, buf)
    return len(buf.getvalue())


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return {"round_robin": measure(ref, ref.shard_keys_by_prefix, 3), "group": measure(ref, by_group, 2),
            "rng_bytes": saved_size(ref.capture_rng_state()), "model_bytes": saved_size(ref.make_model(16, 24, 4).state_dict())}


def verify(result):
    rr, gr = result["round_robin"], result["group"]
    ratio = [r["meta_bytes"] / r["shard_bytes"] for r in (rr, gr)]
    return [
        practice.Check(
            "ANSWER: both layouts resume exactly; by group one file per group, round robin spreads both over all",
            (rr["diff"], gr["diff"]) == (0.0, 0.0)
            and gr["keys"] == [["0.weight", "2.weight", "4.weight"], ["0.bias", "2.bias", "4.bias"]]
            and (rr["files_per_group"], gr["files_per_group"]) == ([3, 3], [1, 1])
            and (rr["numel"], gr["numel"]) == ([600, 388, 120], [1056, 52]),
            f"loss diff {rr['diff']} / {gr['diff']}; files per (weight, bias) group {rr['files_per_group']} "
            f"round robin vs {gr['files_per_group']}; numel {rr['numel']} vs "
            f"{gr['numel']}",
        ),
        practice.Check(
            "FINDING: meta.pt, 'small', outweighs all model shards together about 3 to 1",
            ratio[0] > 2.5 and ratio[1] > 3.0 and result["rng_bytes"] > 2 * result["model_bytes"],
            f"meta {rr['meta_bytes']} B vs shards {rr['shard_bytes']} B ({ratio[0]:.1f}x) round robin, "
            f"{ratio[1]:.1f}x by group; RNG state {result['rng_bytes']} B vs model {result['model_bytes']} B",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
