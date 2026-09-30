"""Exercise 5 — the lesson's loader accepts any schema starting with "ckpt", and a renamed field fails after the weights are already overwritten.

    Implement a `migrate_v1_to_v2` function that adds a new field to the payload and bumps the schema string. Make load tolerate both versions.

Reading of the exercise: the new field is the one v1 is missing to resume
safely -- `loader`, the data-order contract `train_until` hardcodes or
takes from the caller (`seed_base` 12345 and `batches_per_epoch`).
`migrate_v1_to_v2` adds it and bumps `ckpt.v1` to `ckpt.v2`; `load_any`
dispatches on the schema string through a table of migrations, rejects
unknown versions, upgrades a v1 file in place atomically with the lesson's
`atomic_save`, checks the contract, and restores through the lesson's own
`load_checkpoint`. A v1 file from the lesson's demo (seed 11, interrupted
at step 12, 5 batches per epoch) is resumed to step 24 through `load_any`,
then resumed again from the v2 file it was upgraded into, and both are
compared with the uninterrupted run.

**ANSWER: `migrate_v1_to_v2` and `load_any` below; both versions resume
exactly.** The v1 file loads, is rewritten as `ckpt.v2`, and resumes with a
max loss diff of 0.0; the v2 file then loads unchanged, again with 0.0. The
new field is enforced: loading with 6 batches per epoch instead of the
recorded 5 raises ValueError. Without it, the lesson's v1 loader takes the
same wrong setting silently: the resumed run matches for 3 steps, then
diverges at step 15 (0-based) by up to 0.4375 in loss.

**FINDING: the lesson's loader does not dispatch on the schema.** The doc
says "future format changes bump the version string and the loader
dispatches". `load_checkpoint` only asserts `schema.startswith("ckpt")`, so
it loads `ckpt.v9`, `ckptX` and the shard schema `ckpt-shard.v1` as if they
were v1; `load_any` rejects all three. A `ckpt.v3` payload that renames
`state` to `train_state` passes the check and fails with KeyError, after
the model's weights have already been overwritten: the caller's model is
left half-restored.

**FINDING: the checkpoint cannot be read by torch's default loader.**
`torch.load` in torch 2.6 and later defaults to `weights_only=True`, and
refuses the lesson's file with UnpicklingError: the NumPy and Python RNG
states are not tensors. The lesson loads with `weights_only=False`, which
runs arbitrary pickle code from the file.

Expected output: three PASS checks.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "47-checkpoint-save-resume"
DIMS = {"batch_size": 4, "in_dim": 16, "out_dim": 4}


def migrate_v1_to_v2(payload, batches_per_epoch):
    """v1 -> v2: record the data-order contract the resume depends on."""
    if payload["schema"] != "ckpt.v1":
        raise ValueError(f"not a v1 payload: {payload['schema']!r}")
    return {**payload, "schema": "ckpt.v2", "loader": {"seed_base": 12345, "batches_per_epoch": batches_per_epoch}}


MIGRATIONS = {"ckpt.v1": migrate_v1_to_v2, "ckpt.v2": lambda payload, _bpe: payload}


def load_any(ref, path, model, opt, sched, batches_per_epoch):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") not in MIGRATIONS:
        raise ValueError(f"unknown checkpoint schema {payload.get('schema')!r}")
    upgraded = MIGRATIONS[payload["schema"]](payload, batches_per_epoch)
    if upgraded["loader"]["batches_per_epoch"] != batches_per_epoch:
        raise ValueError(f"checkpoint was written with {upgraded['loader']['batches_per_epoch']} batches per epoch")
    if upgraded is not payload:
        ref.atomic_save(upgraded, path)
    return ref.load_checkpoint(path, model, opt, sched)


def fresh(ref):
    ref.seed_everything(11)
    model = ref.make_model(16, 24, 4)
    return (model, *ref.make_optimizer_and_scheduler(model, lr=0.01, total_steps=24))


def resume(ref, path, full, bpe=5, loader=None):
    model, opt, sched = fresh(ref)
    state = loader(model, opt, sched) if loader else ref.load_checkpoint(path, model, opt, sched)
    ref.train_until(model, opt, sched, torch.nn.CrossEntropyLoss(), state, stop_step=24, batches_per_epoch=bpe, **DIMS)
    diffs = [abs(a - b) for a, b in zip(full[12:], state.losses[12:], strict=True)]
    return max(diffs), next((12 + i for i, d in enumerate(diffs) if d > 0), None)


def outcome(fn):
    try:
        fn()
        return "loaded"
    except Exception as exc:  # noqa: BLE001 -- the exception type is the measurement
        return type(exc).__name__


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    out = {}
    with tempfile.TemporaryDirectory(prefix="ex05-") as tmp:
        v1 = Path(tmp) / "ckpt.pt"
        full = ref.run_resume_demo(ckpt_dir=Path(tmp), total_steps=24, interrupt_at=12)["full_losses"]
        base = torch.load(v1, weights_only=False)
        out["default_load"] = outcome(lambda: torch.load(v1))
        out["wrong_bpe_v1"] = resume(ref, v1, full, bpe=6)
        out["v1"] = resume(ref, v1, full, loader=lambda m, o, s: load_any(ref, v1, m, o, s, 5))
        out["upgraded_schema"] = torch.load(v1, weights_only=False)["schema"]
        out["v2"] = resume(ref, v1, full, loader=lambda m, o, s: load_any(ref, v1, m, o, s, 5))
        out["wrong_bpe_v2"] = outcome(lambda: load_any(ref, v1, *fresh(ref), 6))
        accepts = {}
        for schema in ("ckpt.v9", "ckptX", "ckpt-shard.v1"):
            ref.atomic_save({**base, "schema": schema}, Path(tmp) / "odd.pt")
            accepts[schema] = (outcome(lambda: ref.load_checkpoint(Path(tmp) / "odd.pt", *fresh(ref))),
                               outcome(lambda: load_any(ref, Path(tmp) / "odd.pt", *fresh(ref), 5)))
        renamed = {k: v for k, v in base.items() if k != "state"} | {"schema": "ckpt.v3", "train_state": base["state"]}
        ref.atomic_save(renamed, Path(tmp) / "v3.pt")
        model, opt, sched = fresh(ref)
        before = [p.detach().clone() for p in model.parameters()]
        out["v3_lesson"] = outcome(lambda: ref.load_checkpoint(Path(tmp) / "v3.pt", model, opt, sched))
        out["v3_mutated"] = any(not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))
    out["accepts"] = accepts
    return out


def verify(result):
    r = result
    diff, first = r["wrong_bpe_v1"]
    return [
        practice.Check(
            "ANSWER: v1 and v2 both resume with diff 0.0, and the new field refuses a wrong batches_per_epoch",
            (r["v1"], r["upgraded_schema"], r["v2"], r["wrong_bpe_v2"]) == ((0.0, None), "ckpt.v2", (0.0, None), "ValueError")
            and first == 15 and abs(diff - 0.4375) < 1e-3,
            f"v1 {r['v1']}, upgraded to {r['upgraded_schema']}, v2 {r['v2']}; 6 batches/epoch: v2 {r['wrong_bpe_v2']}, "
            f"v1 silently diverges from step {first} by {diff:.4f}",
        ),
        practice.Check(
            "FINDING: the lesson's loader takes any 'ckpt*' schema, and a renamed field half-restores the model",
            set(r["accepts"].values()) == {("loaded", "ValueError")} and (r["v3_lesson"], r["v3_mutated"]) == ("KeyError", True),
            f"(lesson, load_any) per schema {r['accepts']}; ckpt.v3 with train_state: {r['v3_lesson']}, "
            f"model weights changed: {r['v3_mutated']}",
        ),
        practice.Check(
            "FINDING: torch.load's default weights_only=True refuses the lesson's checkpoint",
            r["default_load"] == "UnpicklingError",
            f"torch {torch.__version__}: {r['default_load']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
