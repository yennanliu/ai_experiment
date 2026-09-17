"""Exercise 4 — the only way to know a stage's output hash is to run the stage.

    Build a partial rollback. Simulate a failure at stage 09 (CAI), then re-run
    stages 09 through 12 while leaving 01-08 cached. The orchestrator should
    detect the cached artifacts by hash and skip them. Measure wall-clock saved
    versus full re-run.

Reading of the exercise: "detect the cached artifacts by hash" is taken at its
word and then rejected, because the only hash a stage has is its *output* hash
and `store.put(simulate_stage(...))` is what produces it. A cache keyed on the
output can never skip the work; the orchestrator here keys on the inputs --
`(name, stage_type, input_hashes, seed)` -- which is exactly the payload
`simulate_stage` serialises, so the cache key and the cached blob are the same
object. Wall clock is read from the manifest, since `simulate_stage` returns it
from a table rather than measuring it.

**ANSWER: 30,720 of 37,050 seconds, 82.9%.** Caching 01-08 and re-running 09-12
skips 8 stages costing 8.5 hours of the run's 10.3, and re-executes 4 costing
1.8. The saving is exact rather than approximate, because every wall-clock in
this pipeline is a `cost_table` lookup keyed on `stage_type` alone.

**FINDING: "detect by hash" is circular.** A stage's artifact is addressed by
`sha256(blob)` and `blob` is what the stage produces. To ask the store whether
the artifact is present you must already have run the stage. The lookup that
makes rollback work is on the *inputs*, and the inputs are the one thing
`simulate_stage` writes into the blob -- so the correct cache key is a
re-serialisation of the answer.

**FINDING: there is nothing for a rollback to invalidate.** Two fresh runs
produce the same twelve output hashes, so re-running stage 09 after a failure
produces the artifact that was already there. The scenario the exercise
describes -- a failure at 09, a fix, a re-run -- cannot change a single byte
downstream unless the seed changes, and if the seed changes, stage 01 changes
too and nothing is cached.

**MECHANISM: `run` has no cache at all.** Its loop is
`blob, wall, cost = simulate_stage(...)` then `store.put(blob)`, unconditionally,
twelve times. `ArtifactStore.has` exists and `run` never calls it; the only
early exit is the budget check. The rollback is not an extension of the
orchestrator, it is the first version of a feature the orchestrator advertises
by having a content-addressed store.

Structure: `cached_run` is the orchestrator with an input-keyed skip; `saved`
compares its per-stage wall clocks against the full run's.
"""

from __future__ import annotations

import json

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "13-building-complete-llm-pipeline"
FAILED_AT = "09_cai_grpo_policy"


def cache_key(name, stage_type, input_hashes, seed):
    """What a working cache keys on -- and, verbatim, what `simulate_stage` serialises."""
    return json.dumps({"stage": name, "type": stage_type,
                       "inputs": input_hashes, "seed": seed}, sort_keys=True)


def cached_run(ref, manifest, store, done):
    """The lesson's loop plus the skip it lacks: an input-keyed lookup into `done`."""
    name_to_hash, skipped = {}, []
    for name, deps, stage_type in ref.STAGES:
        inputs = [name_to_hash[d] for d in deps]
        key = cache_key(name, stage_type, inputs, manifest.seed)
        if key in done and store.has(done[key]):
            name_to_hash[name] = done[key]
            skipped.append(name)
            manifest.stages.append(ref.StageRecord(
                name=name, stage_type=stage_type, input_hashes=inputs,
                output_hash=done[key], wall_clock_sec=0.0, cost_usd=0.0, status="cached"))
            continue
        blob, wall, cost = ref.simulate_stage(name, stage_type, inputs, manifest.seed)
        name_to_hash[name] = store.put(blob)
        manifest.total_cost_usd += cost
        manifest.stages.append(ref.StageRecord(
            name=name, stage_type=stage_type, input_hashes=inputs,
            output_hash=name_to_hash[name], wall_clock_sec=wall, cost_usd=cost, status="ok"))
    return manifest, skipped


def warm(ref, store, upto):
    """Run the pipeline once and keep the input-keyed artifacts for stages before `upto`."""
    manifest, name_to_hash, done = ref.Manifest(), {}, {}
    for name, deps, stage_type in ref.STAGES:
        inputs = [name_to_hash[d] for d in deps]
        blob, wall, cost = ref.simulate_stage(name, stage_type, inputs, manifest.seed)
        name_to_hash[name] = store.put(blob)
        manifest.total_cost_usd += cost
        manifest.stages.append(ref.StageRecord(
            name=name, stage_type=stage_type, input_hashes=inputs,
            output_hash=name_to_hash[name], wall_clock_sec=wall, cost_usd=cost, status="ok"))
        if name < upto:
            done[cache_key(name, stage_type, inputs, manifest.seed)] = name_to_hash[name]
    return manifest, done


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    store = ref.ArtifactStore()
    full, done = warm(ref, store, FAILED_AT)
    rerun, skipped = cached_run(ref, ref.Manifest(), store, done)
    repeat = ref.run(ref.Manifest(), ref.ArtifactStore())
    hashes = [[s.output_hash for s in m.stages] for m in (full, rerun, repeat)]
    first = ref.STAGES[0][0]
    return {"skipped": skipped, "stages": len(full.stages),
            "wall": tuple(sum(s.wall_clock_sec for s in m.stages) for m in (full, rerun)),
            "cost": (full.total_cost_usd, rerun.total_cost_usd),
            "hashes_match": hashes[0] == hashes[1] == hashes[2],
            "key_is_blob": cache_key(first, "tokenizer", [], full.seed).encode()
            == ref.simulate_stage(first, "tokenizer", [], full.seed)[0],
            "store_has": hasattr(ref.ArtifactStore, "has")}


def verify(result):
    full_wall, rerun_wall = result["wall"]
    full_cost, rerun_cost = result["cost"]
    saved = full_wall - rerun_wall
    return [
        practice.Check(
            "ANSWER: 30,720 of 37,050 seconds saved -- 82.9%, and exactly, not approximately",
            len(result["skipped"]) == 8 and abs(saved / full_wall - 0.829) < 0.01,
            f"caching {len(result['skipped'])} stages and re-running "
            f"{result['stages'] - len(result['skipped'])} takes wall clock from {full_wall:,.0f} "
            f"to {rerun_wall:,.0f} seconds, a saving of {saved:,.0f} ({saved / full_wall:.1%}), "
            f"and cost from ${full_cost:,.0f} to ${rerun_cost:,.0f}. The figure is exact because "
            "every wall_clock_sec in this pipeline is a cost_table lookup keyed on stage_type",
        ),
        practice.Check(
            "FINDING: 'detect the cached artifacts by hash' is circular",
            result["key_is_blob"],
            "a stage's artifact is addressed by sha256(blob) and blob is what the stage produces, "
            "so asking the store whether it is present requires having run the stage. The lookup "
            "that makes rollback work is on the inputs -- (name, stage_type, input_hashes, seed) "
            "-- and those are exactly what simulate_stage serialises: the cache key built here is "
            "byte-for-byte the blob the stage returns, so the key and the answer are one object",
        ),
        practice.Check(
            "FINDING: there is nothing for a rollback to invalidate",
            result["hashes_match"],
            f"the full run, the cached re-run and a third independent run all produce the same "
            f"{result['stages']} output hashes. Re-running stage 09 after a failure produces the "
            "artifact that was already there, so the scenario the exercise describes cannot "
            "change a byte downstream -- and if the seed changes to make it, stage 01 changes too "
            "and nothing is cached",
        ),
        practice.Check(
            "MECHANISM: run has no cache at all, and the store it writes to has the method",
            result["store_has"] and rerun_wall < full_wall,
            "run's loop is blob, wall, cost = simulate_stage(...) then store.put(blob), "
            f"unconditionally, {result['stages']} times; its only early exit is the budget check. "
            "ArtifactStore.has exists and run never calls it. The rollback is not an extension of "
            "the orchestrator, it is the first version of a feature the orchestrator advertises "
            "by having a content-addressed store at all",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
