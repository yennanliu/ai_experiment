"""Exercise 1 — stage 09's input hash is deterministic, and one list comprehension away from not being.

    Extend the orchestrator to support parallel execution of stages 07 and 08.
    Use the stdlib `concurrent.futures` module. Confirm the final manifest
    records both stages' outputs and that stage 09's input hash is a
    deterministic combination of both.

Reading of the exercise: the two stages are run through a
`ThreadPoolExecutor` exactly as asked, and the manifest that comes back is
compared field by field against the serial one. The counterfactual is then run
directly -- stage 09's blob rebuilt from the same two hashes in the order a
parallel scheduler would have *completed* them -- because "confirm it is
deterministic" is only worth confirming if it could have failed.

**ANSWER: the two manifests are identical, and there is nothing to
parallelise.** `simulate_stage` builds a JSON blob and returns a wall-clock and
a cost looked up from a table keyed on `stage_type` alone -- `policy` is
`(5400.0, 300.0)` whatever its inputs are, whatever it computes. Running 07 and
08 concurrently changes no field of the manifest, because no field of the
manifest is measured.

**FINDING: `Manifest` has no wall-clock total to improve.** Its nine fields are
`pipeline_version, seed, git_commit, stages, gates, eval_metrics, budget_usd,
total_cost_usd, shippable`. `run` accumulates `total_cost_usd` and accumulates
nothing else, so the 10.3 hours of per-stage `wall_clock_sec` are recorded
twelve times and summed zero times. The quantity parallelism improves is the one
the pipeline does not track.

**MECHANISM: the determinism comes from `deps`, not from the scheduler.**
`input_hashes = [name_to_hash[d] for d in deps]` walks the *declared* dependency
list, so completion order cannot reach it. Feed the same two hashes in the other
order and the blob hashes differ -- `bd76c49ae2918783` against
`e2ea3d82e40c4404` -- because `json.dumps(..., sort_keys=True)` sorts dictionary
keys and leaves list order alone. The property the exercise asks you to confirm
holds because of one comprehension, and an implementation that appended hashes
as futures resolved would break it silently.

**FINDING: the pipeline is a pure function of its seed.** Two fresh runs produce
the same twelve output hashes, and changing the seed changes all twelve. Nothing
in `simulate_stage` reads a clock, a file or a random number, so "deterministic
combination" is not a property of the hashing -- it is a property of a pipeline
that does no work.

Structure: `parallel_run` is the orchestrator with 07 and 08 in a thread pool;
`rebuild` recomputes one stage's blob from a given input order.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import hashlib

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "13-building-complete-llm-pipeline"
PARALLEL = ("07_reward_ppo_policy", "08_dpo_policy")
JOIN = "09_cai_grpo_policy"


def parallel_run(ref, manifest, store):
    """The lesson's own loop, with the two independent policy stages in a thread pool."""
    name_to_hash, pool = {}, concurrent.futures.ThreadPoolExecutor(max_workers=2)
    pending = {}
    for name, deps, stage_type in ref.STAGES:
        if name in PARALLEL:
            inputs = [name_to_hash[d] for d in deps]
            pending[name] = (pool.submit(ref.simulate_stage, name, stage_type,
                                         inputs, manifest.seed), inputs, stage_type)
            name_to_hash[name] = None
            continue
        for waiting, (future, inputs, kind) in list(pending.items()):
            blob, wall, cost = future.result()
            name_to_hash[waiting] = store.put(blob)
            record(ref, manifest, waiting, kind, inputs, name_to_hash[waiting], wall, cost)
        pending.clear()
        inputs = [name_to_hash[d] for d in deps]
        blob, wall, cost = ref.simulate_stage(name, stage_type, inputs, manifest.seed)
        name_to_hash[name] = store.put(blob)
        record(ref, manifest, name, stage_type, inputs, name_to_hash[name], wall, cost)
    pool.shutdown()
    return manifest


def record(ref, manifest, name, stage_type, inputs, output, wall, cost):
    manifest.total_cost_usd += cost
    manifest.stages.append(ref.StageRecord(
        name=name, stage_type=stage_type, input_hashes=inputs, output_hash=output,
        wall_clock_sec=wall, cost_usd=cost, status="ok"))


def rebuild(ref, seed, inputs):
    """Stage 09's blob from one ordering of its two input hashes."""
    blob, _, _ = ref.simulate_stage(JOIN, "policy", inputs, seed)
    return hashlib.sha256(blob).hexdigest()


def outputs(manifest):
    return [s.output_hash for s in manifest.stages]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    serial = ref.run(ref.Manifest(), ref.ArtifactStore())
    side = parallel_run(ref, ref.Manifest(), ref.ArtifactStore())
    by_name = {s.name: s for s in serial.stages}
    hashes = [by_name[n].output_hash for n in PARALLEL]
    other = ref.run(ref.Manifest(seed=serial.seed + 1), ref.ArtifactStore())
    return {"serial": [dataclasses.asdict(s) for s in serial.stages],
            "parallel": [dataclasses.asdict(s) for s in side.stages],
            "join_order": (rebuild(ref, serial.seed, hashes),
                           rebuild(ref, serial.seed, hashes[::-1])),
            "manifest_fields": [f.name for f in dataclasses.fields(ref.Manifest)],
            "wall_total": sum(s.wall_clock_sec for s in serial.stages),
            "policy_table": ref.simulate_stage("probe", "policy", [], serial.seed)[1:],
            "seed_hashes": (outputs(serial), outputs(other))}


def verify(result):
    serial, side = result["serial"], result["parallel"]
    declared, completed = result["join_order"]
    same_seed, other_seed = result["seed_hashes"]
    wall, cost = result["policy_table"]
    return [
        practice.Check(
            "ANSWER: the parallel manifest is identical, and there is nothing to parallelise",
            serial == side,
            f"running {list(PARALLEL)} in a ThreadPoolExecutor produces a manifest equal to the "
            f"serial one field for field, across all {len(serial)} stage records. simulate_stage "
            f"builds a JSON blob and looks its wall-clock and cost up in a table keyed on "
            f"stage_type alone -- a policy stage is ({wall:.0f}, {cost:.0f}) whatever its inputs "
            "are -- so concurrency changes nothing because nothing is measured",
        ),
        practice.Check(
            "FINDING: Manifest has no wall-clock total to improve",
            "total_cost_usd" in result["manifest_fields"]
            and not any("wall" in f for f in result["manifest_fields"]),
            f"the manifest's fields are {result['manifest_fields']}. run accumulates "
            f"total_cost_usd and accumulates nothing else, so the "
            f"{result['wall_total'] / 3600:.1f} hours of per-stage wall_clock_sec are recorded "
            f"{len(serial)} times and summed zero times. The quantity parallel execution improves "
            "is the one quantity the pipeline does not track",
        ),
        practice.Check(
            "MECHANISM: the determinism comes from `deps`, not from the scheduler",
            declared != completed,
            f"input_hashes = [name_to_hash[d] for d in deps] walks the declared dependency list, "
            f"so completion order cannot reach it. Hand the same two hashes to simulate_stage in "
            f"the other order and the blob hashes differ -- {declared[:16]} against "
            f"{completed[:16]} -- because json.dumps(..., sort_keys=True) sorts dictionary keys "
            "and leaves list order alone. An implementation that appended hashes as futures "
            "resolved would break this silently",
        ),
        practice.Check(
            "FINDING: the pipeline is a pure function of its seed",
            same_seed == [s["output_hash"] for s in side]
            and not set(same_seed) & set(other_seed),
            f"two fresh runs produce the same {len(same_seed)} output hashes and changing the "
            f"seed changes all {len(other_seed)} of them. Nothing in simulate_stage reads a "
            "clock, a file or a random number, so 'a deterministic combination of both' is not a "
            "property of the hashing scheme -- it is a property of a pipeline that does no work",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
