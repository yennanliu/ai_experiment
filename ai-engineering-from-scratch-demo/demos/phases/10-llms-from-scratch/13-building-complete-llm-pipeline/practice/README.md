<!-- generated:start -->
# 10-llms-from-scratch / 13-building-complete-llm-pipeline

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/10-llms-from-scratch/13-building-complete-llm-pipeline/) · upstream spec
`phases/10-llms-from-scratch/13-building-complete-llm-pipeline/docs/en.md`

```bash
uv run demo practice run 13-building-complete-llm-pipeline --ex 1
uv run demo explain 13-building-complete-llm-pipeline --ex 1
uv run pytest demos/phases/10-llms-from-scratch/13-building-complete-llm-pipeline
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Extend the orchestrator to support parallel execution of stages 07 and 08. Use the stdlib `co… | code | T0 | `ex01_there_is_nothing_to_run_in_parallel.py` |
| 2 | Add a "contamination check" gate. Given the eval dataset hash and the training dataset shards… | code | T0 | `ex02_the_gate_holds_the_ship_not_the_run.py` |
| 3 | Implement a cost estimator from first principles. For stage 04 (pre-training), estimate FLOPs… | code | T0 | `ex03_six_n_d_lands_within_one_and_a_half_percent.py` |
| 4 | Build a partial rollback. Simulate a failure at stage 09 (CAI), then re-run stages 09 through… | code | T0 | `ex04_the_cache_key_is_the_output.py` |
| 5 | Add observability. Emit OpenTelemetry spans for each stage, with attributes for params, token… | code | T0 | `ex05_three_of_the_four_attributes_do_not_exist.py` |
<!-- generated:end -->

## Answers

`code/main.py` is a twelve-stage pipeline orchestrator: a DAG, a
content-addressed artifact store, a manifest, and a gate. It is the only lesson
in the phase whose reference code is infrastructure rather than arithmetic, and
the five exercises all land on the same seam — `simulate_stage` returns a JSON
blob and a wall-clock looked up from a table, so every quantity the exercises
ask you to measure is either a constant or a hash of one.

All five are **T0** on **no** dependency group (stdlib only).

### 1 — there is nothing to run in parallel

**ANSWER: the parallel manifest is identical to the serial one, field for
field.** `simulate_stage` looks its wall-clock and cost up in a table keyed on
`stage_type` alone — a `policy` stage is `(5400.0, 300.0)` whatever its inputs
are — so concurrency changes nothing, because nothing is measured.

**FINDING: `Manifest` has no wall-clock total to improve.** `run` accumulates
`total_cost_usd` and accumulates nothing else, so the 10.3 hours of per-stage
`wall_clock_sec` are recorded twelve times and summed zero times.

**MECHANISM: the determinism comes from `deps`, not from the scheduler.**

```python
input_hashes = [name_to_hash[d] for d in deps]   # declared order, not completion order
```

Hand the same two hashes to `simulate_stage` in the other order and the blob
hashes differ (`bd76c49a…` against `e2ea3d82…`), because `sort_keys=True` sorts
dictionary keys and leaves list order alone. An implementation that appended
hashes as futures resolved would break this silently.

**FINDING: the pipeline is a pure function of its seed.** Nothing in
`simulate_stage` reads a clock, a file or a random number.

### 2 — the gate holds the ship, not the run

**ANSWER: `run` spends the whole budget before `gate` is ever called.** All
twelve stages execute, `total_cost_usd` reaches **$1,941**, and only then does
`run` assign `eval_metrics`. `gate` is a separate call. Nothing in `run`
consults a gate except the budget check.

**FINDING: the two measures the exercise offers disagree exactly where
contamination checks are for.**

| 200 eval items vs 2,000 shards | 13-gram | exact match |
|---|---:|---:|
| clean | 0.000% | 0.000% |
| 1 verbatim leak | 0.500% | 0.500% |
| 5 verbatim leaks | 2.500% | 2.500% |
| **5 embedded leaks** | **2.500%** | **0.000%** |

An eval item pasted into the middle of a longer training line is invisible to
exact match. "Exact string match **or** 13-gram match" is the one choice in the
exercise that changes the answer.

**FINDING: 0.1% is below the resolution of the eval set.** One leaked item of
200 is 0.500% — five times the threshold. To distinguish 0.1% from 0% the eval
set needs ≥1,000 items; on anything smaller the gate is binary.

**MECHANISM: the eval dataset *hash* cannot be used for this.** A SHA-256
supports equality. Overlap needs contents, which the pipeline never stores.

### 3 — 6ND lands within one and a half percent

```text
6 × 7e9 params × 2e12 tokens               = 8.400e22 FLOPs
÷ (989 TFLOPs × 40% MFU) ÷ 3600            = 58,982 H100-hours   = $147,455
÷ (312 TFLOPs × 40% MFU) ÷ 3600            = 186,966 A100-hours
Meta's published Llama 2 7B                = 184,320 A100-hours   → ratio 1.014
```

**ANSWER: $147,455, and the check against Llama 2 passes at 1.4%.** A
back-of-envelope with two constants and one published utilisation figure
reproduces a real pre-training run.

**FINDING: the pipeline books the same stage at $400** — a factor of **369**.
The whole twelve-stage pipeline costs $1,941 in the manifest; stage 04 alone
costs $147,455 in reality.

**FINDING: the budget gate is the wrong order of magnitude in both
directions.** The simulated pipeline spends 3.9% of the $50,000 budget, so it
never binds; the real stage 04 is 2.95× the whole budget, so it would halt the
run at stage 04 and never reach the gate.

**MECHANISM: the 6 is arithmetic; the 40% is the only measured term.**

### 4 — the cache key is the output

**ANSWER: 30,720 of 37,050 seconds — 82.9%, exactly.** Caching 01–08 and
re-running 09–12 takes cost from $1,941 to $331. The figure is exact because
every `wall_clock_sec` is a `cost_table` lookup.

**FINDING: "detect the cached artifacts by hash" is circular.** A stage's
artifact is addressed by `sha256(blob)`, and `blob` is what the stage produces.
The lookup that makes rollback work is on the *inputs* — and those are exactly
what `simulate_stage` serialises, so the cache key built here is byte-for-byte
the blob the stage returns.

**FINDING: there is nothing for a rollback to invalidate.** The full run, the
cached re-run and a third independent run produce the same twelve hashes.

**MECHANISM: `run` has no cache at all.** `ArtifactStore.has` exists and `run`
never calls it.

### 5 — three of the four attributes do not exist

**ANSWER: the trace is connected, and 1 of the 4 named attributes exists.**
Thirteen spans share one `traceId`, every `parentSpanId` resolves, no orphans.
`StageRecord` is `name, stage_type, input_hashes, output_hash, wall_clock_sec,
cost_usd, status` — of `params`, `tokens_seen`, `loss` and `cost`, only `cost`
has a source.

**FINDING: three of the four have to be invented.** There is no parameter count,
token count or loss anywhere in the pipeline. A span reporting them is reporting
a literal the observability layer made up — the failure mode observability
exists to prevent.

**FINDING: the one attribute that exists is fabricated upstream.** Both
checkpoint stages report `400.0`; the twelve stages carry 8 distinct costs
between them. Tracing it recovers the table, not the run.

**MECHANISM: the untestable half of the exercise is the half that proves
nothing.** Whether a span reached an OTLP endpoint says nothing about whether
the trace is well formed; whether every `parentSpanId` resolves does — and that
is checkable without the collector or the package.
