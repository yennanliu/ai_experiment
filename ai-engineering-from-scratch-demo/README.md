# ai-engineering-from-scratch-demo

Runnable, **verified** solutions to every exercise in
[`ai-engineering-from-scratch`](https://yennj12.js.org/ai-engineering-from-scratch/index.html)
— 2,090 of them across 20 phases.

Design: [`DESIGN.md`](./DESIGN.md) · Build sequence:
[`IMPLEMENTATION_PLAN.md`](./IMPLEMENTATION_PLAN.md)

## Quick start

```bash
uv sync --extra math --extra dev            # uv is the only supported toolchain
uv run demo list -v                         # what is built
uv run demo verify                          # what CI runs
uv run demo practice run 01-linear-algebra-intuition --ex 4
uv run demo explain 01-linear-algebra-intuition --ex 1
uv run demo coverage --phase 01
```

The reference curriculum must be checked out where the harness can find it —
any ancestor directory, or `AIEFS_REFERENCE=/path/to/ai-engineering-from-scratch`.
Solutions **import** the lesson's own `code/` rather than copying it, so a check
compares against the reference implementation and not against a fork of it.

## What a solution looks like

Every exercise ships one file, index-identical to the lesson's numbered list
(`ex04_orthonormal_check.py` is exercise 4), and every file is its own grader:

```bash
$ uv run demo practice run 01-linear-algebra-intuition --ex 1
  ex01_angle_between: PASS
    [PASS] all 6 pairs scored — 6 pairs
    [PASS] matches the lesson's angle_between within 1e-09° — worst deviation 7.11e-15°
    [PASS] degenerate ends are exact — 0° case -> 0, 180° case -> 180
    [PASS] near-parallel: Kahan beats acos — exact 5.729578e-07°, Kahan err 0.00e+00, acos err 5.73e-07
```

That last check is the point of the repo: running the exercise surfaced that the
lesson's `angle_between` returns `0.0` where the true angle is `5.73e-7°`. A
hand-written "simulation" of the answer could not have found it.

## Coverage

<!-- coverage:start -->

**67 / 2090 exercises** (3.2%) across 20 phases.

| Phase | Exercises | Solved | Lessons started |
|---|---:|---:|---:|
| ⬚ `00-setup-and-tooling` | 45 | 0 | 0 |
| 🚧 `01-math-foundations` | 92 | 67 | 16 |
| ⬚ `02-ml-fundamentals` | 78 | 0 | 0 |
| ⬚ `03-deep-learning-core` | 63 | 0 | 0 |
| ⬚ `04-computer-vision` | 84 | 0 | 0 |
| ⬚ `05-nlp-foundations-to-advanced` | 87 | 0 | 0 |
| ⬚ `06-speech-and-audio` | 51 | 0 | 0 |
| ⬚ `07-transformers-deep-dive` | 52 | 0 | 0 |
| ⬚ `08-generative-ai` | 47 | 0 | 0 |
| ⬚ `09-reinforcement-learning` | 36 | 0 | 0 |
| ⬚ `10-llms-from-scratch` | 116 | 0 | 0 |
| ⬚ `11-llm-engineering` | 79 | 0 | 0 |
| ⬚ `12-multimodal-ai` | 125 | 0 | 0 |
| ⬚ `13-tools-and-protocols` | 148 | 0 | 0 |
| ⬚ `14-agent-engineering` | 212 | 0 | 0 |
| ⬚ `15-autonomous-systems` | 110 | 0 | 0 |
| ⬚ `16-multi-agent-and-swarms` | 123 | 0 | 0 |
| ⬚ `17-infrastructure-and-production` | 140 | 0 | 0 |
| ⬚ `18-ethics-safety-alignment` | 150 | 0 | 0 |
| ⬚ `19-capstone-projects` | 252 | 0 | 0 |

Regenerate with `uv run python scripts/coverage.py`.

<!-- coverage:end -->

## Layout

```text
harness/     stdlib-only: manifests, tiers, parity, cassettes, coverage, the runner
scripts/     the gates — audit_practice, check_deps, coverage, census, notebooks
demos/       phases/<phase>/<lesson>/practice/ — path-identical to the reference
tests/       harness tests, plus one test per gate proving it fails when it should
```

## The gates

Nothing merges on eyeball review. `uv run demo verify` is the same command CI
runs, and each gate has a test that breaks a fixture on purpose to prove the gate
catches it (`tests/test_gates.py`):

| Gate | Rejects |
|---|---|
| `scripts/audit_practice.py` | >120 lines, complexity >8, missing `PRACTICE_IMPL`, surviving `TODO`, no "Reading of the exercise:" line, unlabelled fixture, missing README or tests |
| `scripts/check_deps.py` | an import not covered by the exercise's `deps_group`; any module-level third-party import in `harness/` |
| `uv run demo coverage --check` | stored exercise text that no longer matches upstream (spec drift) |
| `scripts/census.py` | re-derives every headline number in `DESIGN §1` so none becomes folklore |
