<!-- generated:start -->
# 00-setup-and-tooling / 01-dev-environment

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/00-setup-and-tooling/01-dev-environment/) · upstream spec
`phases/00-setup-and-tooling/01-dev-environment/docs/en.md`

```bash
uv run demo practice run 01-dev-environment --ex 1
uv run demo explain 01-dev-environment --ex 1
uv run pytest demos/phases/00-setup-and-tooling/01-dev-environment
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run the verification script and fix any failures | code | T0 | `ex01_python_3_10_passes_two_of_the_three_verifiers.py` |
| 2 | Create a Python virtual environment for this course and install PyTorch | code | T0 | `ex02_the_fix_for_a_missing_torch_needs_a_pip_uv_venv_lacks.py` |
| 3 | Write a "hello world" in all four languages and run each one | code | T0 | `ex03_no_verifier_requires_a_typescript_runner_or_julia.py` |
<!-- generated:end -->

## Answers

The lesson ships three verifiers -- `verify.py` (route-aware), `verify.ts` and
`main.rs` -- and asks the reader to install four toolchains. All three exercises
are **T0** and stdlib-only: nothing here grades the host. `verify.py` is driven
through its own `main()` with simulated probes, or run for real inside a fresh
virtual environment; the two ports are read as source.

### 1 — the loop works; the three scripts disagree on what "passing" means

**ANSWER: fail, fix, pass.** On a simulated fresh machine the beginner route
prints **0/2**, exits **1**, and pairs both FAIL lines with a `Fix:` line. With
the fixes applied it prints **2/2**, exits **0**, and names `vectors.py`.

**FINDING: Python 3.10 passes two of the three.** `verify.py` wants **3.11**;
`verify.ts` and `main.rs` accept **3.10**. The lesson's `python_result` on a
3.10 interpreter returns `ok=False`.

**FINDING: required sets of 2, 3 and 5.** Beginner requires python and git;
`verify.ts` adds node; `main.rs` adds node, rustc and cargo. A machine
`verify.py` calls ready can fail **3** of `main.rs`'s required checks.

**CONTROL:** the **6** routes ending in `python3 phases/...` name **5** distinct
files, and all **5** exist.

### 2 — the printed fix needs a pip that `uv venv` does not ship

A real venv is built in a temp dir without pip (what Step 2's `uv venv` makes),
and `verify.py` is run by its interpreter.

**ANSWER:** the venv is isolated -- PyTorch is reported LATER, "not importable
by" the venv's own python.

**FINDING: the remedy fails where the lesson puts the reader.** **4** fix lines
say `python3 -m pip install ...`; in the pip-less venv `-m pip` exits **1** with
"No module named pip". The lesson itself installs with `uv pip install`.

**FINDING: PyTorch is required by 0 of 7 routes**, so skipping the exercise
never turns the verifier red.

**CONTROL:** in a venv seeded with pip, the same command exits **0**.

### 3 — no verifier requires a TypeScript runner or Julia

| program | needs | required by (of verify.py, verify.ts, main.rs) |
|---|---|---:|
| `hello.py` | python | 3 |
| `hello.ts` | tsx | **0** |
| `hello.rs` | rustc | 1 (main.rs, which needs rustc to be built) |
| `hello.jl` | julia | **0** |

**ANSWER:** all four are written; the Python one runs and prints exactly
`hello world`.

**FINDING: no TypeScript runner is ever installed.** `tsx` appears **0** times in
the lesson and in `verify.py`; `verify.ts` probes it with `npx -y tsx`, which
downloads it unprompted -- the probe fetches what it claims to check.

**FINDING:** `verify.ts` probes `deno`, which the lesson mentions **0** times, and
has no `julia` probe.
