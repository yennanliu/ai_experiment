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
| 1 | Run the verification script and fix any failures | code | T0 | `ex01_the_preflight_grades_seventeen_of_seventy_seven_pairs.py` |
| 2 | Create a Python virtual environment for this course and install PyTorch | code | T0 | `ex02_a_zero_byte_torch_py_passes_and_then_kills_the_run.py` |
| 3 | Write a "hello world" in all four languages and run each one | code | T0 | `ex03_every_row_of_the_language_table_points_at_the_wrong_phases.py` |
<!-- generated:end -->

## Answers

### 1 — the preflight grades seventeen of seventy-seven pairs

It runs. On this machine the beginner route reports `2/2 required checks
passed` and prints `Ready to start Beginner course.` The interesting question
is what that sentence is worth.

Seven routes times eleven probes is **77** combinations. **17** of them are in
some route's `required` tuple, and only those 17 are executed by default —
optional probes are skipped entirely without `--show-later`, and passing
`--show-later` cannot change the exit code, because `print_probe`'s return
value is discarded for them. Six probes are required on **zero** routes:

| probe | required in | optional in |
|---|---:|---:|
| python, git | 7 | 0 |
| node, npx | 1 | 5 |
| numpy | 1 | 3 |
| cargo, julia, jupyter, matplotlib | 0 | 2 |
| torch | 0 | 4 |
| gpu | 0 | 3 |

No failure of the bottom six can make this script exit non-zero on any route
it offers.

`gpu_result` is the sharpest case. It has five returns. Both
`Result(False, ...)` fire *before* the `import torch` — "PyTorch is not
installed" and "PyTorch could not be imported". Every return after the import
succeeds is `Result(True, ...)`, including the one reading "CPU only; a GPU is
optional". A machine with no accelerator gets a PASS on the probe labelled
**Accelerator backend**. The probe reports on PyTorch.

Two smaller things fall out of the same read. `uv` installs the interpreter,
the virtual environment and three libraries in Build It, and one fix string
tells the reader to run `uv python install 3.12` — and `uv` is not among the
eleven probes and is never looked for. And `beginner` and `ml-foundations`
both end at `01-linear-algebra-intuition/code/vectors.py`; only
`ml-foundations` requires numpy to get there, and `vectors.py` has **zero**
imports.

### 2 — a zero-byte torch.py passes, and then kills the run

The exercise has two halves and the shipped preflight can check neither.

For the first, `sys.prefix != sys.base_prefix` settles "is a virtual
environment active" in one line. **Zero** of the eleven probes read
`sys.prefix`, `VIRTUAL_ENV`, or anything else that would tell a course
environment from the system interpreter.

For the second, `module_result` is `importlib.util.find_spec`, which asks
whether a name is reachable on `sys.path` — not whether anything is installed.
Write a **0-byte** file named `torch.py`, put its directory at the front of
`sys.path`, and:

```
module_result("torch")  ->  ok=True, "importable by .../.venv/bin/python3"
gpu_result()            ->  AttributeError: module 'torch' has no attribute 'cuda'
```

The PyTorch probe passes on an empty file. And `gpu_result`, the one probe
that really imports and could therefore have caught the fake, is the reason
the run produces nothing: its `try`/`except Exception` wraps the import
statement alone, so the `AttributeError` escapes `gpu_result`, escapes
`print_probe`, escapes `main`, and the preflight exits on a traceback instead
of a verdict.

Four fix strings tell the reader to run `python3 -m pip install ...`, which
installs into whatever `python3` resolves to on `PATH`. Zero name
`sys.executable`, which is what the probe reports against. Following the fix
is only guaranteed to fix the probe when those two are the same file.

### 3 — every row of the language table points at the wrong phases

Four hello worlds, and exactly one is guaranteed to run: Python, because it is
the interpreter grading this. `node`, `cargo` and `julia` are each a
`shutil.which` from absent, and all three are optional on the route the
preflight defaults to — so "run each one" is an instruction the lesson's own
environment check does not require the reader to be able to follow.

Then the table under Use It, checked against the tree:

| row says | actual files | inside the claimed phases |
|---|---:|---:|
| Python — Phases 1-12 | 643 | 238 (37%) |
| TypeScript — Phases 13-17 | 129 | 15 (12%) |
| Rust — Phases 12, 15-17 | 10 | **0** |
| Julia — Phase 1 | 20 | 8 (40%) |

Every row misses. The Rust row misses completely: all ten `.rs` files are in
phases 00, 04, 06, 07 and 10, none of which the row names, and one of them is
this lesson's own `main.rs`. The Python row is the most misleading, because it
is the one a beginner plans around — it describes 37% of the Python in the
repository and stops seven phases early.

The lesson also cannot say how many languages it uses. The header reads
`**Languages:** Python, Node.js, Rust` — three. Thirteen lines later the intro
says "Python, TypeScript, Rust, and Julia" — four. The exercise says "all four
languages". `code/` ships three of them, and the one missing is Julia, whose
table row points at the phase the reader starts next.
