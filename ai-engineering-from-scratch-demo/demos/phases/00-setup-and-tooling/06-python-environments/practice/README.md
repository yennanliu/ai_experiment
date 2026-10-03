<!-- generated:start -->
# 00-setup-and-tooling / 06-python-environments

Solutions to all 4 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/00-setup-and-tooling/06-python-environments/) · upstream spec
`phases/00-setup-and-tooling/06-python-environments/docs/en.md`

```bash
uv run demo practice run 06-python-environments --ex 1
uv run demo explain 06-python-environments --ex 1
uv run pytest demos/phases/00-setup-and-tooling/06-python-environments
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `env_setup.sh` and verify all checks pass | code | T0 | `ex01_one_missing_package_aborts_verification_before_the_failure_count_is_printed.py` |
| 2 | Create a second virtual environment, install a different version of numpy in it, and confirm… | code | T0 | `ex02_isolation_holds_until_one_pythonpath_and_the_scripts_venv_check_passes_a_non_venv.py` |
| 3 | Write a `pyproject.toml` for a project that needs both PyTorch and the Anthropic SDK | code | T0 | `ex03_the_lessons_template_installs_neither_torch_nor_anthropic_and_drops_the_pandas_its_script_installs.py` |
| 4 | Deliberately install a package globally (without activating a venv), notice where it goes, th… | code | T0 | `ex04_inside_an_activated_uv_style_venv_pip_still_resolves_to_the_global_one.py` |
<!-- generated:end -->

## Answers

The lesson's code is one bash script, `env_setup.sh`. Running it for real
downloads packages into the reference checkout, so it is read as text from the
reference `phases/` tree and **its own blocks are executed in bash** with only
the outside world faked (`python`/`python3` as shell functions, a fake
`which`). Exercises 2 and 4 build real but pip-less venvs with the stdlib
`venv` module in a temp dir. Nothing touches the network or the machine's
Python. All four are **T0**, stdlib only.

### 1 — one missing package aborts verification before the failure count is printed

**ANSWER:** with every import present, the tail ends **"All checks passed"**,
exit 0, after five packages, the matmul and a torch WARN.

**FINDING: `set -e` turns the first failure into an exit.** `verify_package`
returns 1 from a bare call under `set -euo pipefail`, so with matplotlib missing
the run stops after **2 of 5** packages: scikit-learn, pandas and jupyter go
unchecked, and the "N package(s) failed verification" summary is unreachable.

**FINDING: the version gate rejects Python 4.0.** It tests `major -ge 3` and
`minor -ge 11` separately:

| version | 3.10 | 3.11 | 3.13 | 4.0 |
|---|---|---|---|---|
| exit | 1 | 0 | 0 | **1** |

### 2 — isolation holds until one PYTHONPATH, and the script's venv check passes a non-venv

Two `.venv`s, each with a stand-in `numpy` (1.26.4 and 2.1.0) planted in the
site-packages its own interpreter reports.

**ANSWER:** A imports 1.26.4, B imports 2.1.0, and a third unplanted venv
imports nothing — none sees the project venv that created them.

**FINDING: one `PYTHONPATH` breaks it.** A's interpreter with `PYTHONPATH` set
to B's site-packages imports **2.1.0**.

**FINDING: activation is only PATH.** A's interpreter, called by absolute path
with an empty environment, reports `sys.prefix != sys.base_prefix`.

**FINDING: the script's venv check is a substring test.** Run in bash it
accepts A, B and `/home/u/.venvs-old/bin/python` (no venv at all) and rejects a
conda env.

### 3 — the lesson's template installs neither torch nor anthropic, and drops the pandas its script installs

```toml
[project]
name = "torch-claude-app"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["torch>=2.3", "anthropic>=0.39"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

**FINDING: the lesson's template puts both behind extras.** Its required
dependencies are numpy, matplotlib, jupyter and scikit-learn — **0 of the 2**
libraries a project that *needs* them must install by default.

**FINDING: the lesson's two dependency lists disagree.** `env_setup.sh`
installs pandas, the template omits it, and lesson 05's `notebook_tips.py`
imports pandas at module top. The template also has no `[build-system]`.

### 4 — inside an activated uv-style venv, `pip` still resolves to the global one

Nothing is installed or uninstalled outside a temp dir; a stand-in global bin
holding `python` and `pip` sits on PATH behind the venv.

**ANSWER:** a global install lands in the base interpreter's site-packages
(named in `pyvenv.cfg`, `sys.prefix == sys.base_prefix`), not in `.venv`.

**FINDING: activation does not protect `pip`.** A `uv venv` has no pip, so after
`source .venv/bin/activate` the shell resolves `python` to `.venv/bin/python`
and **`pip` to the global bin**. The lesson's "GOOD" activate-then-`pip install`
is a global install for anyone who followed Option 1.

**FINDING: `python -m pip` fails loudly instead.** It exits 1 with "No module
named pip" rather than writing somewhere else; the lesson's `which pip` check
would also have caught it.
