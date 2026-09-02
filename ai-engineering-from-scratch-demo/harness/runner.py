"""`demo` -- run one demo, a phase, or a tier. (D6)

One runner drives every demo in the repo, which is only possible because every
demo honours the same contract: a `demo.yaml`, an entrypoint that exits 0, and
tests that assert the lesson's claim. The runner enforces the two halves of that
contract a manifest cannot express on its own -- the demo actually runs, and it
runs inside its declared time budget.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import coverage as coverage_mod
from . import tiers
from .manifest import Demo, ManifestError, discover

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMOS_ROOT = REPO_ROOT / "demos"

PASS, SKIP, FAIL, OVER_BUDGET = "PASS", "SKIP", "FAIL", "OVER_BUDGET"
GRACE_SECONDS = 30


@dataclass
class Result:
    demo: Demo
    status: str
    seconds: float
    detail: str = ""
    kind: str = "run"

    @property
    def ok(self) -> bool:
        return self.status in (PASS, SKIP)

    def line(self) -> str:
        return (
            f"{self.status:<11} {self.demo.tier}  {self.kind:<5} {self.seconds:6.2f}s  "
            f"{self.demo.lesson}"
            + (f"\n            {self.detail}" if self.detail else "")
        )


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


def select(target: str | None = None, *, phase: str | None = None,
           tier: str | None = None, parity_only: bool = False) -> list[Demo]:
    """Resolve a CLI selection to demos, newest filters applied in order."""
    demos = discover(DEMOS_ROOT)

    if target:
        needle = target.rstrip("/").removeprefix("demos/")
        matches = [d for d in demos if d.lesson == needle or needle in d.lesson]
        if not matches:
            raise SystemExit(
                f"no demo matches {target!r}.\n"
                "Run `demo list` to see what is built, or "
                f"`demo scaffold {needle}` to start one."
            )
        demos = matches
    if phase:
        want = phase.zfill(2)
        demos = [d for d in demos if d.phase_number == want]
    if tier:
        demos = [d for d in demos if d.tier == tier]
    if parity_only:
        demos = [d for d in demos if d.has_parity]
    return demos


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------


def _child_env() -> dict[str, str]:
    """Env for a demo subprocess: the repo on the path so `harness` imports."""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{existing}" if existing else str(REPO_ROOT)
    return env


def run_demo(demo: Demo, *, extra_args: list[str] | None = None,
             quiet: bool = False) -> Result:
    """Run one demo's entrypoint, honouring its tier and its time budget."""
    skip = tiers.check(demo)
    if skip:
        if not quiet:
            print(skip.render())
        return Result(demo, SKIP, 0.0, skip.reason)

    if not demo.entrypoint_path.exists():
        return Result(demo, FAIL, 0.0, f"no entrypoint at {demo.entrypoint}")

    argv = [sys.executable, demo.entrypoint, *(extra_args or [])]
    started = time.monotonic()
    try:
        proc = subprocess.run(
            argv,
            cwd=demo.path,
            env=_child_env(),
            timeout=demo.runtime_seconds + GRACE_SECONDS,
            capture_output=quiet,
            text=True,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - started
        return Result(demo, OVER_BUDGET, elapsed,
                      f"killed after {elapsed:.0f}s; manifest declares "
                      f"runtime_seconds: {demo.runtime_seconds}")
    elapsed = time.monotonic() - started

    if proc.returncode != 0:
        detail = f"exit {proc.returncode}"
        if quiet and proc.stderr:
            detail += "\n" + proc.stderr.strip().splitlines()[-1]
        return Result(demo, FAIL, elapsed, detail)
    if elapsed > demo.runtime_seconds:
        return Result(demo, OVER_BUDGET, elapsed,
                      f"took {elapsed:.1f}s but declares "
                      f"runtime_seconds: {demo.runtime_seconds}")
    return Result(demo, PASS, elapsed)


def run_tests(demo: Demo, *, quiet: bool = True) -> Result:
    """Run a demo's own test suite."""
    tests = demo.path / "tests"
    if not tests.is_dir():
        return Result(demo, FAIL, 0.0, "no tests/ directory", kind="tests")

    skip = tiers.check(demo)
    if skip:
        return Result(demo, SKIP, 0.0, skip.reason, kind="tests")

    started = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q", "--no-header"],
        cwd=demo.path,
        env=_child_env(),
        timeout=demo.runtime_seconds + GRACE_SECONDS,
        capture_output=quiet,
        text=True,
    )
    elapsed = time.monotonic() - started
    if proc.returncode != 0:
        tail = (proc.stdout or "").strip().splitlines()[-1:] if quiet else []
        return Result(demo, FAIL, elapsed,
                      "; ".join(tail) or f"pytest exit {proc.returncode}", kind="tests")
    return Result(demo, PASS, elapsed, kind="tests")


def report(results: list[Result], *, header: str) -> int:
    """Print a run summary and return the process exit code."""
    print(f"\n{header}")
    for result in results:
        print("  " + result.line())
    failed = [r for r in results if not r.ok]
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    print("  " + ", ".join(f"{n} {status.lower()}" for status, n in sorted(counts.items())))
    return 1 if failed else 0


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_list(args) -> int:
    demos = select(args.target, phase=args.phase, tier=args.tier,
                   parity_only=args.parity)
    if not demos:
        print("no demos match")
        return 0
    for demo in demos:
        flag = "parity" if demo.has_parity else "      "
        print(f"  {demo.tier}  {flag}  {demo.runtime_seconds:>4}s  "
              f"{demo.lesson}\n            {demo.title}")
    print(f"\n{len(demos)} demo(s)")
    return 0


def cmd_run(args) -> int:
    demos = select(args.target, phase=args.phase, tier=args.tier,
                   parity_only=args.parity)
    extra = ["--explain"] if args.explain else []
    single = len(demos) == 1
    results = [run_demo(d, extra_args=extra, quiet=not single) for d in demos]
    return report(results, header=f"run: {len(results)} demo(s), mode={tiers.mode()}")


def cmd_verify(args) -> int:
    """What CI runs: every selected demo executes, then its tests pass."""
    demos = select(args.target, phase=args.phase, tier=args.tier,
                   parity_only=args.parity)
    if not demos:
        print("no demos match -- nothing to verify")
        return 0

    results: list[Result] = []
    for demo in demos:
        ran = run_demo(demo, quiet=True)
        results.append(ran)
        if ran.status == PASS:
            results.append(run_tests(demo))
    return report(results, header=f"verify: mode={tiers.mode()}")


def cmd_coverage(args) -> int:
    statuses = coverage_mod.survey(DEMOS_ROOT)
    counts = coverage_mod.summary(statuses)
    print(coverage_mod.phase_table(statuses))
    print(
        f"\n{counts[coverage_mod.BUILT]} built, "
        f"{counts[coverage_mod.STALE]} stale, "
        f"{counts[coverage_mod.MISSING]} not started"
    )
    stale = [s for s in statuses if s.state == coverage_mod.STALE]
    for status in stale:
        print(f"  ⚠️  {status.lesson}: {status.note}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="demo",
        description="Run the ai-engineering-from-scratch demo suite.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_selection(p):
        p.add_argument("target", nargs="?", help="lesson path, or a substring of one")
        p.add_argument("--phase", help="phase number, e.g. 11")
        p.add_argument("--tier", choices=tiers.TIER_ORDER, help="only this tier")
        p.add_argument("--parity", action="store_true",
                       help="only demos that carry a parity assertion")

    p_list = sub.add_parser("list", help="list built demos")
    add_selection(p_list)
    p_list.set_defaults(func=cmd_list)

    p_run = sub.add_parser("run", help="run demos")
    add_selection(p_run)
    p_run.add_argument("--explain", action="store_true",
                       help="print what the demo proves instead of running it")
    p_run.set_defaults(func=cmd_run)

    p_verify = sub.add_parser("verify", help="run demos and their tests (what CI runs)")
    add_selection(p_verify)
    p_verify.set_defaults(func=cmd_verify)

    p_cov = sub.add_parser("coverage", help="lessons covered, computed from the tree")
    p_cov.set_defaults(func=cmd_coverage)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ManifestError as exc:
        print(f"manifest error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
