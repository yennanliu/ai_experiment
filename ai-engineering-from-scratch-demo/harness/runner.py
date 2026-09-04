"""`demo` — list, run, verify, coverage, practice (`DESIGN §4`)."""

from __future__ import annotations

import argparse
import pathlib
import sys

from . import coverage, explain, manifest, practice, tiers

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEMOS = ROOT / "demos" / "phases"


def _manifests(selector: str | None = None):
    if not DEMOS.is_dir():
        return []
    found = sorted(DEMOS.glob("*/*/practice/practice.yaml"))
    if selector:
        needle = selector.strip("/").replace("phases/", "")
        found = [p for p in found if needle in str(p.parent.parent.relative_to(DEMOS))]
    return found


def _load(path):
    return manifest.load_practice(path)


def cmd_list(args) -> int:
    paths = _manifests(args.lesson)
    if not paths:
        print("no practice manifests found")
        return 0
    for path in paths:
        pack = _load(path)
        built = sum(1 for e in pack.code_exercises
                    if (path.parent / e.filename).is_file())
        print(f"{pack.phase}/{pack.lesson}  {len(pack.exercises)} exercises "
              f"({built} files, {len(pack.exercises) - len(pack.code_exercises)} prose)")
        if args.verbose:
            for ex in pack.exercises:
                mark = "·" if ex.kind != "explain" else "¶"
                print(f"   {mark} ex{ex.index:02d} [{ex.kind}/{ex.tier}] {ex.slug}")
    return 0


def cmd_practice_run(args) -> int:
    paths = _manifests(args.lesson)
    if not paths:
        print(f"no lesson matching {args.lesson!r}", file=sys.stderr)
        return 2
    failures = 0
    for path in paths:
        pack = _load(path)
        print(f"{pack.phase}/{pack.lesson}")
        for ex in pack.exercises:
            if args.ex and ex.index != args.ex:
                continue
            if not tiers.selected(ex.tier):
                continue
            if ex.kind == "explain":
                answered = (path.parent / "README.md").read_text(encoding="utf-8")
                ok = ex.cites and ex.cites in answered
                print(f"  ex{ex.index:02d}: {'PASS' if ok else 'FAIL'} (prose, cites {ex.cites!r})")
                failures += 0 if ok else 1
                continue
            capability = tiers.probe(ex.tier)
            if not capability.ok:
                print(f"  ex{ex.index:02d}: SKIP — {capability.remedy}")
                continue
            result = practice.grade_file(path.parent / ex.filename, ex.stem)
            print(practice.report(result))
            failures += 0 if result.ok else 1
    return 1 if failures else 0


def cmd_verify(args) -> int:
    rc = cmd_practice_run(args)
    print()
    print("tier capabilities:")
    print(tiers.describe())
    return rc


def cmd_coverage(args) -> int:
    rows = coverage.scan(args.phase)
    print(coverage.table(rows))
    drifted = [r for r in rows if r.drifted]
    if drifted:
        print()
        for row in drifted:
            print(f"⚠ spec drift: {row.phase}/{row.lesson} exercises {list(row.drifted)}")
    if args.check and drifted:
        return 1
    return 0


def cmd_explain(args) -> int:
    paths = _manifests(args.lesson)
    if not paths:
        print(f"no lesson matching {args.lesson!r}", file=sys.stderr)
        return 2
    pack = _load(paths[0])
    ex = pack.by_index(args.ex) if args.ex else None
    print(explain.render(pack.phase, pack.lesson, ex))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="demo", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="lessons with a practice manifest")
    p.add_argument("lesson", nargs="?")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("verify", help="run everything the CI gate runs")
    p.add_argument("lesson", nargs="?")
    p.add_argument("--ex", type=int)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("coverage", help="reference tree vs demo tree")
    p.add_argument("--phase")
    p.add_argument("--check", action="store_true", help="exit non-zero on spec drift")
    p.set_defaults(func=cmd_coverage)

    p = sub.add_parser("explain", help="exercise text, source link, thresholds")
    p.add_argument("lesson")
    p.add_argument("--ex", type=int)
    p.set_defaults(func=cmd_explain)

    prac = sub.add_parser("practice", help="run graded solutions")
    prac_sub = prac.add_subparsers(dest="practice_command", required=True)
    p = prac_sub.add_parser("run")
    p.add_argument("lesson")
    p.add_argument("--ex", type=int)
    p.set_defaults(func=cmd_practice_run)
    p = prac_sub.add_parser("list")
    p.add_argument("lesson", nargs="?")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_list)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":                      # pragma: no cover
    raise SystemExit(main())
