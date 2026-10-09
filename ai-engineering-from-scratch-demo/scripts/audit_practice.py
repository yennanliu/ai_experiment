#!/usr/bin/env python3
"""D14's mechanical ceilings, plus D10 and D12 structure. No human in the loop.

Exit non-zero on any violation. D14's line ceiling is the one rule with two numbers —
"<= 120 lines of code per file excluding the docstring; hard fail over 150" — so a file
between the two is reported rather than rejected, which is what §6.4's phase-batch review
reads. The docstring is excluded because it is mandated content (the exercise text verbatim
plus the "Reading of the exercise:" line), and charging a solution for how long its own
exercise is measures the wrong thing. Every rule here is one `DESIGN §6` lists as a
rejection reason, so a solution that passes this passes the generation gate.

`ci_hazards` adds the three CI-only failures that PRs #30 and #32 hit only after
merge-time CI. Each one passes locally, so only a static check finds it before push.
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from harness import manifest, parity, runner  # noqa: E402

MAX_LINES = 120          # D14's target: over it the audit says so, for §6.4 to read
HARD_LINES = 150         # D14's "hard fail over 150" — the number that exits non-zero
MAX_COMPLEXITY = 8
BANNED = ("TODO", "FIXME", "XXX", "<<<", "raise NotImplementedError")


def complexity(node) -> int:
    """Cyclomatic complexity: one plus each branch point."""
    score = 1
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.For, ast.While, ast.ExceptHandler,
                              ast.Assert, ast.IfExp, ast.comprehension)):
            score += 1
        elif isinstance(child, ast.BoolOp):
            score += len(child.values) - 1
    return score


def docstring_lines(tree) -> int:
    """The module docstring's own line count.

    D14 puts the ceiling on *code*, "excluding the docstring", and the docstring is
    mandated content — the exercise text verbatim plus the "Reading of the exercise:"
    line — so counting it would charge a solution for how long its exercise is.
    """
    first = tree.body[0] if tree.body else None
    if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)):
        return first.end_lineno - first.lineno + 1
    return 0


HISTORY = {"log", "show", "diff", "blame", "rev-list"}
CHECKOUT = {"find_reference_root", "getcwd", "cwd"}   # calls that name the checkout itself


class Bindings:
    """Each name's latest binding before a use, in the use's own function.

    Resolving a name this way, rather than asking whether the file mentions a
    marker anywhere, keeps `root = root / "phases"` and an unrelated
    `import tempfile` from deciding what a later call does.
    """

    def __init__(self, tree):
        self.scope, self.binds = {}, {}
        self.module = tree
        for scope in [tree] + [n for n in ast.walk(tree)
                               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            for node in ast.walk(scope):    # walk is outer-first, so the innermost wins
                self.scope[node] = scope
        for node in ast.walk(tree):
            for target, value in _bound(node):
                key = (self.scope[node], target.id)
                self.binds.setdefault(key, []).append((node.lineno, value))

    def resolve(self, name):
        """The value last bound to `name` before it is used, or None (a parameter, say).

        A name its function never binds is a module global, and a module runs to the
        end before any of its functions is called, so the last module binding counts.
        """
        scope = self.scope.get(name)
        rows = [value for line, value in self.binds.get((scope, name.id), [])
                if line < name.lineno]
        if scope is not self.module and (scope, name.id) not in self.binds:
            rows = [value for _, value in self.binds.get((self.module, name.id), [])]
        return rows[-1] if rows else None

    def reaches(self, node, found, depth=4):
        """Whether `found` holds anywhere in `node`, following names to their values."""
        for sub in ast.walk(node):
            if found(sub):
                return True
            if depth and isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                value = self.resolve(sub)
                if value is not None and self.reaches(value, found, depth - 1):
                    return True
        return False


def _bound(node):
    """(Name target, value) pairs for an assignment or a `with ... as name`."""
    if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return [(t, node.value) for t in targets if isinstance(t, ast.Name)]
    if isinstance(node, ast.With):
        return [(i.optional_vars, i.context_expr) for i in node.items
                if isinstance(i.optional_vars, ast.Name)]
    return []


def _calls(tree, attr):
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute) and n.func.attr == attr]


def _called(node, names) -> bool:
    func = getattr(node, "func", None)
    return isinstance(node, ast.Call) and getattr(func, "attr", getattr(func, "id", "")) in names


def _is_environ(node) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "environ"


def _is_key(node) -> bool:
    return isinstance(node, ast.Constant) and node.value == "AIEFS_REFERENCE"


def drops_reference(tree, names) -> bool:
    """An in-process clear of os.environ whose replacement has no AIEFS_REFERENCE.

    `patch.dict(os.environ, values, clear=True)` must carry the key in `values`
    (followed through names, so `{**KEEP, **env}` counts); `os.environ.clear()`
    must be followed in its function by a store or update that puts it back.
    """
    for call in _calls(tree, "dict"):
        clear = any(k.arg == "clear" and getattr(k.value, "value", None) is True
                    for k in call.keywords)
        if call.args and _is_environ(call.args[0]) and clear:
            values = call.args[1:] + [k.value for k in call.keywords if k.arg in (None, "values")]
            named = any(k.arg == "AIEFS_REFERENCE" for k in call.keywords)
            if not named and not any(names.reaches(v, _is_key) for v in values):
                return True
    for call in _calls(tree, "clear"):
        if _is_environ(call.func.value) and not restores_reference(names.scope[call], names):
            return True
    return False


def restores_reference(scope, names) -> bool:
    stored = any(isinstance(n, ast.Subscript) and _is_environ(n.value) and _is_key(n.slice)
                 and isinstance(n.ctx, ast.Store) for n in ast.walk(scope))
    return stored or any(_is_environ(c.func.value) and any(names.reaches(a, _is_key)
                                                           for a in c.args)
                         for c in _calls(scope, "update"))


def scans_reference_root(tree, names) -> bool:
    """An rglob/os.walk whose root is `find_reference_root()` itself, not its phases/."""
    def rooted(node):
        if isinstance(node, ast.Name):
            node = names.resolve(node)
        return _called(node, {"find_reference_root"})
    globbed = any(rooted(c.func.value) for c in _calls(tree, "rglob"))
    return globbed or any(c.args and rooted(c.args[0]) for c in _calls(tree, "walk"))


def git_commands(tree):
    """(subcommand, repo expression or None, call) for every git invocation.

    A literal argv `["git", "log", ...]` names its subcommand directly; a wrapper
    such as `def git(root, *args)` that runs `["git", *args]` with `cwd=root` is
    followed to each of its call sites, where the subcommand is a string argument.
    """
    rows, wrappers = [], {}
    for call in [n for n in ast.walk(tree) if isinstance(n, ast.Call) and n.args]:
        argv = call.args[0]
        if not (isinstance(argv, (ast.List, ast.Tuple)) and argv.elts
                and getattr(argv.elts[0], "value", None) == "git"):
            continue
        cwd = next((k.value for k in call.keywords if k.arg == "cwd"), None)
        words = [e.value for e in argv.elts[1:] if isinstance(e, ast.Constant)]
        if words:
            rows.append((_subcommand(words), cwd, call))
        elif any(isinstance(e, ast.Starred) for e in argv.elts):
            wrappers.update(_wrapper(tree, call, cwd))
    for call in [n for n in ast.walk(tree) if _called(n, set(wrappers))]:
        index = wrappers[getattr(call.func, "attr", getattr(call.func, "id", ""))]
        words = [a.value for a in call.args if isinstance(a, ast.Constant)]
        repo = call.args[index] if index is not None and index < len(call.args) else None
        rows.append((_subcommand(words), repo, call))
    return rows


def _subcommand(words):
    rest = [w for w in words if isinstance(w, str)]
    while rest and rest[0] in ("-C", "-c"):
        rest = rest[2:]
    return next((w for w in rest if not w.startswith("-")), None)


def _wrapper(tree, call, cwd):
    """{function name: index of the parameter it runs git in} for a `*args` git helper."""
    for func in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        if call in ast.walk(func):
            params = [a.arg for a in func.args.args]
            name = cwd.id if isinstance(cwd, ast.Name) else None
            return {func.name: params.index(name) if name in params else None}
    return {}


def reads_checkout_history(tree, names) -> bool:
    """A git history command run in this checkout or the reference, not a scratch repo.

    The repo is the checkout when no cwd is given, or when the cwd is built from
    `find_reference_root()`, `__file__`, `os.getcwd()` or `Path.cwd()`. A cwd that is
    a parameter or a temporary directory is somebody else's repo.
    """
    def checkout(node):
        return _called(node, CHECKOUT) or (isinstance(node, ast.Name) and node.id == "__file__")
    return any(sub in HISTORY and (repo is None or names.reaches(repo, checkout))
               for sub, repo, _ in git_commands(tree))


def ci_hazards(tree) -> list:
    """Code that passes locally and fails only in CI (PRs #30 and #32).

    CI finds the reference through `AIEFS_REFERENCE` alone, checks it out with
    `fetch-depth 1`, and the checkout root holds more than `phases/`.
    """
    names, problems = Bindings(tree), []
    if drops_reference(tree, names):
        problems.append("clears os.environ without carrying AIEFS_REFERENCE, "
                        "so CI cannot find the reference")
    if scans_reference_root(tree, names):
        problems.append("scans the reference checkout root; anchor the scan to phases/")
    if reads_checkout_history(tree, names):
        problems.append("reads git history of a checkout; CI clones with fetch-depth 1, "
                        "so build a temporary repo instead")
    return problems


def audit_solution(path: pathlib.Path, exercise, warnings=None) -> list:
    """Returns the problems. D14's soft ceiling lands in `warnings` when one is passed."""
    problems, warnings = [], [] if warnings is None else warnings
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    tree = ast.parse(text)
    code = len(lines) - docstring_lines(tree)
    if code > HARD_LINES:
        problems.append(f"{code} lines of code > hard ceiling {HARD_LINES} (D14)")
    elif code > MAX_LINES:
        warnings.append(f"{code} lines of code > target {MAX_LINES} (D14)")
    for banned in BANNED:
        if banned in text:
            problems.append(f"contains {banned!r} — a surviving scaffold marker")
    doc = ast.get_docstring(tree) or ""
    if "Reading of the exercise:" not in doc:
        problems.append("docstring has no 'Reading of the exercise:' line (DESIGN §6.4)")
    if exercise.en.split()[0] not in doc:
        problems.append("docstring does not quote the exercise text (D12)")
    for func in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
        score = complexity(func)
        if score > MAX_COMPLEXITY:
            problems.append(f"{func.name}() complexity {score} > {MAX_COMPLEXITY} (D14)")
    names = {n.targets[0].id for n in tree.body
             if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    if "PRACTICE_IMPL" not in names:
        problems.append("no PRACTICE_IMPL (D13)")
    return problems + ci_hazards(tree)


def audit_explain(exercise, readme, headings) -> list:
    """DESIGN §6's gate for prose items: a *resolvable* citation.

    The answer must name a lesson section, that section must actually exist in
    the reference `docs/en.md`, and the README must carry the answer. A citation
    that points nowhere is the prose equivalent of a test that asserts nothing.
    """
    problems = []
    body = readme.read_text(encoding="utf-8") if readme.is_file() else ""
    if exercise.cites not in body:
        problems.append(f"ex{exercise.index:02d}: citation {exercise.cites!r} "
                        f"not answered in README")
    if headings is not None and exercise.cites not in headings:
        problems.append(f"ex{exercise.index:02d}: cites {exercise.cites!r}, which is not "
                        f"a heading in the lesson's docs/en.md")
    return problems


def _reference_headings(pack):
    """The lesson's own section titles, or None if the reference is unreachable."""
    try:
        text = parity.doc_text(pack.phase, pack.lesson, "en")
    except Exception:
        return None
    return {line.lstrip("#").strip() for line in text.splitlines()
            if line.startswith("#")}


def plain_scalars_with_colons(man_path: pathlib.Path) -> list:
    """Prose keys holding a colon-space in a *plain* scalar are not YAML.

    `harness/yamlite` reads them happily; every strict YAML parser stops with
    "mapping values are not allowed here". Since a manifest is a data file other
    tools are entitled to read, a `verifies` or `cites` that contains ": " has to be
    a block scalar. `finalize_practice` writes `>-` for exactly this reason.
    """
    offenders = []
    for number, line in enumerate(man_path.read_text(encoding="utf-8").splitlines(), 1):
        match = re.match(r"^    (verifies|cites): (?![|>])(.*)$", line)
        if match and ": " in match.group(2):
            offenders.append(f"line {number}: {match.group(1)!r} holds ': ' in a plain scalar, "
                             "which no strict YAML parser accepts (use '>-')")
    return offenders


def audit_lesson(man_path: pathlib.Path, warnings=None) -> list:
    """Returns the problems that fail the gate; soft findings go to `warnings` if given."""
    pack = manifest.load_practice(man_path)
    directory = man_path.parent
    problems, warnings = [], [] if warnings is None else warnings
    readme = directory / "README.md"
    if not readme.is_file():
        problems.append(f"{directory}: no README.md")
    problems += plain_scalars_with_colons(man_path)
    headings = _reference_headings(pack) if any(
        e.kind == "explain" for e in pack.exercises) else None
    for ex in pack.exercises:
        if ex.kind == "explain":
            problems += audit_explain(ex, readme, headings)
            continue
        path = directory / ex.filename
        if not path.is_file():
            problems.append(f"ex{ex.index:02d}: missing {ex.filename} (D10)")
            continue
        soft = []
        problems += [f"{ex.filename}: {p}" for p in audit_solution(path, ex, soft)]
        warnings += [f"{ex.filename}: {w}" for w in soft]
        for fixture in ex.fixtures:
            if not (directory / fixture).is_file():
                problems.append(f"ex{ex.index:02d}: fixture {fixture} missing")
            elif '"_meta"' not in (directory / fixture).read_text(encoding="utf-8"):
                problems.append(f"ex{ex.index:02d}: fixture {fixture} is unlabelled (D14)")
    tests = directory / "tests"
    n_tests = len(list(tests.glob("test_*.py"))) if tests.is_dir() else 0
    if n_tests < 1:
        problems.append(f"{directory}: no tests/test_*.py")
    return problems


def main(argv=None) -> int:
    paths = runner._manifests(argv[0] if argv else None)
    if not paths:
        print("audit_practice: no manifests found", file=sys.stderr)
        return 1
    failed = 0
    for path in paths:
        warnings = []
        problems = audit_lesson(path, warnings)
        label = path.parent.parent.name
        failed += bool(problems)
        print(f"{'FAIL' if problems else 'warn' if warnings else 'ok  '} {label}")
        for line in problems + warnings:
            print(f"     {line}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
