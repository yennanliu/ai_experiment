"""Exercise 4 — a string check cannot see a link, because the link is a fact.

    Add a symbolic-link policy for a real filesystem implementation. Explain
    why URI lexical containment alone cannot stop a symlink escape.

Reading of the exercise: the explanation is demonstrated rather than argued,
so a real temporary directory is built with a real symlink pointing out of it
and the lesson's own `uri_within_workspace` is asked about the escaping path.
The policy is then written against that failure, and the policy's *own*
residual weakness is measured too, because "resolve then check" is the answer
most people stop at.

**WHY LEXICAL CONTAINMENT CANNOT WORK: the check and the fact live in
different places.** `uri_within_workspace` is `posixpath.normpath`, `unquote`
and `commonpath` -- **0** filesystem calls. It decides what a path *spells*.
A symlink is a decision the filesystem makes about what a name *means*, and no
amount of string analysis can read it, because the same string means different
things on two machines, or on one machine a second later. `normpath` even
makes this worse on purpose: it cancels `..` textually, which is correct for a
pure path and wrong the moment a component is a link.

**ANSWER: `uri_within_workspace` says the escaping path is inside, and
`realpath` says it is not.** A link at `<root>/escape.md` pointing to a file
outside answers **True** lexically and resolves to a path the same predicate
calls **False**. The policy is to resolve first and compare the resolved path,
and to refuse a resolution that leaves the resolved root.

**FINDING: the policy has to resolve the root as well.** If the workspace root
is itself reached through a link, comparing a resolved candidate against an
unresolved root fails a legitimate path. Resolving both makes the containment
question well-posed: **1** comparison, both sides in the same namespace.

**FINDING: resolve-then-check is still time-of-check to time-of-use.**
Repointing the link after the check passes and before the open leaves the
policy's verdict stale -- the same name resolves somewhere else, and the
check has already returned. The durable fix is not a better predicate but
opening the file and verifying the handle, `O_NOFOLLOW` or a `st_dev`/`st_ino`
comparison after the fact.

Structure: `lexical` and `resolved` are the two containment predicates over
one real directory tree; `uri` turns a path into the `file://` URI the lesson
compares.
"""

from __future__ import annotations

import inspect
import os
import pathlib
import tempfile

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "12-mcp-roots-and-elicitation"


def uri(path):
    return pathlib.Path(path).as_uri()


def lexical(ref, root, candidate):
    """The lesson's check: strings only."""
    return ref.uri_within_workspace(uri(root), uri(candidate))


def resolved(ref, root, candidate):
    """The policy: resolve both sides, then ask the same question."""
    try:
        real_root = os.path.realpath(root)
        real_candidate = os.path.realpath(candidate)
    except OSError:
        return False
    return ref.uri_within_workspace(uri(real_root), uri(real_candidate))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    source = inspect.getsource(ref._normalized_uri_parts) + inspect.getsource(
        ref.uri_within_workspace)
    with tempfile.TemporaryDirectory() as base:
        root = pathlib.Path(base) / "workspace"
        (root / "notes").mkdir(parents=True)
        inside = root / "notes" / "ok.md"
        inside.write_text("inside")
        outside = pathlib.Path(base) / "secret.md"
        outside.write_text("outside")
        escape = root / "escape.md"
        escape.symlink_to(outside)

        linked_root = pathlib.Path(base) / "linked-workspace"
        linked_root.symlink_to(root)

        swapped = pathlib.Path(base) / "swap.md"
        swapped.write_text("swap")
        moving = root / "moving.md"
        moving.symlink_to(inside)
        before_swap = resolved(ref, root, moving)
        moving.unlink()
        moving.symlink_to(swapped)  # the link is repointed after the check returned
        return {
            "fs_calls": sum(token in source for token in ("os.", "realpath", "lstat")),
            "inside_lexical": lexical(ref, root, inside),
            "inside_resolved": resolved(ref, root, inside),
            "escape_lexical": lexical(ref, root, escape),
            "escape_resolved": resolved(ref, root, escape),
            "escape_target": os.path.realpath(escape) == os.path.realpath(outside),
            "linked_root_lexical": lexical(ref, linked_root, inside),
            "linked_root_resolved": resolved(ref, linked_root, inside),
            "before_swap": before_swap, "after_swap": resolved(ref, root, moving),
            "swap_target": os.path.realpath(moving) == os.path.realpath(swapped),
        }


def verify(result):
    return [
        practice.Check(
            "ANSWER: lexical containment calls the escape inside, and realpath does not",
            all([result["escape_lexical"], not result["escape_resolved"],
                 result["escape_target"], result["inside_lexical"],
                 result["inside_resolved"]]),
            f"a link at <root>/escape.md pointing outside answers "
            f"{result['escape_lexical']} lexically and {result['escape_resolved']} once "
            f"resolved, while an ordinary file inside answers {result['inside_lexical']} to "
            "both. The policy is to resolve first and compare the resolved paths",
        ),
        practice.Check(
            "WHY: the predicate makes no filesystem calls at all",
            result["fs_calls"] == 0,
            f"uri_within_workspace and _normalized_uri_parts contain {result['fs_calls']} "
            "references to os., realpath or lstat -- they are normpath, unquote and "
            "commonpath. They decide what a path spells; a symlink is the filesystem's "
            "decision about what a name means, and normpath cancels '..' textually, which "
            "is right for a pure path and wrong when a component is a link",
        ),
        practice.Check(
            "FINDING: the policy has to resolve the root as well",
            all([not result["linked_root_lexical"], result["linked_root_resolved"]]),
            f"with the workspace root itself reached through a link, a legitimate inside "
            f"file answers {result['linked_root_lexical']} against the unresolved root and "
            f"{result['linked_root_resolved']} once both sides are resolved. Resolving one "
            "side makes the question ill-posed in the other direction",
        ),
        practice.Check(
            "FINDING: resolve-then-check is still time-of-check to time-of-use",
            all([result["before_swap"], not result["after_swap"], result["swap_target"]]),
            f"a link that resolved inside ({result['before_swap']}) and is repointed before "
            f"the open resolves outside ({result['after_swap']}). The verdict was correct "
            "when it was made and is stale when it is used, so the durable fix is opening "
            "the file and checking the handle -- O_NOFOLLOW, or st_dev/st_ino after the "
            "fact -- rather than a better predicate",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
