"""Exercise 7 — authority is a two-table function, and neither table is a document.

    Write `AGENTS.md` for repository-wide maintenance rules and a separate
    skill bundle for the reusable research procedure. Explain why neither
    file grants tool authority.

Reading of the exercise: "explain why" is worth writing only if the
explanation is checkable, so the claim is turned into a closed question --
which names does the decision actually read? -- and answered from the
reference's own AST. Both artifacts are then written for real, because the
second half of the answer is that the two documents differ from each other in
a way the first half does not care about at all.

**ANSWER: both artifacts written, and the gateway's answers are unchanged.**
Alice is allowed and bob refused `research:write` identically with the files
present and absent. `gateway_call` reads **4** module globals
directly -- `USERS`, `REQUIRED_SCOPE`, `TOOLS`, `AUDIT` -- and `PINNED`
through `pin_ok`, and **0** of the five is a document, a path or a
filesystem call. Authority is a lookup in two tables,
and a Markdown file cannot be in either.

**FINDING: the two documents differ by version, not by content.** The bundle
carries `name`, `description` and `version`; `AGENTS.md` carries **0**
frontmatter fields. So the procedure can be pinned, diffed and selected by a
router, and the repository rules can only be read -- which is the reason they
are two files rather than two sections.

**FINDING: a directory bundle is invisible to the course installer.** The
bundle is **4** files and **0** of them matches the flat `skill-*.md`
convention the catalog globs for. The lesson's own output artifact is flat
for that reason, and the lesson says so -- the portable format and the
catalog format are simply different contracts.

**FINDING: the minimal parser reads top-level keys, so nesting them loses
them.** Flat frontmatter yields **6** keys; the same fields under `metadata:`
yield **2**, one of them the `metadata` key itself, and `--tag capstone` then matches nothing. A safe YAML parser
would read both and still not agree with the catalog's schema.

Structure: `reads()` answers the authority question from the AST;
`frontmatter()` is the minimal parser the docs describe, applied to both
layouts.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import tempfile
import textwrap

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "23-capstone-tool-ecosystem"
AGENTS = """# Repository agent rules

- Run `python3 code/main.py` before proposing a change to this lesson.
- Keep `code/` stdlib-only; new dependencies belong in a separate lesson.
- Never widen a tool description without re-pinning its hash in `PINNED`.
- Record allow and deny cases together when a policy check changes.
"""
SKILL = """---
name: arxiv-research-report
description: Search arXiv, delegate a summary, and return a report with its trace.
version: "1.0.0"
---

# arXiv research report

1. Call `arxiv_search` with the user's keywords; stop if it returns no hits.
2. Delegate the summary and record the boundary span.
3. Poll the task at its declared `pollIntervalMs` until it is terminal.
4. Save the trace id with the report; a report without one is not evidence.
"""
BUNDLE = {"SKILL.md": SKILL, "references/contracts.md": "# Tool contracts\n",
          "scripts/poll.py": "# poll a task id\n", "evals/cases.jsonl": "{}\n"}


def reads(ref, name):
    """The module globals a function actually loads -- the authority question."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(ref, name))))
    return sorted({node.id for node in ast.walk(tree)
                   if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                   and node.id in vars(ref)})


def frontmatter(text):
    """The minimal parser the lesson describes: top-level `key: value` lines only."""
    if not text.startswith("---\n"):
        return {}
    block = text.split("---\n", 2)[1]
    pairs = [line.split(":", 1) for line in block.splitlines()
             if line and not line.startswith((" ", "\t")) and ":" in line]
    return {key.strip(): value.strip() for key, value in pairs}


def write_bundle(root):
    for name, body in BUNDLE.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def decide(ref, token):
    result = ref.gateway_call(token, "generate_report", {}, ref._hex(16), None,
                              ref.request_meta(tasks=True))
    return result.get("error", result.get("status"))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    before = [decide(ref, "tok_alice"), decide(ref, "tok_bob")]
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        (root / "AGENTS.md").write_text(AGENTS, encoding="utf-8")
        files = write_bundle(root / "arxiv-research-report")
        after = [decide(ref, "tok_alice"), decide(ref, "tok_bob")]
        installable = sorted(p.name for p in (root / "arxiv-research-report").glob("skill-*.md"))

    catalog = "\n".join(["---", "name: ecosystem-blueprint", "description: One page.",
                         'version: "1.0.0"', 'phase: "13"', 'lesson: "23"',
                         "tags: [mcp, capstone]", "---", ""])
    nested = "\n".join(["---", "name: ecosystem-blueprint", "metadata:",
                        '  version: "1.0.0"', "  tags: [mcp, capstone]", "---", ""])
    globals_read = reads(ref, "gateway_call")
    return {
        "before": before, "after": after, "unchanged": before == after,
        "reads": globals_read, "pin_tables": [n for n in reads(ref, "pin_ok") if n.isupper()],
        "tables": [name for name in globals_read if name.isupper()],
        "document_names": [name for name in globals_read
                           if any(word in name.lower()
                                  for word in ("path", "file", "doc", "skill", "agents"))],
        "bundle_files": files, "agents_fields": len(frontmatter(AGENTS)),
        "skill_fields": sorted(frontmatter(SKILL)),
        "installable": installable,
        "flat_keys": sorted(frontmatter(catalog)),
        "nested_keys": sorted(frontmatter(nested)),
        "tag_match": "tags" in frontmatter(nested),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: both artifacts written, and the gateway's answers are unchanged",
            all([result["unchanged"], result["before"][0] == "working",
                 result["before"][1] == "insufficient_scope",
                 sorted(result["tables"]) == ["AUDIT", "REQUIRED_SCOPE", "TOOLS", "USERS"],
                 result["pin_tables"] == ["PINNED"],
                 result["document_names"] == []]),
            f"alice gets {result['before'][0]!r} and bob {result['before'][1]!r}, identically "
            f"with the files present and absent. gateway_call reads {result['reads']}, whose "
            f"data tables are {sorted(result['tables'])} plus {result['pin_tables']} through "
            f"pin_ok, and of which {len(result['document_names'])} is a document, a path or "
            f"a filesystem call. "
            "Authority is a lookup in two tables and a Markdown file cannot be in either",
        ),
        practice.Check(
            "FINDING: the two documents differ by version, not by content",
            all([result["agents_fields"] == 0,
                 result["skill_fields"] == ["description", "name", "version"]]),
            f"the bundle carries {result['skill_fields']} and AGENTS.md carries "
            f"{result['agents_fields']} frontmatter fields. The procedure can be pinned, "
            "diffed and selected by a router; the repository rules can only be read -- which "
            "is why they are two files rather than two sections of one",
        ),
        practice.Check(
            "FINDING: a directory bundle is invisible to the course installer",
            all([len(result["bundle_files"]) == 4, result["installable"] == [],
                 "SKILL.md" in result["bundle_files"]]),
            f"the bundle is {result['bundle_files']} and {len(result['installable'])} of "
            "those match the flat skill-*.md convention the catalog globs for. The lesson's "
            "own output artifact is flat for that reason: the portable format and the "
            "catalog format are different contracts, not two spellings of one",
        ),
        practice.Check(
            "FINDING: the minimal parser reads top-level keys, so nesting them loses them",
            all([len(result["flat_keys"]) == 6, len(result["nested_keys"]) == 2,
                 not result["tag_match"]]),
            f"flat frontmatter yields {result['flat_keys']} and the same fields under "
            f"metadata: yield {result['nested_keys']}, so --tag capstone matches nothing and "
            "the catalog records an empty version. A safe YAML parser would read both and "
            "still not agree with the catalog's schema",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
