"""Exercise 6 — lint reads the prose and the artifact layer reads a fixture.

    Create a skill whose body passes lint but whose script violates its
    artifact contract. Identify which release layer blocks it.

Reading of the exercise: to answer "which layer" the package has to reach
that layer, so the bundle is built to lint completely clean -- both required
sections, every companion referenced, nothing orphaned -- and the violation
is put in the one place lint does not look. Running the script and feeding
its real output to the gate then names the layer, and shows that the layer
only works when someone does exactly that.

**ANSWER: the artifact-contract layer blocks it, and only because the real
output was used.** `lint_package` returns `valid=True` with **0** issues; the
script's output is missing the `Evidence` heading, so `with_skill_artifact`
and `artifact_improvement` are the **2** failing checks of **11**, both in
the artifact layer. Swapping in a hand-written fixture that satisfies the
contract turns every check green while the script is unchanged.

**FINDING: lint reads the prose and never the code.** It checks the
frontmatter, the two required sections, the referenced paths, file types,
sizes and obvious secrets -- and the only thing it reads inside
`scripts/render.py` is a regex for credential material. A script that
contradicts the `## Output contract` section directly above it is invisible
to the layer that just validated that section.

**FINDING: the artifact layer grades a string the caller supplies.**
`compare_artifacts` takes `baseline` and `with_skill` as arguments and
`EvaluationProvenance.artifact_mode` defaults to `"fixture"`. Nothing in the
gate runs the script, so the layer that blocks this package blocks it only if
the person wiring the gate pasted the real output in.

**FINDING: `artifact_improvement` is a second question, not a second
opinion.** A failing `with_skill` artifact forces it false as well, which is
why the two fail together here. It earns its place in the other direction:
with a baseline that already passes, `with_skill_artifact` is `True` and
`artifact_improvement` is the only failure -- a package that produces a
correct artifact nobody needed.

Structure: `render()` is the script's behaviour, executed rather than
described, so the artifact handed to the gate is the artifact the package
produces.
"""

from __future__ import annotations

import pathlib
import shutil
import tempfile

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "27-skill-evals-packaging-and-portability"
SCRIPT = '''"""Render the release summary."""


def render(findings):
    lines = ["# Decision", "", "Pass." if findings == 0 else "Hold."]
    return "\\n".join(lines) + "\\n"
'''
BODY = """# Release summary

Run `scripts/render.py` and read `references/format.md` before writing.

## Output contract

A Markdown document with a `# Decision` heading and a `# Evidence` heading
naming precision and recall.

## Failure behavior

If the findings count is unavailable, stop and report that instead.
"""
FIXTURE = "# Decision\n\nPass.\n\n# Evidence\n\nPrecision: 1.0. Recall: 1.0.\n"


def render(findings):
    """What scripts/render.py actually produces, executed rather than described."""
    namespace = {}
    exec(compile(SCRIPT, "render.py", "exec"), namespace)  # noqa: S102 - the package's own code
    return namespace["render"](findings)


def plant(root):
    (root / "references").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "SKILL.md").write_text(
        f"---\nname: release-summary\ndescription: Summarize a release decision.\n"
        f"---\n\n{BODY}", encoding="utf-8")
    (root / "references" / "format.md").write_text("# Format\n", encoding="utf-8")
    (root / "scripts" / "render.py").write_text(SCRIPT, encoding="utf-8")
    return root


def gate(ref, source, installed, artifact, baseline="Release looks fine."):
    cases = (ref.TriggerCase("pos", "summarize this release decision", True),
             ref.TriggerCase("near", "install the release dependencies", False))
    checks = (ref.EvidenceCheck("fixture", True, "Deterministic fixture passed."),)
    contract = ref.ArtifactContract(required_headings=("Decision", "Evidence"),
                                    required_terms=("precision", "recall"))
    return ref.run_release_gate(
        source, cases, ref.KeywordRouter(("summarize", "release", "decision"), 2), 1,
        baseline, artifact, contract, ref.PackageRequirements(),
        (ref.HostCapabilities("native", True, True, True),), checks, checks, installed,
        ref.build_manifest(source))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    with tempfile.TemporaryDirectory() as temp:
        workspace = pathlib.Path(temp)
        source = plant(workspace / "release-summary")
        installed = workspace / "installed" / "release-summary"
        installed.parent.mkdir()
        shutil.copytree(source, installed)
        lint, produced = ref.lint_package(source), render(0)
        real = gate(ref, source, installed, produced)
        faked = gate(ref, source, installed, FIXTURE)
        redundant = gate(ref, source, installed, FIXTURE, baseline=FIXTURE)
        evaluated = ref.evaluate_artifact(produced, ref.ArtifactContract(
            required_headings=("Decision", "Evidence"),
            required_terms=("precision", "recall")))
        return {
            "lint_valid": lint.valid, "lint_references": list(lint.references),
            "lint_issues": [issue.code for issue in lint.issues],
            "missing_headings": evaluated["missing_headings"],
            "real_failed": sorted(name for name, ok in real["checks"].items() if not ok),
            "faked_failed": sorted(name for name, ok in faked["checks"].items() if not ok),
            "checks": len(real["checks"]), "artifact_mode": real["provenance"]["artifactMode"],
            "improvement": faked["checks"]["artifact_improvement"],
            "baseline_passed": faked["artifacts"]["baseline"]["passed"],
            "redundant_failed": sorted(name for name, ok in redundant["checks"].items()
                                       if not ok),
            "redundant_baseline": redundant["artifacts"]["baseline"]["passed"],
            "script_read": "render" in ref.lint_package(source).references[1],
        }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the artifact-contract layer blocks it, and only with the real output",
            all([result["lint_valid"], result["lint_issues"] == [],
                 result["real_failed"] == ["artifact_improvement",
                                           "with_skill_artifact"],
                 result["faked_failed"] == [], result["checks"] == 11,
                 result["missing_headings"] == ["Evidence"]]),
            f"lint_package returns valid={result['lint_valid']} with "
            f"{len(result['lint_issues'])} issues, and the script's real output is missing "
            f"{result['missing_headings']} -- so {result['real_failed']} are the failing "
            f"checks of {result['checks']}, both in the artifact layer. A fixture turns "
            "every check green with the script unchanged",
        ),
        practice.Check(
            "FINDING: lint reads the prose and never the code",
            all([result["lint_valid"], result["script_read"],
                 result["lint_references"] == ["references/format.md",
                                               "scripts/render.py"],
                 result["missing_headings"] == ["Evidence"]]),
            f"lint resolves {result['lint_references']}, checks the two required sections, "
            "the file types, the sizes and obvious secrets -- and the only thing it reads "
            "inside the script is a credential regex. A script contradicting the "
            "## Output contract section above it is invisible to the layer that just "
            "validated that section",
        ),
        practice.Check(
            "FINDING: the artifact layer grades a string the caller supplies",
            all([result["artifact_mode"] == "fixture",
                 result["real_failed"] != result["faked_failed"]]),
            f"compare_artifacts takes baseline and with_skill as arguments and "
            f"EvaluationProvenance.artifact_mode defaults to {result['artifact_mode']!r}. "
            f"Nothing in the gate runs the script, so the same package fails "
            f"{result['real_failed']} or nothing at all, depending on which string was "
            "pasted in",
        ),
        practice.Check(
            "FINDING: artifact_improvement is a second question, not a second opinion",
            all([result["improvement"], not result["baseline_passed"],
                 result["redundant_baseline"],
                 result["redundant_failed"] == ["artifact_improvement"],
                 "artifact_improvement" in result["real_failed"]]),
            f"a failing with_skill artifact forces artifact_improvement false too, which "
            f"is why {result['real_failed']} fail together. It earns its place in the "
            f"other direction: with a baseline that already passes the gate fails "
            f"{result['redundant_failed']} alone -- a correct artifact nobody needed",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
