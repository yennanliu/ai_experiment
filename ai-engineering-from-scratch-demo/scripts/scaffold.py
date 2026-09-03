#!/usr/bin/env python3
"""Generate a demo skeleton from a lesson path. (Section 7, step 1)

    python scripts/scaffold.py phases/09-reinforcement-learning/03-q-learning

Reads the lesson's own `docs/en.md` and `code/`, guesses the tier and dependency
group from what the lesson imports, and writes a `demo.yaml`, a `run.py` stub, a
test stub and a README stub. What it cannot guess it leaves as a `TODO:` marker,
which `audit_demos.py` then refuses to pass -- a scaffold is a starting point,
never a deliverable.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.parity import reference_repo  # noqa: E402
from harness.runner import DEMOS_ROOT  # noqa: E402

# What a lesson imports is a good enough first guess at what it costs to run.
TIER_SIGNALS = [
    (("anthropic", "openai", "mcp"), "T2", "llm"),
    (("diffusers", "bitsandbytes", "nerfstudio"), "T3", "vision"),
    (("transformers", "peft", "accelerate"), "T1", "llm"),
    (("torchaudio", "librosa", "whisper"), "T1", "audio"),
    (("torchvision", "PIL", "cv2"), "T1", "vision"),
    (("torch",), "T1", "llm"),
    (("sklearn", "scipy", "numpy"), "T0", "math"),
]
DEFAULT_TIER, DEFAULT_GROUP = "T0", "math"

RUNTIME_BY_TIER = {"T0": 10, "T1": 120, "T2": 45, "T3": 1800}


def lesson_title(doc: Path) -> str:
    if doc.is_file():
        for line in doc.read_text(encoding="utf-8").splitlines():
            if line.startswith("# "):
                return line[2:].strip()
    return "TODO: title"


def guess_tier(code_dir: Path) -> tuple[str, str]:
    imports: set[str] = set()
    for source in code_dir.glob("*.py"):
        for match in re.finditer(r"^\s*(?:import|from)\s+([\w.]+)",
                                 source.read_text(encoding="utf-8"), re.MULTILINE):
            imports.add(match.group(1).split(".")[0])
    for names, tier, group in TIER_SIGNALS:
        if imports & set(names):
            return tier, group
    return DEFAULT_TIER, DEFAULT_GROUP


def biggest_code_file(code_dir: Path) -> Path | None:
    candidates = [p for p in code_dir.glob("*.py") if p.name != "__init__.py"]
    return max(candidates, key=lambda p: p.stat().st_size) if candidates else None


def scaffold(lesson: str, *, force: bool = False) -> Path:
    reference = reference_repo()
    lesson_dir = reference / lesson
    if not lesson_dir.is_dir():
        raise SystemExit(f"no such lesson in the reference repo: {lesson}")

    target = DEMOS_ROOT / lesson
    if target.exists() and not force:
        raise SystemExit(f"{target} already exists (pass --force to overwrite)")
    (target / "tests").mkdir(parents=True, exist_ok=True)

    doc = lesson_dir / "docs" / "en.md"
    code_dir = lesson_dir / "code"
    tier, group = guess_tier(code_dir)
    primary = biggest_code_file(code_dir)
    parity = f"{lesson}/code/{primary.name}" if primary else None
    doc_sha = (
        hashlib.sha256(doc.read_bytes()).hexdigest()[:16] if doc.is_file() else ""
    )

    manifest = [
        f"lesson: {lesson}",
        f"title: {lesson_title(doc)}",
        f"tier: {tier}",
        "entrypoint: run.py",
        f"runtime_seconds: {RUNTIME_BY_TIER[tier]}",
        "needs_env: []" if tier != "T2" else "needs_env:\n  - ANTHROPIC_API_KEY",
        f"deps_group: {group}",
    ]
    if tier == "T2":
        manifest.append("cassette: responses.json")
    if tier == "T3":
        manifest.append("skip_reason: >\n  TODO: name the GPU this needs and what renting it costs.")
    manifest.append(
        "proves: >\n  TODO: one sentence. What does running this establish that reading\n"
        "  the lesson does not? If you cannot finish this sentence, the demo is\n"
        "  not worth writing."
    )
    if parity:
        manifest.append(f"parity_with: {parity}")
    if doc_sha:
        manifest += [f"reference_doc: {lesson}/docs/en.md",
                     f"reference_doc_sha256: {doc_sha}"]
    (target / "demo.yaml").write_text("\n".join(manifest) + "\n", encoding="utf-8")

    depth = len(Path(lesson).parts) + 1   # demos/ + phases/<phase>/<lesson>
    (target / "run.py").write_text(
        RUN_TEMPLATE.format(lesson=lesson, parity=parity or "", depth=depth),
        encoding="utf-8",
    )
    (target / "tests" / "test_parity.py").write_text(
        TEST_TEMPLATE.format(lesson=lesson, depth=depth), encoding="utf-8"
    )
    (target / "README.md").write_text(
        README_TEMPLATE.format(title=lesson_title(doc), lesson=lesson, tier=tier,
                               group=group),
        encoding="utf-8",
    )
    return target


RUN_TEMPLATE = '''"""TODO: one paragraph -- what the lesson builds by hand, and what this runs
instead. Delete this file rather than shipping it half-written.

Run:  uv run demo run {lesson}
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[{depth}]))

from harness.explain import explain          # noqa: E402
from harness.parity import assert_close, load_reference, report  # noqa: E402

LESSON = "{parity}"


def main() -> int:
    ref = load_reference(LESSON)  # noqa: F841 -- TODO: use it
    checks = []
    # TODO: run the lesson's implementation and the library's on the same input,
    # then assert_close(mine, theirs, label=..., atol=...) for each claim.
    raise NotImplementedError("scaffolded, not written")
    report(checks, title="{lesson}")
    return 0


if __name__ == "__main__":
    if explain(__file__):
        raise SystemExit(0)
    raise SystemExit(main())
'''

TEST_TEMPLATE = '''"""TODO: at least three assertions on what the lesson *claims*, not on plumbing."""

import sys
from pathlib import Path

DEMO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO))
sys.path.insert(0, str(DEMO.parents[{depth_minus_one}]))

import pytest  # noqa: E402


@pytest.mark.skip(reason="TODO: scaffolded, not written")
def test_the_lessons_claim_holds():
    raise NotImplementedError
'''.replace("{depth_minus_one}", "{depth}")

README_TEMPLATE = """# {title}

**Lesson:** `{lesson}`
**Tier:** {tier} · **Install:** `uv sync --extra {group}`

## What it proves

TODO: two or three sentences, and the headline number the demo prints.

## Run

```bash
uv run demo run {lesson}
uv run python run.py --explain
```
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("lesson", help="e.g. phases/09-reinforcement-learning/03-q-learning")
    parser.add_argument("--force", action="store_true", help="overwrite an existing demo")
    args = parser.parse_args()

    target = scaffold(args.lesson.rstrip("/"), force=args.force)
    print(f"scaffolded {target}")
    print("Next: fill every TODO, then `python scripts/audit_demos.py` will tell you")
    print("what is still missing. A scaffold does not pass the audit on its own.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
