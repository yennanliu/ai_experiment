"""Read and validate `demo.yaml` (D3) with the standard library only.

`demo.yaml` is a *closed* schema, not arbitrary YAML, so a full parser would be
a dependency we cannot afford (see the module docstring in `harness/__init__`).
`load_yaml` therefore implements the strict subset the schema uses:

    key: scalar          # scalars are str / int / float / bool / null
    key: [a, b]          # inline list of scalars
    key:                 # block list
      - a
      - b
    key: >               # folded block scalar (newlines become spaces)
      text
    key: |               # literal block scalar (newlines kept)
      text

Anything outside that subset raises `ManifestError` rather than being silently
misread -- a demo whose manifest cannot be parsed must fail loudly, because the
manifest is what CI, the dependency check and the coverage report all read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

TIERS = ("T0", "T1", "T2", "T3")
DEPS_GROUPS = ("core", "math", "vision", "audio", "llm", "agents", "infra")

REQUIRED_FIELDS = ("lesson", "title", "tier", "runtime_seconds", "deps_group", "proves")
KNOWN_FIELDS = REQUIRED_FIELDS + (
    "entrypoint",
    "needs_env",
    "parity_with",
    "reference_doc",
    "reference_doc_sha256",
    "cassette",
    "skip_reason",
)


class ManifestError(ValueError):
    """A demo.yaml is missing, unparseable, or violates the schema."""


# --------------------------------------------------------------------------
# minimal YAML subset
# --------------------------------------------------------------------------

_INT_RE = re.compile(r"^-?\d+$")
_FLOAT_RE = re.compile(r"^-?\d+\.\d+$")


def _scalar(raw: str):
    """Convert one YAML scalar token to a Python value."""
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "'\"":
        return text[1:-1]
    if text in ("", "null", "~"):
        return None
    if text in ("true", "True"):
        return True
    if text in ("false", "False"):
        return False
    if _INT_RE.match(text):
        return int(text)
    if _FLOAT_RE.match(text):
        return float(text)
    return text


def _strip_comment(line: str) -> str:
    """Drop a trailing `# comment`, respecting quotes."""
    quote = None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1].isspace()):
            return line[:i]
    return line


def load_yaml(text: str) -> dict:
    """Parse the supported YAML subset into a flat dict."""
    lines = text.splitlines()
    out: dict = {}
    i = 0
    while i < len(lines):
        raw = lines[i]
        i += 1
        if not raw.strip() or raw.lstrip().startswith("#") or raw.strip() == "---":
            continue
        if raw[:1].isspace() or raw.startswith("-"):
            raise ManifestError(f"unexpected indentation at top level: {raw!r}")

        stripped = _strip_comment(raw)
        if ":" not in stripped:
            raise ManifestError(f"line is not `key: value`: {raw!r}")
        key, _, rest = stripped.partition(":")
        key = key.strip()
        rest = rest.strip()

        if rest in (">", "|", ">-", "|-"):
            block, i = _read_block(lines, i)
            joiner = " " if rest.startswith(">") else "\n"
            value = joiner.join(block)
            out[key] = value.strip() if rest.startswith(">") else value.rstrip("\n")
        elif rest.startswith("[") and rest.endswith("]"):
            inner = rest[1:-1].strip()
            out[key] = [_scalar(p) for p in inner.split(",")] if inner else []
        elif rest == "":
            items, i = _read_list(lines, i)
            out[key] = items
        else:
            out[key] = _scalar(rest)
    return out


def _read_block(lines: list[str], i: int) -> tuple[list[str], int]:
    """Consume an indented block scalar starting at `i`."""
    body: list[str] = []
    indent = None
    while i < len(lines):
        line = lines[i]
        if line.strip() and not line[:1].isspace():
            break
        if line.strip():
            here = len(line) - len(line.lstrip())
            indent = here if indent is None else min(indent, here)
        body.append(line)
        i += 1
    while body and not body[-1].strip():
        body.pop()
    return [b[indent:] if len(b) >= (indent or 0) else "" for b in body], i


def _read_list(lines: list[str], i: int) -> tuple[list, int]:
    """Consume an indented `- item` block starting at `i`. Empty block -> []."""
    items: list = []
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if not line[:1].isspace():
            break
        entry = _strip_comment(line).strip()
        if not entry.startswith("- "):
            raise ManifestError(f"expected a `- item` list entry, got: {line!r}")
        items.append(_scalar(entry[2:]))
        i += 1
    return items, i


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Demo:
    """One demo directory, as declared by its manifest."""

    lesson: str
    title: str
    tier: str
    runtime_seconds: int
    deps_group: str
    proves: str
    path: Path
    entrypoint: str = "run.py"
    needs_env: list[str] = field(default_factory=list)
    parity_with: str | None = None
    reference_doc: str | None = None
    reference_doc_sha256: str | None = None
    cassette: str | None = None
    skip_reason: str | None = None

    @property
    def phase(self) -> str:
        """`11-llm-engineering` for a lesson under `phases/11-llm-engineering/`."""
        return self.lesson.split("/")[1]

    @property
    def phase_number(self) -> str:
        return self.phase.split("-")[0]

    @property
    def entrypoint_path(self) -> Path:
        return self.path / self.entrypoint

    @property
    def has_parity(self) -> bool:
        return self.parity_with is not None


def parse(text: str, path: Path) -> Demo:
    """Validate a manifest body and build a `Demo`. Raises `ManifestError`."""
    data = load_yaml(text)

    unknown = sorted(set(data) - set(KNOWN_FIELDS))
    if unknown:
        raise ManifestError(f"{path}: unknown field(s): {', '.join(unknown)}")
    missing = [f for f in REQUIRED_FIELDS if data.get(f) in (None, "")]
    if missing:
        raise ManifestError(f"{path}: missing required field(s): {', '.join(missing)}")

    if data["tier"] not in TIERS:
        raise ManifestError(f"{path}: tier must be one of {TIERS}, got {data['tier']!r}")
    if data["deps_group"] not in DEPS_GROUPS:
        raise ManifestError(
            f"{path}: deps_group must be one of {DEPS_GROUPS}, got {data['deps_group']!r}"
        )
    if not isinstance(data["runtime_seconds"], int) or data["runtime_seconds"] <= 0:
        raise ManifestError(f"{path}: runtime_seconds must be a positive integer")
    if not str(data["lesson"]).startswith("phases/"):
        raise ManifestError(f"{path}: lesson must be a `phases/...` path")

    needs_env = data.get("needs_env") or []
    if not isinstance(needs_env, list):
        raise ManifestError(f"{path}: needs_env must be a list")

    # A T2 demo that never names a cassette cannot run in replay mode, which
    # would quietly turn every CI run into a live billed API call (D4).
    if data["tier"] == "T2" and not data.get("cassette"):
        raise ManifestError(f"{path}: tier T2 requires a `cassette` field")
    # A T3 demo must say what it would have done, or it is just a crash (D8).
    if data["tier"] == "T3" and not data.get("skip_reason"):
        raise ManifestError(f"{path}: tier T3 requires a `skip_reason` field")

    return Demo(
        lesson=data["lesson"],
        title=data["title"],
        tier=data["tier"],
        runtime_seconds=data["runtime_seconds"],
        deps_group=data["deps_group"],
        proves=data["proves"],
        path=path.parent,
        entrypoint=data.get("entrypoint") or "run.py",
        needs_env=[str(e) for e in needs_env],
        parity_with=data.get("parity_with"),
        reference_doc=data.get("reference_doc"),
        reference_doc_sha256=data.get("reference_doc_sha256"),
        cassette=data.get("cassette"),
        skip_reason=data.get("skip_reason"),
    )


def load(path: Path) -> Demo:
    """Load a single `demo.yaml`."""
    if not path.exists():
        raise ManifestError(f"{path}: no manifest")
    return parse(path.read_text(encoding="utf-8"), path)


def discover(demos_root: Path) -> list[Demo]:
    """Every demo under `demos_root`, ordered by lesson path."""
    found = [load(p) for p in sorted(demos_root.rglob("demo.yaml"))]
    return sorted(found, key=lambda d: d.lesson)
