"""`practice.yaml` / `demo.yaml` schemas (`DESIGN D3`, `D12`).

Validation is strict and total: an unknown key is an error, not a warning
(`DESIGN §3` M0 item 2). A manifest that parses is a manifest the runner can
trust, so nothing downstream re-checks these fields.
"""

from __future__ import annotations

import dataclasses
import pathlib

from . import yamlite

TIERS = ("T0", "T1", "T2", "T3")
KINDS = ("code", "explain", "lab")
DEPS_GROUPS = ("math", "vision", "audio", "llm", "agents", "infra", "none")

_EX_KEYS = {"index", "slug", "kind", "tier", "deps_group", "en", "zh",
            "verifies", "fixtures", "uses_reference", "cites"}
_EX_REQUIRED = {"index", "slug", "kind", "tier", "en", "zh"}
_PRACTICE_KEYS = {"lesson", "phase", "source", "exercises"}


class ManifestError(ValueError):
    """A manifest that is malformed or outside the schema."""


@dataclasses.dataclass(frozen=True)
class Exercise:
    index: int
    slug: str
    kind: str
    tier: str
    en: str
    zh: str
    deps_group: str = "none"
    verifies: str | None = None
    fixtures: tuple = ()
    uses_reference: tuple = ()
    cites: str | None = None

    @property
    def stem(self) -> str:
        """`ex03_sliding_window_rate_limit` — D10's index-identical filename."""
        return f"ex{self.index:02d}_{self.slug}"

    @property
    def filename(self) -> str:
        return f"{self.stem}.py"


@dataclasses.dataclass(frozen=True)
class Practice:
    lesson: str
    phase: str
    source: str
    exercises: tuple
    path: pathlib.Path | None = None

    def by_index(self, index: int) -> Exercise:
        for ex in self.exercises:
            if ex.index == index:
                return ex
        raise KeyError(f"{self.lesson}: no exercise {index}")

    @property
    def code_exercises(self) -> tuple:
        """The ones that ship a runnable file — `explain` items ship prose."""
        return tuple(e for e in self.exercises if e.kind in ("code", "lab"))


def _require(mapping, keys, allowed, where):
    if not isinstance(mapping, dict):
        raise ManifestError(f"{where}: expected a mapping, got {type(mapping).__name__}")
    missing = keys - mapping.keys()
    if missing:
        raise ManifestError(f"{where}: missing {sorted(missing)}")
    unknown = mapping.keys() - allowed
    if unknown:
        raise ManifestError(f"{where}: unknown key(s) {sorted(unknown)}")


def _exercise(raw, where) -> Exercise:
    _require(raw, _EX_REQUIRED, _EX_KEYS, where)
    kind, tier = raw["kind"], raw["tier"]
    if kind not in KINDS:
        raise ManifestError(f"{where}: kind {kind!r} not in {KINDS}")
    if tier not in TIERS:
        raise ManifestError(f"{where}: tier {tier!r} not in {TIERS}")
    group = raw.get("deps_group") or "none"
    if group not in DEPS_GROUPS:
        raise ManifestError(f"{where}: deps_group {group!r} not in {DEPS_GROUPS}")
    if kind in ("code", "lab") and not raw.get("verifies"):
        raise ManifestError(f"{where}: kind {kind!r} requires a 'verifies' threshold")
    if kind == "explain" and not raw.get("cites"):
        # the explain-item gate from DESIGN §6: a resolvable citation
        raise ManifestError(f"{where}: kind 'explain' requires a 'cites' anchor")
    for text_key in ("en", "zh"):
        if not str(raw[text_key]).strip():
            raise ManifestError(f"{where}: {text_key} text is empty (D12 wants it verbatim)")
    return Exercise(
        index=raw["index"], slug=raw["slug"], kind=kind, tier=tier,
        en=str(raw["en"]).strip(), zh=str(raw["zh"]).strip(), deps_group=group,
        verifies=raw.get("verifies"), cites=raw.get("cites"),
        fixtures=tuple(raw.get("fixtures") or ()),
        uses_reference=tuple(raw.get("uses_reference") or ()),
    )


def load_practice(path) -> Practice:
    path = pathlib.Path(path)
    try:
        raw = yamlite.load(path)
    except yamlite.YamlError as exc:
        raise ManifestError(f"{path}: {exc}") from exc
    _require(raw, _PRACTICE_KEYS, _PRACTICE_KEYS, str(path))
    items = raw["exercises"]
    if not isinstance(items, list) or not items:
        raise ManifestError(f"{path}: 'exercises' must be a non-empty list")
    exercises = tuple(
        _exercise(item, f"{path}[exercise {n}]") for n, item in enumerate(items, 1)
    )
    seen = [e.index for e in exercises]
    if seen != sorted(seen) or len(set(seen)) != len(seen):
        raise ManifestError(f"{path}: exercise indices must be unique and ascending, got {seen}")
    if seen[0] != 1:
        raise ManifestError(f"{path}: exercise indices must start at 1 (D10), got {seen[0]}")
    return Practice(lesson=raw["lesson"], phase=raw["phase"], source=raw["source"],
                    exercises=exercises, path=path)
