from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def should_skip(path: Path, root: Path, skip_names: frozenset[str]) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return True
    parts = rel.parts
    if not parts:
        return False
    if parts[0] in skip_names:
        return True
    return any(p.startswith(".") and p not in (".", "..") for p in parts)


def build_file_map(root: Path, skip_names: frozenset[str] | None = None) -> dict[str, str]:
    root = root.resolve()
    skip = skip_names or frozenset({".dvs", ".git", "__pycache__", ".venv", "venv", "node_modules"})
    files: dict[str, str] = {}
    for p in sorted(root.rglob("*")):
        if p.is_dir():
            continue
        if should_skip(p, root, skip):
            continue
        rel = p.relative_to(root).as_posix()
        files[rel] = file_digest(p)
    return files


def canonical_manifest_payload(root: str, files: dict[str, str]) -> str:
    """Deterministic JSON for version id."""
    payload = {"files": dict(sorted(files.items())), "root": root}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def dataset_version_id(root: str, files: dict[str, str]) -> str:
    raw = canonical_manifest_payload(root, files)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


@dataclass
class DatasetManifest:
    version_id: str
    root: str
    files: dict[str, str]
    created: str = field(default_factory=_utc_now)
    tag: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "root": self.root,
            "files": self.files,
            "created": self.created,
            "tag": self.tag,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> DatasetManifest:
        return cls(
            version_id=data["version_id"],
            root=data["root"],
            files=dict(data["files"]),
            created=data["created"],
            tag=data.get("tag"),
        )


def diff_manifests(
    a: dict[str, str], b: dict[str, str]
) -> tuple[list[str], list[str], list[str]]:
    """Returns (added_paths, removed_paths, changed_paths)."""
    ka, kb = set(a), set(b)
    added = sorted(kb - ka)
    removed = sorted(ka - kb)
    changed = sorted(p for p in (ka & kb) if a[p] != b[p])
    return added, removed, changed


@dataclass
class ExperimentRecord:
    experiment_id: str
    dataset_version_id: str
    command: list[str]
    workdir: str
    created: str = field(default_factory=_utc_now)
    name: str | None = None
    notes: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "dataset_version_id": self.dataset_version_id,
            "command": self.command,
            "workdir": self.workdir,
            "created": self.created,
            "name": self.name,
            "notes": self.notes,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ExperimentRecord:
        return cls(
            experiment_id=data["experiment_id"],
            dataset_version_id=data["dataset_version_id"],
            command=list(data["command"]),
            workdir=data["workdir"],
            created=data["created"],
            name=data.get("name"),
            notes=data.get("notes"),
        )


def experiment_id(dataset_version_id: str, command: list[str], workdir: str) -> str:
    payload = json.dumps(
        {"dataset_version_id": dataset_version_id, "command": command, "workdir": workdir},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]
