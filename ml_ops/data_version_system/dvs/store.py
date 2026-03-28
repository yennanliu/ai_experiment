from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dvs.core import DatasetManifest, ExperimentRecord


class Store:
    def __init__(self, project_root: Path) -> None:
        self.root = project_root.resolve()
        self.dvs_dir = self.root / ".dvs"
        self.datasets = self.dvs_dir / "datasets"
        self.experiments = self.dvs_dir / "experiments"

    def ensure_init(self) -> None:
        self.datasets.mkdir(parents=True, exist_ok=True)
        self.experiments.mkdir(parents=True, exist_ok=True)

    def is_initialized(self) -> bool:
        return self.dvs_dir.is_dir()

    @staticmethod
    def _write_json(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def save_dataset(self, manifest: DatasetManifest) -> Path:
        path = self.datasets / f"{manifest.version_id}.json"
        self._write_json(path, manifest.to_json())
        return path

    def load_dataset(self, version_id: str) -> DatasetManifest:
        path = self.datasets / f"{version_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Unknown dataset version: {version_id}")
        return DatasetManifest.from_json(self._read_json(path))

    def list_datasets(self) -> list[DatasetManifest]:
        if not self.datasets.is_dir():
            return []
        out: list[DatasetManifest] = []
        for p in sorted(self.datasets.glob("*.json")):
            out.append(DatasetManifest.from_json(self._read_json(p)))
        return sorted(out, key=lambda m: m.created)

    def save_experiment(self, exp: ExperimentRecord) -> Path:
        path = self.experiments / f"{exp.experiment_id}.json"
        self._write_json(path, exp.to_json())
        return path

    def load_experiment(self, exp_id: str) -> ExperimentRecord:
        path = self.experiments / f"{exp_id}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Unknown experiment: {exp_id}")
        return ExperimentRecord.from_json(self._read_json(path))

    def list_experiments(self) -> list[ExperimentRecord]:
        if not self.experiments.is_dir():
            return []
        out = [ExperimentRecord.from_json(self._read_json(p)) for p in sorted(self.experiments.glob("*.json"))]
        return sorted(out, key=lambda e: e.created)
