"""Recorded provider responses, not hand-written simulations (`DESIGN D4`).

`DEMO_MODE=replay` (default) is deterministic, free, offline and CI-safe.
`DEMO_MODE=live` hits the provider, re-records, and prints token cost.
Redaction happens **at the write boundary**, so a key can never reach the disk.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
import re

CASSETTE_DIR = pathlib.Path(__file__).resolve().parent.parent / "cassettes"
_SECRET = re.compile(
    r"(sk-[A-Za-z0-9_\-]{8,}|Bearer\s+[A-Za-z0-9._\-]{8,}|(?i:api[_-]?key\"?\s*[:=]\s*)\"?[A-Za-z0-9._\-]{8,})"
)


class CassetteMiss(RuntimeError):
    """Replay was asked for a request that was never recorded."""


def mode() -> str:
    value = os.environ.get("DEMO_MODE", "replay")
    if value not in ("replay", "live"):
        raise ValueError(f"DEMO_MODE must be 'replay' or 'live', got {value!r}")
    return value


def redact(text: str) -> str:
    return _SECRET.sub("<redacted>", text)


def _key(request: dict) -> str:
    blob = json.dumps(request, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclasses.dataclass
class Cassette:
    name: str
    provider: str = "openai"          # DESIGN §8 Q3
    model: str = ""
    recorded: str = ""
    entries: dict = dataclasses.field(default_factory=dict)

    @property
    def path(self) -> pathlib.Path:
        return CASSETTE_DIR / f"{self.name}.json"

    @classmethod
    def load(cls, name: str) -> "Cassette":
        path = CASSETTE_DIR / f"{name}.json"
        if not path.is_file():
            return cls(name=name)
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(name=name, provider=raw.get("provider", "openai"),
                   model=raw.get("model", ""), recorded=raw.get("recorded", ""),
                   entries=raw.get("entries", {}))

    def save(self) -> pathlib.Path:
        CASSETTE_DIR.mkdir(parents=True, exist_ok=True)
        body = {"provider": self.provider, "model": self.model,
                "recorded": self.recorded, "entries": self.entries}
        text = redact(json.dumps(body, indent=2, ensure_ascii=False, sort_keys=True))
        self.path.write_text(text + "\n", encoding="utf-8")
        return self.path

    def play(self, request: dict, live):
        """Replay `request`, or call `live()` and record it."""
        key = _key(request)
        if mode() == "replay":
            if key not in self.entries:
                raise CassetteMiss(
                    f"{self.name}: no recording for this request. "
                    f"Record it once with: DEMO_MODE=live uv run demo practice run ..."
                )
            return self.entries[key]["response"]
        import datetime
        response = live()
        self.entries[key] = {"request": json.loads(redact(json.dumps(request, ensure_ascii=False))),
                             "response": response}
        self.recorded = datetime.date.today().isoformat()
        self.save()
        return response

    @property
    def is_empty(self) -> bool:
        return not self.entries
