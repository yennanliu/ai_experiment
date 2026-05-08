import json
from pathlib import Path

_STORE = Path(__file__).parent.parent / "memory" / "store.json"


def load() -> dict:
    _STORE.parent.mkdir(exist_ok=True)
    return json.loads(_STORE.read_text()) if _STORE.exists() else {}


def save(data: dict) -> None:
    _STORE.parent.mkdir(exist_ok=True)
    _STORE.write_text(json.dumps(data, indent=2))
