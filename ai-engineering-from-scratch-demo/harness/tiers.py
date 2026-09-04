"""Capability probe (`DESIGN D2`).

The whole point: a missing GPU or a missing key produces a **skip with a remedy
string**, never a stack trace (`DESIGN §3` M0 item 3).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import shutil

TIER_ORDER = {"T0": 0, "T1": 1, "T2": 2, "T3": 3}


@dataclasses.dataclass(frozen=True)
class Capability:
    ok: bool
    remedy: str = ""


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def has_numpy() -> Capability:
    return Capability(True) if _has("numpy") else Capability(
        False, "uv sync --extra math")


def has_torch() -> Capability:
    return Capability(True) if _has("torch") else Capability(
        False, "uv sync --extra llm  # torch-CPU is enough for T1")


def has_api_key() -> Capability:
    if os.environ.get("OPENAI_API_KEY"):
        return Capability(True)
    return Capability(False, "export OPENAI_API_KEY=...  (or DEMO_MODE=replay)")


def has_gpu() -> Capability:
    if not _has("torch"):
        return Capability(False, "uv sync --extra llm, then run on a CUDA host")
    try:
        import torch
    except Exception as exc:                       # pragma: no cover - import guard
        return Capability(False, f"torch present but unimportable: {exc}")
    if torch.cuda.is_available():
        return Capability(True)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return Capability(False, "MPS is present but T3 wants CUDA/bf16; rent a GPU host")
    return Capability(False, "no CUDA device; rent a GPU host or run the D11 scaled-down path")


def probe(tier: str) -> Capability:
    """Can this machine run `tier` right now?"""
    if tier not in TIER_ORDER:
        raise ValueError(f"unknown tier {tier!r}")
    if tier == "T0":
        return Capability(True)
    if tier == "T1":
        return has_numpy()
    if tier == "T2":
        if os.environ.get("DEMO_MODE", "replay") == "replay":
            return Capability(True)                # cassettes make T2 offline-safe
        return has_api_key()
    return has_gpu()


def selected(tier: str) -> bool:
    """Honour `--tier`/`DEMO_TIER` as a ceiling, not an exact match."""
    ceiling = os.environ.get("DEMO_TIER")
    if not ceiling:
        return True
    return TIER_ORDER[tier] <= TIER_ORDER[ceiling]


def describe() -> str:
    rows = [("T0", probe("T0")), ("T1", probe("T1")), ("T2", probe("T2")), ("T3", probe("T3"))]
    width = max(len(t) for t, _ in rows)
    out = []
    for tier, cap in rows:
        mark = "ok" if cap.ok else "skip"
        line = f"  {tier:<{width}}  {mark}"
        if not cap.ok:
            line += f"   -> {cap.remedy}"
        out.append(line)
    if shutil.which("uv") is None:
        out.append("  note: uv not on PATH; DESIGN §8 Q2 makes it the only supported toolchain")
    return "\n".join(out)
