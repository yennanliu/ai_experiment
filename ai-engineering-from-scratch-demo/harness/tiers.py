"""Capability probe: what can this machine actually run? (D2)

The tier table in DESIGN.md is a promise about cost and hardware, and this
module is what makes it enforceable. Its job is to turn "you don't have a GPU"
into a *clean skip with an explanation* rather than a stack trace three minutes
into a download.

Nothing here imports torch. Probing must not cost a two-second import on a
machine that is only running T0 demos.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import socket
from dataclasses import dataclass

REPLAY = "replay"
LIVE = "live"

TIER_ORDER = {"T0": 0, "T1": 1, "T2": 2, "T3": 3}

TIER_DESCRIPTION = {
    "T0": "cpu-instant  (stdlib / numpy / sklearn, <10s, no network)",
    "T1": "cpu-heavy    (torch-CPU or a small HF checkpoint, <5min)",
    "T2": "api          (a real provider call; replayed from a cassette by default)",
    "T3": "gpu          (needs CUDA / >=16GB VRAM; explain-and-skip elsewhere)",
}


def mode() -> str:
    """`DEMO_MODE`, defaulting to replay so nothing is billed by accident (D4)."""
    value = os.environ.get("DEMO_MODE", REPLAY).strip().lower()
    if value not in (REPLAY, LIVE):
        raise SystemExit(f"DEMO_MODE must be {REPLAY!r} or {LIVE!r}, got {value!r}")
    return value


def has_module(name: str) -> bool:
    """True if `name` is importable without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def has_cuda() -> bool:
    """True if torch reports a usable CUDA device.

    Checked via `nvidia-smi` first so a machine with no GPU never pays for the
    torch import just to be told it has no GPU.
    """
    if not shutil.which("nvidia-smi"):
        return False
    if not has_module("torch"):
        return False
    import torch  # noqa: PLC0415 -- deliberately lazy, see docstring

    return torch.cuda.is_available()


def has_network(host: str = "pypi.org", port: int = 443, timeout: float = 1.5) -> bool:
    """True if a TCP connect to `host` succeeds inside `timeout`."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@dataclass(frozen=True)
class Skip:
    """Why a demo is not going to run here, and what to do about it."""

    reason: str
    remedy: str

    def render(self) -> str:
        return f"SKIP: {self.reason}\n      {self.remedy}"


def check(demo, *, run_mode: str | None = None) -> Skip | None:
    """Return a `Skip` if `demo` cannot run on this machine, else None.

    A `Skip` is never an error. Per D2 a demo the machine cannot host must exit
    0 with an explanation, so the caller prints this and returns 0.
    """
    run_mode = run_mode or mode()

    if demo.tier == "T1" and not has_module("torch"):
        return Skip(
            reason="tier T1 needs torch, which is not installed",
            remedy=f"uv sync --extra {demo.deps_group}",
        )

    if demo.tier == "T2":
        cassette_missing = not (demo.path / "cassettes" / demo.cassette).exists()
        if run_mode == REPLAY and cassette_missing:
            return Skip(
                reason=f"no recorded cassette at cassettes/{demo.cassette}",
                remedy="DEMO_MODE=live uv run demo run "
                f"{demo.lesson}   # records it once, then replay is free",
            )
        if run_mode == LIVE:
            absent = [key for key in demo.needs_env if not os.environ.get(key)]
            if absent:
                return Skip(
                    reason=f"DEMO_MODE=live needs {', '.join(absent)}",
                    remedy="export the key, or drop DEMO_MODE to replay the cassette",
                )
            if not has_network():
                return Skip(
                    reason="DEMO_MODE=live but there is no network",
                    remedy="drop DEMO_MODE to replay the cassette offline",
                )

    if demo.tier == "T3" and not has_cuda():
        return Skip(
            reason="tier T3 needs a CUDA GPU and this machine has none",
            remedy=demo.skip_reason or "rent a GPU runner to run this demo",
        )

    return None
