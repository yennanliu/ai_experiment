"""Record real provider responses once, replay them forever. (D4)

The reference curriculum fakes model output with hand-written `simulate_llm_call`
functions. A cassette is strictly better on three axes: the text is what the
model actually said, replaying it is free and offline, and CI can assert on it.

    DEMO_MODE=replay   # default -- deterministic, free, offline, CI-safe
    DEMO_MODE=live     # hits the API, re-records, prints the token cost

Cassettes are committed to the repo. Each one carries the model ID and the date
it was recorded, so a cassette that has drifted away from what the current model
would say is *visible* rather than silently authoritative.

Only `record_live` imports `anthropic`, and only in live mode -- replay must work
on a machine with no SDK and no key installed.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .tiers import LIVE, mode

CASSETTE_VERSION = 1

DEFAULT_MODEL = "claude-opus-5"

# USD per million tokens, first-party Anthropic API rates. Printed with the
# as-of date so a stale number is obvious rather than quietly wrong.
PRICES_AS_OF = "2026-06-24"
PRICE_PER_MTOK = {
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-fable-5-1": (10.00, 50.00),
}

# Never let one of these reach a committed cassette, however it got into a
# request dict.
_SECRET_KEYS = ("api_key", "authorization", "x-api-key", "auth_token", "token")


class CassetteError(RuntimeError):
    """A cassette is missing an interaction that replay needs."""


@dataclass(frozen=True)
class Recording:
    """One recorded request/response pair."""

    key: str
    request: dict
    text: str
    model: str
    stop_reason: str | None
    usage: dict
    recorded_at: str

    def cost_usd(self) -> float | None:
        """What this interaction cost when it was recorded, if we can price it."""
        price = PRICE_PER_MTOK.get(self.model)
        if not price:
            return None
        rate_in, rate_out = price
        tokens_in = self.usage.get("input_tokens", 0)
        tokens_out = self.usage.get("output_tokens", 0)
        return (tokens_in * rate_in + tokens_out * rate_out) / 1_000_000


def _redact(value: Any) -> Any:
    """Strip anything key-shaped out of a structure bound for disk."""
    if isinstance(value, dict):
        return {
            k: ("<redacted>" if k.lower() in _SECRET_KEYS else _redact(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def request_key(request: dict) -> str:
    """Stable id for a request. Same request text -> same cassette entry."""
    canonical = json.dumps(_redact(request), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


@dataclass
class Cassette:
    """A committed tape of provider interactions for one demo."""

    path: Path
    provider: str = "anthropic"
    interactions: dict[str, Recording] = field(default_factory=dict)
    _dirty: bool = False

    @classmethod
    def load(cls, path: Path, *, provider: str = "anthropic") -> "Cassette":
        tape = cls(path=path, provider=provider)
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            if raw.get("version") != CASSETTE_VERSION:
                raise CassetteError(
                    f"{path}: cassette version {raw.get('version')}, "
                    f"this harness writes version {CASSETTE_VERSION}"
                )
            tape.provider = raw.get("provider", provider)
            for entry in raw.get("interactions", []):
                tape.interactions[entry["key"]] = Recording(**entry)
        return tape

    def save(self) -> None:
        """Write the tape back, sorted, only if something changed."""
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = {
            "version": CASSETTE_VERSION,
            "provider": self.provider,
            "harness_version": __version__,
            "interactions": [
                vars(self.interactions[k]) for k in sorted(self.interactions)
            ],
        }
        self.path.write_text(json.dumps(body, indent=2, sort_keys=False) + "\n",
                             encoding="utf-8")
        self._dirty = False

    @property
    def models(self) -> list[str]:
        return sorted({r.model for r in self.interactions.values()})

    @property
    def recorded_dates(self) -> list[str]:
        return sorted({r.recorded_at[:10] for r in self.interactions.values()})

    def total_cost_usd(self) -> float:
        return sum(r.cost_usd() or 0.0 for r in self.interactions.values())

    def complete(self, request: dict, *, run_mode: str | None = None,
                 recorder=None) -> Recording:
        """Return the response for `request`, replaying or recording as needed.

        `recorder` exists so the record/replay machinery can be tested without
        spending money: it defaults to the real provider call.
        """
        run_mode = run_mode or mode()
        key = request_key(request)

        if run_mode != LIVE:
            hit = self.interactions.get(key)
            if hit is None:
                raise CassetteError(
                    f"{self.path.name} has no recording for this request "
                    f"(key {key}).\nThe prompt changed since the tape was cut. "
                    f"Re-record it:\n  DEMO_MODE=live uv run demo run <lesson>"
                )
            return hit

        recording = (recorder or record_live)(request, key=key)
        # Redact here, at the boundary where a recording enters the tape, rather
        # than trusting each recorder to have done it. This is the only path to
        # disk, so it is the only place the guarantee has to hold.
        recording = replace(recording, request=_redact(recording.request))
        self.interactions[key] = recording
        self._dirty = True
        return recording


def record_live(request: dict, *, key: str) -> Recording:
    """Make one real Claude call and wrap it as a `Recording`.

    Imported lazily: a replay run must not need the `anthropic` package.
    """
    import anthropic  # noqa: PLC0415 -- live mode only, see docstring

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        # Not fatal: the SDK also resolves an `ant auth login` profile.
        print("note: no ANTHROPIC_API_KEY set; falling back to the SDK's "
              "credential chain (ant auth login profile)")

    client = anthropic.Anthropic()
    payload = dict(request)
    model = payload.pop("model", DEFAULT_MODEL)

    response = client.beta.messages.create(
        model=model,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        **payload,
    )

    text = "".join(b.text for b in response.content if b.type == "text")
    if response.stop_reason == "refusal":
        detail = getattr(response, "stop_details", None)
        raise CassetteError(
            "the model declined this request, so there is nothing to record"
            + (f" (category: {detail.category})" if detail else "")
        )

    return Recording(
        key=key,
        request=_redact(request),
        text=text,
        model=response.model,
        stop_reason=response.stop_reason,
        usage={
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
        recorded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def report(tape: Cassette, *, run_mode: str | None = None) -> None:
    """Print the provenance line every T2 demo owes the reader."""
    run_mode = run_mode or mode()
    if not tape.interactions:
        print(f"\ncassette {tape.path.name}: empty")
        return
    print(f"\ncassette {tape.path.name}  [{run_mode}]")
    print(f"  {len(tape.interactions)} interaction(s), "
          f"model(s) {', '.join(tape.models)}, recorded {', '.join(tape.recorded_dates)}")
    cost = tape.total_cost_usd()
    if run_mode == LIVE:
        print(f"  this run cost ~${cost:.4f} "
              f"(list price as of {PRICES_AS_OF}); replay is free")
    else:
        print(f"  replayed offline for $0.00 "
              f"(recording it live cost ~${cost:.4f})")
