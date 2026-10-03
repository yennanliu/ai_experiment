"""Exercise 1 -- the Python never reads the .env file the lesson recommends.

    Get an Anthropic API key and make your first API call

Reading of the exercise: no key and no network here, so the first call is made
by the lesson's own `first_api_call.call_with_sdk` against a recording fake
`anthropic` module, inside an environment built from scratch (`os.environ`
cleared, a dummy key), so neither a real key nor the host's variables can
reach it. "Get a key" is read as the lesson's Step 1 -- putting the key where
the code will find it -- and that step is tested against both places the
lesson names: an exported variable and a `.env` file.

**ANSWER: the first call sends `claude-sonnet-5`, `max_tokens=256` and one
user message, and prints the reply plus "12 in, 28 out".** The client is built
with no arguments: the key travels only through `ANTHROPIC_API_KEY`.

**FINDING: a `.env` file is never read.** Step 1 says "Or use a `.env` file";
with the key only in `./.env`, `call_raw_http` prints "Set ANTHROPIC_API_KEY
environment variable first" and makes 0 requests. The TypeScript port does
load `./.env` (`loadDotenv`); the Python has no loader.

**FINDING: `LLM_MODEL=""` sends an empty model id.** `os.environ.get` only
falls back when the variable is absent, so an empty one ships `model: ""`;
the TS port's `.trim() || "claude-sonnet-5"` falls back. And the model is
read once at import: setting `LLM_MODEL` afterwards changes nothing.

**CONTROL: `LLM_MODEL` set before import is honoured** -- `claude-test-model`
is the model sent.

Structure: `fake_sdk` records calls; `load` imports the lesson in a clean env.
"""

from __future__ import annotations

import contextlib
import io
import os
import pathlib
import sys
import tempfile
import types
from unittest import mock

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "04-apis-and-keys"
DUMMY_KEY = "sk-ant-dummy-not-a-real-key"


def fake_sdk(calls):
    reply = types.SimpleNamespace(
        content=[types.SimpleNamespace(text="A neural network is a learned function.")],
        usage=types.SimpleNamespace(input_tokens=12, output_tokens=28))

    def create(**kwargs):
        calls.append(kwargs)
        return reply

    def client(**kwargs):
        calls.append({"client_args": kwargs})
        return types.SimpleNamespace(messages=types.SimpleNamespace(create=create))

    return types.SimpleNamespace(Anthropic=client)


def load(env):
    with mock.patch.dict(os.environ, env, clear=True):
        return parity.load_reference(PHASE, LESSON, "first_api_call")


def sdk_call(import_env, call_env=None):
    """(stdout, recorded calls) of call_with_sdk under a clean environment."""
    ref, calls, out = load(import_env), [], io.StringIO()
    with mock.patch.dict(os.environ, call_env or import_env, clear=True):
        with mock.patch.dict(sys.modules, {"anthropic": fake_sdk(calls)}):
            with contextlib.redirect_stdout(out):
                ref.call_with_sdk()
    return out.getvalue(), calls


def dotenv_only():
    """call_raw_http with the key only in ./.env: (stdout, requests attempted)."""
    ref, sent, out = load({}), [], io.StringIO()
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        pathlib.Path(tmp, ".env").write_text(f"ANTHROPIC_API_KEY={DUMMY_KEY}\n")
        os.chdir(tmp)
        try:
            with mock.patch.dict(os.environ, {}, clear=True), \
                    mock.patch("urllib.request.urlopen", lambda *a, **k: sent.append(a)), \
                    contextlib.redirect_stdout(out):
                ref.call_raw_http()
        finally:
            os.chdir(cwd)
    return out.getvalue().strip(), len(sent)


def solve():
    key = {"ANTHROPIC_API_KEY": DUMMY_KEY}
    text, calls = sdk_call(key)
    ts = (parity.lesson_dir(PHASE, LESSON) / "code" / "first_api_call.ts").read_text()
    return {
        "printed": text.strip().splitlines(), "client_args": calls[0]["client_args"],
        "request": calls[1], "dotenv": dotenv_only(),
        "doc_says_dotenv": "use a `.env` file" in parity.doc_text(PHASE, LESSON),
        "ts_loads_dotenv": "loadDotenv(" in ts and '".env"' in ts,
        "empty_model": sdk_call({**key, "LLM_MODEL": ""})[1][1]["model"],
        "ts_falls_back": '.trim() || "claude-sonnet-5"' in ts,
        "late_model": sdk_call(key, {**key, "LLM_MODEL": "claude-late"})[1][1]["model"],
        "set_model": sdk_call({**key, "LLM_MODEL": "claude-test-model"})[1][1]["model"],
    }


def verify(r):
    req, (dotenv_out, dotenv_sent) = r["request"], r["dotenv"]
    return [
        practice.Check(
            "ANSWER: claude-sonnet-5, 256 tokens, one user message; key only via env",
            all((req["model"] == "claude-sonnet-5", req["max_tokens"] == 256,
                 len(req["messages"]) == 1, r["client_args"] == {},
                 r["printed"][1] == "Tokens used: 12 in, 28 out")),
            f"request model={req['model']!r} max_tokens={req['max_tokens']} "
            f"messages={len(req['messages'])}, client args {r['client_args']}; printed "
            f"{r['printed']}",
        ),
        practice.Check(
            "FINDING: a .env file is never read by the Python, though the TS port reads it",
            all((r["doc_says_dotenv"], dotenv_sent == 0,
                 "Set ANTHROPIC_API_KEY" in dotenv_out, r["ts_loads_dotenv"])),
            f"key only in ./.env: prints {dotenv_out!r}, {dotenv_sent} requests; "
            f"first_api_call.ts has loadDotenv: {r['ts_loads_dotenv']}",
        ),
        practice.Check(
            "FINDING: LLM_MODEL='' ships an empty model id, and LLM_MODEL is read once",
            all((r["empty_model"] == "", r["ts_falls_back"],
                 r["late_model"] == "claude-sonnet-5")),
            f"LLM_MODEL='' sends model={r['empty_model']!r} (the TS port falls back); "
            f"LLM_MODEL set after import still sends {r['late_model']!r}",
        ),
        practice.Check(
            "CONTROL: LLM_MODEL set before import is the model sent",
            r["set_model"] == "claude-test-model",
            f"sent model={r['set_model']!r}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
