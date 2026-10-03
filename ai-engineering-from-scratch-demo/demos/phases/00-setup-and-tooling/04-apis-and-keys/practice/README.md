<!-- generated:start -->
# 00-setup-and-tooling / 04-apis-and-keys

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/00-setup-and-tooling/04-apis-and-keys/) · upstream spec
`phases/00-setup-and-tooling/04-apis-and-keys/docs/en.md`

```bash
uv run demo practice run 04-apis-and-keys --ex 1
uv run demo explain 04-apis-and-keys --ex 1
uv run pytest demos/phases/00-setup-and-tooling/04-apis-and-keys
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Get an Anthropic API key and make your first API call | code | T0 | `ex01_the_python_never_reads_the_dotenv_the_lesson_recommends.py` |
| 2 | Try the raw HTTP version and compare the response format to the SDK version | code | T0 | `ex02_both_paths_print_a_truncated_reply_as_if_it_were_complete.py` |
| 3 | Intentionally use a wrong API key and read the error message | code | T0 | `ex03_the_401_body_is_never_shown_and_a_newline_key_leaks_whole.py` |
<!-- generated:end -->

## Answers

The lesson's Python is `first_api_call.py`: `call_with_sdk` and
`call_raw_http`, one request each. There is no key and no network here, so both
run in a cleared environment (no host variable, no real key can reach them)
against a fake `anthropic` module, a fake `urlopen`, or — in exercise 3 — the
real `urllib` stack behind a `socket.create_connection` tripwire. All three
are **T0**, stdlib only.

### 1 — the Python never reads the `.env` the lesson recommends

**ANSWER:** the first call sends `model="claude-sonnet-5"`, `max_tokens=256`,
one user message; the client is built with no arguments, so the key travels
only through `ANTHROPIC_API_KEY`.

| where the key / model is | what the Python does | TS port |
|---|---|---|
| key only in `./.env` | prints "Set ANTHROPIC_API_KEY…", **0 requests** | loads it (`loadDotenv`) |
| `LLM_MODEL=""` | sends `model: ""` | falls back to `claude-sonnet-5` |
| `LLM_MODEL` set after import | still sends `claude-sonnet-5` | — |
| `LLM_MODEL` set before import | sends it (CONTROL) | — |

**FINDING:** Step 1's "Or use a `.env` file" works for the TypeScript port and
silently does nothing for the Python one.

### 2 — both paths print a truncated reply as if it were complete

One fixture in the documented response shape, fed to both functions.

**ANSWER:** the raw JSON body equals the SDK's keyword arguments; the raw path
adds `x-api-key`, `anthropic-version: 2023-06-01` and `Content-Type` by hand.
The reply is a dict on one side and objects on the other, and both print the
same two lines — reading only `content` and `usage`, 2 of the 8 top-level
fields (a response cut to those two prints the same).

**FINDING: truncation is invisible.** With `stop_reason: "max_tokens"` instead
of `"end_turn"` both paths print byte-identical output.

**FINDING: the raw call has no timeout** — `urlopen(req)` takes 1 argument,
so the socket default (`None`, wait forever) applies.

### 3 — the 401's message is never shown, and a newline key leaks whole

**ANSWER: the learner reads `HTTP Error 401: Unauthorized`.** The API's reason,
`authentication_error: invalid x-api-key`, is in the response body, which the
lesson never reads: it appears in the traceback **0 times**, and only
`e.read()` recovers it. The TS port prints the body.

**FINDING: a key with a trailing newline fails before the network and prints
the whole key.** Through the real `urllib`, `http.client` rejects the header
with 0 connection attempts: `ValueError: Invalid header value
b'sk-ant-wrong-dummy-key-00000\n'` — all 28 characters. A wrong key costs a
401; a malformed one puts the secret in your logs.

**CONTROL:** the fake server received exactly the dummy key in `x-api-key`, so
the 401 is the wrong-key path, not a missing-key one.
