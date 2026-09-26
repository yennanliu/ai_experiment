"""Exercise 2 — a Chrome version check breaks three of nine clients, and the same API still needs a model-id map.

    WebGPU on Android requires Chrome v121+. Design a fallback for older
    browsers -- server-side via the same OpenAI-compatible API.

Reading of the exercise: the fallback is a router in the page that picks
WebLLM in-browser or a server speaking `/v1/chat/completions`, and sends the
same request body either way. Two routers are compared over nine labelled
client profiles: the rule the exercise states ("Android Chrome >= 121 ->
local") and capability detection. The server is a stub (T0, DESIGN D11); the
real one is `mlc_llm serve <model>`, which exposes the OpenAI-compatible API.

Facts used, each checked on 2026-09-26: Chrome 121 enables WebGPU "on devices
running Android 12 and greater powered by Qualcomm and ARM GPUs" (Chrome
blog, "New in WebGPU 121"). Firefox Android is "behind a flag" and Safari iOS
ships WebGPU in 26 (gpuweb wiki, Implementation Status). WebLLM v0.2.84's
`src/config.ts` gives Llama-3.2-3B-Instruct-q4f16_1-MLC 2263.69 MB and
Llama-3.1-8B-Instruct-q4f16_1-MLC 5001.0 MB at a 4K context; Mistral-7B q4f16
lists `required_features: ["shader-f16"]`. The per-device memory budgets and
f16 support in `CLIENTS` are assumptions.

**ANSWER: route on capability, not version, and fall back per request.** The
page asks for a WebGPU adapter, checks the model's `required_features` and
its `vram_required_MB` against the device budget, and otherwise, or if engine
load throws, sends the identical body to the server. Over the nine clients
detection routes all nine to a path that works: three local, six server.

**FINDING: the version check breaks three of nine clients.** Chrome 121 on
Android 11, a Mistral model on a GPU without shader-f16, and an 8B model on a
4 GB budget all pass "Chrome >= 121" and then fail to load. The same rule
also sends Safari iOS 26, which has WebGPU, to the server. With a catch on
engine load the three broken sessions reach the server anyway -- after paying
for a failed model download.

**FINDING: the same API still needs a model-id map.** Both paths take an
identical chat-completions body except `model`. WebLLM names models
`Llama-3.2-3B-Instruct-q4f16_1-MLC`; a server names the model, not its
quantized build. The reference code has no routing or fallback of any kind:
the only browser fact in it is the note string "mobile browser Chrome 121+".

Structure: `webgpu()` encodes the verified browser facts, `detect()` and
`version_rule()` are the two routers, `send()` is the fallback path.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "12-edge-inference"
MODELS = {  # WebLLM id -> (server name, vram_required_MB, required_features)
    "Llama-3.2-3B-Instruct-q4f16_1-MLC": ("llama-3.2-3b-instruct", 2263.69, ()),
    "Llama-3.1-8B-Instruct-q4f16_1-MLC": ("llama-3.1-8b-instruct", 5001.0, ()),
    "Mistral-7B-Instruct-v0.3-q4f16_1-MLC": ("mistral-7b-instruct", 4573.39, ("shader-f16",)),
}
S, L, M = (f"{m}-q4f16_1-MLC" for m in ("Llama-3.2-3B-Instruct", "Llama-3.1-8B-Instruct",
                                         "Mistral-7B-Instruct-v0.3"))
CLIENTS = [  # (label, browser, version, os version, gpu vendor, f16, budget MB, model) -- assumed
    ("old chrome", "chrome-android", 120, 14, "qualcomm", True, 6000, S),
    ("chrome 121 on android 11", "chrome-android", 121, 11, "arm", True, 6000, S),
    ("no shader-f16", "chrome-android", 125, 14, "arm", False, 6000, M),
    ("8B on a 4 GB budget", "chrome-android", 130, 14, "qualcomm", True, 4000, L),
    ("3B on flagship", "chrome-android", 130, 14, "qualcomm", True, 6000, S),
    ("3B on 8 Gen 3", "chrome-android", 121, 14, "qualcomm", True, 4000, S),
    ("firefox android", "firefox-android", 140, 14, "qualcomm", True, 6000, S),
    ("safari ios 26", "safari-ios", 26, 26, "apple", True, 6000, S),
    ("safari ios 18", "safari-ios", 18, 18, "apple", True, 6000, S),
]


def webgpu(browser, version, os_version, gpu):
    """Verified status: does the browser hand the page a WebGPU adapter?"""
    if browser == "chrome-android":
        return version >= 121 and os_version >= 12 and gpu in ("qualcomm", "arm")
    return browser == "safari-ios" and version >= 26  # Firefox Android: behind a flag


def loads(client):
    _, browser, version, os_v, gpu, f16, budget, model = client
    _, vram, needs = MODELS[model]
    features = {"shader-f16"} if f16 else set()
    return webgpu(browser, version, os_v, gpu) and set(needs) <= features and vram <= budget


def version_rule(client):
    return "local" if client[1] == "chrome-android" and client[2] >= 121 else "server"


def detect(client):
    return "local" if loads(client) else "server"


def request(model_id, route, messages):
    """The one chat-completions body; only `model` depends on the route."""
    model = model_id if route == "local" else MODELS[model_id][0]
    return {"model": model, "messages": messages, "stream": True, "max_tokens": 256}


def send(client, route, messages):
    """Local if routed there and the engine loads; else the server. Returns the path used."""
    if route == "local" and loads(client):
        return "local", request(client[7], "local", messages)
    return "server", request(client[7], "server", messages)


def summarize(version, det, bodies):
    local, server = bodies
    return {
        "broken": [k for k, (route, ok) in version.items() if route == "local" and not ok],
        "missed": [k for k, (route, ok) in version.items() if route == "server" and ok],
        "all_good": all((route == "local") == ok for route, ok in det.values()),
        "n_local": sum(route == "local" for route, _ in det.values()),
        "diff": sorted(k for k in local if local[k] != server[k]),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    msgs = [{"role": "user", "content": "hi"}]
    version = {c[0]: (version_rule(c), loads(c)) for c in CLIENTS}
    det = {c[0]: (detect(c), loads(c)) for c in CLIENTS}
    bodies = (request(S, "local", msgs), request(S, "server", msgs))
    return {
        **summarize(version, det, bodies),
        "fallback": {c[0]: send(c, version_rule(c), msgs)[0] for c in CLIENTS},
        "bodies": bodies,
        "ref_routing": [n for n in dir(ref) if any(k in n.lower() for k in ("route", "fallback"))],
        "ref_browser_notes": [t.notes for t in ref.TARGETS if "Chrome" in t.notes],
    }


def verify(result):
    broken, missed, diff, n_local = (result[k] for k in ("broken", "missed", "diff", "n_local"))
    local, server = result["bodies"]
    fell_back = [result["fallback"][k] for k in broken]
    return [
        practice.Check(
            "ANSWER: route on capability, not version, and fall back per request",
            result["all_good"] and n_local == 3,
            f"detection routes {n_local} local and {len(CLIENTS) - n_local} server, "
            "every one to a path that loads",
        ),
        practice.Check(
            "FINDING: the version check breaks three of nine clients",
            len(broken) == 3 and missed == ["safari ios 26"] and set(fell_back) == {"server"},
            f"broken {broken}; sent to the server although local works: {missed}; "
            "a catch on engine load moves the broken ones to the server",
        ),
        practice.Check(
            "FINDING: the same API still needs a model-id map",
            diff == ["model"] and not result["ref_routing"]
            and result["ref_browser_notes"] == ["mobile browser Chrome 121+"],
            f"bodies differ only in {diff}: {local['model']} vs {server['model']}; reference "
            f"routing symbols {result['ref_routing']}, browser notes {result['ref_browser_notes']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
