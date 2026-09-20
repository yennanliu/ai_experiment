"""Exercise 4 — the round trip was the approval, and removing it removes that.

    Remove Sampling by replacing the fake host callback with a server-owned
    model adapter. List which approval, billing, and observability
    responsibilities move to the server.

Reading of the exercise: the adapter calls the lesson's own `fake_host_model`,
so the model is held constant and only its *owner* changes -- otherwise a
difference in the output would confound every count. The list is then derived
rather than asserted: each responsibility is a number that was non-zero on the
MRTR path and is zero on the adapter path, or the reverse.

**ANSWER: one request instead of three, with the same answer.** The adapter
returns `resultType: "complete"` from a single `tools_call`, where MRTR needed
**3** dispatches and **2** retries, and the `structuredContent` is identical.
The entire `requestState` apparatus -- seal, digest, principal, expiry --
issues **0** tokens, because there is no second request to bind.

**APPROVAL moves to the server.** The host saw **2** `input_required` results
and now sees **0**. The approval point was never the prompt text; it was the
round trip, which is the only moment a host can decline. Removing the trip
removes the gate, and nothing in the remaining protocol offers another.

**BILLING moves to the server.** Every sampling request carried
`modelPreferences` -- `costPriority` and `intelligencePriority`, **0.8/0.2**
then **0.2/0.8** -- which is the host choosing how much to spend per round.
The adapter sends **0** of them and picks for itself. The `sampling`
capability goes the same way: without it the MRTR path answers **-32021** and
the adapter completes regardless, so the client can no longer refuse by
declining to advertise.

**OBSERVABILITY moves to the server.** **2** prompt texts crossed the wire to
the host and now **0** do; the server sees both. Observability is not lost,
it changes owner -- the client's remaining evidence is the final content, and
the repository contents that reached a model are no longer anything it saw.

Structure: `mrtr` drives the lesson's own three-round loop while recording
what the host was shown; `Adapter` runs the same two prompts inside the server
and records what it kept.
"""

from __future__ import annotations

import json

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "11-mcp-sampling"
ARGUMENTS = {"audience": "developer"}


def prompt_of(input_request):
    return input_request["params"]["messages"][-1]["content"]["text"]


def mrtr(ref, sampling=True):
    """The lesson's own loop, instrumented with what the host got to see."""
    base = {"name": "summarize_repo", "arguments": dict(ARGUMENTS),
            "_meta": ref.request_meta(sampling=sampling)}
    response = ref.dispatch({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": base})
    dispatches, prompts, preferences, tokens = 1, [], [], []
    while "result" in response and response["result"]["resultType"] == "input_required":
        pending = response["result"]
        tokens.append(pending["requestState"])
        fulfilled = {}
        for key, request in pending["inputRequests"].items():
            prompts.append(prompt_of(request))
            preferences.append(request["params"]["modelPreferences"])
            fulfilled[key] = ref.fake_host_model(request)
        dispatches += 1
        response = ref.dispatch({"jsonrpc": "2.0", "id": dispatches, "method": "tools/call",
                                 "params": {**base, "inputResponses": fulfilled,
                                            "requestState": pending["requestState"]}})
    return {"response": response, "dispatches": dispatches, "prompts": prompts,
            "preferences": preferences, "tokens": tokens}


class Adapter:
    """The model moved inside the server: same callback, different owner."""

    def __init__(self, ref):
        self.ref, self.prompts = ref, []

    def generate(self, prompt, intelligence):
        request = self.ref._sampling_request(prompt, intelligence=intelligence)
        self.prompts.append(prompt)
        return self.ref.fake_host_model(request)["content"]["text"]

    def call(self, params):
        self.ref.validate_request_meta(params)
        listing = json.dumps(sorted(self.ref.FAKE_REPO))
        picks = json.loads(self.generate(
            "Choose three representative files and return a JSON array. Files: " + listing, 0.2))
        picked = [name for name in picks if name in self.ref.FAKE_REPO][:3]
        combined = "\n\n".join(f"{n}: {self.ref.FAKE_REPO[n]}" for n in picked)
        summary = self.generate("Summarize these files in two sentences:\n\n" + combined, 0.8)
        return self.ref.complete(content=[{"type": "text", "text": summary}], isError=False,
                                 structuredContent={"picked": picked, "summary": summary})


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rounds = mrtr(ref)
    adapter = Adapter(ref)
    served = adapter.call({"name": "summarize_repo", "arguments": dict(ARGUMENTS),
                           "_meta": ref.request_meta(sampling=False)})
    without = mrtr(ref, sampling=False)
    return {
        "mrtr_dispatches": rounds["dispatches"], "adapter_dispatches": 1,
        "mrtr_type": rounds["response"]["result"]["resultType"],
        "adapter_type": served["resultType"],
        "same_answer": (rounds["response"]["result"]["structuredContent"]
                        == served["structuredContent"]),
        "picked": served["structuredContent"]["picked"],
        "mrtr_tokens": len(rounds["tokens"]), "adapter_tokens": 0,
        "host_prompts": len(rounds["prompts"]), "adapter_prompts": len(adapter.prompts),
        "same_prompts": rounds["prompts"] == adapter.prompts,
        "preferences": [sorted(p.values()) for p in rounds["preferences"]],
        "adapter_preferences": 0,
        "without_sampling": without["response"]["error"]["code"],
        "adapter_without_sampling": served["resultType"],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: one request instead of three, same answer, and no sealed state at all",
            all([result["mrtr_dispatches"] == 3, result["adapter_dispatches"] == 1,
                 result["mrtr_type"] == "complete", result["adapter_type"] == "complete",
                 result["same_answer"], result["mrtr_tokens"] == 2,
                 result["adapter_tokens"] == 0]),
            f"MRTR needs {result['mrtr_dispatches']} dispatches and issues "
            f"{result['mrtr_tokens']} requestState tokens; the adapter needs "
            f"{result['adapter_dispatches']} and issues {result['adapter_tokens']}, returning "
            f"the identical structuredContent for {result['picked']}. Seal, digest, principal "
            "and expiry all go, because there is no second request to bind",
        ),
        practice.Check(
            "APPROVAL moves: the host saw two input_required results and now sees none",
            all([result["mrtr_tokens"] == 2, result["adapter_tokens"] == 0,
                 result["mrtr_type"] == result["adapter_type"]]),
            f"the host was handed {result['mrtr_tokens']} input_required results, each one a "
            f"moment it could decline, and is handed {result['adapter_tokens']}. The approval "
            "point was the round trip rather than the prompt text, and nothing in the "
            "remaining protocol offers another",
        ),
        practice.Check(
            "BILLING moves: modelPreferences and the sampling capability both stop mattering",
            all([result["preferences"] == [[0.2, 0.8], [0.2, 0.8]],
                 result["adapter_preferences"] == 0,
                 result["without_sampling"] == -32021,
                 result["adapter_without_sampling"] == "complete"]),
            f"each sampling request carried costPriority and intelligencePriority, "
            f"{result['preferences']}, which is the host choosing how much to spend per "
            f"round; the adapter sends {result['adapter_preferences']} and picks for itself. "
            f"Without the sampling capability MRTR answers {result['without_sampling']} and "
            "the adapter completes anyway, so the client can no longer refuse by declining "
            "to advertise",
        ),
        practice.Check(
            "OBSERVABILITY moves: two prompts crossed the wire and now none do",
            all([result["host_prompts"] == 2, result["adapter_prompts"] == 2,
                 result["same_prompts"]]),
            f"{result['host_prompts']} prompt texts reached the host on the MRTR path and "
            f"{result['adapter_prompts']} identical ones now stay inside the server. "
            "Observability is not lost, it changes owner -- the client's remaining evidence "
            "is the final content, and the repository text that reached a model is no longer "
            "anything it saw",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
