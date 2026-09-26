"""Exercise 2 — route by modality behind one gateway, because chat on the multimodal vendor costs 34x.

    Your product serves image generation plus chat plus speech-to-text. Pick
    platforms for each modality and name the gateway pattern that unifies them.

Reading of the exercise: each modality is sent to the platform whose billing
unit matches it, and the picks are costed with the reference `cost_per_day`
where it can express them. The alternative a team reaches for first is one
multimodal vendor for everything. The day's workload is the lesson's Scenario
A for chat (10,000 chats, 2M output tokens), plus 1,000 images and 1,000
transcriptions.

**ANSWER: chat on Together, images on Replicate, speech-to-text on
Replicate's Whisper, all behind an AI gateway.** The gateway is Lesson 19's
pattern: "one process with one API (typically OpenAI-compatible) that fans
out to providers". Here it also routes by modality and turns every bill into
a cost per request.

- **Chat goes to Together.** It is the code's cheapest per-token vendor,
  $1.76/day for the chat load, with Fireworks at $1.80 as the fallback.
- **Images go to Replicate**, which the lesson calls "the default platform
  for image, video, and audio models". The module has no image price, so this
  leg uses its only per-prediction price, $0.006: $6.00/day for 1,000 images.
- **Speech-to-text goes to `openai/whisper` on Replicate.** Its model page
  (fetched 2026-09-26) says "approximately $0.0027 to run", on a T4 billed by
  GPU time, typically within 13 seconds. The lesson's Groq is the swap when
  transcription is real-time. 1,000 runs cost $2.70/day.

**FINDING: sending chat to the multimodal vendor costs 34x.** The reference
prices Replicate at $0.006 a prediction, so the same 10,000 chats cost
$60.00/day there against $1.76 on Together. The gateway exists so that
image traffic can stay on Replicate without dragging chat along.

**FINDING: the lesson's comparator can price only the chat leg.** All 6
`VENDORS` serve a Llama 70B, and `Vendor` has no modality field.
`cost_per_day` takes output tokens and predictions only, with no audio
minutes and no image size. Replicate's $0.006 is one flat price for any
call, so a 1-second clip, a 1-hour clip and an image all cost the same.
The image and speech legs need a unit the module does not have.

Structure: `ROUTES` is the gateway's table; `gateway()` resolves a request to
a route; `leg_cost()` uses the reference for per-token and per-prediction
routes and a fetched per-run price for Whisper.
"""

from __future__ import annotations

import dataclasses

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "02-inference-platform-economics"
WHISPER_RUN = 0.0027  # $/run, replicate.com/openai/whisper, fetched 2026-09-26
WORKLOAD = {"chat": (2_000_000, 10_000), "image": (0, 1_000), "stt": (0, 1_000)}
ROUTES = {
    "chat": ("Together", "per-token"),
    "image": ("Replicate", "per-prediction"),
    "stt": ("Replicate openai/whisper", "per GPU-second"),
}
GATEWAY = "one process with one API (typically OpenAI-compatible) that fans out to providers"


def gateway(request):
    """The gateway's routing decision: modality -> (platform, billing unit)."""
    return ROUTES[request["modality"]]


def leg_cost(ref, modality, platform):
    tokens, calls = WORKLOAD[modality]
    vendors = {v.name: v for v in ref.VENDORS}
    if modality == "stt":
        return round(calls * WHISPER_RUN, 2)
    return round(ref.cost_per_day(vendors[platform.split()[0]], tokens, calls), 2)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    routed = {m: gateway({"modality": m})[0] for m in WORKLOAD}
    chat = {v.name: ref.cost_per_day(v, *WORKLOAD["chat"]) for v in ref.VENDORS
            if v.per_mtok_output is not None}
    gateway_doc = parity.doc_text(PHASE, "19-ai-gateways")
    return {
        "routed": routed,
        "costs": {m: leg_cost(ref, m, p) for m, p in routed.items()},
        "chat_options": {k: round(c, 2) for k, c in sorted(chat.items(), key=lambda kv: kv[1])},
        "single_vendor_chat": leg_cost(ref, "chat", "Replicate"),
        "models": sorted({v.model for v in ref.VENDORS}),
        "fields": [f.name for f in dataclasses.fields(ref.Vendor)],
        "flat": [ref.cost_per_day(ref.VENDORS[4], t, 1) for t in (10, 1_000_000)],
        "gateway_quoted": GATEWAY in gateway_doc,
    }


def verify(result):
    costs, options = result["costs"], result["chat_options"]
    ratio = result["single_vendor_chat"] / costs["chat"]
    return [
        practice.Check(
            "ANSWER: chat on Together, images and speech on Replicate, behind an AI gateway",
            all([result["routed"]["chat"] == next(iter(options)) == "Together",
                 costs == {"chat": 1.76, "image": 6.0, "stt": 2.7},
                 result["gateway_quoted"]]),
            f"routes {result['routed']}, $/day {costs}; per-token chat options {options}; "
            "the gateway is Lesson 19's single OpenAI-compatible fan-out",
        ),
        practice.Check(
            "FINDING: sending chat to the multimodal vendor costs 34x",
            round(ratio) == 34,
            f"10,000 chats cost ${result['single_vendor_chat']} on Replicate vs "
            f"${costs['chat']} on Together, {ratio:.1f}x",
        ),
        practice.Check(
            "FINDING: the lesson's comparator can price only the chat leg",
            all(["70B" in m for m in result["models"]] + [len(set(result["flat"])) == 1,
                 not any("modal" in f or "audio" in f for f in result["fields"])]),
            f"models {result['models']}; Vendor fields {result['fields']}; Replicate "
            f"charges {result['flat']} for a 10-token and a 1M-token call",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
