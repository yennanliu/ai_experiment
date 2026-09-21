"""Exercise 5 — text-level control costs the STT and TTS bands.

    Measure an OpenAI Realtime vs STT+LLM+TTS cascade on the same query. What
    latency cost does text-level control carry?

Reading of the exercise: measuring this on one laptop would produce a fact
about the laptop, so the comparison is built from the lesson's own published
bands, which is what a latency budget is for. A Realtime path is
`VAD -> model -> transport`; the cascade inserts STT before the model and TTS
after it. Subtracting the two is the answer, and it is exact rather than
sampled.

**ANSWER: text-level control costs 200-450ms, and that is precisely the STT
and TTS bands.** The cascade sums to **400-990ms** (midpoint **695ms**);
Realtime, carrying audio straight into the model, sums to **200-540ms**
(midpoint **370ms**). The difference is **200-450ms** -- STT's **100-250**
plus TTS's **100-200**, with nothing else changing.

**FINDING: the two land on opposite sides of the premium band.** The lesson
calls **450-600ms** premium, **800-1200ms** common and **>1500ms** broken.
Realtime's midpoint of **370ms** is below premium; the cascade's **695ms**
is above it, and the cascade's ceiling of **990ms** is squarely in the common
band. Neither is ever broken, so the choice is between "better than premium"
and "ordinary".

**FINDING: what the 200-450ms buys is the input to 3 of the 5 stages.** STT
emits text, the LLM consumes and emits text, TTS consumes it -- so a Realtime
path leaves **3** of **5** processors with no frame they recognise. Every
feature built in this lesson reads that text: the confidence gate, the turn
rule and `LLM.replies` all key on a `transcript` or a `text` frame.

**FINDING: the shipped cascade pays the cost and does not deliver the
barge-in.** `TTS.process` sets `self.cancelled = False` and then loops over
words checking it, with nothing able to mutate it in between -- so a TTS
pre-set to cancelled still emits all **14** words, and `"cut mid-word"` appears
**0** times in the trace. The branch that makes the cascade interruptible is
unreachable, which is the one advantage a text pipeline is supposed to have
over direct audio.

Structure: `chain()` sums a stage list from the published bands; `barge_in()`
drives the shipped TTS against its own cancel flag.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "22-voice-agents-pipecat-livekit"
# "Typical 2026 latencies", in milliseconds.
BANDS = {"vad": (20, 60), "stt": (100, 250), "model": (150, 400),
         "tts": (100, 200), "transport": (30, 80)}
CASCADE = ("vad", "stt", "model", "tts", "transport")
REALTIME = ("vad", "model", "transport")
PREMIUM, COMMON, BROKEN = (450, 600), (800, 1200), 1500
REPLY = "sure, I can help with a refund; what order number should I look up?"


def chain(stages):
    low = sum(BANDS[name][0] for name in stages)
    high = sum(BANDS[name][1] for name in stages)
    return {"low": low, "high": high, "mid": (low + high) // 2}


def difference(a, b):
    return {"low": a["low"] - b["low"], "high": a["high"] - b["high"]}


def text_stages(ref):
    """Processors whose recognised frame kind carries text."""
    import inspect
    kinds = {"stt": "transcript", "llm": "text", "tts": "text"}
    present = [name for name, kind in kinds.items()
               if kind in inspect.getsource(getattr(ref, name.upper()))]
    return present


def barge_in(ref):
    """Cancel the TTS before it synthesises, then count what it emitted."""
    tts, transport = ref.TTS("tts"), ref.Transport("transport")
    ref.link(tts, transport)
    tts.cancelled = True
    tts.process(ref.Frame("text", REPLY))
    return {"words": len(REPLY.split()),
            "emitted": len(transport.delivered[-1]),
            "cuts": sum("cut mid-word" in line for line in tts.trace),
            "flag_after": tts.cancelled}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    cascade, realtime = chain(CASCADE), chain(REALTIME)
    cost = difference(cascade, realtime)
    return {
        "cascade": cascade, "realtime": realtime, "cost": cost,
        "stt_tts": {"low": BANDS["stt"][0] + BANDS["tts"][0],
                    "high": BANDS["stt"][1] + BANDS["tts"][1]},
        "premium": PREMIUM, "common": COMMON, "broken": BROKEN,
        "realtime_premium": realtime["mid"] < PREMIUM[0],
        "cascade_premium": cascade["mid"] > PREMIUM[1],
        "ceiling_common": COMMON[0] <= cascade["high"] <= COMMON[1],
        "ever_broken": max(cascade["high"], realtime["high"]) > BROKEN,
        "text_stages": text_stages(ref), "stages": len(CASCADE),
        "barge": barge_in(ref),
    }


def verify(result):
    cascade, realtime, cost = result["cascade"], result["realtime"], result["cost"]
    barge = result["barge"]
    return [
        practice.Check(
            "ANSWER: text-level control costs 200-450ms, the STT and TTS bands",
            all([cascade["low"] == 400, cascade["high"] == 990, cascade["mid"] == 695,
                 realtime["low"] == 200, realtime["high"] == 540,
                 realtime["mid"] == 370, cost["low"] == 200, cost["high"] == 450,
                 result["stt_tts"] == {"low": 200, "high": 450}]),
            f"the cascade sums to {cascade['low']}-{cascade['high']}ms (midpoint "
            f"{cascade['mid']}) and Realtime to {realtime['low']}-{realtime['high']}ms "
            f"(midpoint {realtime['mid']}). The difference is "
            f"{cost['low']}-{cost['high']}ms, which is exactly STT's 100-250 plus TTS's "
            "100-200 with nothing else changing",
        ),
        practice.Check(
            "FINDING: the two land on opposite sides of the premium band",
            all([result["realtime_premium"] is True,
                 result["cascade_premium"] is True,
                 result["ceiling_common"] is True,
                 result["ever_broken"] is False]),
            f"the lesson calls {result['premium']} premium, {result['common']} common and "
            f">{result['broken']}ms broken. Realtime's {realtime['mid']}ms midpoint is "
            f"below premium, the cascade's {cascade['mid']}ms is above it, and the "
            f"cascade's {cascade['high']}ms ceiling sits in the common band. Neither is "
            "ever broken, so the choice is better-than-premium against ordinary",
        ),
        practice.Check(
            "FINDING: what the cost buys is the input to 3 of the 5 stages",
            all([result["text_stages"] == ["stt", "llm", "tts"],
                 result["stages"] == 5]),
            f"{result['text_stages']} all name a text frame kind in their own source, so a "
            f"Realtime path leaves {len(result['text_stages'])} of {result['stages']} "
            "processors with no frame they recognise. The confidence gate, the turn rule "
            "and LLM.replies every one key on a transcript or a text frame",
        ),
        practice.Check(
            "FINDING: the shipped cascade pays the cost and skips the barge-in",
            all([barge["emitted"] == 14, barge["words"] == 14, barge["cuts"] == 0,
                 barge["flag_after"] is False]),
            f"TTS.process sets self.cancelled = False and then loops over words checking "
            f"it, with nothing able to mutate it in between, so a TTS pre-set to cancelled "
            f"still emits all {barge['emitted']}/{barge['words']} words and logs "
            f"{barge['cuts']} cuts. The branch that makes a text pipeline interruptible is "
            "unreachable -- the one advantage it has over direct audio",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
