"""Exercise 2 — the payload is the transcript, so there is nothing to doubt.

    Implement confidence-gated STT: below threshold, request "could you
    repeat that?"

Reading of the exercise: `STT.process` does `transcript = str(frame.payload)`
-- the audio *is* the answer, exactly, every time. A confidence gate
therefore needs a confidence to exist first, so the port widens the payload
to `(text, confidence)` and leaves everything else alone. Where the threshold
goes is then a cost question, because a re-ask costs a whole turn.

**ANSWER: a gate at 0.60 re-asks on 3 of 10 utterances and cuts bad
transcripts from 3 to 0.** Below threshold the STT emits a `text` frame
carrying "could you repeat that?" instead of a `transcript`, so the LLM is
never called on a guess. Over the fixture that is **3** clarifications and
**7** answered turns.

**FINDING: the threshold is decided by a ratio, and 0.60 is the optimum over
a wide range.** Scoring a wrong answer at 5x a re-ask puts the optimum at
**0.60** with a cost of **3**; at 2x and at 10x it is still **0.60**. The
number is stable because this recogniser's errors all sit at confidence
**0.55** or below -- one cut separates them at any price, which is a
statement about calibration rather than about the ratio.

**FINDING: every gate costs a full round trip, not a stage.** A clarification
is a complete turn -- the user hears it, answers, and the chain runs again --
so at the lesson's **450-600ms** premium band, **3** gates add **1350-1800ms**
to the conversation. Gating is cheap per frame and expensive per dialogue,
which is the opposite of how a per-stage latency budget makes it look.

**FINDING: the re-ask has no route that skips the LLM.** Sending it as a
`transcript` puts it through `LLM.replies`, which has no entry for it and
answers `'[no canned reply]'` -- the user is asked to repeat and then told
nothing. Emitting a `text` frame from the STT instead reaches TTS in **1**
hop and bypasses **1** stage, which is only possible because `Processor`
forwards by kind rather than by contract.

Structure: `GatedSTT` subclasses the lesson's own `STT`; `sweep()` prices
every threshold against a stated cost model.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "22-voice-agents-pipecat-livekit"
REASK = "could you repeat that?"
# (heard, confidence, is_correct) -- a recogniser that is wrong when unsure.
HEARD = (("hello", 0.98, True), ("refund please", 0.91, True),
         ("refund please", 0.45, False), ("hello", 0.72, True),
         ("cancel my order", 0.88, True), ("hello", 0.31, False),
         ("refund please", 0.55, False), ("cancel my order", 0.66, True),
         ("hello", 0.95, True), ("cancel my order", 0.79, True))
THRESHOLDS = tuple(round(0.1 * k, 2) for k in range(11))
PREMIUM = (450, 600)


def gated_stt(ref, threshold):
    class GatedSTT(ref.STT):
        """The lesson's STT, with the payload widened to (text, confidence)."""

        def process(self, frame):
            if frame.kind != "vad_speech":
                return ref.Processor.process(self, frame)
            text, confidence = frame.payload
            if confidence < threshold:
                self.trace.append(f"STT: low confidence {confidence}")
                return ref.Processor.process(self, ref.Frame("text", REASK))
            return ref.Processor.process(self, ref.Frame("transcript", text))
    return GatedSTT("stt")


def pipeline(ref, threshold):
    stages = (ref.VAD("vad"), gated_stt(ref, threshold),
              ref.LLM("llm", replies={"hello": "hi there",
                                      "refund please": "what order number?",
                                      "cancel my order": "which order?"}),
              ref.TTS("tts"), ref.Transport("transport"))
    ref.link(*stages)
    return stages


def run(ref, threshold):
    stages = pipeline(ref, threshold)
    reasked = answered = wrong = 0
    for text, confidence, correct in HEARD:
        before = len(stages[-1].delivered)
        stages[0].process(ref.Frame("audio_chunk", (text, confidence)))
        spoken = " ".join(stages[-1].delivered[-1]) if (
            len(stages[-1].delivered) > before) else ""
        if spoken == REASK:
            reasked += 1
        else:
            answered += 1
            wrong += not correct
    return {"reasked": reasked, "answered": answered, "wrong": wrong}


def sweep(ref, ratio):
    rows = [(ratio * row["wrong"] + row["reasked"], threshold, row)
            for threshold in THRESHOLDS for row in [run(ref, threshold)]]
    return sorted(rows, key=lambda row: (row[0], row[1]))[0]


def reask_through_llm(ref):
    """The clarification routed as a transcript instead of as text."""
    stages = pipeline(ref, 0.0)
    stages[1].next.process(ref.Frame("transcript", REASK))
    return " ".join(stages[-1].delivered[-1])


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    best = sweep(ref, 5)
    gated = run(ref, best[1])
    ungated = run(ref, 0.0)
    return {
        "utterances": len(HEARD), "threshold": best[1], "cost": best[0],
        "gated": gated, "ungated": ungated,
        "by_ratio": {ratio: sweep(ref, ratio)[1] for ratio in (2, 5, 10)},
        "worst_error": max(c for _, c, ok in HEARD if not ok),
        "turn_cost": (gated["reasked"] * PREMIUM[0], gated["reasked"] * PREMIUM[1]),
        "premium": PREMIUM,
        "through_llm": reask_through_llm(ref),
        "frame_fields": list(ref.Frame.__dataclass_fields__),
    }


def verify(result):
    gated, ungated = result["gated"], result["ungated"]
    return [
        practice.Check(
            "ANSWER: a gate at 0.60 re-asks on 3 of 10 and cuts bad transcripts to 0",
            all([result["threshold"] == 0.6, gated["reasked"] == 3,
                 gated["wrong"] == 0, gated["answered"] == 7,
                 ungated["wrong"] == 3, result["utterances"] == 10]),
            f"below threshold the STT emits the clarification instead of a transcript, so "
            f"the LLM is never called on a guess: {gated['reasked']} clarifications and "
            f"{gated['answered']} answered turns over {result['utterances']} utterances, "
            f"with wrong transcripts falling from {ungated['wrong']} to {gated['wrong']}",
        ),
        practice.Check(
            "FINDING: the threshold is stable because the recogniser is calibrated",
            all([set(result["by_ratio"].values()) == {0.6}, result["cost"] == 3,
                 result["worst_error"] == 0.55]),
            f"at wrong-answer-to-re-ask ratios of 2, 5 and 10 the optimum stays at "
            f"{result['by_ratio'][5]}, cost {result['cost']}. Every error this recogniser "
            f"makes sits at confidence {result['worst_error']} or below, so one cut "
            "separates them at any price -- that is calibration, not the ratio",
        ),
        practice.Check(
            "FINDING: every gate costs a full round trip, not a stage",
            all([result["turn_cost"] == (1350, 1800),
                 result["premium"] == (450, 600), gated["reasked"] == 3]),
            f"a clarification is a complete turn -- heard, answered, chain rerun -- so at "
            f"the lesson's {result['premium'][0]}-{result['premium'][1]}ms premium band "
            f"{gated['reasked']} gates add {result['turn_cost'][0]}-"
            f"{result['turn_cost'][1]}ms. Gating is cheap per frame and expensive per "
            "dialogue, the opposite of how a per-stage budget makes it look",
        ),
        practice.Check(
            "FINDING: the re-ask has no route that skips the LLM",
            all([result["through_llm"] == "[no canned reply]",
                 result["frame_fields"] == ["kind", "payload", "direction"]]),
            f"routed as a transcript the clarification reaches LLM.replies, which has no "
            f"entry for it and answers {result['through_llm']!r} -- the user is asked to "
            "repeat and then told nothing. Emitting a text frame from the STT bypasses the "
            "LLM entirely, which works only because Processor forwards by kind",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
