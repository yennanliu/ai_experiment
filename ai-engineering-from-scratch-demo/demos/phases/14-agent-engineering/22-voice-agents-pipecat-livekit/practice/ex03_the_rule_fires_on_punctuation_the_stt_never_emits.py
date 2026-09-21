"""Exercise 3 — the rule fires on punctuation the STT never emits.

    Add semantic turn detection: simple rule — if transcript ends with "?",
    end of turn.

Reading of the exercise: the rule is stated as a straw man and behaves like
one, but not for the reason it looks like. `STT.process` emits
`str(frame.payload)` verbatim, and the lesson's own payloads are `"hello"`
and `"refund please"` -- no recogniser in the pipeline ever produces a
question mark. So the rule is measured on realistic unpunctuated transcripts,
then against a rule that reads the words instead.

**ANSWER: the "?" rule ends 0 of 12 turns; a word-level rule gets 11 of 12.**
On unpunctuated transcripts `endswith("?")` is never true, so every turn
runs to the pipeline's default -- which is to answer immediately, and the
rule scores **4/12** purely by agreeing with the non-endings. Checking for a
trailing filler instead gives **11/12**, with **1** false end and **0**
missed.

**FINDING: the pipeline has no turn to detect.** Every `vad_speech` frame
produces a `transcript` and an LLM call on the spot, so a three-chunk
utterance calls the LLM **3** times and emits **3** replies before the
speaker has finished. Turn detection needs a buffer, and `Processor` has
**3** attributes -- `name`, `next`, `prev` -- and no place to keep one.

**FINDING: punctuation is a property of the recogniser, not the speaker.**
Adding it back to the same **12** transcripts takes the "?" rule from
**4/12** to **10/12** -- better, and still short of the word rule's
**11/12**, because **2** of the turn-ending utterances are not questions at
all ("tell me the status", "hello"). Whether the rule works is a fact about
the STT's formatting policy, not about the speaker.

**FINDING: the two rules fail in opposite directions, and the costs differ.**
The word rule makes **1** error and it is a false end; the punctuated "?"
rule makes **2** and both are missed ends. A missed end costs one extra turn
-- **450-600ms** each, so **900-1200ms** across the fixture -- and is merely
slow. A false end interrupts the speaker and needs a barge-in the shipped
`TTS` cannot perform, because nothing mutates `self.cancelled` during its
emit loop. Picking a rule means picking which of those to have.

Structure: `ends_turn()` holds the three rules; `score()` runs each over the
same labelled transcripts.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "22-voice-agents-pipecat-livekit"
OPENERS = ("what", "where", "when", "why", "how", "who", "can", "could",
           "do", "does", "is", "are", "should", "which")
FILLERS = ("um", "uh", "and", "so", "but", "then")
# (raw transcript, punctuated form, is end of turn)
LINES = (
    ("what is my order status", "What is my order status?", True),
    ("um", "Um,", False),
    ("can you refund this", "Can you refund this?", True),
    ("i want to and", "I want to and", False),
    ("how long does shipping take", "How long does shipping take?", True),
    ("tell me the status", "Tell me the status.", True),
    ("the order number is", "The order number is", False),
    ("is that the right address", "Is that the right address?", True),
    ("hello", "Hello.", True),
    ("so", "So", False),
    ("cancel my order please", "Cancel my order please?", True),
    ("do you have it in blue", "Do you have it in blue?", True),
)
PREMIUM = (450, 600)


def ends_turn(text, rule):
    if rule == "question_mark":
        return text.endswith("?")
    words = text.lower().rstrip(".,?").split()
    if not words:
        return False
    if rule == "words":
        return not (words[-1] in FILLERS or words[0] in OPENERS
                    and len(words) < 2)
    return False


def predictions(rule, punctuated):
    return [(ends_turn(shown if punctuated else raw, rule), truth)
            for raw, shown, truth in LINES]


def score(rule, punctuated=False):
    rows = predictions(rule, punctuated)
    return {"correct": sum(p == t for p, t in rows),
            "false_end": sum(p > t for p, t in rows),
            "missed": sum(t > p for p, t in rows),
            "fired": sum(p for p, _ in rows)}


def chunked_turn(ref):
    """One utterance arriving as three audio chunks, through the shipped chain."""
    stages = (ref.VAD("vad"), ref.STT("stt"),
              ref.LLM("llm", replies={}), ref.TTS("tts"), ref.Transport("transport"))
    ref.link(*stages)
    for chunk in ("i want to", "cancel my", "order please"):
        stages[0].process(ref.Frame("audio_chunk", chunk))
    llm_calls = sum(line.startswith("LLM:") for line in stages[2].trace)
    return {"calls": llm_calls, "delivered": len(stages[-1].delivered),
            "attributes": sorted(vars(ref.Processor("p")))}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    marks = score("question_mark")
    words = score("words")
    punctuated = score("question_mark", punctuated=True)
    return {
        "lines": len(LINES), "marks": marks, "words": words,
        "punctuated": punctuated, "chunked": chunked_turn(ref),
        "premium": PREMIUM,
        "raw_with_marks": sum(raw.endswith("?") for raw, _, _ in LINES),
    }


def verify(result):
    marks, words, punctuated = result["marks"], result["words"], result["punctuated"]
    chunked = result["chunked"]
    return [
        practice.Check(
            "ANSWER: the '?' rule ends 0 of 12 turns, a word-level rule gets 11 of 12",
            all([marks["fired"] == 0, marks["correct"] == 4, result["lines"] == 12,
                 words["correct"] == 11, words["false_end"] == 1,
                 words["missed"] == 0, result["raw_with_marks"] == 0]),
            f"{result['raw_with_marks']} of {result['lines']} raw transcripts end in a "
            f"question mark, so the rule fires {marks['fired']} times and is right "
            f"{marks['correct']}/{result['lines']} only by agreeing with the "
            f"non-endings. Reading the words gives {words['correct']}/{result['lines']}, "
            f"{words['false_end']} false end and {words['missed']} missed",
        ),
        practice.Check(
            "FINDING: the pipeline has no turn to detect",
            all([chunked["calls"] == 3, chunked["delivered"] == 3,
                 chunked["attributes"] == ["name", "next", "prev", "trace"]]),
            f"every vad_speech frame produces a transcript and an LLM call on the spot, so "
            f"a three-chunk utterance makes {chunked['calls']} calls and delivers "
            f"{chunked['delivered']} replies before the speaker has finished. A detector "
            f"needs a buffer and Processor carries {chunked['attributes']}",
        ),
        practice.Check(
            "FINDING: punctuation is a property of the recogniser, not the speaker",
            all([punctuated["correct"] == 10, marks["correct"] == 4,
                 punctuated["correct"] < words["correct"],
                 punctuated["missed"] == 2, punctuated["false_end"] == 0]),
            f"restoring punctuation takes the same rule from {marks['correct']}/12 to "
            f"{punctuated['correct']}/12 -- still short of the word rule's "
            f"{words['correct']}/12, because {punctuated['missed']} of the turn-ending "
            "utterances are not questions at all. Whether the rule works is a fact about "
            "the STT's formatting policy",
        ),
        practice.Check(
            "FINDING: the two rules fail in opposite directions, and the costs differ",
            all([words["false_end"] == 1, words["missed"] == 0,
                 punctuated["missed"] == 2, punctuated["false_end"] == 0,
                 result["premium"] == (450, 600)]),
            f"the word rule makes {words['false_end']} error and it is a false end; the "
            f"punctuated rule makes {punctuated['missed']} and both are missed ends. A "
            f"missed end is merely slow, {result['premium'][0]}-{result['premium'][1]}ms "
            f"of extra turn each; a false end interrupts the speaker and needs a barge-in "
            "the shipped TTS cannot perform, since nothing mutates self.cancelled during "
            "its emit loop",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
