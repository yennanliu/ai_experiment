"""Exercise 3 -- faithfulness stays 1.000 for 400 turns, and one turn of memory pins every citation to one chunk.

    Add multi-turn memory with a 10k-token summary buffer. Measure whether
    faithfulness drops as the conversation grows.

Reading of the exercise: the buffer works like a summary-buffer memory.
Recent turns (question plus answer) are kept verbatim while the whole buffer
fits in 10,000 tokens, counted with the lesson's `tokenize`. When it does not
fit, the oldest turn is folded into a running summary, which keeps only the
anchors that turn cited. A 400-turn analyst/GDPR conversation cycles through
six questions with known gold chunks. The memory reaches the pipeline the
only way the lesson's `chat_turn` accepts it, through the query text: summary
plus recent turns plus the new question. This is the "condense the history
into the query" pattern with no model to rewrite it. A no-memory run is the
control. Faithfulness is measured RAGAS-style: the fraction of the answer's
claims (its "anchor -> text" segments) whose text appears verbatim in a
retrieved chunk. Citation accuracy means the gold chunk is ranked first.

**ANSWER: no, faithfulness does not drop. It is 1.000 on every turn from 1
to 400, with or without memory, before and after the buffer first overflows
(turn 180).** The lesson's synthesiser only copies retrieved text, so it
cannot state anything the context lacks. What the growing conversation
damages is retrieval. With memory in the query, citation accuracy is 100% on
turn 1, then 33.3% over turns 2-10, 32.5% over 11-50, 33.3% over 51-179 and
33.5% over 180-400. The no-memory control stays at 100% throughout.

**FINDING: the damage comes at turn 2, not at 10k tokens, and it is total.**
One previous turn (58 tokens) is enough. From turn 2 to turn 400 the top
citation is `MSA-2024-03-11 s12.4` every time, whatever was asked. The 1/3
accuracy is just the share of questions whose gold chunk is that one. The
copied chunk text in the memory gives every eligible chunk's words to the
query, and `bm25_score` and the Jaccard `dense_score` then favour the same
chunk on every turn. Folding old turns into an anchors-only summary from
turn 180 on does not bring accuracy back.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "08-production-rag-chatbot"
BUDGET, TURNS, WHO = 10_000, 400, ("analyst", "GDPR")
MSA, DPA, FAQ = "MSA-2024-03-11 s12.4", "DPA-v2.1 s5", "general-privacy-faq Q1"
QUESTIONS = [("how many days to delete EU user profiles after termination", MSA),
             ("deletion deadline for the restricted data category", DPA), ("how can users request a data export", FAQ),
             ("when must EU user profiles be deleted", MSA),
             ("how fast is restricted data deleted after a termination notice", DPA),
             ("where do users export their data", FAQ)]
WINDOWS = [(0, 1), (1, 10), (10, 50), (50, 179), (179, 400)]


def faithfulness(ref, reply):
    """Share of 'anchor -> text' claims whose text is verbatim inside the chunk the claim cites."""
    text = {c.anchor(): c.text for c in ref.CORPUS}
    claims = reply["answer"].split(": ", 1)[1].split("; ")
    return sum(c.split(" -> ", 1)[1] in text[c.split(" -> ", 1)[0]] for c in claims) / len(claims)


def conversation(ref, with_memory):
    recent, summary, turns, filled = [], [], [], None
    count = lambda parts: len(ref.tokenize(" ".join(parts)))
    for t in range(TURNS):
        q, gold = QUESTIONS[t % len(QUESTIONS)]
        memory = " ".join(summary + recent) if with_memory else ""
        reply = ref.chat_turn(f"{memory} {q}".strip(), *WHO, ref.CORPUS, ref.PromptCache())
        turns.append((reply["citations"][0] == gold, faithfulness(ref, reply), reply["citations"][0]))
        recent.append(f"{q} {reply['answer']}")
        while count(summary + recent) > BUDGET:
            filled = t + 1 if filled is None else filled
            oldest = recent.pop(0)
            summary.append(" ".join(w for w in oldest.split() if "-" in w and w[0].isupper()))
    return turns, filled


def windows(turns):
    return [round(sum(t[0] for t in turns[a:b]) / (b - a), 3) for a, b in WINDOWS]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    mem, filled = conversation(ref, True)
    ctrl, _ = conversation(ref, False)
    q0 = QUESTIONS[0][0]
    return {
        "filled_at": filled, "memory_accuracy": windows(mem), "control_accuracy": windows(ctrl),
        "faith": sorted({t[1] for t in mem + ctrl}), "turn2_correct": mem[1][0],
        "memory_tops": sorted({t[2] for t in mem[1:]}), "control_tops": len({t[2] for t in ctrl}),
        "first_turn_tokens": len(ref.tokenize(f"{q0} {ref.chat_turn(q0, *WHO, ref.CORPUS, ref.PromptCache())['answer']}")),
    }


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: faithfulness is 1.000 on all 400 turns, memory or not, before and after the buffer fills",
            r["faith"] == [1.0] and r["filled_at"] == 180,
            f"distinct faithfulness values {r['faith']}; 10k-token buffer first overflows at turn {r['filled_at']}",
        ),
        practice.Check(
            "ANSWER: with memory in the query, citation accuracy falls from 100% to 1/3; the control holds",
            r["memory_accuracy"] == [1.0, 0.333, 0.325, 0.333, 0.335] and r["control_accuracy"] == [1.0] * 5,
            f"accuracy by turn window {WINDOWS}: memory {r['memory_accuracy']}, no memory {r['control_accuracy']}",
        ),
        practice.Check(
            "FINDING: one 58-token turn pins the top citation to MSA s12.4 for the rest of the conversation",
            r["turn2_correct"] is False and r["first_turn_tokens"] == 58
            and r["memory_tops"] == [MSA] and r["control_tops"] == 3,
            f"turn 2 correct: {r['turn2_correct']}, carrying one {r['first_turn_tokens']}-token turn; top citation "
            f"on turns 2-400: {r['memory_tops']} (control uses {r['control_tops']} distinct)",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
