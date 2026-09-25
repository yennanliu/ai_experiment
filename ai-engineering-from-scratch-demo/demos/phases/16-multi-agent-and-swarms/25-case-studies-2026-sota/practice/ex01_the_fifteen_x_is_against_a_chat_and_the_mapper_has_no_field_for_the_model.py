"""Exercise 1 — the 15x is against a chat, and the mapper has no field for the model.

    Read the Anthropic Research system post end-to-end. Identify three design
    decisions that would change if you replaced Opus 4 with a smaller model
    (e.g., Haiku 4).

Reading of the exercise: the post was read in full (fetched 2026-09-25) and
each decision is anchored on a sentence it actually contains; the lesson's
summary of the post and the lesson's mapper are then checked against those
sentences, since the mapper is what the lesson ships to "apply" the case.

**ANSWER: the lead's model, the effort-scaling rules, and the token budget.**
(1) *Who leads.* The 90.2% is "Claude Opus 4 as the lead agent and Claude
Sonnet 4 subagents" against single-agent Opus 4, and the post finds
"upgrading to Claude Sonnet 4 is a larger performance gain than doubling the
token budget on Claude Sonnet 3.7" -- so a smaller model goes into the
subagent slots first, and the lead is the last seat to downgrade. (2) *Effort
scaling.* The lead decides "1 agent with 3-10 tool calls", "2-4 subagents
with 10-15 calls each" or "more than 10 subagents"; a smaller lead judges
complexity worse, so those tiers move out of its prompt into explicit rules.
(3) *Budget.* "Token usage by itself explains 80% of the variance", so a
cheaper model is bought back with tokens -- more subagents or more calls
each -- which is exactly the trade the post measured as the weaker lever.

What the check measures: none of the three can reach the lesson's mapper.
`Design` has 7 fields -- none is a model, a token budget or a tool-call
count -- and `CASES["anthropic_research"]` lists 4 patterns, 0 of them about
the model.
Swapping Opus for Haiku changes no input, so it changes no recommendation.

**FINDING: "15x tokens per query vs single-agent" is 15x against a chat.**
The post: "agents typically use about 4x more tokens than chat interactions,
and multi-agent systems use about 15x more tokens than chats". Against a
single agent the multi-agent system costs 15 / 4 = 3.75x -- the lesson
overstates the bill by the factor of 4 it dropped.

**FINDING: the post has no verifier role.** The lesson says the system "was
observed to hallucinate without explicit verifier roles", and the mapper's
Research case lists "verification role". The post's only hallucination
sentence is about *human testers* finding "hallucinated answers on unusual
queries"; its extra agent is a CitationAgent, which attaches citations -- "This
ensures all claims are properly attributed to their sources" -- attribution,
not verification. Evaluation is an offline LLM judge with a rubric.

Structure: `POST` holds the quoted sentences; `solve()` reads the reference
`Design`, `CASES` and the lesson text against them.
"""

from __future__ import annotations

import dataclasses

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "25-case-studies-2026-sota"
POST = {  # anthropic.com/engineering/multi-agent-research-system, fetched 2026-09-25
    "agent_vs_chat": 4, "multi_vs_chat": 15,
    "hallucination": "hallucinated answers on unusual queries",
    "citation": "ensures all claims are properly attributed to their sources",
    "roles": ("LeadResearcher", "subagents", "CitationAgent"),
}
MODEL_WORDS = ("model", "token", "budget", "tool", "calls")


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    doc = parity.doc_text(PHASE, LESSON)
    fields = [f.name for f in dataclasses.fields(ref.Design)]
    patterns = ref.CASES["anthropic_research"]["patterns"]
    return {
        "fields": fields,
        "model_fields": [f for f in fields if any(w in f for w in MODEL_WORDS)],
        "patterns": patterns,
        "model_patterns": [p for p in patterns if any(w in p for w in MODEL_WORDS)],
        "doc_15x": "**15x tokens per query** vs single-agent" in doc,
        "vs_single": POST["multi_vs_chat"] / POST["agent_vs_chat"],
        "doc_verifier": "hallucinate without explicit verifier roles" in doc,
        "case_verifier": "verification role" in patterns,
        "post_roles": POST["roles"],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the lead's model, the effort-scaling rules, and the token budget",
            all([len(result["fields"]) == 7, result["model_fields"] == [],
                 result["model_patterns"] == []]),
            f"the reference Design's fields are {result['fields']} and the Research "
            f"case's patterns {result['patterns']}: none names a model, a budget or a "
            "call count, so none of the three decisions a Haiku swap changes has an input",
        ),
        practice.Check(
            "FINDING: '15x tokens per query vs single-agent' is 15x against a chat",
            result["doc_15x"] and result["vs_single"] == 3.75,
            f"the post gives agents at {POST['agent_vs_chat']}x a chat and multi-agent at "
            f"{POST['multi_vs_chat']}x a chat, so against a single agent the multiple is "
            f"{result['vs_single']}x -- the lesson's 15x drops the factor of 4",
        ),
        practice.Check(
            "FINDING: the post has no verifier role",
            result["doc_verifier"] and result["case_verifier"],
            f"the lesson and the mapper both claim a verifier; the post's agents are "
            f"{list(result['post_roles'])}, its CitationAgent '{POST['citation']}', and "
            f"its one hallucination sentence is about human testers finding "
            f"'{POST['hallucination']}'",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
