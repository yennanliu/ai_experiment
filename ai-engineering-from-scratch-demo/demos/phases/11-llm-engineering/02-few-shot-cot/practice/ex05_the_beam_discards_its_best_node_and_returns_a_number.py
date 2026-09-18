r"""Exercise 5 — the beam discards its best node, and returns a number for a story.

    **ToT for creative tasks**: Adapt the Tree-of-Thought solver for a creative
    writing task: "Write a 6-word story that is both funny and sad." Use the LLM
    as evaluator. Does branching exploration produce better creative outputs
    than single-shot generation?

Reading of the exercise: the adaptation is a prompt swap, so it is done as one
-- `tree_of_thought_solve` driven through the `client` parameter, with a
scripted generator returning six-word stories and a scripted evaluator standing
in for "use the LLM as evaluator". What the search does with those scores is
then a property of the search, and is measured exactly.

**ANSWER: the adapted solver returns a number, or the empty string.**
`tree_of_thought_solve` ends with `extract_answer(best_thought)`, so "For sale:
baby shoes, never worn." comes back as `''` -- the `[\d,]+` class matches the
comma in the prose -- and "Born 1947. Died 1947. Lived anyway." as `'1947.'`.
`''` is not `None`, so Exercise 3's vote counts it as an answer. The story
survives only in the second return value, which the escalation pipeline throws
away: adapting this means changing the return contract, not the prompts.

**FINDING: the search replaces its frontier instead of extending it.** Each
depth does `scored = sorted(next_thoughts, ...)` over the *children only*, so a
parent that beats all of its children is thrown away. With an evaluator scoring
the three initial stories 0.95 and every continuation 0.40, the function
returns a 0.40 node. It is a beam search with no incumbent.

**FINDING: the evaluator's parser takes the first number in the reply, and
clamps up.** `re.search(r"([\d.]+)")` reads "Approach #2. Score: 0.9" as 2.0 ->
1.0 and "Score: 8/10" as 8.0 -> 1.0. Two of six labelled evaluator replies land
on the maximum, so a formatting slip outranks every honest score.

**FINDING: an evaluator that gives no number scores 0.5, and 0.5 ties.**
`float(".")` raises and the except branch returns 0.5, so every thought ties,
`sorted` is stable, and the beam follows generation order. Branching then
returns thought #1 -- single-shot with extra steps.

**ANSWER: branching costs 30 calls against 1.** Three generations, fifteen
evaluations, twelve extensions at breadth 3 and depth 3 -- and the harness
cannot hand back the artefact it selected.

Structure: `Stub` routes on the system prompt the lesson writes for each role;
`EVALUATOR_REPLIES` is the labelled scoring corpus.
"""

from __future__ import annotations

import re
import types

from harness import parity, practice

PHASE, LESSON = "11-llm-engineering", "02-few-shot-cot"
STORIES = ["For sale: baby shoes, never worn.", "He proposed. She laughed. Then cried.",
           "Wrong number, saved his life anyway."]
NUMERIC = "Born 1947. Died 1947. Lived anyway."
# (evaluator reply, the score a reader would give it)
EVALUATOR_REPLIES = [("0.9", 0.9), ("Score: 0.9", 0.9), ("0.4", 0.4),
                     ("Approach #2. Score: 0.9", 0.9), ("Score: 8/10", 0.8),
                     ("Excellent, but not six words.", 0.5)]


class Stub:
    """A scripted `client`, routed on the system prompt each ToT role writes."""

    def __init__(self, score_children=True):
        self.score_children, self.counts = score_children, {"gen": 0, "eval": 0, "ext": 0}

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        system, user = (m["content"] for m in kwargs["messages"][:2])
        if "evaluator" in system:
            self.counts["eval"] += 1
            child = "EXT" in user.split("Reasoning so far:")[1]
            text = "0.40" if child and self.score_children else "0.95"
            text = text if self.score_children else "no number here"
        elif "continuing a line of reasoning" in system:
            self.counts["ext"] += 1
            text = "EXT continuation."
        else:
            self.counts["gen"] += 1
            text = STORIES[self.counts["gen"] % len(STORIES)]
        message = types.SimpleNamespace(content=text)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


def scored_reply(text):
    """`evaluate_thought`'s parsing step, verbatim."""
    try:
        return min(max(float(re.search(r"([\d.]+)", text).group(1)), 0.0), 1.0)
    except (AttributeError, ValueError):
        return 0.5


def run(ref, score_children=True):
    stub = Stub(score_children)
    answer, best = ref.tree_of_thought_solve("Write a 6-word story.", stub, "model",
                                             breadth=3, depth=3)
    return {"answer": answer, "best": best, "counts": dict(stub.counts),
            "calls": sum(stub.counts.values()), "is_child": "EXT" in best,
            "lineage": best.split("\n\n")[0]}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "advanced_prompting")
    graded, tied = run(ref), run(ref, score_children=False)
    measured = [scored_reply(text) for text, _ in EVALUATOR_REPLIES]
    return {
        "graded": graded, "tied": tied,
        "story_answers": [ref.extract_answer(s) for s in STORIES],
        "numeric_answer": ref.extract_answer(NUMERIC),
        "measured": measured, "intended": [want for _, want in EVALUATOR_REPLIES],
        "wrong": [t for (t, want), got in zip(EVALUATOR_REPLIES, measured) if got != want],
        "clamped": sum(got == 1.0 for got in measured),
        "first_story": STORIES[1], "single_shot": 1,
    }


def verify(result):
    graded, tied = result["graded"], result["tied"]
    return [
        practice.Check(
            "ANSWER: the adapted solver returns a number, not the story",
            all([result["story_answers"] == ["", None, ""], result["numeric_answer"] == "1947.",
                 graded["answer"] is None, len(graded["best"]) > 0]),
            f"`tree_of_thought_solve` ends with extract_answer(best_thought), so the three "
            f"stories come back as {result['story_answers']} and {NUMERIC!r} as "
            f"{result['numeric_answer']!r}. Two are the empty string, because [\\d,]+ "
            "matches the comma in the prose -- and '' is not None, so Exercise 3's vote "
            "counts it. The story survives only in the discarded second return value",
        ),
        practice.Check(
            "FINDING: the frontier is replaced, so a parent beating its children is lost",
            all([graded["is_child"], graded["lineage"] in STORIES]),
            f"the evaluator scored the three initial stories 0.95 and every continuation "
            f"0.40, and the function returned a continuation -- {graded['best'][:46]!r}. "
            "`scored = sorted(next_thoughts, ...)` keeps only the children, so the beam "
            "carries no incumbent and the best node found can be discarded at any depth",
        ),
        practice.Check(
            "FINDING: the evaluator's parser reads the first number and clamps up",
            all([result["measured"] != result["intended"], result["clamped"] == 2]),
            f"measured {result['measured']} against {result['intended']} over six labelled "
            f"replies. {len(result['wrong'])} are wrong and {result['clamped']} land on the "
            f"clamp maximum 1.0 -- {result['wrong'][0]!r} reads as 2.0 and "
            f"{result['wrong'][1]!r} as 8.0, so a formatting slip outranks an honest score",
        ),
        practice.Check(
            "FINDING: no number scores 0.5, every thought ties, and the beam follows order",
            all([scored_reply("no number here") == 0.5,
                 tied["lineage"] == result["first_story"]]),
            f"`float('.')` raises and the except branch returns 0.5, so with an evaluator "
            f"that never emits a number all thoughts tie, `sorted` is stable, and the "
            f"lineage returned is the first generated -- {tied['lineage']!r}. Branching "
            "under a silent evaluator is single-shot with extra steps",
        ),
        practice.Check(
            "ANSWER: branching costs 30 calls against 1",
            all([graded["calls"] == 30, tied["calls"] == 30]),
            f"{graded['counts']} at breadth 3 and depth 3 -- {graded['calls']} calls "
            f"against {result['single_shot']} for single-shot generation. Whether the "
            "extra 29 bought a better story cannot be read off the return value, which is "
            "a number extracted from prose",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
