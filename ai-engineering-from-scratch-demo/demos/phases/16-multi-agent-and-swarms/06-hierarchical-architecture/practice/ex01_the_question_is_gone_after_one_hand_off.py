"""Exercise 1 — the question is gone after one hand-off.

    Run `code/main.py` and compare happy vs perturbed. How many levels of
    manager hand-off does it take before the top output fully diverges from
    the user's question?

Reading of the exercise: measure the overlap between the user's words and
what exists at each level, on both paths -- because the question presupposes
that the happy path keeps the question and the perturbed one loses it, and
neither is true.

**ANSWER: one hand-off, on the happy path.** Taking the content words of
"Ship the premium tier feature to production" and counting how many survive
at each level:

| level | overlap |
|---|---|
| top task | 5/5 |
| branch task (`task -- branch: X`) | 5/5 |
| leaf question | **1/5** |
| leaf answer | **0/5** |
| sub summary | 0/5 |
| top synthesis | 0/5 |

Four of the five words are lost at the top-to-sub hand-off and the last one,
"feature", survives only because three of the four split strings happen to
contain it. By the leaf *answer* -- still two levels below the top -- nothing
of the question remains.

**FINDING: the sub-manager discards the task it was given.**
`SubManager.run` computes `self.split.get(w.name, task)`, and every worker has
an entry, so the fallback never fires: **4** of **4** workers receive a
constant string fixed at construction. The `f"{task} -- branch: {label}"` the
top manager carefully builds reaches **0** leaves. Decomposition drift is not
what this demonstrates; the decomposition is a lookup table.

**FINDING: the happy path diverges exactly as far as the perturbed one.**
Both runs reach **0/5** overlap at the top synthesis. The demo contrasts them
as truthful against drifted, but measured against the user's question they are
equally divorced from it -- the perturbation changes *which* canned answers
appear, not whether any of them answers what was asked.

**FINDING: the one error a human could catch is never triggered.** An unknown
branch label produces a `MISSING[...]` summary whose text -- "no such
sub-manager" -- propagates verbatim into the top synthesis, so it is visible
at the top, not hidden a level below it. `main()` passes only labels that
exist, in **2** of **2** runs, so that path executes **0** times and the
closing claim about errors being one level removed is untested.

Structure: `overlap()` counts surviving content words; `levels()` collects the
text that exists at each depth of one run.
"""

from __future__ import annotations

import inspect
import re

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "06-hierarchical-architecture"
TASK = "Ship the premium tier feature to production."
STOP = {"the", "to", "of", "a", "and", "for", "in", "on", "is"}


def words(text):
    """Content words, lowercased, with the obvious stopwords dropped."""
    return {word for word in re.findall(r"[a-z]+", text.lower())} - STOP


def overlap(texts, asked):
    """How many of the user's words survive in this collection of strings."""
    present = set().union(*(words(text) for text in texts)) if texts else set()
    return sorted(asked & present)


def levels(run, labels):
    """Every string that exists at each depth of one hierarchy run."""
    leaves = [leaf for branch in run.branches for leaf in branch.leaves]
    return {"top task": [TASK],
            "branch task": [f"{TASK} -- branch: {label}" for label in labels],
            "leaf question": [leaf.question for leaf in leaves],
            "leaf answer": [leaf.answer for leaf in leaves],
            "sub summary": [branch.summary for branch in run.branches],
            "top synthesis": [run.synthesis]}


def briefs(top):
    """Workers holding a constant brief, and workers in total."""
    workers = [worker for sub in top.subs.values() for worker in sub.workers]
    fixed = [worker for sub in top.subs.values()
             for worker in sub.workers if worker.name in sub.split]
    return len(fixed), len(workers)


def first(surviving, predicate):
    """The first level name whose surviving count satisfies `predicate`."""
    return next(name for name, count in surviving.items() if predicate(count))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    top = ref.build_hierarchy()
    asked = words(TASK)
    happy_labels, perturbed_labels = ["engineering", "legal"], ["engineering", "finance"]
    happy = top.run(TASK, branch_labels=happy_labels)
    perturbed = top.run(TASK, branch_labels=perturbed_labels)
    surviving = {name: len(overlap(texts, asked))
                 for name, texts in levels(happy, happy_labels).items()}
    constants, workers = briefs(top)
    missing = top.run(TASK, branch_labels=["engineering", "nonexistent"])
    calls = re.findall(r"branch_labels=\[([^\]]+)\]", inspect.getsource(ref.main))
    leaf_questions = [leaf.question for branch in happy.branches for leaf in branch.leaves]
    return {
        "asked": len(asked), "surviving": surviving,
        "first_loss": first(surviving, lambda n: n < len(asked)),
        "zero_at": first(surviving, lambda n: n == 0),
        "constants": constants, "workers": workers,
        "task_reaches_leaf": any(TASK in question for question in leaf_questions),
        "perturbed_top": len(overlap([perturbed.synthesis], asked)),
        "happy_top": surviving["top synthesis"],
        "missing_visible": "no such sub-manager" in missing.synthesis,
        "main_labels": calls,
        "invalid_runs": sum("nonexistent" in call for call in calls),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: one hand-off, on the happy path",
            all([result["asked"] == 5, result["first_loss"] == "leaf question",
                 result["surviving"]["leaf question"] == 1,
                 result["zero_at"] == "leaf answer"]),
            f"of {result['asked']} content words, the branch task keeps all "
            f"{result['surviving']['branch task']}, the leaf question keeps "
            f"{result['surviving']['leaf question']} -- 'feature', and only because "
            f"three split strings contain it -- and the leaf answer keeps "
            f"{result['surviving']['leaf answer']}, two levels below the top",
        ),
        practice.Check(
            "FINDING: the sub-manager discards the task it was given",
            all([result["constants"] == result["workers"] == 4,
                 not result["task_reaches_leaf"]]),
            f"SubManager.run reads self.split.get(w.name, task) and all "
            f"{result['constants']} of {result['workers']} workers have an entry, so the "
            "fallback never fires and the branch task reaches 0 leaves -- the "
            "decomposition is a lookup table fixed at construction",
        ),
        practice.Check(
            "FINDING: the happy path diverges exactly as far as the perturbed one",
            all([result["happy_top"] == 0, result["perturbed_top"] == 0]),
            f"both runs reach {result['happy_top']}/{result['asked']} overlap at the top "
            "synthesis, so the perturbation changes which canned answers appear and not "
            "whether any of them answers what was asked",
        ),
        practice.Check(
            "FINDING: the one error a human could catch is never triggered",
            all([result["missing_visible"], result["invalid_runs"] == 0,
                 len(result["main_labels"]) == 2]),
            f"an unknown label propagates 'no such sub-manager' verbatim into the top "
            f"synthesis, so it is visible at the top; main() passes only labels that "
            f"exist in {len(result['main_labels'])} of "
            f"{len(result['main_labels'])} runs, so that path executes "
            f"{result['invalid_runs']} times",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
