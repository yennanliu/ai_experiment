"""Exercise 3 — the canary cannot answer the question it is asked.

    Implement a "canary" worker at each sub-manager that is always asked the
    original user question unchanged. Use the canary answer to detect
    decomposition drift. How should the manager react when the canary
    disagrees with the synthesized answer?

Reading of the exercise: build the canary, run it, and read what comes back --
because every worker here answers by keyword lookup, and the original question
contains none of their keywords. The canary does detect drift. It detects it
by failing.

**ANSWER: escalate only when the canary *answered* and disagreed; a canary
that cannot answer is a different alarm.** Asking all **4** shipped workers
the user's question unchanged returns the fallback string
`[no canned answer for ...]` **4** times out of 4 -- `Worker._match_key`
scans its canned keys for a substring of the question, finds none, and returns
`"default"`, which is a key **0** of the four workers define. So a naive rule
"canary disagrees, therefore drift" fires on **4/4** sub-managers on the happy
path, which makes it useless. The rule that carries information is
*answered and disagreed*; *could not answer* means no worker is competent for
the question as the user phrased it, which is a decomposition problem of a
different kind and wants a different response -- re-plan, not re-run.

**FINDING: the fallback fires twice for one condition.** `_match_key` returns
the literal `"default"` when nothing matches, and `run` then calls
`self.canned.get(key, f"[no canned answer for '{question}']")`. Since no
worker declares a `"default"` key, both fallbacks fire on the same input, and
the sentinel that was meant to name a catch-all answer instead guarantees the
catch-all string. **0** of **4** canned dictionaries contain it.

**FINDING: agreement cannot be computed even in principle here.** The canary
answer and the sub-manager's summary share **0** content words on every
branch, because one is a fallback sentence about a missing answer and the
other is a join of canned findings. Any comparison that is not exact-match
returns "disagree" for structural reasons, so the canary's signal is
indistinguishable from the drift it is meant to detect.

**FINDING: the canary is the only component that would see the question.**
The shipped workers receive a constant brief in **4** of **4** cases, so the
canary would be the sole path by which the user's words reach a leaf at all --
which is the strongest argument for adding it, and also why it has nothing to
compare against once it gets there.

Structure: `canaries()` asks every worker the untouched question;
`agreement()` compares what comes back with the branch summary.
"""

from __future__ import annotations

import inspect
import re

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "06-hierarchical-architecture"
TASK = "Ship the premium tier feature to production."
STOP = {"the", "to", "of", "a", "and", "for", "in", "on", "is"}
FALLBACK = "no canned answer"


def words(text):
    """Content words, lowercased, with the obvious stopwords dropped."""
    return {word for word in re.findall(r"[a-z]+", text.lower())} - STOP


def canaries(top):
    """Every sub-manager's canary: one worker asked the user's question unchanged."""
    asked = {}
    for name, sub in top.subs.items():
        worker = sub.workers[0]
        asked[name] = worker.run(TASK)
    return asked


def agreement(canary, summary):
    """Do the canary answer and the branch summary say anything in common?"""
    return sorted(words(canary.answer) & words(summary))


def briefs(top):
    """Workers holding a constant brief, and workers in total."""
    workers = [worker for sub in top.subs.values() for worker in sub.workers]
    fixed = [worker for sub in top.subs.values()
             for worker in sub.workers if worker.name in sub.split]
    return len(fixed), len(workers)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    top = ref.build_hierarchy()
    run = top.run(TASK, branch_labels=["engineering", "legal"])
    asked = canaries(top)
    summaries = {branch.sub_manager.split("-")[0]: branch.summary for branch in run.branches}
    overlaps = [len(agreement(asked[name], summary))
                for name, summary in (("engineering", summaries.get("eng", "")),
                                      ("legal", summaries.get("legal", "")))]
    constant_briefs, worker_count = briefs(top)
    workers = [worker for sub in top.subs.values() for worker in sub.workers]
    return {
        "subs": len(top.subs),
        "unanswered": sum(FALLBACK in out.answer for out in asked.values()),
        "canaries": len(asked),
        "keys": [worker._match_key(TASK) for worker in workers],
        "default_declared": sum("default" in worker.canned for worker in workers),
        "workers": worker_count,
        "fallback_in_run": "no canned answer" in inspect.getsource(ref.Worker.run),
        "sentinel_in_match": "default" in inspect.getsource(ref.Worker._match_key),
        "overlaps": overlaps,
        "constant_briefs": constant_briefs,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: escalate only when the canary answered and disagreed",
            all([result["unanswered"] == result["canaries"] == 3,
                 set(result["keys"]) == {"default"},
                 result["default_declared"] == 0]),
            f"all {result['canaries']} canaries return the fallback: _match_key finds no "
            f"keyword in the user's question and returns 'default', a key "
            f"{result['default_declared']} of {result['workers']} workers declare -- so "
            "'canary disagrees therefore drift' fires on every branch of the happy path, "
            "and only 'answered and disagreed' carries information",
        ),
        practice.Check(
            "FINDING: the fallback fires twice for one condition",
            all([result["sentinel_in_match"], result["fallback_in_run"],
                 result["default_declared"] == 0]),
            f"_match_key returns the literal 'default' and run() then calls canned.get "
            f"with a second fallback string; {result['default_declared']} of "
            f"{result['workers']} canned dictionaries define that key, so both fire on "
            "the same input",
        ),
        practice.Check(
            "FINDING: agreement cannot be computed even in principle here",
            result["overlaps"] == [0, 0],
            f"the canary answer and the branch summary share {result['overlaps']} "
            "content words -- one is a sentence about a missing answer, the other a join "
            "of canned findings -- so any non-exact comparison returns 'disagree' for "
            "structural reasons",
        ),
        practice.Check(
            "FINDING: the canary is the only component that would see the question",
            all([result["constant_briefs"] == result["workers"] == 4]),
            f"the shipped workers receive a constant brief in {result['constant_briefs']} "
            f"of {result['workers']} cases, so the canary would be the only path by "
            "which the user's words reach a leaf -- which is the argument for it, and "
            "why it has nothing to compare against once there",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
