"""Exercise 4 — the detector is a field the loop computes and throws away.

    Read Beel et al. Section 4 on presentation-quality gap. Design one
    additional evaluator that would catch polished-looking but
    experimentally flawed papers.

Reading of the exercise: "additional" is taken seriously -- the evaluator has
to run beside the existing stages on something they already produce, not
replace them with a better reviewer. So the design is a *provenance* check
rather than a quality check: record how the experiment reached its result, and
refuse any paper whose experiment was retry-recovered without an independent
re-run.

**ANSWER: the retry flag catches 1959 of 3173 flawed submissions -- 61.7% --
with 0 false positives.** In this loop a residual flaw is *defined* as a
retry-recovered experiment, so the flag is a complete detector for that class:
recall **1.000** on experiment flaws, precision **1.000**, and no model,
reviewer or figure is involved.

**FINDING: the loop computes the signal and discards it.** `run_one` binds
`recovered` and uses it only to decide whether to abandon; **0** of
`Outcome`'s **6** fields record that a retry happened. The evaluator the
exercise asks for needs one boolean that already exists in a local variable.

**FINDING: what provenance cannot reach.** The remaining **1214** flawed
submissions -- **38.3%** -- carry a novelty flaw alone. `has_novelty_flaw` is
drawn before any stage runs and never gates one, so no downstream artifact
correlates with it: the same paper, the same figures, the same retries. A
presentation-quality evaluator cannot reach that class at any threshold, and
saying so is the difference between an evaluator and a claim.

**FINDING: the figures are the wrong place to look.** The stage the exercise
points at writes nothing -- `polished_hides_weakness` is assigned and never
read -- so an evaluator aimed at the polish output would be reading a stage
with no effect on any outcome. The presentation-quality gap in this simulator
is entirely upstream of presentation.

Structure: `detector()` scores the retry flag as a classifier against the
ground-truth flags the loop already carries.
"""

from __future__ import annotations

import ast
import inspect
import random

from harness import parity, practice

PHASE, LESSON = "15-autonomous-systems", "05-ai-scientist-v2"

TRIALS, SEED = 20000, 42


def submissions(ref, config):
    random.seed(SEED)
    return [row for row in (ref.run_one(config) for _ in range(TRIALS)) if row.submitted]


def detector(rows):
    """The retry flag as a classifier over flawed submissions."""
    flawed = [row for row in rows if row.polished_but_flawed]
    flagged = [row for row in rows if row.has_experiment_flaw]
    caught = [row for row in flagged if row.polished_but_flawed]
    return {
        "submissions": len(rows),
        "flawed": len(flawed),
        "flagged": len(flagged),
        "caught": len(caught),
        "recall": round(len(caught) / len(flawed), 4),
        "precision": round(len(caught) / len(flagged), 4),
        "missed": len(flawed) - len(caught),
        "missed_share": round((len(flawed) - len(caught)) / len(flawed), 4),
    }


def name_uses(source, wanted):
    """How often a bare name is bound or read in the function -- comments excluded."""
    tree = ast.parse(source)
    return sum(isinstance(node, ast.Name) and node.id == wanted for node in ast.walk(tree))


def novelty_only(rows):
    return [row for row in rows
            if row.polished_but_flawed and not row.has_experiment_flaw]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = submissions(ref, ref.LoopConfig())
    scored = detector(rows)
    source = inspect.getsource(ref.run_one)
    fields = list(ref.Outcome.__dataclass_fields__)
    missed = novelty_only(rows)
    return {
        **scored,
        "outcome_fields": fields,
        "provenance_fields": [name for name in fields
                              if any(word in name for word in ("retry", "recovered",
                                                               "attempt"))],
        "recovered_mentions": name_uses(source, "recovered"),
        "all_missed_are_novelty": all(row.has_novelty_flaw for row in missed),
        "missed_with_experiment_flaw": sum(row.has_experiment_flaw for row in missed),
        "polish_dead": name_uses(source, "polished_hides_weakness") == 1,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the retry flag catches 61.7% of flawed submissions, precision 1.000",
            all([result["flawed"] == 3173, result["caught"] == 1959,
                 result["recall"] == 0.6174, result["precision"] == 1.0]),
            f"of {result['submissions']} submissions, {result['flawed']} are flawed and "
            f"the retry flag catches {result['caught']} of them -- recall "
            f"{result['recall']}, precision {result['precision']}, with no reviewer in "
            "the loop",
        ),
        practice.Check(
            "FINDING: the loop computes the signal and discards it",
            all([result["provenance_fields"] == [], len(result["outcome_fields"]) == 6,
                 result["recovered_mentions"] == 2]),
            f"run_one names `recovered` {result['recovered_mentions']} times, to bind it "
            f"and to branch on it, and {len(result['provenance_fields'])} of Outcome's "
            f"{len(result['outcome_fields'])} fields records that a retry happened",
        ),
        practice.Check(
            "FINDING: what provenance cannot reach",
            all([result["missed"] == 1214, result["missed_share"] == 0.3826,
                 result["all_missed_are_novelty"],
                 result["missed_with_experiment_flaw"] == 0]),
            f"the {result['missed']} misses -- {result['missed_share']:.1%} of flawed "
            "submissions -- carry a novelty flaw alone, drawn before any stage runs and "
            "gating none, so no downstream artifact distinguishes them",
        ),
        practice.Check(
            "FINDING: the figures are the wrong place to look",
            result["polish_dead"],
            "the polish stage assigns polished_hides_weakness and never reads it, so an "
            "evaluator aimed at its output would read a stage with no effect on any "
            "outcome -- the gap here is entirely upstream of presentation",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
