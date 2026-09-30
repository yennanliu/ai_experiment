"""Exercise 1 -- recipes alone pass 11 of 50 repos, the agent alone passes 19 more, and neither choice reads the repo.

    Run the migrate pipeline with OpenRewrite only (no agent). Compare pass rate to the full pipeline. Identify the cases where the agent alone is the difference.

Reading of the exercise: the lesson's `migrate` is run twice on its own
50-repo bench (`synth_bench`, `main()`'s seed 19): once as shipped, and
once with `agent_loop` replaced by a stub that files the repo instead of
iterating, which is "OpenRewrite only". The cases where the agent alone is
the difference are the passes that needed at least one agent turn. Because
`main()` shares one rng across all 50 repos, switching the agent off also
re-rolls every later repo, so the paired answer reads both outcomes off
the same run, and the rerun is reported separately. Rates are then taken
over 1,000 seeds.

**ANSWER: OpenRewrite only passes 11 of 50 (22%); the full pipeline passes
30 (60%).** The agent alone is the difference on 19 repos, after 1 to 8
turns. They are not a harder class: their hardness spans 0.219-0.823 and
the recipe-only passes span 0.220-0.735. A repo passes on recipes alone when
one draw lands under `0.55 * (1 - hardness)`, which caps at 52.25% even for
the easiest repo, so the agent-alone cases are the repos whose draw missed.
Over 1,000 seeds the rates are 19.45% recipe-only and 56.61% full; the
formula predicts 19.42%.

**FINDING: rerunning with the agent off rescues 4 repos the full pipeline
failed.** The recipe-only rerun of seed 19 passes 15, and 4 of them failed
with the agent on. The agent loop consumes rng draws, so the naive A/B
compares different coin flips, not the same repo with and without an agent.

**FINDING: the lesson's own "70-80%" is 19.45% in its code.** Build It step
1 says recipes "catch the 70-80% of migrations that are mechanical".

**FINDING: the failure class is drawn from the rng, not from the repo.**
Over 1,000 seeds, 1,186 of the 3,013 repos filed as `custom_annotation`
(39.4%) are Python repos, the same share Python has in the bench (40%).
`classify_failure` reads no build log; it samples fixed weights.

Structure: `run()` swaps `agent_loop` for `no_agent` and restores it;
`shipped()` is the paired seed-19 comparison; `sweep()` the 1,000 seeds.
"""

from __future__ import annotations

import random
import statistics

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "09-code-migration-agent"
SHIPPED_SEED, SEEDS = 19, 1000


def no_agent(attempt, rng):
    """The recipe-only pipeline: a repo the recipes did not finish is filed, not handed on."""
    attempt.status, attempt.failure_class = "fail", "recipe_only"


def run(ref, seed, agent=True):
    original, rng = ref.agent_loop, random.Random(seed)
    bench = ref.synth_bench(rng)
    ref.agent_loop = original if agent else no_agent
    try:
        return [ref.migrate(repo, rng) for repo in bench]
    finally:
        ref.agent_loop = original


def passed(results):
    return {a.repo.name for a in results if a.status == "pass"}


def span(values):
    values = list(values)
    return [round(min(values), 3), round(max(values), 3)]


def shipped(ref):
    full = run(ref, SHIPPED_SEED)
    recipe = [a for a in full if a.status == "pass" and a.agent_turns == 0]
    alone = [a for a in full if a.status == "pass" and a.agent_turns > 0]
    rerun = passed(run(ref, SHIPPED_SEED, agent=False))
    return {
        "full": len(passed(full)), "recipe": len(recipe), "alone": len(alone),
        "alone_turns": span(a.agent_turns for a in alone),
        "hard_recipe": span(a.repo.hardness for a in recipe), "hard_alone": span(a.repo.hardness for a in alone),
        "rerun": len(rerun), "rerun_not_in_full": len(rerun - passed(full)),
    }


def sweep(ref):
    full = recipe = mislabelled = annotated = 0
    hard = []
    for seed in range(SEEDS):
        f, r = run(ref, seed), run(ref, seed, agent=False)
        full, recipe = full + len(passed(f)), recipe + len(passed(r))
        hard += [a.repo.hardness for a in f]
        tagged = [a for a in f if a.failure_class == "custom_annotation"]
        annotated += len(tagged)
        mislabelled += sum(a.repo.lang == "python" for a in tagged)
    return {
        "full_rate": round(full / SEEDS / 50, 4), "recipe_rate": round(recipe / SEEDS / 50, 4),
        "predicted": round(0.55 * (1 - statistics.mean(hard)), 4),
        "annotated": annotated, "python_annotated": mislabelled,
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    doc = parity.doc_text(PHASE, LESSON, "en")
    return {**shipped(ref), **sweep(ref), "doc_claim": "Catch the 70-80% of migrations that are mechanical" in doc,
            "cap": round(0.55 * (1 - 0.05), 4)}


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: OpenRewrite only passes 11 of 50 (22%), the full pipeline 30 (60%)",
            (r["full"], r["recipe"], r["alone"], r["alone_turns"]) == (30, 11, 19, [1, 8])
            and (r["hard_alone"], r["hard_recipe"], r["cap"]) == ([0.219, 0.823], [0.22, 0.735], 0.5225)
            and (r["recipe_rate"], r["full_rate"], r["predicted"]) == (0.1945, 0.5661, 0.1942),
            f"seed 19: {r['recipe']} recipe-only + {r['alone']} agent-alone ({r['alone_turns']} turns) = "
            f"{r['full']}; hardness {r['hard_recipe']} vs {r['hard_alone']}; 1,000 seeds: "
            f"{r['recipe_rate']:.2%} vs {r['full_rate']:.2%} (formula {r['predicted']:.2%})",
        ),
        practice.Check(
            "FINDING: rerunning with the agent off rescues 4 repos the full pipeline failed",
            (r["rerun"], r["rerun_not_in_full"]) == (15, 4),
            f"the recipe-only rerun passes {r['rerun']}, {r['rerun_not_in_full']} of them failed with the agent",
        ),
        practice.Check(
            "FINDING: the lesson's own '70-80%' is 19.45% in its code",
            r["doc_claim"] and r["recipe_rate"] < 0.2,
            f"doc says 70-80%: {r['doc_claim']}; measured {r['recipe_rate']:.2%}",
        ),
        practice.Check(
            "FINDING: the failure class is drawn from the rng, not from the repo",
            (r["annotated"], r["python_annotated"]) == (3013, 1186),
            f"{r['python_annotated']} of {r['annotated']} custom_annotation repos are Python",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
