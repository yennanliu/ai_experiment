"""Exercise 5 — the lesson's code has no self-critique, no model and no training step, so its delta is 0 at every size.

    Run the constitutional self-critique on a 30B model and measure whether the delta scales.

Reading of the exercise: the self-critique run exists only in the prose
("Build It" step 6: draft, critique against a constitution, rewrite,
fine-tune, measure on a held-out eval). The solution therefore measures what
the lesson's `code/main.py` can do. It parses the module for any critique,
revise, train or model entry point, and asks whether `SafetyPipeline` takes
a model or a size at all. It then runs the only critic the code has, the
`llama_guard_4` + `x_guard` gate, as a self-critique over the lesson's own
12 red-team probes. The drafts it critiques are the stub target's responses.
No new prompts are written; the fixture is the lesson's own strings.

**ANSWER: the delta cannot scale because there is no delta. It is 0 at 8B,
at 30B and at any other size.** `main.py` defines 18 functions, methods and
classes, and none of them critiques, revises, trains or loads a model.
`SafetyPipeline` has one field, `domain`, so `SafetyPipeline(model="30B")`
raises `TypeError`. The target's "response" is the f-string
`(target response for: <first 60 prompt characters>...)`. The pipeline
blocks 10 of the 12 range probes before and after any critique pass,
because a pass has nothing to update.

**FINDING: a self-critique that uses the code's own critic would fix 0 of
the 2 successful attacks.** The two successes are the base64 `encoding`
probes. The gate objects to 0 of their 2 drafts, and to 0 of their 2
sanitized prompts, because the payload stays encoded below `sanitize`'s
32-character cut-off. A critique-and-rewrite loop bounded by this critic
has no target to rewrite. Across all 12 drafts the gate objects to 0: the
stub echoes the first 60 characters, which for every probe are the banking
pretext (the `_bank` prefix alone is 58), and no pretext has a lexicon word.
"""

from __future__ import annotations

import ast
import dataclasses
import re

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "15-constitutional-safety-harness"
TRAINING_WORDS = re.compile(r"critique|revis|train|fine_?tune|sft|constitution|model|llm|harmless", re.I)


def defined_names():
    src = (parity.lesson_dir(PHASE, LESSON) / "code" / "main.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    return [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))]


def objects(ref, text):
    """The code's only critic: its two classifier stand-ins."""
    return not (ref.llama_guard_4(text)[0] and ref.x_guard(text)[0])


def size_error(ref):
    try:
        ref.SafetyPipeline(model="30B")
    except TypeError as exc:
        return type(exc).__name__
    return None


def structure(ref):
    names = defined_names()
    return {"names": names, "training_names": [n for n in names if TRAINING_WORDS.search(n)],
            "fields": [f.name for f in dataclasses.fields(ref.SafetyPipeline)], "size_error": size_error(ref)}


def critique_pass(ref):
    """Run the code's critic over the stub drafts of the lesson's own 12 range probes."""
    pipe = ref.SafetyPipeline()
    family = {a.prompt: f for f, attacks in ref.run_range(pipe).items() for a in attacks}
    runs = {p: pipe.process(p) for p in family}
    drafts = {p: f"(target response for: {ref.sanitize(p)[:60]}...)" for p in family}
    passed = [p for p, r in runs.items() if not r["blocked"]]
    return {"n_probes": len(family), "blocked": len(family) - len(passed),
            "stable": runs == {p: pipe.process(p) for p in family},
            "passed_families": [family[p] for p in passed], **critic_counts(ref, runs, drafts, passed)}


def critic_counts(ref, runs, drafts, passed):
    return {"response_is_stub": [runs[p]["response"] == drafts[p] for p in passed],
            "critic_on_passed_drafts": sum(objects(ref, runs[p]["response"]) for p in passed),
            "critic_on_passed_prompts": sum(objects(ref, ref.sanitize(p)) for p in passed),
            "critic_on_all_drafts": sum(objects(ref, d) for d in drafts.values())}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return {**structure(ref), **critique_pass(ref), "prefix_len": len(ref._bank(""))}


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: no critique, training or model entry point; no size parameter; delta 0 at any size",
            (r["training_names"], len(r["names"]), r["fields"], r["size_error"], r["blocked"], r["n_probes"],
             r["stable"], r["response_is_stub"]) == ([], 18, ["domain"], "TypeError", 10, 12, True, [True, True]),
            f"{len(r['names'])} definitions, training-related {r['training_names']}; SafetyPipeline "
            f"fields {r['fields']}; model='30B' -> {r['size_error']}; blocks {r['blocked']}/"
            f"{r['n_probes']} range probes on every run",
        ),
        practice.Check(
            "FINDING: the code's own critic objects to 0 of the 2 successful attacks and 0 of 12 drafts",
            (r["passed_families"], r["critic_on_passed_drafts"], r["critic_on_passed_prompts"],
             r["critic_on_all_drafts"], r["prefix_len"]) == (["encoding", "encoding"], 0, 0, 0, 58),
            f"successes {r['passed_families']}; critic objects to {r['critic_on_passed_drafts']}/2 drafts, "
            f"{r['critic_on_passed_prompts']}/2 sanitized prompts, {r['critic_on_all_drafts']}/12 drafts",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
