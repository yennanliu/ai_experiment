"""Exercise 4 -- the codemod clears 346 of 401 Node 22 hazards, and swapping it for no recipes at all changes 0 of 50 outcomes.

    Extend to a third migration: Node 18 to Node 22. Reuse the sandbox wrapping; swap the recipe layer for a custom codemod.

Reading of the exercise: the codemod is five regex rules for four Node 18
to 22 breaks: import assertions (`assert { type: 'json' }`, a SyntaxError
on Node 22, checked on node v22.23.2 while writing this), `util.isArray`
and `util.isDate` (DEP0044/DEP0047, runtime-deprecated in v22.0.0),
`new Buffer()` (DEP0005) and `require('punycode')` (DEP0040,
runtime-deprecated in v21.0.0), per https://nodejs.org/api/deprecations.html
(read 2026-09-29). `new Buffer(x)` with a non-literal argument is left for
the agent: from its text alone it could mean `Buffer.from` or
`Buffer.alloc`. The fixture is 50 generated 12-line files (`make_fixture`,
seed 19). The sandbox wrapping reused is the lesson's own `migrate`, with
its `agent_loop` and budget, run on the `synth_bench` repos relabelled
`lang="node"`, with `run_recipes` replaced by the codemod.

**ANSWER: the codemod resolves 346 of the 401 hazards (86.3%) and plugs
into `migrate` unchanged.** It removes all 50 import assertions, 127
`util.is*` calls and 55 punycode requires, and 114 of the 169
`new Buffer` calls; the 55 left are `new Buffer(payload)`, the agent's
share. 19 of the 50 files come out with nothing left to fix. Through the
lesson's pipeline 30 of the 50 Node repos pass.

**FINDING: the recipe layer does not affect the pipeline at all.** With
the codemod, the lesson's `run_recipes` or a layer that rewrites nothing,
all 50 repos get the same status, failure class, turn count and cost.
`migrate` stores the rewrite count and never reads it; `lang` is never
read either, so "node" passes through without any change.

**FINDING: with no recipes, 9 repos pass without an agent turn while their
source still does not parse on Node 22.** 36 of the 50 files contain an
import assertion; 11 repos pass straight after the (empty) recipe pass,
and 9 of those 11 still hold one. With the codemod, 9 straight passes
still hold a deferred `new Buffer(payload)`.

Structure: `codemod()` applies `RULES`; `hazards()` counts what Node 22
rejects or warns on; `outcomes()` swaps `run_recipes` and runs `migrate`.
"""

import dataclasses
import random
import re

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "09-code-migration-agent"
SEED, REPOS, LINES_PER_FILE = 19, 50, 12
POOL = [  # Node 18 idioms a file is built from
    "import cfg from './config.json' assert { type: 'json' };", "if (util.isArray(items)) return items.length;",
    "if (util.isDate(when)) return when.toISOString();", "const greeting = new Buffer('hello', 'utf8');",
    "const pad = new Buffer(16);", "const raw = new Buffer(payload);", "const puny = require('punycode');",
    "const fs = require('node:fs');", "export function add(a, b) { return a + b; }",
    "const url = new URL(req.url, 'http://localhost');",
]
RULES = [  # the codemod: (pattern, replacement); `new Buffer(<variable>)` is left for the agent
    (r"\bassert(\s*\{\s*type:)", r"with\1"),
    (r"\butil\.(isArray|isDate)\((\w+)\)",
     lambda m: f"Array.isArray({m[2]})" if m[1] == "isArray" else f"({m[2]} instanceof Date)"),
    (r"\bnew Buffer\((\d+)\)", r"Buffer.alloc(\1)"),
    (r"\bnew Buffer\(('[^']*'(?:, '\w+')?)\)", r"Buffer.from(\1)"),
    (r"require\('punycode'\)", "require('punycode/')"),
]
HAZARDS = {  # what Node 22 rejects or warns on; the agent's to-do list after the codemod
    "import_attributes": r"\bassert\s*\{\s*type:", "util_is": r"\butil\.is\w+\(",
    "buffer": r"\bnew Buffer\(", "punycode": r"require\('punycode'\)",
}


def make_fixture(seed=SEED):
    rng = random.Random(seed)
    return {f"repo-{i:02d}-node": [rng.choice(POOL) for _ in range(LINES_PER_FILE)] for i in range(REPOS)}


def codemod(lines):
    out, applied = [], 0
    for line in lines:
        for pattern, repl in RULES:
            line, n = re.subn(pattern, repl, line)
            applied += n
        out.append(line)
    return out, applied


def hazards(lines):
    return {rule: sum(len(re.findall(p, line)) for line in lines) for rule, p in HAZARDS.items()}


def outcomes(ref, recipe_layer, files):
    """The lesson's sandbox wrapping (`migrate`, `agent_loop`, the budget) with one recipe layer swapped in."""
    rng, original = random.Random(SEED), ref.run_recipes
    bench = [dataclasses.replace(r, name=name, lang="node") for r, name in zip(ref.synth_bench(rng), files)]
    ref.run_recipes, migrate = recipe_layer, lambda repo: ref.migrate(repo, rng)
    try:
        return [(a.status, a.failure_class, a.agent_turns, round(a.cost_usd, 4)) for a in map(migrate, bench)]
    finally:
        ref.run_recipes = original


def fixture_stats(files):
    before = [hazards(f) for f in files.values()]
    after = [hazards(codemod(f)[0]) for f in files.values()]
    asserted, residual = [b["import_attributes"] > 0 for b in before], [any(a.values()) for a in after]
    total = lambda rows: {k: sum(r[k] for r in rows) for k in HAZARDS}
    return asserted, residual, {
        "before": total(before), "after": total(after), "sample": codemod(POOL)[0],
        "clean_repos": REPOS - sum(residual), "with_assert": sum(asserted),
        "applied": sum(codemod(f)[1] for f in files.values()),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    files = make_fixture()
    asserted, residual, stats = fixture_stats(files)
    layers = {"codemod": lambda repo: codemod(files[repo.name])[1], "lesson": ref.run_recipes, "none": lambda _: 0}
    runs = {k: outcomes(ref, f, files) for k, f in layers.items()}
    # repos that pass with no agent turn while their source still holds the marked hazard
    straight = lambda run, marks: sum(row[0] == "pass" and row[2] == 0 and m for row, m in zip(run, marks))
    return {
        **stats,
        "passes": {k: sum(row[0] == "pass" for row in v) for k, v in runs.items()},
        "identical": runs["codemod"] == runs["lesson"] == runs["none"],
        "broken_straight": straight(runs["none"], asserted),
        "residual_straight": straight(runs["codemod"], residual),
        "straight": straight(runs["none"], [True] * REPOS),
    }


def verify(r):
    want_before = {"import_attributes": 50, "util_is": 127, "buffer": 169, "punycode": 55}
    want_after = {"import_attributes": 0, "util_is": 0, "buffer": 55, "punycode": 0}
    return [
        practice.Check(
            "ANSWER: the codemod resolves 346 of 401 Node 22 hazards and plugs into migrate unchanged",
            (r["before"], r["after"], r["applied"], r["clean_repos"], r["passes"]["codemod"])
            == (want_before, want_after, 346, 19, 30) and r["sample"][5] == "const raw = new Buffer(payload);",
            f"hazards {r['before']} -> {r['after']}; {r['applied']} rewrites; "
            f"{r['clean_repos']}/50 files clean; {r['passes']['codemod']}/50 pass",
        ),
        practice.Check(
            "FINDING: the recipe layer does not affect the pipeline at all",
            r["identical"] and r["passes"] == {"codemod": 30, "lesson": 30, "none": 30},
            f"passes per layer {r['passes']}; per-repo outcomes identical: {r['identical']}",
        ),
        practice.Check(
            "FINDING: with no recipes, 9 repos pass without an agent turn while their source cannot parse",
            (r["with_assert"], r["straight"], r["broken_straight"], r["residual_straight"]) == (36, 11, 9, 9),
            f"{r['with_assert']} files hold an import assertion; {r['broken_straight']} of {r['straight']} "
            f"straight passes still do; {r['residual_straight']} with the codemod keep a hazard",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
