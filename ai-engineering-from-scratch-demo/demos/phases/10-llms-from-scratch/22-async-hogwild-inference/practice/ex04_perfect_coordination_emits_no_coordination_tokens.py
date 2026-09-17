"""Exercise 4 — the simulator's two failure modes are redundancy and a ceiling, and coordinating is free.

    Read the Hogwild! paper's Section 4 (preliminary evaluation). Identify the
    two failure modes the authors report. Describe how a better coordination
    prompt might mitigate each.

Reading of the exercise: the failure modes are looked for in the simulator,
because a prompt that mitigates one has to change a number the simulator
produces, and this module is what the lesson gives to produce numbers with.
`run_hogwild` tags every token `unique` or `redundant` and returns per-category
counts, so both modes are observable in its own output.

**ANSWER: the simulator exhibits two, and only one of them is a coordination
problem.**

    failure mode                weight 0.0          weight 1.0
    redundancy at N=2           146 of 343 work     0 of 385
    redundancy at N=8         1,152 of 1,352      1,111 of 1,511
    unique progress at N=8      200                  400  (cap 400)

At N=2 a better coordination prompt removes every redundant token. At N=8 it
removes almost none, because the second failure mode has taken over: `run_hogwild`
credits at most one token per category per step and there are two categories, so
above two workers the ceiling binds whatever the prompt says.

**FINDING: coordinating costs nothing in this model, which inverts the trade.**
At weight 1.0 the workers emit **0** coordination tokens; at weight 0.0 they emit
**163**. `decide_next_category` reaches its `coord` branch only when the
coordination branch did *not* fire, so the better a worker coordinates the less
it spends saying so. In the paper the coordination prompt is the overhead; here
it is a discount.

**FINDING: noise is the one cost that survives, and it is 5.6%.** The 5% noise
branch fires before everything else, so it is the only token category
independent of the weight: 89 at weight 1.0 and 85 at weight 0.0 out of 1,600.
It is also the only thing keeping `progress_per_step` below its ceiling of 2.

**MECHANISM: `target_per_category` is a parameter of `decide_next_category` and
never appears in its body.** The function's signature promises that workers know
how much of each category is wanted; the implementation sorts the two categories
by current count and takes the smaller. A coordination prompt that said "stop at
N of each" has nowhere to land.

Structure: `modes` measures both failure modes at a given weight; `costs`
separates the coordination and noise token counts.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "10-llms-from-scratch", "22-async-hogwild-inference"
STEPS, TARGET, WORKERS = 200, 100, 8
CATEGORIES = 2


def modes(ref, weight, workers=WORKERS):
    row = ref.run_hogwild(workers, STEPS, TARGET, weight)
    return {
        "redundant": row["work_tokens"] - row["unique_progress"],
        "work": row["work_tokens"],
        "unique": row["unique_progress"],
        "ceiling": CATEGORIES * STEPS,
        "coord": row["coord_tokens"],
        "noise": row["noise_tokens"],
        "tokens": row["tokens_emitted"],
        "per_step": row["progress_per_step"],
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rows = {weight: modes(ref, weight) for weight in (1.0, 0.0)}
    pair = {weight: modes(ref, weight, workers=2) for weight in (1.0, 0.0)}
    source = inspect.getsource(ref.decide_next_category)
    body = source.split('"""')[2]
    return {
        "rows": rows,
        "pair": pair,
        "params": list(inspect.signature(ref.decide_next_category).parameters),
        "target_used": "target_per_category" in body,
        "noise_stable": abs(rows[1.0]["noise"] - rows[0.0]["noise"]) / rows[1.0]["tokens"],
        "noise_share": rows[1.0]["noise"] / rows[1.0]["tokens"],
        "categories": CATEGORIES,
    }


def verify(result):
    rows, pair = result["rows"], result["pair"]
    perfect, none = rows[1.0], rows[0.0]
    two_on, two_off = pair[1.0], pair[0.0]
    return [
        practice.Check(
            "ANSWER: two failure modes, and only one is a coordination problem",
            two_off["redundant"] > 100 and two_on["redundant"] == 0
            and perfect["unique"] == perfect["ceiling"] and perfect["redundant"] > 1000,
            f"at N=2 with no coordination, {two_off['redundant']} of {two_off['work']} work "
            f"tokens are redundant; with perfect coordination, {two_on['redundant']}. That mode a "
            f"better prompt fixes entirely. The second is the ceiling, and at N={WORKERS} it "
            f"binds whatever the prompt says: perfect coordination still leaves "
            f"{perfect['redundant']:,} of {perfect['work']:,} redundant, because unique progress "
            f"is capped at {perfect['ceiling']} -- one token per category per step, and there "
            f"are {result['categories']} categories",
        ),
        practice.Check(
            "FINDING: coordinating costs nothing in this model, which inverts the trade",
            perfect["coord"] == 0 < none["coord"],
            f"at weight 1.0 the workers emit {perfect['coord']} coordination tokens; at weight "
            f"0.0 they emit {none['coord']}. decide_next_category reaches its coord branch only "
            "when the coordination branch did not fire, so the better a worker coordinates the "
            "less it spends saying so. In the paper the coordination prompt is the overhead; "
            "here it is a discount",
        ),
        practice.Check(
            "FINDING: noise is the one cost that survives the weight",
            result["noise_stable"] < 0.01 and 0.04 < result["noise_share"] < 0.07,
            f"the 5% noise branch fires before everything else, so it is the only category "
            f"independent of the weight: {perfect['noise']} tokens at 1.0 and {none['noise']} at "
            f"0.0 out of {perfect['tokens']:,}, a difference of "
            f"{result['noise_stable']:.2%} of the stream. It is also what keeps "
            f"progress_per_step at {two_on['per_step']:.2f} rather than "
            f"{result['categories']}.00 when N=2 -- at N={WORKERS} there are enough workers that "
            "a noise token no longer costs a category",
        ),
        practice.Check(
            "MECHANISM: target_per_category is a parameter that never appears in the body",
            "target_per_category" in result["params"] and not result["target_used"],
            f"decide_next_category's signature is {result['params']}, promising that workers know "
            "how much of each category is wanted, and the implementation sorts the two categories "
            "by current count and takes the smaller. A coordination prompt that said 'stop at N "
            "of each' has nowhere to land, and every caller must pass a number that cannot "
            "change the result",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
