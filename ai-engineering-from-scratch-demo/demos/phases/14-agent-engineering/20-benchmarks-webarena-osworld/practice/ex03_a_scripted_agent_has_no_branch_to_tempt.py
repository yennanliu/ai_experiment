"""Exercise 3 — a scripted agent has no branch to tempt.

    Implement a "distractor" tool — one the gold trajectory never uses. Does
    the scripted agent get tempted?

Reading of the exercise: the answer to the question as asked is no, and the
reason is not restraint. `_agent_task_1` through `_agent_task_3` are fixed
call sequences: no conditionals, no tool registry, nothing that reads a tool
result and decides. A distractor can only tempt a policy, so the experiment
is run twice -- against the shipped scripts, and against the smallest agent
that actually chooses.

**ANSWER: the scripted agent is untemptable and a keyword policy is tempted
on 3 of 3 tasks.** A `search` tool -- plausible, never in gold -- is called
**0** times by the shipped agents, which contain **0** `if` statements
between them. The same tool, offered to a policy that picks by description
overlap, is called on all **3** tasks and adds **3** steps.

**FINDING: the distractor is invisible to the metric that decides.** A
tempted run still satisfies **3** of **3** `success` predicates, because they
read `app.orders` and a search leaves no trace there. Efficiency does not
move either, because `search` *replaces* `list_items` rather than adding to
it: both runs sit at **0.92x**, under gold, since the chooser never makes the
mistake gold's trajectory records. Only padding moves the number, to
**1.17x**. Scoring success-rate alone -- the lesson's second pitfall --
cannot see a distractor at all.

**FINDING: `gold_steps` is a count, so it cannot say which actions were
gold.** Comparing the tempted run against a gold *sequence* flags **3** of
**3** tasks as containing a non-gold action, and **2** of those match the
gold step count exactly -- identical under the shipped `int`, different under
the sequence. The trajectory-efficiency gap is defined against a number that
has thrown the trajectory away.

**FINDING: a distractor that is cheap enough is free.** Padding the policy's
run with **1** extra search per task raises efficiency from **0.92x** to
**1.17x** and leaves success at **3/3**; it takes **4** per task to double
the headline, and none of the **12** searches change a verdict. The
distractor tax is paid in a metric nobody gates on.

Structure: `POLICY_TOOLS` is the registry the lesson lacks; `tempted()` runs
the same tasks under a chooser rather than a script.
"""

from __future__ import annotations

import ast
import inspect

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "20-benchmarks-webarena-osworld"
GOLD = {"buy_headphones": ("list_items", "add_to_cart", "checkout"),
        "buy_bundle": ("list_items", "add_to_cart", "add_to_cart", "checkout"),
        "revised_order": ("list_items", "add_to_cart", "remove_from_cart",
                          "add_to_cart", "checkout")}
WANTS = {"buy_headphones": ("sku-001",), "buy_bundle": ("sku-002", "sku-003"),
         "revised_order": ("sku-001", "sku-003")}
DESCRIPTIONS = {"list_items": "list every item in the catalogue",
                "search": "search the store to find a product by name",
                "add_to_cart": "add a product to the cart",
                "checkout": "place the order in the cart"}
PROMPTS = {"buy_headphones": "search and find the headphones product to order",
           "buy_bundle": "search and find the keyboard and mouse to order",
           "revised_order": "search and find the headphones and mouse to order"}


def search(app, query):
    """The distractor: plausible, useful-looking, and never in a gold trajectory."""
    return [row for row in app.list_items() if query in row["name"]]


def choose(prompt, offered):
    """The smallest agent that can be tempted: pick by description overlap."""
    words = set(prompt.lower().split())
    ranked = sorted(((len(words & set(DESCRIPTIONS[name].split())), name)
                     for name in offered), reverse=True)
    return ranked[0][1]


def tempted(ref, tid, offered, pad=0):
    """Run one task under the chooser, recording every tool it calls."""
    app = ref.ShoppingApp()
    first = choose(PROMPTS[tid], offered)
    search(app, "x") if first == "search" else app.list_items()
    for _ in range(pad):
        search(app, "x")
    for sku in WANTS[tid]:
        app.add_to_cart(sku)
    app.checkout()
    return app, [first] + ["search"] * pad + ["add_to_cart"] * len(WANTS[tid]) + [
        "checkout"]


def script_searches(ref, agent):
    return sum("search" in line for line in agent(ref.ShoppingApp()))


def succeeds(app, tid):
    return any(all(order["items"].get(sku) == 1 for sku in WANTS[tid])
               for order in app.orders)


def branches(ref):
    src = "".join(inspect.getsource(getattr(ref, f"_agent_task_{n}")) for n in (1, 2, 3))
    return sum(isinstance(node, ast.If) for node in ast.walk(ast.parse(src)))


def off_gold(tid, calls):
    """Two verdicts on the same run: by sequence, and by the shipped count."""
    wrong = any(call not in GOLD[tid] for call in calls)
    return wrong, wrong and len(calls) == len(GOLD[tid])


def tool_stats(runs):
    return {"searches": sum(calls.count("search") for _, calls in runs.values()),
            "tempted_tasks": sum("search" in calls for _, calls in runs.values()),
            "ok": sum(succeeds(app, tid) for tid, (app, _) in runs.items())}


def summarise(ref, offered, pad=0):
    runs = {tid: tempted(ref, tid, offered, pad) for tid in GOLD}
    flags = [off_gold(tid, calls) for tid, (_, calls) in runs.items()]
    steps = sum(len(calls) for _, calls in runs.values())
    return {**tool_stats(runs), "steps": steps, "off_by_count": sum(c for _, c in flags),
            "off_gold": sum(w for w, _ in flags),
            "efficiency": round(steps / sum(len(g) for g in GOLD.values()), 2)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    agents = (ref._agent_task_1, ref._agent_task_2, ref._agent_task_3)
    offered, honest = tuple(DESCRIPTIONS), ("list_items", "add_to_cart", "checkout")
    base = summarise(ref, honest)
    return {"branches": branches(ref), "baseline": base,
            "script_searches": sum(script_searches(ref, fn) for fn in agents),
            "with_distractor": summarise(ref, offered),
            "padded": summarise(ref, offered, pad=1),
            "pad_to_double": next(n for n in range(1, 40)
                                  if summarise(ref, offered, pad=n)["efficiency"]
                                  >= 2 * base["efficiency"])}


def verify(result):
    plain, tempt, pad = result["baseline"], result["with_distractor"], result["padded"]
    return [
        practice.Check(
            "ANSWER: the scripts are untemptable, the policy is tempted 3 of 3",
            all([result["script_searches"] == 0, result["branches"] == 0,
                 tempt["tempted_tasks"] == 3, tempt["searches"] == 3,
                 tempt["steps"] - plain["steps"] == 0]),
            f"the shipped agents call search {result['script_searches']} times and hold "
            f"{result['branches']} if statements, so 'tempted' is not a state they can "
            f"enter. Offered to a chooser it is picked {tempt['searches']} times, on "
            f"{tempt['tempted_tasks']}/3 tasks",
        ),
        practice.Check(
            "FINDING: the distractor is invisible to the metric that decides",
            all([tempt["ok"] == 3, plain["ok"] == 3, plain["efficiency"] == 0.92,
                 tempt["efficiency"] == plain["efficiency"],
                 pad["efficiency"] == 1.17]),
            f"a tempted run still satisfies {tempt['ok']}/3 success predicates: they read "
            f"app.orders and a search leaves nothing there. Efficiency does not move "
            f"either -- search replaces list_items, so both runs sit at "
            f"{plain['efficiency']}x, under gold. Only padding moves it, to "
            f"{pad['efficiency']}x",
        ),
        practice.Check(
            "FINDING: gold_steps is a count, so it cannot say which actions were gold",
            all([tempt["off_gold"] == 3, tempt["off_by_count"] == 2,
                 plain["off_gold"] == 0]),
            f"against a gold sequence {tempt['off_gold']}/3 tasks contain a non-gold "
            f"action and {tempt['off_by_count']} of those match the gold step *count* "
            "exactly. The shipped int cannot separate them: the efficiency gap is defined "
            "against a number that threw the trajectory away",
        ),
        practice.Check(
            "FINDING: a distractor that is cheap enough is free",
            all([pad["ok"] == 3, pad["efficiency"] == 1.17,
                 result["pad_to_double"] == 4, plain["efficiency"] == 0.92]),
            f"one extra search per task raises efficiency from {plain['efficiency']}x to "
            f"{pad['efficiency']}x with success unchanged at {pad['ok']}/3, and "
            f"{result['pad_to_double']} per task doubles the headline without changing a "
            "verdict. The distractor tax is paid in a metric nobody gates on",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
