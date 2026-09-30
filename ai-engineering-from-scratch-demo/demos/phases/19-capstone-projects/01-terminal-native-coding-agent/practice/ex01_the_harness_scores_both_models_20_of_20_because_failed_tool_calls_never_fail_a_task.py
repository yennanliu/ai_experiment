"""Exercise 1 — the harness scores both models 20/20: failed tool calls never fail a task, and a $0 model never meets the $5 cap.

    Swap the backing model from Claude Sonnet 4.7 to Qwen3-Coder-30B served on vLLM. Compare pass@1 and $-per-task. Report where the open model underperforms.

Reading of the exercise: no GPU and no key here, so this is the D11
scaled-down run of the comparison, not a leaderboard number. The swap point
is the lesson's own seam, the module-level `model_step` that `run_agent`
calls each turn. Two stand-ins go through it on 20 tasks, each "find the
FIX-nn token in task_nn.txt" in a temp worktree. The frontier stand-in
makes clean tool calls and prices its turns the way the shipped script does.
The open stand-in makes the tool-call slips smaller open models are known
for, one kind per 4 tasks: clean, wrong tool name, wrong argument key,
absolute path, and never stopping; self-hosted, it reports $0. pass@1 is
scored twice: the harness's own signal (every plan item `[x]`), and
evidence (a tool output held the token and the run ended before the turn
cap). The real run is `vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct
--enable-auto-tool-choice --tool-call-parser qwen3_xml` behind the same
`model_step`.

**ANSWER: the open model underperforms on tool-call format, and the lesson's
harness cannot see it.** By the harness's signal both score 20/20. By
evidence the frontier stand-in scores 20/20 and the open one 4/20: the
wrong-name, wrong-key and absolute-path calls are caught as errors and the
plan is still marked done, and the 4 looping runs hit the 50-turn cap with
all items `[x]`. $-per-task is $0.03 against $0, but the open arm's looping
runs burn 60,000 tokens each (33.3x a clean run) and the $5 ceiling cannot
stop them.

**FINDING: the shipped demo already "passes" with a failed read.** `main()`
reads `README.md` from `code/`, which has none; the trace shows `ok: False`
and the plan prints 3/3 `[x]`. The model never sees the observation.

**FINDING: dollars are whatever the model says.** `Budget.step` adds the
model's self-reported cost, with no price table and one token total (no
input/output split). The shipped script reports $0.05 for 2,700 tokens,
$18.52/MTok, above even Sonnet 4.6's $15/MTok output price, and "Claude
Sonnet 4.7" is not on Anthropic's price list (read 2026-09-29).
"""

from __future__ import annotations

import os
import tempfile

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "01-terminal-native-coding-agent"
SONNET_46 = {"input": 3.0, "output": 15.0}  # $/MTok, platform.claude.com pricing, read 2026-09-29
SLIPS = ["clean", "wrong_tool", "wrong_key", "absolute_path", "never_stops"]
TASKS = 20


def tool_call(i, slip):
    path = f"task_{i:02d}.txt"
    return {"wrong_tool": ("read", {"path": path}),
            "wrong_key": ("read_file", {"file": path}),
            "absolute_path": ("read_file", {"path": "/" + path})}.get(slip, ("read_file", {"path": path}))


def stand_in(ref, i, slip, cost):
    def model_step(plan, turn):
        done = [ref.TodoItem(1, f"find FIX-{i:02d}", "done")]
        if turn == 0 or slip == "never_stops":
            return {"plan": done, "tool": tool_call(i, slip), "tokens": 1200, "cost": cost[0]}
        return {"plan": done, "tool": None, "tokens": 600, "cost": cost[1]}
    return model_step


def score(i, run, outputs):
    b = run["budget"]
    hit_cap = b["turns_used"] >= b["max_turns"]
    return {"harness_pass": "[ ]" not in run["plan"] and "[x]" in run["plan"],
            "evidence_pass": any(f"FIX-{i:02d}" in o for o in outputs) and not hit_cap,
            "tokens": b["tokens_used"], "dollars": b["dollars_used"], "hit_cap": hit_cap}


def run_arm(ref, sandbox, slip_of, cost):
    outputs, rows = [], []
    real = ref.TOOLS["read_file"]
    ref.TOOLS["read_file"] = lambda sb, **kw: outputs.append(real(sb, **kw)) or outputs[-1]
    try:
        for i in range(TASKS):
            outputs.clear()
            ref.model_step = stand_in(ref, i, slip_of(i), cost)
            rows.append(score(i, ref.run_agent(f"find FIX-{i:02d}", sandbox), outputs))
    finally:
        ref.TOOLS["read_file"] = real
    total = {k: sum(r[k] for r in rows) for k in ("harness_pass", "evidence_pass", "hit_cap", "dollars")}
    return {**total, "per_task_dollars": round(total.pop("dollars") / TASKS, 4),
            "max_tokens": max(r["tokens"] for r in rows)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    code = str(parity.lesson_dir(PHASE, LESSON) / "code")
    shipped = ref.run_agent("demo", code)
    step = ref.model_step
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(TASKS):
            with open(os.path.join(tmp, f"task_{i:02d}.txt"), "w") as fh:
                fh.write(f"the bug is marked FIX-{i:02d}\n")
        try:
            frontier = run_arm(ref, tmp, lambda i: "clean", (0.02, 0.01))
            open_arm = run_arm(ref, tmp, lambda i: SLIPS[i // 4], (0.0, 0.0))
        finally:
            ref.model_step = step
    b = shipped["budget"]
    return {"frontier": frontier, "open": open_arm, "shipped_plan_done": shipped["plan"].count("[x]"),
            "shipped_ok": [e.get("ok") for e in shipped["trace"] if e["event"] == "tool"],
            "shipped_per_mtok": round(b["dollars_used"] / b["tokens_used"] * 1e6, 2),
            "doc_names_47": "Sonnet 4.7" in parity.doc_text(PHASE, LESSON)}


def verify(result):
    f, o = result["frontier"], result["open"]
    return [
        practice.Check(
            "ANSWER: open model loses on tool-call format; harness scores both 20/20",
            (f["harness_pass"], o["harness_pass"], f["evidence_pass"], o["evidence_pass"],
             f["per_task_dollars"], o["per_task_dollars"], o["hit_cap"], o["max_tokens"],
             round(o["max_tokens"] / f["max_tokens"], 1)) == (20, 20, 20, 4, 0.03, 0.0, 4, 60000, 33.3),
            f"harness pass@1 {f['harness_pass']}/20 vs {o['harness_pass']}/20; evidence "
            f"{f['evidence_pass']}/20 vs {o['evidence_pass']}/20; $/task {f['per_task_dollars']} vs "
            f"{o['per_task_dollars']}; {o['hit_cap']} open runs hit the turn cap at {o['max_tokens']} tokens",
        ),
        practice.Check(
            "FINDING: the shipped demo marks 3/3 done while its read_file failed",
            (result["shipped_plan_done"], result["shipped_ok"]) == (3, [True, False]),
            f"plan [x] {result['shipped_plan_done']}/3; tool ok flags {result['shipped_ok']}",
        ),
        practice.Check(
            "FINDING: dollars are the model's self-report, above Sonnet 4.6's output price",
            result["shipped_per_mtok"] == 18.52 and result["shipped_per_mtok"] > SONNET_46["output"]
            and result["doc_names_47"],
            f"shipped script: ${result['shipped_per_mtok']}/MTok vs ${SONNET_46['output']}/MTok output; "
            f"doc names 'Sonnet 4.7': {result['doc_names_47']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
