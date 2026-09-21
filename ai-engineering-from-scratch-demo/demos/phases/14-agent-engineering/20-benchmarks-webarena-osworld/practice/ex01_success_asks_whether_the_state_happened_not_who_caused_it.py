"""Exercise 1 — success asks whether the state happened, not who caused it.

    Extend the toy harness with a second app (a forum). Write 3 tasks plus
    gold trajectories.

Reading of the exercise: `Task` is typed `Callable[[ShoppingApp], ...]` but
nothing enforces the annotation, so a second app is a drop-in. What the port
exposes is that there is nothing to drop it *into*: the run loop lives inside
`main()`, so "extend the harness" means writing the loop again. Both halves
are done here -- a forum with three tasks and gold trajectories, and the
`run()` the lesson does not have.

**ANSWER: a forum, three tasks, 3/3 successful at 1.08x over gold.** Replying
to the pinned thread takes **3** steps against gold **3**; upvoting every
reply in the busiest thread takes **5** against **4**; removing the spam post
takes **5** against **5**. The shipped `Task` dataclass is reused unchanged.

**FINDING: the harness is a print loop, so the port has to rebuild it.** The
lesson's module defines **4** top-level functions and **0** of them runs a
task -- `main` computes success, steps and efficiency inline and prints them.
A second app cannot reuse any of that, which is why the port's `run()` is
**5** lines that already existed as unreachable logic.

**FINDING: a brute-forcer passes every task.** Each `success` predicate scans
the whole app for a state, so an agent that clears every post and then
replies to and upvotes everything satisfies **3** of **3** at **27** steps --
**2.25x** gold against the honest run's **1.08x** -- without ever reading a
task description. Execution-based evaluation scores the world, not the
trajectory, and the step count is the only thing that notices.

**FINDING: the gold number is an input, not a measurement.** `gold_steps` is
an `int` a human typed; nothing in the module derives it or checks it against
a trajectory. Setting all three golds to **1** turns the same **13** steps
into **4.33x** with **0** code changes and **3** still-successful tasks.

Structure: `ForumApp` is the second app; `run()` is the missing harness.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "20-benchmarks-webarena-osworld"
SPAM = "buy cheap watches"


class ForumApp:
    """A second WebArena-style app: threads, replies, votes, moderation."""

    def __init__(self):
        self.threads = {"th-1": {"title": "welcome", "pinned": True},
                        "th-2": {"title": "bug reports", "pinned": False}}
        self.posts = {"p-001": {"tid": "th-2", "text": "repro attached", "votes": 0},
                      "p-002": {"tid": "th-2", "text": SPAM, "votes": 0}}
        self.next_id = 3

    def list_threads(self):
        return [{"tid": tid, **meta} for tid, meta in self.threads.items()]

    def open_thread(self, tid):
        return ([pid for pid, post in self.posts.items() if post["tid"] == tid]
                if tid in self.threads else "error: unknown thread")

    def reply(self, tid, text):
        if tid not in self.threads:
            return "error: unknown thread"
        pid, self.next_id = f"p-{self.next_id:03d}", self.next_id + 1
        self.posts[pid] = {"tid": tid, "text": text, "votes": 0}
        return pid

    def upvote(self, pid):
        if pid not in self.posts:
            return "error: unknown post"
        self.posts[pid]["votes"] += 1
        return f"votes {self.posts[pid]['votes']}"

    def delete_post(self, pid):
        return f"deleted {pid}" if self.posts.pop(pid, None) else "error: unknown post"


def agent_reply(app):
    pinned = [row["tid"] for row in app.list_threads() if row["pinned"]][0]
    return ["list_threads", f"open_thread {app.open_thread(pinned)}",
            f"reply -> {app.reply(pinned, 'thanks')}"]


def agent_upvote(app):
    """Gold is 4; this one re-reads the listing it already had."""
    app.list_threads()
    return ["list_threads", "list_threads (re-read, the agent forgot)"] + [
        f"upvote {pid} -> {app.upvote(pid)}" for pid in app.open_thread("th-2")
    ] + ["open_thread th-2"]


def agent_moderate(app):
    app.list_threads()
    spam = [pid for pid in app.open_thread("th-2") if app.posts[pid]["text"] == SPAM]
    return ["list_threads", "open_thread th-2"] + [
        f"read {pid}" for pid in app.open_thread("th-2")
    ] + [f"delete_post {pid} -> {app.delete_post(pid)}" for pid in spam]


def brute_force(app):
    """Every action the app offers, in a fixed sweep, reading no description."""
    app.list_threads()
    trace = ["list_threads"] + [f"delete_post {pid} -> {app.delete_post(pid)}"
                                for pid in list(app.posts)]
    for tid in list(app.threads):
        app.open_thread(tid)
        trace += [f"open_thread {tid}", f"reply -> {app.reply(tid, 'x')}"]
    return trace + [f"upvote {pid} -> {app.upvote(pid)}" for pid in list(app.posts)]


ROWS = (("reply_to_pinned", "reply to pinned", agent_reply, 3,
         lambda app: any(p["tid"] == "th-1" for p in app.posts.values())),
        ("upvote_thread", "upvote every reply in the busiest thread", agent_upvote, 4,
         lambda app: all(p["votes"] for p in app.posts.values() if p["tid"] == "th-2")),
        ("remove_spam", "remove the spam post", agent_moderate, 5,
         lambda app: all(p["text"] != SPAM for p in app.posts.values())))


def run(task, app):
    """The four lines main() inlines and never exposes."""
    steps = len(task.agent(app))
    return {"ok": task.success(app), "steps": steps, "gold": task.gold_steps}


def sweep(ref, agent=None, golds=None):
    rows = [run(ref.Task(tid, text, agent or fn, golds[i] if golds else gold, ok),
                ForumApp())
            for i, (tid, text, fn, gold, ok) in enumerate(ROWS)]
    steps = sum(r["steps"] for r in rows)
    return {"rows": rows, "ok": sum(r["ok"] for r in rows), "steps": steps,
            "over_gold": round(steps / sum(r["gold"] for r in rows), 2)}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    names = [n for n in vars(ref) if callable(getattr(ref, n)) and n[:2] != "__"]
    return {"honest": sweep(ref), "brute": sweep(ref, brute_force),
            "retimed": sweep(ref, golds=[1, 1, 1]),
            "top_level": sorted(n for n in names if n.startswith(("_agent", "main"))),
            "runner": [n for n in names if "run" in n.lower()],
            "gold_type": type(ROWS[0][3]).__name__}


def verify(result):
    honest, brute, retimed = result["honest"], result["brute"], result["retimed"]
    return [
        practice.Check(
            "ANSWER: a forum, three tasks, 3/3 successful at 1.08x over gold",
            all([honest["ok"] == 3, honest["over_gold"] == 1.08,
                 [row["steps"] for row in honest["rows"]] == [3, 5, 5],
                 [row["gold"] for row in honest["rows"]] == [3, 4, 5]]),
            f"the forum tasks take {[r['steps'] for r in honest['rows']]} steps against "
            f"golds {[r['gold'] for r in honest['rows']]}, succeeding {honest['ok']}/3 at "
            f"{honest['over_gold']}x, reusing Task unchanged (it annotates ShoppingApp)"),
        practice.Check(
            "FINDING: the harness is a print loop, so the port has to rebuild it",
            all([len(result["top_level"]) == 4, result["runner"] == []]),
            f"the module defines {result['top_level']}, {len(result['runner'])} of them "
            "a task runner: main computes success, steps and efficiency inline, prints "
            "them, and a second app reuses none of it",
        ),
        practice.Check(
            "FINDING: a brute-forcer passes every task",
            all([brute["ok"] == 3, brute["steps"] == 27, brute["over_gold"] == 2.25,
                 brute["over_gold"] > honest["over_gold"]]),
            f"an agent that clears every post then replies to and upvotes everything, "
            f"reading no description, satisfies {brute['ok']}/3 predicates in "
            f"{brute['steps']} steps -- {brute['over_gold']}x against "
            f"{honest['over_gold']}x. It scores the world, not the trajectory",
        ),
        practice.Check(
            "FINDING: the gold number is an input, not a measurement",
            all([result["gold_type"] == "int", retimed["over_gold"] == 4.33,
                 retimed["ok"] == 3, retimed["steps"] == honest["steps"]]),
            f"gold_steps is an {result['gold_type']} nothing derives or checks: setting "
            f"all three golds to 1 turns the same {honest['steps']} steps into "
            f"{retimed['over_gold']}x, {retimed['ok']} tasks still successful"),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
