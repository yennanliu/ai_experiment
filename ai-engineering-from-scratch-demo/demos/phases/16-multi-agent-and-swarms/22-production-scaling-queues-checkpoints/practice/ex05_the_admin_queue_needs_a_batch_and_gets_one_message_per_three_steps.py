"""Exercise 5 — the admin queue needs a batch and gets one message per three steps.

    Read MegaAgent (arXiv:2408.09955) Section 3. The two-layer coordination
    (intra-group + inter-group admin chat) is explicit. Sketch how you would
    map this to a message queue with two queue families.

Reading of the exercise: the sketch is built rather than drawn -- a `group.*`
family with one reference `AgentQueue` per agent, an `admin.*` family with one
per group admin, and a router that consumes each `out_queue` -- and sized at
the paper's largest run, 590 agents, as 59 groups of 10. The mechanism the
exercise points at is in the paper's §2.2 (§3 is Experiments).

**ANSWER: two families, one routing rule.** `group.<g>.<agent>` carries
intra-group chat; `admin.<g>` carries inter-group chat and is the only way
out of a group, since in MegaAgent ordinary agents cannot talk across groups.
A cross-group message takes 3 hops -- sender's admin, receiver's admin,
receiver -- where an intra-group one takes 1. One all-hands round costs
8,732 deliveries (5,310 intra-group plus 3,422 between admins) against
347,510 for a flat mesh: 40x fewer, bought by admins summarising.

**FINDING: the reference queue drains one message per three steps.**
`AgentQueue.step` pops one message per Processing state and spends a step in
each of the three states, so an admin holding one message from each of the
other 58 groups needs 174 steps to empty its inbox. MegaAgent's Processing
state takes "the message batch"; the same queue with a batch Processing
state needs 3. The admin family is exactly where the batch matters, because
it is where fan-in concentrates.

**FINDING: Response does nothing and nobody reads the out_queue.** The reply
is appended during Processing; Response is one assignment back to Idle, where
the paper has it verify outputs "before being dispatched through designated
function call". And no code consumes `out_queue` -- after the lesson's demo it
still holds both replies. In the mapping the router is that consumer, which
makes it the one component the reference leaves out.

Structure: `Mesh` builds both families from the reference `AgentQueue` and
counts deliveries; `BatchQueue` overrides only the Processing step.
"""

from __future__ import annotations

import inspect

from harness import parity, practice

PHASE, LESSON = "16-multi-agent-and-swarms", "22-production-scaling-queues-checkpoints"
GROUPS, SIZE = 59, 10


class Mesh:
    def __init__(self, ref, groups=GROUPS, size=SIZE):
        self.queues = {f"group.{g}.{a}": ref.AgentQueue(f"{g}.{a}")
                       for g in range(groups) for a in range(size)}
        self.queues.update({f"admin.{g}": ref.AgentQueue(f"admin-{g}") for g in range(groups)})
        self.delivered = 0

    def route(self, src_group, dst_group, dst_agent):
        """The queue names one message passes through."""
        hops = [f"admin.{src_group}", f"admin.{dst_group}"] if src_group != dst_group else []
        return hops + [f"group.{dst_group}.{dst_agent}"]

    def send(self, names, msg):
        for name in names:
            self.queues[name].enqueue(msg)
            self.delivered += 1

    def all_hands(self, groups=GROUPS, size=SIZE):
        for g in range(groups):
            for a in range(size):
                self.send([f"group.{g}.{b}" for b in range(size) if b != a], {"from": a})
            self.send([f"admin.{h}" for h in range(groups) if h != g], {"summary": g})
        return self.delivered


def drain_steps(queue):
    steps = 0
    while queue.in_queue or queue.state.value != "idle":
        queue.step()
        steps += 1
    return steps


def batch_class(ref):
    class BatchQueue(ref.AgentQueue):
        def step(self):
            if self.state == ref.AgentState.PROCESSING:
                batch, self.in_queue = self.in_queue, []
                self.out_queue.append({"reply_to": batch, "body": f"{self.agent_id} batch"})
                self.state = ref.AgentState.RESPONSE
            else:
                super().step()
    return BatchQueue


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    mesh = Mesh(ref)
    single, batch = ref.AgentQueue("admin-0"), batch_class(ref)("admin-0")
    for queue in (single, batch):
        for g in range(1, GROUPS):
            queue.enqueue({"summary": g})
    step_src = inspect.getsource(ref.AgentQueue.step)
    demo = ref.AgentQueue("agent-a")
    for task in ("compress logs", "write summary"):  # the lesson's demo_queue
        demo.enqueue({"task": task})
    for _ in range(7):
        demo.step()
    agents = GROUPS * SIZE
    return {
        "agents": agents, "two_layer": mesh.all_hands(), "flat": agents * (agents - 1),
        "hops": (len(mesh.route(0, 0, 3)), len(mesh.route(0, 5, 3))),
        "single_steps": drain_steps(single), "batch_steps": drain_steps(batch),
        "response_body": step_src.split("AgentState.RESPONSE:")[1].strip().splitlines(),
        "undelivered": len(demo.out_queue),
        "consumers": inspect.getsource(ref).count("out_queue.pop"),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: two families, one routing rule",
            all([result["agents"] == 590, result["two_layer"] == 8732,
                 result["flat"] == 347510, result["hops"] == (1, 3)]),
            f"{result['agents']} agents: an all-hands round costs {result['two_layer']:,} "
            f"deliveries against {result['flat']:,} flat "
            f"({result['flat'] / result['two_layer']:.0f}x); hops intra/cross = {result['hops']}",
        ),
        practice.Check(
            "FINDING: the reference queue drains one message per three steps",
            result["single_steps"] == 174 and result["batch_steps"] == 3,
            f"an admin with 58 inbound summaries needs {result['single_steps']} steps on the "
            f"reference AgentQueue and {result['batch_steps']} with a batch Processing state",
        ),
        practice.Check(
            "FINDING: Response does nothing and nobody reads the out_queue",
            all([result["response_body"] == ["self.state = AgentState.IDLE"],
                 result["undelivered"] == 2, result["consumers"] == 0]),
            f"the RESPONSE branch is {result['response_body']}; after the demo "
            f"{result['undelivered']} replies sit in out_queue and the module pops from "
            f"it {result['consumers']} times",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
