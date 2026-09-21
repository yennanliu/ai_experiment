"""Exercise 4 — the chain forgets the input and pays for the summary.

    Combine prompt chaining with routing: a router picks one of three chains.
    Measure token cost vs a single big-prompt alternative.

Reading of the exercise: cost is measured in prompt characters, because that
is the only unit this code exposes -- `ScriptedLLM` records every prompt it
was given in `calls`, so the bill is `sum(len(p) for p in llm.calls)` and
nothing has to be estimated. The comparison is the same three tasks routed
into three chains against one prompt that does all of it.

**ANSWER: routed chains cost 507 characters across 9 calls; one big prompt
costs 283 across 3.** Chaining is **1.8x** the characters and **3.0x** the
calls for the same three inputs -- and the single prompt is a lower bound
that a real system rarely reaches, because it needs one prompt per task
shape rather than one per step.

**FINDING: the chain pays for its own intermediate output.** Each step
formats the *previous output* into the next prompt, so a step's cost is the
template plus everything the last step emitted. Across the routed run the
intermediate outputs account for **90** of the **507** characters -- **17.8%**
of the bill is text the system wrote itself.

**FINDING: the router is 3 of the 9 calls.** Classification is a model call
in this pattern, so routing adds **3** calls and **193** characters
before any work happens -- **38.1%** of the total. On short inputs the
routing decision can cost more than the step it routes to.

**FINDING: the chain forgets the input after step one.** `prompt_chain` sets
`current = output`, so the original text appears in **3** of the **6** chain
prompts -- once per input, at step one only. Anything step 2 needs from the input has to have survived step 1's
summary, which is a correctness property the cost comparison does not show.

Structure: `chained()` and `single()` run the same three inputs through the
lesson's own `route` and `prompt_chain`, and count what the LLM was handed.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "12-anthropic-workflow-patterns"
INPUTS = ("i want a refund for this charge",
          "the cli crash dumps a stack",
          "do you offer volume pricing")
CHAINS = {
    "refund": (("extract", "extract the order id: {text}"),
               ("reply", "write a refund reply: {text}")),
    "bug": (("extract", "extract the stack frame: {text}"),
            ("reply", "write a triage note: {text}")),
    "sales": (("extract", "extract the account size: {text}"),
              ("reply", "write a pricing reply: {text}")),
}
BIG = ("classify, extract the key field, and write the reply in one pass: {text}")


class Counting:
    """The lesson's ScriptedLLM, with the prompts kept so the bill is countable."""

    def __init__(self):
        self.calls, self.outputs = [], []

    def __call__(self, prompt):
        self.calls.append(prompt)
        output = f"[{len(self.outputs) + 1}] handled {prompt.split(':')[0][:18]}"
        self.outputs.append(output)
        return output


def label_for(text):
    for name in CHAINS:
        if name == "refund" and "refund" in text:
            return name
        if name == "bug" and "crash" in text:
            return name
    return "sales"


def chained(ref):
    llm = Counting()
    router_calls = []
    for text in INPUTS:
        def classifier(value, _llm=llm):
            router_calls.append(_llm(f"classify into refund, bug or sales: {value}"))
            return label_for(value)
        ref.route(text, classifier, {
            name: (lambda t, n=name: ref.prompt_chain(t, llm, list(CHAINS[n])))
            for name in CHAINS})
    return llm, len(router_calls)


def single(ref):
    llm = Counting()
    for text in INPUTS:
        llm(BIG.format(text=text))
    return llm


def bill(llm, router_calls):
    calls = llm.calls
    total = sum(len(prompt) for prompt in calls)
    router = sum(len(p) for p in calls if p.startswith("classify into"))
    echoed = sum(len(out) for out in llm.outputs if any(out in p for p in calls))
    return {
        "chain_cost": total, "chain_calls": len(calls),
        "router_calls": router_calls, "router_cost": router,
        "router_share": round(router / total, 3),
        "echoed": echoed, "echo_share": round(echoed / total, 3),
        "chain_prompts": len(calls) - router_calls,
        "carries_input": sum(any(text in p for text in INPUTS) for p in calls
                             if not p.startswith("classify into")),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    chain_llm, router_calls = chained(ref)
    big_llm = single(ref)
    big_cost = sum(len(prompt) for prompt in big_llm.calls)
    rows = bill(chain_llm, router_calls)
    return {
        **rows, "big_cost": big_cost, "big_calls": len(big_llm.calls),
        "ratio": round(rows["chain_cost"] / big_cost, 1),
        "call_ratio": round(rows["chain_calls"] / len(big_llm.calls), 1),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 507 characters over 9 calls against 283 over 3",
            all([result["chain_cost"] == 507, result["chain_calls"] == 9,
                 result["big_cost"] == 283, result["big_calls"] == 3,
                 result["ratio"] == 1.8, result["call_ratio"] == 3.0]),
            f"the routed chains cost {result['chain_cost']} characters across "
            f"{result['chain_calls']} calls and the single big prompt "
            f"{result['big_cost']} across {result['big_calls']} -- "
            f"{result['ratio']}x the characters and {result['call_ratio']}x the calls "
            "for the same three inputs",
        ),
        practice.Check(
            "FINDING: the chain pays for its own intermediate output",
            all([result["echoed"] == 90, result["echo_share"] == 0.178,
                 result["echoed"] < result["chain_cost"]]),
            f"each step formats the previous output into the next prompt, so "
            f"{result['echoed']} of the {result['chain_cost']} characters -- "
            f"{result['echo_share']:.1%} of the bill -- is text the system wrote itself "
            "and then paid to read back",
        ),
        practice.Check(
            "FINDING: the router is 3 of the 9 calls",
            all([result["router_calls"] == 3, result["router_cost"] == 193,
                 result["router_share"] == 0.381]),
            f"classification is a model call, so routing adds {result['router_calls']} "
            f"calls and {result['router_cost']} characters -- "
            f"{result['router_share']:.1%} of the total -- before any work happens. On "
            "short inputs the decision can cost more than the step it routes to",
        ),
        practice.Check(
            "FINDING: the chain forgets the input after step one",
            all([result["chain_prompts"] == 6, result["carries_input"] == 3,
                 result["carries_input"] * 2 == result["chain_prompts"]]),
            f"prompt_chain sets current = output, so the original text appears in "
            f"{result['carries_input']} of the {result['chain_prompts']} chain prompts. "
            "Anything step 2 needs has to have survived step 1's summary -- a "
            "correctness property the cost comparison does not show",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
