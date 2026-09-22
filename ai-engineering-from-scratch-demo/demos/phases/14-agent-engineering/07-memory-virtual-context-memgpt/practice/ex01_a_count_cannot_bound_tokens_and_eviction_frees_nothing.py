"""Exercise 1 — a count cannot bound tokens, and eviction frees nothing.

    Add a `max_main_context_tokens` cap measured in tokens (approximate with
    `len(text.split())` * 1.3). Compact the oldest messages into a summary
    when the cap is exceeded. Compare behavior with and without the
    summarizer.

Reading of the exercise: `MainContext` already evicts, but on
`max_messages`, which is a count. The exercise replaces the unit, so the
first thing to measure is how far apart the two units are on the same buffer.
The comparison is then run twice over the same eight messages -- cap only,
and cap plus a summarizer that folds each evicted message into a `summary`
core section -- with the rendered prompt as the thing being bounded.

**ANSWER: the same `max_messages=4` holds a prompt of **13.0** tokens or
**79.3**.** A 6.1x spread on the identical setting, because the count does
not know how long a message is. With the token cap at **40**, both arms keep
the render at or under it: **3** messages survive without the summarizer and
**1** with it, the difference being the tokens the summary itself occupies --
compaction buys recall of old turns by spending live context.

**FINDING: the summary is what survives the eviction.** Without it, the
earliest message is gone from the render and `ava` appears **0** times in the
prompt the model sees. With it, the summary section carries **4** gists and
`ava` is still there, at **0.26** tokens kept per token evicted. The rolling
summary has its own budget, so **3** further evictions are dropped
unrecorded: compaction is lossy twice over.

**FINDING: eviction frees nothing.** `MainContext.append` moves the oldest
message into `self.evicted`, which is an unbounded list on the same object,
and `conversation_search` reads `evicted + messages`. After **8** appends the
"fixed-size" tier holds **3** rendered messages and **5** evicted ones, and
the evicted text is still retrievable in full. This tier is a *render* bound,
not a memory bound -- the paging in MemGPT's analogy has not happened yet.

**FINDING: the cap binds the render, and the render is not the prompt.**
`render()` emits the core sections and the messages and nothing else -- no
tool definitions, no system preamble, no retrieved observations. Every
archival hit the agent pages in lands outside the number this cap controls.

Structure: `Bounded` drives the lesson's own `MainContext` with the count cap
disabled, so the only policy under test is the one the exercise adds.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "14-agent-engineering", "07-memory-virtual-context-memgpt"
CAP, GIST_WORDS, MAX_GISTS = 40.0, 4, 4
SHORT = (("user", "hi"), ("assistant", "hello"), ("user", "ok"), ("assistant", "sure"))
LONG = (
    ("user", "my name is ava and I ship agents for a living at a mid sized company"),
    ("assistant", "noted, what are you building right now and which tools are involved"),
    ("user", "a retrieval bot for our sales org with twelve tools registered so far"),
    ("assistant", "twelve tools is in the long horizon band so plan for drift and checkpoints"),
)
MESSAGES = (
    ("user", "my name is ava and I ship agents for a living"),
    ("assistant", "noted. what are you building right now?"),
    ("user", "a retrieval bot for our sales org, twelve tools so far"),
    ("assistant", "twelve tools is in the long-horizon band; plan for drift"),
    ("user", "we also added a summarizer last week"),
    ("assistant", "good. does it run on eviction or on a timer?"),
    ("user", "on eviction, same as memgpt"),
    ("assistant", "then watch the compression ratio"),
)


def tokens(text):
    """The exercise's own approximation."""
    return round(len(text.split()) * 1.3, 1)


class Bounded:
    """`MainContext` with the count cap disabled and a token cap in its place."""

    def __init__(self, ref, cap, summarize):
        self.ctx = ref.MainContext(max_messages=10 ** 6)
        self.cap, self.summarize = cap, summarize
        self.folded, self.dropped = 0, 0

    def append(self, role, text):
        self.ctx.append(role, text)
        while tokens(self.ctx.render()) > self.cap and len(self.ctx.messages) > 1:
            oldest = self.ctx.messages.pop(0)
            self.ctx.evicted.append(oldest)
            if self.summarize:
                self.fold(oldest)

    def fold(self, message):
        """A rolling summary has a budget too: the first MAX_GISTS evictions fit."""
        if self.folded >= MAX_GISTS:
            self.dropped += 1
            return
        gist = " ".join(message.text.split()[:GIST_WORDS])
        prior = self.ctx.core.get("summary", "")
        self.ctx.core["summary"] = f"{prior}; {gist}".strip("; ")
        self.folded += 1


def fill(ref, cap, summarize):
    buffer = Bounded(ref, cap, summarize)
    for role, text in MESSAGES:
        buffer.append(role, text)
    return buffer


def shipped(ref, rows):
    context = ref.MainContext(max_messages=4)
    for role, text in rows:
        context.append(role, text)
    return tokens(context.render())


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    plain, summed = fill(ref, CAP, False), fill(ref, CAP, True)
    evicted_tokens = sum(tokens(m.text) for m in summed.ctx.evicted)
    tools = ref.MemoryTools(plain.ctx, ref.ArchivalStore())
    return {
        "short": shipped(ref, SHORT), "long": shipped(ref, LONG),
        "plain_tokens": tokens(plain.ctx.render()),
        "summed_tokens": tokens(summed.ctx.render()),
        "plain_messages": len(plain.ctx.messages),
        "summed_messages": len(summed.ctx.messages),
        "plain_has_ava": plain.ctx.render().count("ava"),
        "summed_has_ava": summed.ctx.render().count("ava"),
        "folded": summed.folded,
        "ratio": round(tokens(summed.ctx.core.get("summary", "")) / evicted_tokens, 2),
        "evicted": len(plain.ctx.evicted), "appended": len(MESSAGES),
        "still_searchable": tools.conversation_search("ship agents"),
        "dropped": summed.dropped,
        "render_sections": [line for line in plain.ctx.render().splitlines()
                            if line.startswith("[")],
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: max_messages=4 holds 13.0 tokens or 79.3, and the cap holds both",
            all([result["short"] == 13.0, result["long"] == 79.3,
                 result["plain_tokens"] <= CAP, result["summed_tokens"] <= CAP,
                 result["plain_messages"] == 3, result["summed_messages"] == 1]),
            f"the same max_messages=4 renders {result['short']} tokens of short turns "
            f"and {result['long']} of long ones -- a "
            f"{result['long'] / result['short']:.1f}x spread on one setting. Under a "
            f"{CAP}-token cap both arms stay inside it, keeping "
            f"{result['plain_messages']} and {result['summed_messages']} messages",
        ),
        practice.Check(
            "FINDING: the summary is what survives the eviction",
            all([result["plain_has_ava"] == 0, result["summed_has_ava"] == 1,
                 result["folded"] == 4, result["dropped"] == 3,
                 result["ratio"] == 0.26]),
            f"without the summarizer 'ava' appears {result['plain_has_ava']} times in the "
            f"rendered prompt; with it, {result['folded']} gists are folded into the "
            f"summary and it appears {result['summed_has_ava']} time, at "
            f"{result['ratio']} tokens kept per token evicted. The rolling summary has a "
            f"budget too, so {result['dropped']} further evictions go unrecorded",
        ),
        practice.Check(
            "FINDING: eviction frees nothing",
            all([result["evicted"] == 5, result["appended"] == 8,
                 result["still_searchable"].startswith("found (user):"),
                 "ship agents" in result["still_searchable"]]),
            f"after {result['appended']} appends the tier holds "
            f"{result['plain_messages']} rendered messages and {result['evicted']} "
            f"evicted ones on the same object, and conversation_search still returns "
            f"{result['still_searchable']!r}. The bound is on the render, not on memory",
        ),
        practice.Check(
            "FINDING: the render is not the prompt",
            all([result["render_sections"] == ["[core]", "[messages]"],
                 len(result["render_sections"]) == 2]),
            f"render() emits {result['render_sections']} and nothing else -- no tool "
            "definitions, no system preamble, no retrieved observations. Every archival "
            "hit the agent pages in lands outside the number this cap controls, which is "
            "the half of the budget the exercise's approximation cannot see",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
