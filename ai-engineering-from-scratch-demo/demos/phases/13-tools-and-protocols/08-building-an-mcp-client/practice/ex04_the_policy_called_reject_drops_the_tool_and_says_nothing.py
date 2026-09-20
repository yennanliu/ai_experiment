"""Exercise 4 — the policy called `reject` drops the tool and says nothing.

    Change the collision policy to rejection and make startup fail with both
    peer names in the error.

Reading of the exercise: the policy is switched to `reject` and the result
inspected before anything is written, because the exercise says to *make* it
fail, which implies it does not. It does not -- it `continue`s. So the work is
turning a silent drop into an error, and the error has to name both peers, which
the shipped code could not do even if it raised, because by the time the
collision is noticed the losing peer is the one still in hand.

**ANSWER: `reject` silently keeps the first peer's tool and discards the
second.** Two peers each exporting `notes_list` merge to a registry of **1**
entry owned by `alpha`. No exception, no diagnostic, and `beta`'s tool is
unreachable through `call` -- which returns `Unknown tool` for the only name
that could have addressed it.

**FINDING: which peer wins is alphabetical, not deliberate.** `merge` iterates
`sorted(self.peers)`, so renaming `alpha` to `zulu` hands the bare name to
`beta` instead. The surviving tool is chosen by the peer's label, and the label
is a client-side configuration string the server never sees.

**FINDING: the strict policy names both peers, because the loser is the only
one it has.** The rewritten `merge` raises `tool name collision: notes_list
exported by alpha and beta` -- assembled from the incumbent's `peer_name`, which
the registry already stores, and the current peer. The shipped code has both
values at the same point; it just does not use them.

**FINDING: and the permissive policy is the one that keeps every tool
addressable.** `prefix-on-collision` produces **2** entries, `notes_list` and
`beta/notes_list`, and both resolve through `call`. So the three policies differ
in what they lose: prefixing loses name stability, rejecting loses a tool
silently, and raising loses the whole startup -- which is the only one of the
three that cannot be discovered in production.

Structure: `two_peers` builds a client over two colliding servers,
`strict_merge` is the rewritten policy, and `resolve` asks the client's own
`call` whether a canonical name reaches anything.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "08-building-an-mcp-client"
COLLIDING = "notes_list"


def two_peers(ref, first="alpha", second="beta"):
    client = ref.MultiServerClient()
    for name in (first, second):
        server = ref.ModernFakeServer(name, [ref.tool(COLLIDING, f"List notes on {name}.")])
        client.add_server(name, server)
    client.connect_all()
    client.discover_tools()
    return client


def strict_merge(client):
    """The rewritten policy: raise, naming the incumbent and the newcomer."""
    registry = {}
    for peer_name in sorted(client.peers):
        peer = client.peers[peer_name]
        for tool in peer.tools:
            local_name = tool["name"]
            if local_name in registry:
                incumbent = registry[local_name]
                raise ValueError(f"tool name collision: {local_name} exported by "
                                 f"{incumbent} and {peer.name}")
            registry[local_name] = peer.name
    return registry


def resolve(client, canonical_name):
    """Does this canonical name reach a tool, per the client's own call()?"""
    result = client.call(canonical_name, {})
    return not result.get("isError", False)


def owners(client):
    return {name: merged.peer_name for name, merged in client.registry.items()}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rejecting = two_peers(ref)
    rejecting.merge(policy="reject")
    reject_owners = owners(rejecting)
    reject_reaches_beta = resolve(rejecting, f"beta/{COLLIDING}")

    prefixing = two_peers(ref)
    prefixing.merge(policy="prefix-on-collision")
    prefix_owners = owners(prefixing)

    renamed = two_peers(ref, first="zulu", second="beta")
    renamed.merge(policy="reject")
    renamed_owners = owners(renamed)

    strict = two_peers(ref)
    strict_failure = None
    try:
        strict_merge(strict)
    except ValueError as error:
        strict_failure = str(error)
    return {
        "reject_owners": reject_owners, "reject_entries": len(reject_owners),
        "reject_reaches_beta": reject_reaches_beta,
        "renamed_owners": renamed_owners,
        "prefix_owners": prefix_owners, "prefix_entries": len(prefix_owners),
        "prefix_reaches_both": [resolve(prefixing, name) for name in prefix_owners],
        "strict_failure": strict_failure,
        "names_both": all(name in (strict_failure or "") for name in ("alpha", "beta")),
        "policies": 3,
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: reject silently keeps the first peer's tool and discards the second",
            all([result["reject_owners"] == {COLLIDING: "alpha"},
                 result["reject_entries"] == 1,
                 not result["reject_reaches_beta"]]),
            f"two peers each exporting {COLLIDING} merge to {result['reject_entries']} entry, "
            f"{result['reject_owners']}. No exception and no diagnostic, and beta's tool is "
            f"unreachable -- call('beta/{COLLIDING}') resolves to "
            f"{result['reject_reaches_beta']}",
        ),
        practice.Check(
            "FINDING: which peer wins is alphabetical, not deliberate",
            all([result["renamed_owners"] == {COLLIDING: "beta"},
                 result["reject_owners"] == {COLLIDING: "alpha"}]),
            f"merge iterates sorted(self.peers), so renaming alpha to zulu moves the bare "
            f"name from {result['reject_owners']} to {result['renamed_owners']}. The "
            "surviving tool is chosen by the peer's label, and the label is a client-side "
            "configuration string the server never sees",
        ),
        practice.Check(
            "FINDING: the strict policy names both peers, because the loser is the one it has",
            all([result["strict_failure"] ==
                 f"tool name collision: {COLLIDING} exported by alpha and beta",
                 result["names_both"]]),
            f"the rewritten merge raises {result['strict_failure']!r} -- assembled from the "
            "incumbent's peer_name, which the registry already stores, and the current peer. "
            "The shipped code has both values at the same point; it just does not use them",
        ),
        practice.Check(
            "FINDING: the permissive policy is the one that keeps every tool addressable",
            all([result["prefix_entries"] == 2,
                 sorted(result["prefix_owners"]) == [f"beta/{COLLIDING}", COLLIDING],
                 all(result["prefix_reaches_both"]), result["policies"] == 3]),
            f"prefix-on-collision produces {result['prefix_entries']} entries, "
            f"{sorted(result['prefix_owners'])}, and both resolve. So the "
            f"{result['policies']} policies differ in what they lose: prefixing loses name "
            "stability, rejecting loses a tool silently, and raising loses the whole startup "
            "-- the only one of the three that cannot go unnoticed in production",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
