"""Exercise 3 — discovery omits it silently and the call says why.

    Change one backend descriptor and prove both discovery and direct call are
    blocked.

Reading of the exercise: "both" invites checking two outcomes and stopping,
but the two blocks are implemented by different code reaching the same pin, so
what they *say* is worth comparing as well as whether they fire. One of them
says nothing, and a client that holds a cached listing meets only the other.

**ANSWER: the tool leaves the listing and the call answers 409.** Editing
`notes.search`'s description takes alice's listing from **4** tools to **3**,
and a direct `tools/call` on the same name answers HTTP **409** with
**-32010 Descriptor changed**. The audit records `pin_mismatch`, a decision
distinct from `deny`.

**FINDING: the two blocks report differently, and only one is an answer.**
`_visible_tools` drops the tool with no error -- a client that re-lists sees
**3** where it saw 4 and is told nothing about the fourth -- while
`tools/call` names the tool and the reason. A client working from a cached
listing gets the useful message; a client that refreshes gets the silent one.

**FINDING: the pin covers the whole descriptor, so prose is load-bearing.**
Only the description changed -- the name, the schema and the required fields
are identical -- and `descriptor_digest` hashes the serialized tool, so the
edit moves the digest. The same edit would be invisible to anything comparing
input schemas.

**FINDING: the deny path and the pin path are told apart in the log and
nowhere else.** The audit distinguishes `deny` from `pin_mismatch`, but from
the client's side a tool alice may not use and a tool whose descriptor moved
both simply fail to appear in her listing. The log can attribute the
disappearance; the protocol cannot.

Structure: `changed` returns a gateway with one descriptor edited, and `probe`
runs a listing and a call against one gateway as one principal.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "17-mcp-gateways-and-registries"
TOOL = "notes.search"


def changed(ref, edit):
    gateway = ref.Gateway()
    edit(gateway.backends["notes"].tools[0])
    return gateway


def probe(ref, gateway, bearer="bearer-alice", tool=TOOL):
    body, headers = ref.make_request("tools/list", 1)
    listed = [item["name"] for item in gateway.handle(bearer, body, headers)[1]
              ["result"]["tools"]]
    body, headers = ref.make_request("tools/call", 2, {"name": tool, "arguments": {}})
    status, response = gateway.handle(bearer, body, headers)
    error = (response or {}).get("error", {})
    return {"listed": listed, "status": status, "code": error.get("code"),
            "message": error.get("message"), "data": error.get("data"),
            "decision": gateway.audit[-1]["decision"] if gateway.audit else None}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    clean = probe(ref, ref.Gateway())
    edited = probe(ref, changed(ref, lambda tool: tool.update(
        description="Search approved notes, quickly.")))

    original = ref.Gateway().backends["notes"].tools[0]
    after = changed(ref, lambda tool: tool.update(
        description="Search approved notes, quickly.")).backends["notes"].tools[0]
    forbidden = probe(ref, ref.Gateway(), bearer="bearer-bob", tool="notes.create")
    return {
        "clean": clean, "edited": edited, "forbidden": forbidden,
        "same_name": original["name"] == after["name"],
        "same_schema": original["inputSchema"] == after["inputSchema"],
        "digest_moved": ref.descriptor_digest(original) != ref.descriptor_digest(after),
        "vanished": sorted(set(clean["listed"]) - set(edited["listed"])),
        "bob_missing": sorted(set(clean["listed"]) - set(forbidden["listed"])),
    }


def verify(result):
    clean, edited, forbidden = result["clean"], result["edited"], result["forbidden"]
    return [
        practice.Check(
            "ANSWER: the tool leaves the listing and the call answers 409",
            all([len(clean["listed"]) == 4, len(edited["listed"]) == 3,
                 result["vanished"] == [TOOL], edited["status"] == 409,
                 edited["code"] == -32010, edited["message"] == "Descriptor changed",
                 edited["decision"] == "pin_mismatch", clean["status"] == 200]),
            f"editing {TOOL}'s description takes alice's listing from "
            f"{len(clean['listed'])} tools to {len(edited['listed'])}, dropping "
            f"{result['vanished']}, and the direct call answers HTTP {edited['status']} "
            f"{edited['code']} {edited['message']!r} with the audit recording "
            f"{edited['decision']!r}",
        ),
        practice.Check(
            "FINDING: the two blocks report differently, and only one is an answer",
            all([edited["data"] == {"tool": TOOL}, result["vanished"] == [TOOL],
                 len(edited["listed"]) < len(clean["listed"])]),
            f"_visible_tools drops the tool with no error -- a client that re-lists sees "
            f"{len(edited['listed'])} where it saw {len(clean['listed'])} and is told nothing "
            f"about the fourth -- while tools/call names it, {edited['data']}. A client "
            "working from a cached listing gets the useful message; one that refreshes gets "
            "the silent one",
        ),
        practice.Check(
            "FINDING: the pin covers the whole descriptor, so prose is load-bearing",
            all([result["same_name"], result["same_schema"], result["digest_moved"]]),
            "only the description changed -- name and inputSchema are identical -- and "
            "descriptor_digest hashes the serialized tool, so the edit moves the digest. The "
            "same edit would be invisible to anything comparing input schemas",
        ),
        practice.Check(
            "FINDING: the deny path and the pin path are told apart in the log and nowhere else",
            all([forbidden["decision"] == "deny", forbidden["status"] == 403,
                 result["bob_missing"] == ["issues.open", "notes.create"],
                 edited["decision"] != forbidden["decision"]]),
            f"the audit separates {forbidden['decision']!r} from {edited['decision']!r}, but "
            f"from the client's side bob's forbidden tools {result['bob_missing']} and a "
            f"pinned-out tool {result['vanished']} both simply fail to appear in the listing. "
            "The log can attribute the disappearance; the protocol cannot",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
