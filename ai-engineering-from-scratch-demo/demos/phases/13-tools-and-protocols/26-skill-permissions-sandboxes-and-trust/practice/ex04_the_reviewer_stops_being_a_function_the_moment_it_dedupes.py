"""Exercise 4 — the reviewer stops being a function the moment it dedupes.

    Extend `ActionRequest` with an idempotency key and require one for
    external writes.

Reading of the exercise: requiring a key for external writes needs a
definition of "external write", and `ActionRequest` does not have one -- it
has `kind`, and every network call is the same kind. So the rule is written
against the only signal present, a non-empty payload, and the mismatch
between that proxy and the real thing is reported rather than hidden. The
key itself then turns the reviewer into something with memory, which is the
larger change.

**ANSWER: a key is required for external writes, and a replayed key is a
duplicate rather than a second write.** The unkeyed publish is denied; the
first keyed one allows; the same key again answers `duplicate` without
allowing; and a different key with the same payload allows, because two
deliberate publishes are two publishes. **4** requests, **3** distinct
verdicts.

**FINDING: the reviewer stops being a function the moment it dedupes.**
`review_action(policy, request)` is pure -- the same arguments give the same
answer forever. A key store makes the second identical call answer
differently, so the component is no longer a policy function but a service
with durability, ordering and eviction questions of its own.

**FINDING: "external write" is not a thing the request can say.**
`ActionRequest` has no method, verb or direction field, so the rule falls
back to `payload != ""`. A fetch that carries a body is then treated as a
write and a delete-by-URL with no body is not -- **2** misclassifications
out of the **4** shapes tested, and neither is detectable from the request.

**FINDING: the key is the caller's word, exactly like `approved`.** A retry
that mints a fresh key gets a second write, so the key protects against a
dropped response and not against a caller that wants two. Idempotency is a
courtesy to honest clients, and the module already showed what
caller-asserted fields are worth.

Structure: `Keyed` adds the field, `Reviewer` holds the store, and the store
is the whole difference between this and `review_action`.
"""

from __future__ import annotations

import pathlib
import tempfile

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "26-skill-permissions-sandboxes-and-trust"
PUBLISH = "https://registry.example.test/packages/report/1.4.0"
BODY = '{"digest": "sha256:abc"}'


def build(ref):
    class Keyed(ref.ActionRequest):
        """The shipped request plus the one field an external write needs."""

        def __init__(self, *args, idempotency_key=None, **kwargs):
            super().__init__(*args, **kwargs)
            object.__setattr__(self, "idempotency_key", idempotency_key)

    class Reviewer:
        """review_action plus a key store -- which is what makes it stateful."""

        def __init__(self, policy):
            self.policy, self.seen = policy, {}

        def review(self, request):
            key = getattr(request, "idempotency_key", None)
            external_write = request.kind == "network" and bool(request.payload)
            if external_write and not key:
                return "deny", "external writes require an idempotency key"
            if key in self.seen:
                return "duplicate", f"key {key} already answered {self.seen[key]!r}"
            verdict = ref.review_action(self.policy, request).verdict.value
            if key:
                self.seen[key] = verdict
            return verdict, "first use of this key"

    return Keyed, Reviewer


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    Keyed, Reviewer = build(ref)
    with tempfile.TemporaryDirectory() as temp:
        workspace = pathlib.Path(temp) / "workspace"
        workspace.mkdir(parents=True)
        policy = ref.SandboxPolicy(
            workspace_root=workspace, allowed_kinds=("network",), approval_kinds=(),
            network_allowlist=("https://registry.example.test",))
        reviewer = Reviewer(policy)
        unkeyed = reviewer.review(Keyed("network", url=PUBLISH, payload=BODY))
        first = reviewer.review(Keyed("network", url=PUBLISH, payload=BODY,
                                      idempotency_key="pub-1"))
        replay = reviewer.review(Keyed("network", url=PUBLISH, payload=BODY,
                                       idempotency_key="pub-1"))
        reminted = reviewer.review(Keyed("network", url=PUBLISH, payload=BODY,
                                         idempotency_key="pub-2"))

        shapes = {
            "publish-with-body": ("network", BODY, True),
            "fetch-with-body": ("network", '{"query": "report"}', False),
            "delete-by-url": ("network", "", True),
            "fetch": ("network", "", False),
        }
        classified = {name: bool(payload) for name, (_, payload, _) in shapes.items()}
        truth = {name: is_write for name, (_, _, is_write) in shapes.items()}
        pure = ref.review_action(policy, ref.ActionRequest("network", url=PUBLISH,
                                                           payload=BODY))
        pure_again = ref.review_action(policy, ref.ActionRequest("network", url=PUBLISH,
                                                                 payload=BODY))
        return {
            "unkeyed": unkeyed, "first": first, "replay": replay, "reminted": reminted,
            "verdicts": [unkeyed[0], first[0], replay[0], reminted[0]],
            "distinct": len({unkeyed[0], first[0], replay[0], reminted[0]}),
            "keys_stored": sorted(reviewer.seen),
            "pure_equal": pure.to_dict() == pure_again.to_dict(),
            "request_fields": list(vars(ref.ActionRequest)["__dataclass_fields__"]),
            "verb_fields": [name for name in vars(ref.ActionRequest)["__dataclass_fields__"]
                            if name in {"method", "verb", "direction", "write"}],
            "classified": classified, "truth": truth,
            "misclassified": sorted(name for name in shapes
                                    if classified[name] != truth[name]),
            "shapes": len(shapes),
        }


def verify(result):
    return [
        practice.Check(
            "ANSWER: a key is required for external writes, and a replay is a duplicate",
            all([result["unkeyed"][0] == "deny", result["first"][0] == "allow",
                 result["replay"][0] == "duplicate", result["reminted"][0] == "allow",
                 result["distinct"] == 3, result["keys_stored"] == ["pub-1", "pub-2"]]),
            f"the four requests answer {result['verdicts']} -- unkeyed denied "
            f"({result['unkeyed'][1]}), first keyed allowed, the same key answering "
            f"{result['replay'][1]!r}, and a second key allowed because two deliberate "
            f"publishes are two publishes. {result['distinct']} distinct verdicts and "
            f"{len(result['keys_stored'])} keys retained",
        ),
        practice.Check(
            "FINDING: the reviewer stops being a function the moment it dedupes",
            all([result["pure_equal"], result["first"][0] != result["replay"][0],
                 result["keys_stored"] != []]),
            "review_action(policy, request) is pure -- the same arguments give the same "
            f"answer forever ({result['pure_equal']}) -- while the keyed reviewer answers "
            f"{result['first'][0]!r} then {result['replay'][0]!r} for identical input. The "
            "component is no longer a policy function but a service, with durability, "
            "ordering and eviction questions of its own",
        ),
        practice.Check(
            "FINDING: external write is not a thing the request can say",
            all([result["verb_fields"] == [], result["shapes"] == 4,
                 result["misclassified"] == ["delete-by-url", "fetch-with-body"]]),
            f"ActionRequest carries {result['request_fields']} with no method, verb or "
            f"direction, so the rule falls back to payload != ''. Of "
            f"{result['shapes']} shapes, {result['misclassified']} are wrong -- a fetch "
            "carrying a body is treated as a write and a delete-by-URL is not -- and "
            "neither error is detectable from the request",
        ),
        practice.Check(
            "FINDING: the key is the caller's word, exactly like approved",
            all([result["reminted"][0] == "allow", result["replay"][0] == "duplicate",
                 "approved" in result["request_fields"]]),
            "a retry that mints a fresh key gets a second write, so the key protects "
            "against a dropped response and not against a caller that wants two. "
            "Idempotency is a courtesy to honest clients, which is the same thing approved "
            "already is -- both are fields the requester fills in",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
