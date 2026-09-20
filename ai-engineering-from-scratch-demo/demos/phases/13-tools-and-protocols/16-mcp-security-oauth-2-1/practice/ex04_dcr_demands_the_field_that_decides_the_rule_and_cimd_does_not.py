"""Exercise 4 — DCR demands the field that decides the rule, and CIMD does not.

    Build a web client variant with a remote HTTPS redirect and compare its
    DCR metadata to the native client.

Reading of the exercise: "compare" is answerable by diffing two documents, and
the diff is small -- two fields. What makes it worth doing is that one of the
two fields is what the server uses to decide which redirect rule applies, so
the comparison leads straight to the question of what happens when that field
is absent. The two enrollment paths answer it differently.

**ANSWER: the two documents differ in exactly two fields.**
`application_type` is `web` against `native`, and `redirect_uris` is
`['https://client.example.com/callback']` against
`['http://127.0.0.1:8765/callback']`. The other **4** -- `client_id`,
`client_name`, `grant_types`, `response_types` -- are identical.

**FINDING: the two fields are not independent, and swapping one is refused.**
Registering the native loopback URI while declaring `web` answers `web
redirect URIs must use remote HTTPS`, because `_validate_application` applies
the HTTPS-and-not-loopback rule only under `application_type == "web"`. The
declaration selects the rule that judges the URI.

**FINDING: DCR requires the deciding field and CIMD does not.**
`dynamic_register` calls `_validate_application(require_application_type=
True)` and refuses a document without it; `enroll_cimd` passes `False` and
accepts the same document. So a CIMD client can register a remote HTTPS
redirect with no `application_type` at all -- and the web rule is then never
applied to it.

**FINDING: the loopback redirect is the thing the native client cannot share.**
`http://127.0.0.1:8765/callback` is accepted for a native client and would be
refused for a web one, and it names a port on the user's own machine -- so the
native document is not portable to a hosted deployment even though its
`client_id` is the same URL. Portability of the identifier, which exercise 2
measured, is not portability of the registration.

Structure: `document` is the client's own metadata, and `register` runs one
document through one enrollment path and reports the outcome either way.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "13-tools-and-protocols", "16-mcp-security-oauth-2-1"


def document(ref, application_type):
    return ref.Client(application_type=application_type)._client_document()


def register(auth, metadata, *, path="dcr"):
    """One document through one enrollment path, reporting either outcome."""
    try:
        if path == "dcr":
            body = {k: v for k, v in metadata.items() if k != "client_id"}
            return auth.dynamic_register(body)
        return auth.enroll_cimd(metadata["client_id"], metadata)
    except ValueError as exc:
        return str(exc)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    auth = ref.AuthorizationServer()
    native, web = document(ref, "native"), document(ref, "web")
    differing = sorted(k for k in set(native) | set(web) if native.get(k) != web.get(k))
    shared = sorted(k for k in native if k not in differing)

    swapped = {**web, "redirect_uris": native["redirect_uris"]}
    untyped = {k: v for k, v in web.items() if k != "application_type"}
    return {
        "differing": differing, "shared": shared,
        "native_type": native["application_type"], "web_type": web["application_type"],
        "native_uris": native["redirect_uris"], "web_uris": web["redirect_uris"],
        "web_dcr": register(auth, web),
        "native_dcr": register(auth, native),
        "swapped": register(auth, swapped),
        "untyped_dcr": register(auth, untyped),
        "untyped_cimd": register(auth, untyped, path="cimd"),
        "loopback_as_web": register(auth, {**native, "application_type": "web"}),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: the two documents differ in exactly two fields",
            all([result["differing"] == ["application_type", "redirect_uris"],
                 result["shared"] == ["client_id", "client_name", "grant_types",
                                      "response_types"],
                 result["native_type"] == "native", result["web_type"] == "web",
                 result["web_uris"] == ["https://client.example.com/callback"],
                 result["native_uris"] == ["http://127.0.0.1:8765/callback"]]),
            f"the documents differ in {result['differing']} -- {result['native_type']!r} "
            f"against {result['web_type']!r} and {result['native_uris']} against "
            f"{result['web_uris']} -- and share {result['shared']}, "
            f"{len(result['shared'])} fields",
        ),
        practice.Check(
            "FINDING: the two fields are not independent, and swapping one is refused",
            all([result["swapped"] == "web redirect URIs must use remote HTTPS",
                 result["loopback_as_web"] == result["swapped"],
                 result["web_dcr"].startswith("dcr_"),
                 result["native_dcr"].startswith("dcr_")]),
            f"declaring web with the native loopback URI answers {result['swapped']!r}, "
            f"while each document registered as itself succeeds "
            f"({result['web_dcr'][:4]}..., {result['native_dcr'][:4]}...). "
            "_validate_application applies the HTTPS-and-not-loopback rule only under "
            "application_type == 'web': the declaration selects the rule that judges the URI",
        ),
        practice.Check(
            "FINDING: DCR requires the deciding field and CIMD does not",
            all([result["untyped_dcr"] == "application_type must be native or web",
                 result["untyped_cimd"] == "https://client.example.com/oauth/metadata.json"]),
            f"the same document without application_type answers "
            f"{result['untyped_dcr']!r} from dynamic_register and enrolls as "
            f"{result['untyped_cimd']} through CIMD, because one passes "
            "require_application_type=True and the other False. A CIMD client can register a "
            "remote HTTPS redirect with no type at all, and the web rule is never applied to "
            "it",
        ),
        practice.Check(
            "FINDING: the loopback redirect is the thing the native client cannot share",
            all([result["native_uris"][0].startswith("http://127.0.0.1"),
                 result["loopback_as_web"] == "web redirect URIs must use remote HTTPS"]),
            f"{result['native_uris'][0]} is accepted for a native client, refused for a web "
            "one, and names a port on the user's own machine. So the native document is not "
            "portable to a hosted deployment even though its client_id is the same URL -- "
            "portability of the identifier is not portability of the registration",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
