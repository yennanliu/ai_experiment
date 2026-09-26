"""Exercise 1 — the same SSN gets one placeholder only if it is spelled the same, in the same Scrubber.

    Run `code/main.py`. Send two prompts referencing the same SSN. Confirm both
    get the same placeholder.

Reading of the exercise: "the same SSN" is a person's number, not a byte
string, so the two prompts are sent as the demo sends them (one Scrubber,
identical spelling) and then the way they arrive in production: written
differently, and handled by two gateway replicas, each with its own Scrubber.

**ANSWER: confirmed for the demo's case -- both prompts get [SSN_001].**
Prompts 1 and 2 of `main()` share 123-45-6789 and both scrub to [SSN_001];
so do two fresh prompts through one Scrubber. The audit section scrubs all
three prompts a second time, and each entry's `prompt_hash` equals the hash
of the first pass, so the table held.

**FINDING: the same SSN written any other way is sent raw.** `SSN` matches
only `ddd-dd-dddd`. "123 45 6789", "123456789" and "123.45.6789" -- 3 of 4
spellings of one number -- pass through unmasked. Consistency is keyed on the
matched string, so one phone number in 4 spellings gets 4 placeholders
([PHONE_001]..[PHONE_004]), one email in two letter cases gets 2, and
"(415) 555-0199" and "+1 415-555-0199" keep a stray "(" and "+" because the
pattern's leading `\\b` cannot sit before them (the demo's own third
prompt ships as "phone ([PHONE_002].").

**FINDING: placeholders are counters local to one Scrubber, so two replicas
give two people the same token.** A Scrubber numbers values in arrival
order and keeps the table in memory. Two instances map 123-45-6789 and
987-65-4321 both to [SSN_001]; the scrubbed prompts are identical and so are
their audit-log `prompt_hash` values. The lesson's example token,
`[SSN_TOKEN_A3F]`, reads like a keyed hash, which would be stable across
replicas; the code does not do that.

Structure: every number comes from the reference `Scrubber` and `hash_short`;
`main()` runs with stdout captured.
"""

from __future__ import annotations

import contextlib
import io
import json
import re
import warnings

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "25-security-secrets-audit"
SSN_SPELLINGS = ("123-45-6789", "123 45 6789", "123456789", "123.45.6789")
PHONE_SPELLINGS = ("415-555-0199", "(415) 555-0199", "+1 415-555-0199", "4155550199")
PLACEHOLDER = re.compile(r"\[[A-Z]+_\d{3}\]")


def demo_run(ref):
    """main() with stdout captured: its scrubbed prompts and its audit entries."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out), warnings.catch_warnings():
        warnings.simplefilter("ignore")  # datetime.utcnow() is deprecated on 3.12+
        ref.main()
    lines = out.getvalue().splitlines()
    scrubbed = [ln.split("scrubbed: ", 1)[1] for ln in lines if "scrubbed: " in ln]
    return scrubbed, [json.loads(ln) for ln in lines if ln.startswith("{")]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    scrubbed, audit = demo_run(ref)
    one = ref.Scrubber()
    pair = [one.scrub("My SSN is 123-45-6789."), one.scrub("Please verify 123-45-6789 today.")]
    spell = ref.Scrubber()
    leaked = [s for s in SSN_SPELLINGS if s in spell.scrub(f"SSN {s}")]
    phones = ref.Scrubber()
    phone_out = [phones.scrub(f"call {p}") for p in PHONE_SPELLINGS]
    emails = ref.Scrubber()
    email_tokens = {emails.scrub(e) for e in ("jane.doe@example.com", "Jane.Doe@Example.com")}
    a, b = ref.Scrubber(), ref.Scrubber()
    replicas = [a.scrub("My SSN is 123-45-6789."), b.scrub("My SSN is 987-65-4321.")]
    return {
        "demo": scrubbed, "rescrub": [e["prompt_hash"] == ref.hash_short(s) for e, s in zip(audit, scrubbed)],
        "pair": pair, "leaked": leaked, "phones": phone_out, "emails": sorted(email_tokens),
        "replicas": replicas, "hashes": [ref.hash_short(r) for r in replicas],
    }


def verify(result):
    demo, pair, phones = result["demo"], result["pair"], result["phones"]
    tokens = [PLACEHOLDER.findall(p) for p in phones]
    return [
        practice.Check(
            "ANSWER: confirmed for the demo's case -- both prompts get [SSN_001]",
            all([demo[0].startswith("My SSN is [SSN_001]"), "[SSN_001]" in demo[1],
                 "[SSN_002]" in demo[2], all("[SSN_001]" in p for p in pair),
                 result["rescrub"] == [True] * 3]),
            f"demo prompts 1-2 -> {demo[0][:19]!r} / {demo[1][:25]!r}; fresh pair {pair}; the audit "
            f"section's re-scrub hashes match the first pass {result['rescrub']}",
        ),
        practice.Check(
            "FINDING: the same SSN written any other way is sent raw",
            all([result["leaked"] == list(SSN_SPELLINGS[1:]), len({t[0] for t in tokens}) == 4,
                 len(result["emails"]) == 2, phones[1].startswith("call (["),
                 phones[2].startswith("call +["), "phone ([PHONE_002]." in demo[2]]),
            f"unmasked SSN spellings {result['leaked']}; one phone in 4 spellings -> "
            f"{phones}; one email in two cases -> {result['emails']}",
        ),
        practice.Check(
            "FINDING: placeholders are counters local to one Scrubber, so two replicas "
            "give two people the same token",
            result["replicas"][0] == result["replicas"][1]
            and result["hashes"][0] == result["hashes"][1],
            f"123-45-6789 and 987-65-4321 through two Scrubbers -> {result['replicas']}, "
            f"prompt_hash {result['hashes']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
