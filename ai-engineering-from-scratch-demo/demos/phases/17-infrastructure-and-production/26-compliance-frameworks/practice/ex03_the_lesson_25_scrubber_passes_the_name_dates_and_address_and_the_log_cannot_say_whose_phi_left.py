"""Exercise 3 — the Lesson 25 scrubber passes the name, dates and address, and the log cannot say whose PHI left.

    You accidentally sent PHI to a provider without BAA. Walk through the
    incident response.

Reading of the exercise: "you" are the LLM vendor serving a covered entity,
so a HIPAA business associate, and the provider is a model API with no
subcontractor BAA. The walk-through is the Breach Notification Rule (45 CFR
164.400-414) as dated steps from a fixed discovery date. It is then run
against the tools this phase ships, Lesson 25's scrubber and audit log, to
see which steps they can support.

**ANSWER: contain, scope, assess, notify the covered entity within 60 days.**
(1) Cut the route and repoint traffic at a provider under BAA. (2) Ask the
provider for deletion and a written attestation; this is 164.402 factor (iv),
mitigation. (3) Scope the window and the individuals. (4) Run the four-factor
assessment. An impermissible disclosure is presumed a breach unless it shows a
low probability of compromise. (5) Notify the covered entity no later than 60
calendar days after discovery (164.410). From discovery on 2 March 2026 that
is 1 May 2026. The covered entity then has its own 60 days for patients
(164.404). Unless you are its agent, its clock starts at your notice, so the
two can chain to 30 June 2026. HHS is told at the same time if 500 or more
people are affected, and otherwise by 1 March 2027 (164.408). Media notice
applies when more than 500 residents of one state are affected (164.406).

**FINDING: Lesson 25's scrubber in front would not have prevented it.** On a
one-line clinical note it masks 4 values: SSN, phone, email, and the 10-digit
MRN, which it catches only because it matches the phone pattern and labels
[PHONE_001]. The patient's name, date of birth, visit date, street address and
ZIP, and health-plan ID pass through. That is 4 of the note's 8 Safe Harbor
identifier categories left (164.514(b)(2) lists 18). The diagnosis passes too.

**FINDING: the audit log cannot answer factor (i), whose PHI and how much.**
None of its 10 fields names a patient, and `prompt_hash` is taken over the
*scrubbed* prompt. Two patients' refill requests, "Refill metformin for SSN
..., call ... or ...", scrub in fresh sessions to the same text and log the
same hash. The
log bounds the window and the call count. It cannot tell you who was affected,
so the count has to be presumed from call volume.

**FINDING: the lesson's map has no incident-response row.** Its only breach
citation, "HIPAA breach-notification", is filed under change management, and
no row cites 164.402-414.

Structure: `timeline()` is the regulation's clocks as date arithmetic;
`scoping()` runs the Lesson 25 reference on a fixed note and
two refill requests.
"""

from __future__ import annotations

import datetime as dt
import re

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "26-compliance-frameworks"
DISCOVERY = dt.date(2026, 3, 2)
NOTE = (
    "Patient John Smith, MRN 4155550123, DOB 03/14/1961, seen 2026-03-02 at 12 Oak St, "
    "Boston MA 02118 for type 2 diabetes. SSN 123-45-6789, phone 415-555-0199, email "
    "jsmith@example.com. Health plan ID HMO-778812."
)
REFILLS = (
    "Refill metformin for SSN 123-45-6789, call 415-555-0199 or jsmith@example.com.",
    "Refill metformin for SSN 987-65-4321, call 202-555-0150 or ana.ruiz@example.org.",
)
# Safe Harbor categories present in NOTE, with the literal that carries each
IDENTIFIERS = {
    "name": "John Smith",
    "mrn": "4155550123",
    "dates": "03/14/1961",
    "address": "12 Oak St",
    "ssn": "123-45-6789",
    "phone": "415-555-0199",
    "email": "jsmith@example.com",
    "health plan id": "HMO-778812",
}


def timeline(discovery=DISCOVERY, affected=120):
    to_entity = discovery + dt.timedelta(days=60)
    to_patients = to_entity + dt.timedelta(days=60)
    year_end = dt.date(discovery.year, 12, 31) + dt.timedelta(days=60)
    return {
        "covered entity (164.410)": to_entity,
        "individuals, chained (164.404)": to_patients,
        "HHS (164.408)": to_patients if affected >= 500 else year_end,
    }


def scoping(sec):
    scrubbed = sec.Scrubber().scrub(NOTE)
    left = [k for k, v in IDENTIFIERS.items() if v in scrubbed]
    hashes = [sec.hash_short(sec.Scrubber().scrub(r)) for r in REFILLS]
    return scrubbed, left, hashes


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    sec = parity.load_reference(PHASE, "25-security-secrets-audit", "main")
    scrubbed, left, hashes = scoping(sec)
    cites = {c: " ".join(v) for c, v in ref.CONTROL_MAP.items()}
    return {
        "small": timeline(),
        "large": timeline(affected=600),
        "scrubbed": scrubbed,
        "left": left,
        "masks": re.findall(r"\[[A-Z]+_\d{3}\]", scrubbed),
        "hashes": hashes,
        "fields": list(sec.AuditEntry.__dataclass_fields__),
        "breach_rows": [c for c, text in cites.items() if "breach" in text],
        "cites_rule": any(
            re.search(r"164\.4(0[2-9]|1[0-4])", t) for t in cites.values()
        ),
    }


def verify(result):
    small, large = result["small"], result["large"]
    return [
        practice.Check(
            "ANSWER: contain, scope, assess, notify the covered entity within 60 days",
            [*small.values(), large["HHS (164.408)"]]
            == [dt.date(2026, 5, 1), dt.date(2026, 6, 30), dt.date(2027, 3, 1)]
            + [dt.date(2026, 6, 30)],
            f"from discovery {DISCOVERY}: {', '.join(f'{k} {v}' for k, v in small.items())}; "
            f"at 500+ affected HHS moves to {large['HHS (164.408)']}",
        ),
        practice.Check(
            "FINDING: Lesson 25's scrubber in front would not have prevented it",
            (result["left"], result["masks"][0], len(result["masks"]))
            == (["name", "dates", "address", "health plan id"], "[PHONE_001]", 4),
            f"masks {result['masks']} (the MRN as PHONE); left in the prompt: "
            f"{result['left']}: {result['scrubbed']}",
        ),
        practice.Check(
            "FINDING: the audit log cannot answer factor (i), whose PHI and how much",
            len(set(result["hashes"])) == 1 and len(result["fields"]) == 10,
            f"two patients' refill requests log prompt_hash {result['hashes']}; no field of "
            f"{result['fields']} names a patient",
        ),
        practice.Check(
            "FINDING: the lesson's map has no incident-response row",
            result["breach_rows"] == ["change management"] and not result["cites_rule"],
            f"'breach' appears only in {result['breach_rows']}; no row cites 164.402-414",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
