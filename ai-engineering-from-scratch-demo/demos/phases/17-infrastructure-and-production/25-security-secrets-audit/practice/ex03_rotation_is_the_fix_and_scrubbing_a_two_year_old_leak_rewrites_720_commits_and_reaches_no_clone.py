"""Exercise 3 — rotation is the fix, and scrubbing a two-year-old leak rewrites 720 commits and reaches no clone.

    You discover a key in git history (2 years old). What's the correct
    response — rotate the key, scrub history, or both? Justify.

Reading of the exercise: "correct" means the key can no longer be used, at
the least cost. So each response is scored on four things. Is the key still
live? Is it still in the canonical history? Is it still in the copies made
over two years? How many commit ids change? The repository is a toy: 730
daily commits, with a `.env` holding two provider keys added on day 10 and
deleted on day 11. The keys are built at run time so no real-looking literal
sits in this file.

**ANSWER: rotate, always and first; scrub only as hygiene afterwards.**
Only the two responses that rotate leave the key dead. Scrubbing alone
leaves it live in every clone, fork and CI cache made in two years, and it
rewrites 720 of 730 commit ids. Changing day 10 changes every descendant,
so open PRs, signed tags and pinned SHAs all break. GitHub's own guidance
says to revoke or rotate first. After rotation, a scanner still flags the
dead key in history, and an allowlist entry for that one fingerprint clears
it more cheaply than a rewrite. Scrub as well only when the repo is public
or the history is being migrated anyway.

**FINDING: a live two-year-old key means the rotation policy already
failed 8 times.** At the lesson's 90-day ceiling, 730 days holds 8 rotation
deadlines. If the key still authenticates, the leak is the smaller incident,
and the exposure window is the full 730 days.

**FINDING: the lesson's code could not have caught this key or measured its
use.** `Scrubber` masks SSNs, emails and phone numbers: it leaves 0 of 2
key formats masked. So its "output regex scrub for leaked secrets" has no
code behind it. `AuditEntry` has 10 fields and none names the credential a
call used, so the audit log cannot answer "was the leaked key used, and
when?". That answer has to come from the provider's usage logs.

Structure: `history()` builds content-addressed commits the way git chains
parent ids; `scan()` is a two-pattern regex scanner; `respond()` scores one
response.
"""

from __future__ import annotations

import dataclasses
import hashlib
import random
import re

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "25-security-secrets-audit"
DAYS, LEAK_DAY, POLICY_DAYS = 730, 10, 90
PATTERNS = (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), re.compile(r"\bsk-proj-[A-Za-z0-9]{24}\b"))


def keys(seed=0):
    rng = random.Random(seed)
    pick = "ABCDEFGHJKLMNPQRSTUVWXYZ234567"
    aws = "AK" + "IA" + "".join(rng.choice(pick) for _ in range(16))
    oai = "sk-" + "proj-" + "".join(rng.choice(pick + "abcdefghjk") for _ in range(24))
    return aws, oai


def history(leaked):
    """[(commit id, tree)] where each id hashes its parent id and its tree."""
    commits, parent = [], ""
    for day in range(DAYS):
        tree = f"app.py v{day}\n" + (leaked if day == LEAK_DAY else "")
        parent = hashlib.sha1(f"{parent}\n{tree}".encode()).hexdigest()
        commits.append((parent, tree))
    return commits


def scan(commits):
    return [i for i, (_, tree) in enumerate(commits) if any(p.search(tree) for p in PATTERNS)]


def respond(rotate, scrub, env):
    commits = history("" if scrub else env)
    original = history(env)  # every clone, fork and CI cache is a copy of this
    return {
        "live": not rotate, "in_history": bool(scan(commits)), "in_clones": bool(scan(original)),
        "rewritten": sum(a[0] != b[0] for a, b in zip(commits, original)),
    }


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    aws, oai = keys()
    env = f"AWS_ACCESS_KEY_ID={aws}\nOPENAI_API_KEY={oai}\n"
    masked = ref.Scrubber().scrub(env)
    return {
        "responses": {
            (r, s): respond(r, s, env) for r in (False, True) for s in (False, True)
        },
        "flagged": scan(history(env)),
        "masked": sum(k not in masked for k in (aws, oai)),
        "fields": [f.name for f in dataclasses.fields(ref.AuditEntry)],
    }


def verify(result):
    resp = result["responses"]
    dead = [k for k, v in resp.items() if not v["live"]]
    scrub_only = resp[(False, True)]
    credential = [f for f in result["fields"] if "key" in f or "cred" in f]
    return [
        practice.Check(
            "ANSWER: rotate, always and first; scrub only as hygiene afterwards",
            all([dead == [(True, False), (True, True)], scrub_only["live"],
                 scrub_only["in_clones"], scrub_only["rewritten"] == DAYS - LEAK_DAY,
                 resp[(True, False)]["rewritten"] == 0,
                 resp[(True, False)]["in_history"], result["flagged"] == [LEAK_DAY]]),
            f"responses leaving the key dead (rotate, scrub): {dead}; scrub alone leaves "
            f"it live and rewrites {scrub_only['rewritten']} of {DAYS} ids; rotate alone "
            f"rewrites 0 and the scanner still flags day {result['flagged']}",
        ),
        practice.Check(
            "FINDING: a live two-year-old key means the rotation policy already failed 8 times",
            DAYS // POLICY_DAYS == 8,
            f"{DAYS} days / {POLICY_DAYS}-day ceiling = {DAYS // POLICY_DAYS} missed rotations",
        ),
        practice.Check(
            "FINDING: the lesson's code could not have caught this key or measured its use",
            result["masked"] == 0 and len(result["fields"]) == 10 and credential == [],
            f"Scrubber masked {result['masked']} of 2 keys; AuditEntry fields "
            f"{result['fields']} name no credential",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
