"""Exercise 4 — the 335-day warm tier is the bill, and one object per entry would cost the cold tier 800x.

    Your audit log grows 10 GB/day. Design retention tiers (hot 30d, warm
    12mo, cold 6yr).

Reading of the exercise: the tier boundaries are ages since the entry was
written -- hot days 0-30, warm to day 365, cold to day 2190 -- because
retention clocks run from creation. The 10 GB/day is stored bytes of the
lesson's own `audit_log_call` lines, so the entry size is measured on 1000
seeded `AuditEntry` records. Prices are the S3 us-east-1 list prices in the
AWS price-list API (publication date 2026-09-18): Standard $0.023,
Standard-IA $0.0125, Glacier Instant Retrieval $0.004, Deep Archive $0.00099
per GB-month. Hot is priced as S3 Standard, which is the floor: a searchable
index costs more.

**ANSWER: at steady state 21,900 GB -- hot 300, warm 3,350, cold 18,250 --
for $66.84 a month.** Hot is S3 Standard, plus the index the on-call queries.
Warm is Standard-IA and cold is Deep Archive, both under Object Lock in
compliance mode, and a lifecycle rule moves each daily object at day 30 and
day 365 and expires it at day 2190. Cold holds 83% of the bytes and costs
27% of the bill. The warm tier is 63% of it. Moving warm to Glacier Instant
Retrieval (90-day minimum, millisecond reads) takes the total to $38.37.

**FINDING: one ~280-byte line per call, archived one object per entry,
costs the cold tier 800x.** The lesson writes one line per call, so 10 GB/day
is 35.7M calls/day.
S3 lifecycle does not transition objects under 128 KB by default. Deep
Archive also adds 40 KB of billed metadata per object: 8 KB at the Standard
rate and 32 KB at the archive rate. Per 280-byte entry that is 797x the
archive price of the bytes themselves; left in Standard because they are
under 128 KB, they cost 23x. So the lines must be batched into large daily
objects before they tier.

**FINDING: a 12-hex `prompt_hash` collides twice a day at this volume.**
`hash_short` keeps 48 bits. 35.7M entries a day gives 2.3 expected colliding
pairs per day, and about 10.8M over the six-year retention, so the hash
cannot prove which prompt an entry refers to.

**FINDING: the "immutable" log is JSON printed to stdout, and an edit leaves
no trace.** Nothing is appended or protected. Change one entry's `cost_usd`
and the plain lines give no signal. Chain each line to its predecessor's
SHA-256 and verification stops at the edited entry, 500, in every tier the
lines move through.

Structure: `entries()` builds records with the reference dataclass and
serializer; the tier arithmetic is closed-form.
"""

from __future__ import annotations

import hashlib
import json
import random

from harness import parity, practice

PHASE, LESSON = "17-infrastructure-and-production", "25-security-secrets-audit"
GB_DAY, GB = 10, 10**9
TIERS = (("hot", 0, 30, 0.023), ("warm", 30, 365, 0.0125), ("cold", 365, 2190, 0.00099))
GIR, STANDARD, KB = 0.004, 0.023, 1024


def entries(ref, n=1000, seed=0):
    rng = random.Random(seed)
    lines = []
    for i in range(n):
        entry = ref.AuditEntry(
            f"2026-09-26T{i // 3600:02}:{i // 60 % 60:02}:{i % 60:02}.{rng.randrange(10**6):06}Z",
            f"user_{rng.randrange(10**4):04}", f"tenant_{rng.randrange(100):02}",
            "anthropic/claude-3.7-sonnet", ref.hash_short(f"p{i}"), ref.hash_short(f"r{i}"),
            rng.randrange(20, 4000), rng.randrange(10, 1500), round(rng.uniform(1e-4, 0.05), 4),
            [] if rng.random() < 0.95 else ["pii_masked"],
        )
        lines.append(ref.audit_log_call(entry))
    return lines


def chain(lines):
    out, prev = [], ""
    for line in lines:
        prev = hashlib.sha256((prev + line).encode()).hexdigest()
        out.append(prev)
    return out


def first_break(lines, stored):
    return next((i for i, (a, b) in enumerate(zip(chain(lines), stored)) if a != b), None)


def tiers():
    gb = {name: GB_DAY * (end - start) for name, start, end, _ in TIERS}
    cost = {name: round(gb[name] * price, 4) for name, *_, price in TIERS}
    return gb, cost


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    lines = entries(ref)
    size = sum(len(line) + 1 for line in lines) / len(lines)
    per_day = GB_DAY * GB / size
    gb, cost = tiers()
    per_object = (size + 32 * KB) * 0.00099 + 8 * KB * STANDARD  # one entry, one object
    tampered = list(lines)
    record = json.loads(tampered[500])
    record["cost_usd"] = 0.0
    tampered[500] = json.dumps(record)
    return {
        "size": size, "per_day": per_day, "gb": gb, "cost": cost,
        "gir_total": round(sum(cost.values()) - cost["warm"] + gb["warm"] * GIR, 2),
        "object_x": per_object / (size * 0.00099),
        "daily_pairs": per_day**2 / 2 / 2**48,
        "six_year_pairs": (per_day * 2190) ** 2 / 2 / 2**48,
        "break": first_break(tampered, chain(lines)), "plain_fields": len(record),
    }


def verify(result):
    gb, cost = result["gb"], result["cost"]
    total = round(sum(cost.values()), 2)
    return [
        practice.Check(
            "ANSWER: at steady state 21,900 GB -- hot 300, warm 3,350, cold 18,250 -- "
            "for $66.84 a month",
            all([gb == {"hot": 300, "warm": 3350, "cold": 18250}, total == 66.84,
                 round(cost["warm"] / total, 2) == 0.63, round(cost["cold"] / total, 2) == 0.27,
                 result["gir_total"] == 38.37]),
            f"GB {gb}, $/month {cost} = {total}; warm on Glacier IR -> {result['gir_total']}",
        ),
        practice.Check(
            "FINDING: one ~280-byte line per call, archived one object per entry, "
            "costs the cold tier 800x",
            round(result["size"]) == 280 and round(result["per_day"] / 1e6, 1) == 35.7
            and 790 < result["object_x"] < 805 and round(STANDARD / 0.00099) == 23,
            f"{result['size']:.1f} B/entry, {result['per_day'] / 1e6:.1f}M entries/day; "
            f"per-object overhead multiplies cold cost {result['object_x']:.0f}x",
        ),
        practice.Check(
            "FINDING: a 12-hex prompt_hash collides twice a day at this volume",
            round(result["daily_pairs"], 1) == 2.3 and round(result["six_year_pairs"] / 1e6, 1) == 10.8,
            f"{result['daily_pairs']:.2f} colliding pairs/day, "
            f"{result['six_year_pairs'] / 1e6:.1f}M over 2190 days (48-bit hash)",
        ),
        practice.Check(
            "FINDING: the 'immutable' log is JSON printed to stdout, and an edit leaves no trace",
            result["break"] == 500 and result["plain_fields"] == 10,
            f"the edited 10-field line is valid JSON with no integrity field; the hash "
            f"chain breaks at entry {result['break']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
