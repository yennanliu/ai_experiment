"""Exercise 5 — the CVSS calculator reproduces the doc's 9.3, and the MFA demonstration is a 2.1-point swing.

    Responsible disclosure for AI vulnerabilities is evolving. Sketch a
    disclosure protocol that includes AI-specific evidence (reproducibility,
    model-version scoping, prompt-injection resistance).

Reading of the exercise: a disclosure protocol for an AI CVE has to turn
AI-specific evidence into the score everyone cites, so this builds a CVSS 3.1
base-score calculator (the standard formula) and runs the protocol's evidence
requirement through it. The score is computed from the metric vector; the vector
is a sourced constant (NVD, read 2026-09-27), not a hand-typed score, and the
result is checked against the number the lesson doc prints. The protocol's point
-- "demand a demonstrated proof-of-concept" -- is then measured as the score
gap the MFA demonstration is worth.

**ANSWER: a disclosure protocol scores the CVE only from demonstrated impact.**
The calculator reproduces the doc's 9.3 exactly from EchoLeak's vector
`AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:L/A:N`. Rated as "information disclosure only"
(confidentiality Low, not High) the same bug scores 7.2. So the protocol's
AI-specific evidence -- the reproducible MFA-code exfiltration Aim Labs
demonstrated -- is worth a 2.1-point swing, from High to Critical, and is what
justifies the 9.3 rather than the initial lower rating.

**FINDING: the confidentiality metric alone moves it across severity bands.** Holding the
rest of the vector, C:H gives 9.3 (Critical), C:L 7.2 and C:N 5.8. The gap the doc calls a
severity-calibration failure is exactly the C:L -> C:H step, which is what a
demonstrated exploit establishes.

**FINDING: the calculator round-trips its own vector.** Parsing the vector
string and re-emitting it returns the sourced string unchanged, so the 9.3 is
computed from the published metrics and not asserted.

Structure: `score()` is the CVSS 3.1 base formula; `SOURCED` holds the vector
read from NVD; the checks compare the computed score with the doc's printed one.
"""

from __future__ import annotations

import math
import re

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "25-echoleak-cves-for-ai"
# CVE-2025-32711 base vector, from NVD (nvd.nist.gov), read 2026-09-27.
SOURCED = "AV:N/AC:L/PR:N/UI:N/S:C/C:H/I:L/A:N"
AV = {"N": 0.85, "A": 0.62, "L": 0.55, "P": 0.2}
AC = {"L": 0.77, "H": 0.44}
UI = {"N": 0.85, "R": 0.62}
CIA = {"H": 0.56, "L": 0.22, "N": 0.0}
PR = {"U": {"N": 0.85, "L": 0.62, "H": 0.27}, "C": {"N": 0.85, "L": 0.68, "H": 0.5}}


def roundup(value):
    """CVSS 3.1 Appendix A roundup: ceiling to one decimal, integer-exact."""
    scaled = round(value * 100000)
    return scaled / 100000 if scaled % 10000 == 0 else (math.floor(scaled / 10000) + 1) / 10


def score(vector):
    """CVSS 3.1 base score from a metric vector string."""
    m = dict(part.split(":") for part in vector.split("/"))
    changed = m["S"] == "C"
    iss = 1 - (1 - CIA[m["C"]]) * (1 - CIA[m["I"]]) * (1 - CIA[m["A"]])
    impact = (7.52 * (iss - 0.029) - 3.25 * (iss - 0.02) ** 15) if changed else 6.42 * iss
    expl = 8.22 * AV[m["AV"]] * AC[m["AC"]] * PR["C" if changed else "U"][m["PR"]] * UI[m["UI"]]
    if impact <= 0:
        return 0.0
    return roundup(min((1.08 if changed else 1.0) * (impact + expl), 10))


def with_c(level):
    return re.sub(r"\bC:\w", f"C:{level}", SOURCED)


def solve():
    doc = parity.doc_text(PHASE, LESSON)
    stated = re.search(r"CVSS (\d\.\d)", doc)
    parsed = dict(part.split(":") for part in SOURCED.split("/"))
    roundtrip = "/".join(f"{k}:{v}" for k, v in parsed.items())
    return {
        "vector": SOURCED, "score": score(SOURCED),
        "doc_stated": float(stated.group(1)) if stated else None,
        "by_conf": {c: score(with_c(c)) for c in ("H", "L", "N")},
        "roundtrip_ok": roundtrip == SOURCED,
    }


def verify(result):
    conf = result["by_conf"]
    return [
        practice.Check(
            "ANSWER: the calculator reproduces the doc's 9.3, and C:L would be 7.2 (a 2.1 swing)",
            result["score"] == 9.3 and result["doc_stated"] == 9.3
            and conf["L"] == 7.2 and round(conf["H"] - conf["L"], 1) == 2.1,
            f"computed {result['score']} from {result['vector']}; doc states "
            f"{result['doc_stated']}; as C:L it is {conf['L']}",
        ),
        practice.Check(
            "FINDING: the confidentiality metric alone moves the score across severity bands",
            conf == {"H": 9.3, "L": 7.2, "N": 5.8},
            f"base score by confidentiality impact: {conf}",
        ),
        practice.Check(
            "FINDING: the calculator round-trips its sourced vector",
            result["roundtrip_ok"],
            f"parsing and re-emitting {result['vector']} returns it unchanged",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
