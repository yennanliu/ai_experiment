"""Exercise 1 -- the Python filter leaks 0 of 60 slots, and the TypeScript chat server leaks on 15 of 20 probes.

    Build a second corpus slice under a different jurisdiction (e.g., HIPAA
    alongside GDPR). Demonstrate role+jurisdiction filtering preventing
    cross-leak on a 20-question cross-jurisdiction probe.

Reading of the exercise: the lesson's `CORPUS` has one HIPAA chunk (counsel
only) next to two GDPR chunks. The second slice adds four HIPAA chunks (three
analyst, one counsel) and three more GDPR chunks (two counsel, one analyst),
all labelled `Chunk`s passed to the lesson's `retrieve`. The probe has 20
questions: 10 from an analyst in GDPR asking HIPAA or SOC2 questions, and 10
from an analyst in HIPAA asking GDPR or counsel-only questions. A leak is a
returned chunk whose jurisdiction is not the user's (or `any`), or whose role
is not the user's (or `public`). As a control, the same probe runs with the
labels stripped, to show the probe really reaches for hidden text. The same
20 questions then go to the lesson's other retriever, `retrieve` in
`code/ts/src/stream.ts`, the one `/chat/stream` calls, run through Node's
type stripping. That is why the tier is T1 (Node 22.6 or newer, no npm
packages).

**ANSWER: the filter holds: 0 leaks in 60 returned slots across the 20
probes.** Without labels the same questions pull 48 out-of-scope chunks into
the top 3, so the probe is adversarial and the filter is what stops it.

**FINDING: the TypeScript chat server leaks on 15 of 20 probes (25 foreign
snippets).** Its `retrieve(query, jurisdiction, k)` adds +2 for a matching
tag and never filters, and it has no role argument at all. A HIPAA session
asking about erasure gets `GDPR-Art-17` text streamed back. The skill file's
hard reject, "any chatbot that leaks cross-jurisdiction data", fits the half
of the lesson that has a UI.

**FINDING: the Python filter blocks leaks, but it does not refuse.** All 20
probes still get three citations and a "Based on the cited sections" answer,
built from in-scope chunks that do not answer the question.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "08-production-rag-chatbot"
NEW_HIPAA = [
    ("HIPAA-BAA-2024", "s4", ("Business associate must report any breach of unsecured protected health information "
                              "within 60 days."), "analyst"),
    ("HIPAA-BAA-2024", "s9", ("Minimum necessary standard: disclose only the protected health information needed "
                              "for the purpose."), "analyst"),
    ("HIPAA-NPP-2025", "s2", ("Patients have the right to obtain a copy of their health records within 30 days "
                              "of a request."), "analyst"),
    ("HIPAA-SEC-2025", "s3", "Audit logs of access to electronic health records are retained for six years.", "counsel"),
]
NEW_GDPR = [
    ("DPA-v2.1", "s8", "Personal data breaches must be notified to the supervisory authority within 72 hours.", "counsel"),
    ("DPA-v2.1", "s3", "Data subjects have the right to obtain a copy of their personal data within one month.", "analyst"),
    ("MSA-2024-03-11", "s14", "Processing logs for EU personal data are retained for three years.", "counsel"),
]
TO_GDPR = ["how many days to report a breach of protected health information",
           "what is the minimum necessary standard for health information",
           "when must PHI be returned or destroyed after termination", "how long are audit logs of health records retained",
           "can patients obtain a copy of their health records", "what does the business associate agreement require",
           "disclose protected health information to a covered entity", "HIPAA breach notification deadline",
           "PHI destruction within 60 days", "access review cadence for privileged users"]
TO_HIPAA = ["how fast must personal data breaches be notified to the supervisory authority",
            "when must EU user profiles be deleted under GDPR Article 17", "right to erasure of personal data",
            "deletion of restricted data category within 14 days",
            "how long are processing logs for EU personal data retained",
            "can data subjects obtain a copy of their personal data", "GDPR 72 hours breach notification",
            "the data subject right to obtain confirmation that personal data are processed",
            "what are the counsel only audit log retention rules", "quarterly access review for privileged users"]
PROBE = [("analyst", "GDPR", q) for q in TO_GDPR] + [("analyst", "HIPAA", q) for q in TO_HIPAA]


def corpus(ref):
    extra = [ref.Chunk(d, s, t, role, "HIPAA") for d, s, t, role in NEW_HIPAA]
    extra += [ref.Chunk(d, s, t, role, "GDPR") for d, s, t, role in NEW_GDPR]
    return ref.CORPUS + extra


def allowed(chunk, role, jurisdiction):
    return chunk.role in (role, "public") and chunk.jurisdiction in (jurisdiction, "any")


def probe_one(ref, labelled, unlabelled, role, j, q):
    truth = {c.anchor(): c for c in labelled}
    hits = ref.retrieve(q, role, j, labelled, k=3)
    stripped = ref.retrieve(q, role, j, unlabelled, k=3)
    reply = ref.chat_turn(q, role, j, labelled, ref.PromptCache())
    return (len(hits), sum(not allowed(c, role, j) for c, _ in hits),
            sum(not allowed(truth[c.anchor()], role, j) for c, _ in stripped),
            int(reply["answer"].startswith("Based on the cited sections") and len(reply["citations"]) == 3))


def python_side(ref):
    labelled = corpus(ref)
    unlabelled = [dataclasses.replace(c, role="public", jurisdiction="any") for c in labelled]
    rows = [probe_one(ref, labelled, unlabelled, *p) for p in PROBE]
    slots, leaks, control, answered = (sum(col) for col in zip(*rows))
    return {"leaks": leaks, "slots": slots, "control": control, "answered": answered,
            "sizes": {j: sum(c.jurisdiction == j for c in labelled) for j in ("GDPR", "HIPAA")}}


def ts_side():
    node = shutil.which("node")
    if node is None:
        raise practice.Skip("needs Node >= 22.6 on PATH for TypeScript type stripping (nvm install 22)")
    stream = parity.lesson_dir(PHASE, LESSON) / "code" / "ts" / "src" / "stream.ts"
    js = (f"const m = await import({json.dumps(str(stream))}); const p = JSON.parse(process.argv[1]);"
          "console.log(JSON.stringify(p.map(([r, j, q]) => m.retrieve(q, j, 3).map(c => c.docId))));")
    out = subprocess.run([node, "--no-warnings", "--input-type=module", "-e", js, json.dumps(PROBE)],
                         capture_output=True, text=True, check=True, timeout=60)
    ids = json.loads(out.stdout)
    foreign = [[d for d in got if d.split("-")[0] != j] for (_, j, _), got in zip(PROBE, ids)]
    signature = next(line for line in stream.read_text().splitlines() if "export function retrieve" in line)
    return {"ts_probes_leaking": sum(bool(f) for f in foreign), "ts_foreign": sum(map(len, foreign)),
            "erasure_to_hipaa": ids[PROBE.index(("analyst", "HIPAA", "right to erasure of personal data"))],
            "ts_signature": signature.strip()}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    return {"probes": len(PROBE), **python_side(ref), **ts_side()}


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: role+jurisdiction filtering leaks 0 of 60 slots on the 20-question probe",
            (r["probes"], r["sizes"], r["leaks"], r["slots"], r["control"]) == (20, {"GDPR": 5, "HIPAA": 5}, 0, 60, 48),
            f"{r['probes']} probes over slices {r['sizes']}: {r['leaks']} leaks in {r['slots']} slots; "
            f"with labels stripped {r['control']} out-of-scope chunks reach the top 3",
        ),
        practice.Check(
            "FINDING: the TypeScript /chat/stream retriever boosts instead of filtering and leaks on 15 of 20",
            (r["ts_probes_leaking"], r["ts_foreign"]) == (15, 25) and "role" not in r["ts_signature"]
            and r["erasure_to_hipaa"][0] == "GDPR-Art-17",
            f"{r['ts_probes_leaking']}/20 probes get foreign-jurisdiction text ({r['ts_foreign']} snippets); "
            f"`{r['ts_signature']}`; HIPAA 'right to erasure' -> {r['erasure_to_hipaa']}",
        ),
        practice.Check(
            "FINDING: the filter stops the leak but still answers all 20 probes with 3 off-topic citations",
            r["answered"] == 20,
            f"{r['answered']}/20 probes answered 'Based on the cited sections' with 3 citations",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
