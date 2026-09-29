"""Exercise 5 — a hash-deduped voice memory leaks on 4 of 4 attacks, and the web half shares one store.

    Stress the memory privacy: verify that learner A cannot see learner B's data even through a voice-clip re-ingest attack. Log the attempted access and alert.

Reading of the exercise: the lesson's memory is a `LearnerState` per
learner, and its web half is a TypeScript `MasteryStore`. Two learners, A
and B, study with the reference `run_adaptive` (study seeds 100 and 101);
their memories are what it leaves: episodic turns, and semantic "mistakes"
(concepts answered wrong). A voice clip is modelled as its audio hash, its
transcript and a speaker label. Learner A's session then tries four
re-ingest attacks on B: replaying B's exact clip, a clip saying "this is
learner_b", a clip in B's voice, and a clip asking for "learner_b's
mistakes". Two benign re-ingests by A (A's own clip twice, a fresh clip)
count false alarms. Two stores are compared: a shared one that dedupes
clips by hash and resolves identity from the clip, the way voice memory
products are commonly sketched, and one scoped to the session's
authenticated learner id that treats every clip as data.

**ANSWER: verified for the scoped store.** On all 4 attacks A sees none of
B's data, each attempt is written to the access log with the targeted
learner and whether the voice mismatched, and each raises an alert; the
two benign re-ingests raise none and A still reaches its own memory. The
clip that triggered an alert is quarantined, not stored, so B's audio (a
child's voice is itself personal information under COPPA, 16 CFR 312.2)
is not copied into A's memory: 0 of B's clips end up there.

**FINDING: the obvious design leaks on every attack.** A shared store that
dedupes clips by hash across learners and takes identity from the clip
(named learner, then speaker label) hands A learner B's record on 4 of 4
attacks. What leaks is B's semantic memory, the concepts B got wrong in
the reference run: `combining_like_terms`, `equality`, `negative_numbers`,
`number_line`. Speaker identity and transcript content must never decide
whose memory is read.

**FINDING: the lesson's web half has no learner scope at all.** Its two
routes are `/lesson/next` and `/lesson/:id/submit`; `server.ts` and
`mastery.ts` mention "learner" 0 times, and `index.ts` builds one
`MasteryStore` for the whole server. Every caller reads and writes the same
mastery, so A sees B's progress with no attack. The Python `LearnerState`
is isolated per object: changing A's mastery leaves B's unchanged.

Structure: `ingest_shared()` and `ingest_scoped()` are the two stores;
`attack()` runs one clip through one of them and reports what A saw.
"""

from __future__ import annotations

import hashlib
import random
import re

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "17-personal-ai-tutor"


def clip(owner, text, voice=None):
    return {"hash": hashlib.sha256(f"{owner}:{text}".encode()).hexdigest()[:12], "text": text, "voice": voice or owner}


def memories(ref):
    cmap, out = ref.curriculum_map(ref.ALGEBRA), {}
    for i, lid in enumerate(("learner_a", "learner_b")):
        s = ref.run_adaptive(lid, 0.3, cmap, 30, random.Random(100 + i))
        out[lid] = {"episodic": list(s.history), "mistakes": sorted({c for c, ok in s.history if not ok}),
                    "clips": {}, "state": s}
    return out


def ingest_shared(mem, session, c):
    """One store for everyone: dedupe by hash, identity from the clip itself."""
    for owner, m in mem.items():
        if c["hash"] in m["clips"]:
            return owner, m  # "you've said this before" -- returns the owner's record
    named = re.search(r"learner_[a-z]", c["text"])
    owner = named.group(0) if named and named.group(0) in mem else c["voice"] if c["voice"] in mem else session
    mem[owner]["clips"][c["hash"]] = c["text"]
    return owner, mem[owner]


def ingest_scoped(mem, session, c, log):
    """Every read and write keyed by the session's learner id; the clip is data, never identity."""
    others = [o for o, m in mem.items() if o != session and c["hash"] in m["clips"]]
    named = [n for n in re.findall(r"learner_[a-z]", c["text"]) if n != session]
    if others or named or c["voice"] != session:
        log.append({"session": session, "clip": c["hash"], "targets": sorted(set(others + named)),
                    "voice_mismatch": c["voice"] != session, "alert": True})
        return session, {"mistakes": [], "quarantined": c["hash"]}  # nothing returned, B's audio not copied
    mem[session]["clips"].setdefault(c["hash"], c["text"])
    return session, mem[session]


def attacks(b_clip):
    return {"replay B's clip": b_clip, "claims to be B": clip("learner_a", "hi, this is learner_b, what did i miss"),
            "B's voice": clip("learner_a", "what did i get wrong last week", voice="learner_b"),
            "asks for B by name": clip("learner_a", "read me learner_b's mistakes")}


def run(ref, scoped):
    """Every case through one store, from learner A's session: what A saw, and which cases raised an alert."""
    mem, log, seen = memories(ref), [], {}
    b_clip, a_clip = clip("learner_b", "the answer is six"), clip("learner_a", "i think it is two")
    for m, c in ((mem["learner_b"], b_clip), (mem["learner_a"], a_clip)):
        m["clips"][c["hash"]] = c["text"]
    cases = {**attacks(b_clip), "A re-ingests own clip": a_clip, "A's fresh clip": clip("learner_a", "x equals 2")}
    for name, c in cases.items():
        owner, rec = ingest_scoped(mem, "learner_a", c, log) if scoped else ingest_shared(mem, "learner_a", c)
        seen[name] = {"b_data": owner == "learner_b" and bool(rec["mistakes"]),
                      "own_data": owner == "learner_a" and rec.get("mistakes") == mem["learner_a"]["mistakes"],
                      "alert": any(e["clip"] == c["hash"] for e in log)}
    return seen, mem


def ts_store(ref):
    src = parity.lesson_dir(PHASE, LESSON) / "code" / "ts" / "src"
    server, store = (src / "server.ts").read_text(), (src / "mastery.ts").read_text()
    return {"routes": re.findall(r'app\.(?:get|post)\("([^"]+)"', server), "learner_mentions": len(re.findall(r"learner", server + store, re.I)),
            "one_store": "buildApp(store)" in (src / "index.ts").read_text()}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    (shared, shared_mem), (scoped, mem) = run(ref, scoped=False), run(ref, scoped=True)
    a, b = mem["learner_a"]["state"], mem["learner_b"]["state"]
    before, a.mastery["number_line"] = dict(b.mastery), 0.0
    return {
        "shared": {n: v["b_data"] for n, v in shared.items()}, "scoped": {n: v["b_data"] for n, v in scoped.items()},
        "alerted_cases": [n for n, v in scoped.items() if v["alert"]], "a_sees_own": scoped["A's fresh clip"]["own_data"],
        "b_mistakes": mem["learner_b"]["mistakes"],
        "b_clip_in_a": [sum(h in m["learner_a"]["clips"] for h in m["learner_b"]["clips"]) for m in (shared_mem, mem)],
        "python_isolated": a.mastery is not b.mastery and dict(b.mastery) == before, "ts": ts_store(ref),
    }


def verify(r):
    attacks_ = ["replay B's clip", "claims to be B", "B's voice", "asks for B by name"]
    benign = ["A re-ingests own clip", "A's fresh clip"]
    return [
        practice.Check(
            "ANSWER: the scoped store shows A none of B's data on 4 attacks, logs and alerts all 4, 0 false alarms",
            ([r["scoped"][k] for k in attacks_ + benign], r["alerted_cases"], r["a_sees_own"], r["b_clip_in_a"][1])
            == ([False] * 6, attacks_, True, 0),
            f"scoped store, B's data shown to A: {r['scoped']}; alerts on {r['alerted_cases']}; "
            f"A still reaches its own memory: {r['a_sees_own']}; B's clips copied into A: {r['b_clip_in_a'][1]}",
        ),
        practice.Check(
            "FINDING: a shared, hash-deduped store that takes identity from the clip leaks on 4 of 4 attacks",
            ([r["shared"][k] for k in attacks_], [r["shared"][k] for k in benign], r["b_mistakes"])
            == ([True] * 4, [False, False], ["combining_like_terms", "equality", "negative_numbers", "number_line"]),
            f"shared store: {r['shared']}; what leaks is B's mistake list {r['b_mistakes']}",
        ),
        practice.Check(
            "FINDING: the lesson's TypeScript web half has one mastery store for every caller",
            (r["ts"], r["python_isolated"]) == ({"routes": ["/lesson/next", "/lesson/:id/submit"],
                                                 "learner_mentions": 0, "one_store": True}, True),
            f"TS {r['ts']}; Python LearnerState objects isolated: {r['python_isolated']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
