"""Exercise 2 -- pooling the three vectors raises MRR from 0.259 to 0.391, because RRF drowns the stream that matched.

    Reduce per-scene frame embedding to one pooled vector instead of multi-vector. Measure the retrieval regression.

Reading of the exercise: the lesson stores one frame vector per scene (one
keyframe), so "multi-vector" is the scene's three named vectors -- caption,
frame, transcript -- queried separately and merged with RRF by
`multi_vector_search`. The pooled variant sums a scene's three unit vectors
into one, re-normalises it, and ranks by cosine. Both run on the lesson's
`Scene.embed` and `fake_embed` over a 50-scene fixture. Each scene draws its
caption, frame tags and transcript from three separate word pools, and gets 3
queries of 2 words from one of its three streams (150 queries). MRR is
averaged over 10 hash salts. The builtin `hash` is swapped for a salted
blake2b so the run repeats; Python salts `hash` per process. Two controls:
the max cosine over the three vectors (multi-vector without RRF), and the
lesson's own 6 scenes with the 4 queries in `main()`.

**ANSWER: there is no regression. Pooling improves retrieval.** MRR goes from
0.259 (RRF, as shipped) to 0.391 (pooled). Max-cosine over the three vectors
scores 0.608, so keeping three vectors helps, but only without RRF. The
lesson's skill file lists pooling as a hard reject ("Pipelines that pool a
single vector per scene"); on this index it is the better of the two.

**FINDING: RRF ranks every scene in every stream, so the two streams that did
not match outvote the one that did.** With k = 60 and 50 scenes, a rank
contributes between 1/61 and 1/110. A stream's first place is worth 0.0073
over its last place, and two random ranks move the sum as much. At
`EMB_DIM = 256`, with fewer hash collisions, RRF gets worse (0.220) while
pooled reaches 0.801. Most cosines are then exactly 0, and `sorted` breaks the
ties by list order: fixture scene 0 is RRF's top hit for 21.0% of queries,
against 2% for a uniform pick.

**FINDING: on the lesson's own 6 scenes the two tie, 31 of 40 each.** Per
demo query, over 10 salts, RRF puts the intended scene first 4, 7, 10, 10
times and pooling 5, 6, 10, 10. Most misses are the headline counting query
("how many cars pass through the intersection"): "intersection" is scene 1's
caption word, and the cars are in scene 2.

Structure: `fixture` builds the scenes; `rankings` returns the three orders
for one query; `run` scores one salt and embedding size.
"""

from __future__ import annotations

import hashlib
import math
import random

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "12-video-understanding-pipeline"
SALTS = 10
CAP = ("sunrise skyline drone intersection pedestrians kitchen chef plating ocean waves sunset forest trail runner "
       "stadium crowd goal market vendor fruit train platform commuters snow mountain skier classroom teacher "
       "whiteboard lab scientist microscope garage mechanic engine beach surfer board concert guitarist stage "
       "office meeting laptop park dog frisbee").split()
FRM = ("orange haze glass steel asphalt signal pan stove garnish foam spray pine mud track seats floodlight net "
       "awning crate rails bench peaks lift desk marker vial lens wrench hood oil sand wax amp spotlight chair "
       "screen grass leash disc").split()
TRN = ("welcome tokyo count vehicles pour stir tasty evening shore listen breathe score cheer price fresh delay "
       "cold steep homework quiz sample result torque repair wave paddle encore loud agenda deadline fetch "
       "goodboy budget launch").split()
FIELDS = ("caption", "frame_tags", "transcript")
DEMO_GOLD = [("vid_001", 2), ("vid_001", 3), ("vid_001", 4), ("vid_002", 0)]  # main()'s 4 queries, in order


def fixture(ref):
    rng = random.Random(0)
    return [ref.Scene(f"v{i // 10}", i % 10, i * 30_000, (i + 1) * 30_000, " ".join(rng.sample(CAP, 4)),
                      " ".join(rng.sample(TRN, 5)), " ".join(rng.sample(FRM, 5))) for i in range(50)]


def salted(ref, salt, dim):
    ref.EMB_DIM = dim
    ref.hash = lambda s: int.from_bytes(hashlib.blake2b(f"{salt}|{s}".encode(), digest_size=8).digest(), "big")


def unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def rankings(ref, query, scenes):
    qv = ref.fake_embed(query)
    vectors = {id(s): (s.caption_emb, s.frame_emb, s.transcript_emb) for s in scenes}
    rrf = [s for s, _ in ref.multi_vector_search(query, scenes, k=len(scenes))]
    best = sorted(scenes, key=lambda s: -max(ref.cosine(qv, v) for v in vectors[id(s)]))
    pooled = sorted(scenes, key=lambda s: -ref.cosine(qv, unit([sum(x) for x in zip(*vectors[id(s)])])))
    return {"rrf": rrf, "max": best, "pooled": pooled}


def run(ref, salt, dim):
    salted(ref, salt, dim)
    scenes, rng = fixture(ref), random.Random(1)
    for s in scenes:
        s.embed()
    rr, first_is_scene0 = {"rrf": 0.0, "max": 0.0, "pooled": 0.0}, 0
    for s in scenes:
        for field in FIELDS:
            ranks = rankings(ref, " ".join(rng.sample(getattr(s, field).split(), 2)), scenes)
            first_is_scene0 += ranks["rrf"][0] is scenes[0]
            for name, order in ranks.items():
                rr[name] += 1 / ([id(x) for x in order].index(id(s)) + 1)
    return {k: v / 150 for k, v in rr.items()}, first_is_scene0


def demo_hits(ref):
    hits = {"rrf": [0] * 4, "pooled": [0] * 4}
    for salt in range(SALTS):
        salted(ref, salt, 24)
        scenes = ref.SAMPLE
        for s in scenes:
            s.embed()
        for i, (q, gold) in enumerate(zip(("how many cars pass through the intersection", "what happened first pour or stir",
                            "plating of the dish", "ocean at sunset"), DEMO_GOLD)):
            ranks = rankings(ref, q, scenes)
            for name in hits:
                hits[name][i] += (ranks[name][0].video_id, ranks[name][0].scene_id) == gold
    return hits


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    out = {}
    for dim in (24, 256):
        runs = [run(ref, salt, dim) for salt in range(SALTS)]
        out[dim] = {k: round(sum(r[0][k] for r in runs) / SALTS, 3) for k in ("rrf", "max", "pooled")}
        out[dim]["scene0_top"] = round(sum(r[1] for r in runs) / (150 * SALTS), 3)
    skill = (parity.lesson_dir(PHASE, LESSON) / "outputs" / "skill-video-qa.md").read_text()
    return {"mrr": out, "demo": demo_hits(ref), "skill_rejects_pooling": "pool a single vector per scene" in skill}


def verify(result):
    shipped, wide = result["mrr"][24], result["mrr"][256]
    return [
        practice.Check(
            "ANSWER: no regression -- pooling raises MRR from 0.259 (RRF) to 0.391",
            (shipped["rrf"], shipped["pooled"], shipped["max"]) == (0.259, 0.391, 0.608)
            and result["skill_rejects_pooling"],
            f"EMB_DIM 24, 150 queries x {SALTS} salts: MRR RRF {shipped['rrf']}, pooled {shipped['pooled']}, "
            f"max-cosine {shipped['max']}; the skill file hard-rejects pooling: {result['skill_rejects_pooling']}",
        ),
        practice.Check(
            "FINDING: RRF lets the two unmatched streams outvote the matched one; ties go to list order",
            (wide["rrf"], wide["pooled"], wide["scene0_top"]) == (0.22, 0.801, 0.21)
            and 1 / 61 - 1 / 110 < 0.0074,
            f"EMB_DIM 256: RRF {wide['rrf']}, pooled {wide['pooled']}; fixture scene 0 is RRF's top hit for "
            f"{wide['scene0_top']:.1%} of queries (uniform: 2%); rank-1 bonus {1 / 61 - 1 / 110:.4f}",
        ),
        practice.Check(
            "FINDING: on the lesson's 6 scenes the two tie, and the counting demo query misses in 5-6 of 10 salts",
            result["demo"] == {"rrf": [4, 7, 10, 10], "pooled": [5, 6, 10, 10]},
            f"intended scene first per demo query, of {SALTS} salts: {result['demo']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
