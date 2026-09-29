"""Exercise 1 -- the lower-rated captioner retrieves better, and `fake_embed` cannot see action order.

    Swap Gemini 2.5 Pro for Qwen3-VL-Max on the captioning pass. Report caption quality delta on a human-rated 50-scene sample.

Reading of the exercise: the lesson calls no VLM -- `Scene.caption` is a
hand-written string -- so the swap is two stand-in captioners over a
50-scene fixture with gold content (a setting, two object kinds with counts,
two actions in order). Each writes the gold caption and then makes errors at
its own rate: a wrong count, the two actions swapped, an object dropped. The
rates are placeholders for the two models, stated below, not measurements of
them. The "human rating" is a 1-5 rubric (5, minus 1 per error) seen by two
raters with Gaussian noise (sd 0.6), rounded. The delta is the mean paired
difference with a 2,000-resample bootstrap 95% interval. Then both caption
sets go into the lesson's `Scene.embed` and `multi_vector_search`, and
hit@1 is measured on 100 queries (object + action, and "how many <object>")
over 10 salts of the lesson's hash (a salted blake2b replaces the builtin,
which Python salts per process).

**ANSWER: the harness reports a delta of +0.34 (Gemini stand-in 4.17, Qwen
stand-in 3.83, mean of two raters), 95% interval [-0.02, 0.71].** This
sample cannot call the swap: the interval includes 0. The raters agree
exactly on 56% of scenes. Over 200 fresh 50-scene studies at the same
rates, 89% find the gap, so about one study in nine would report "no
difference".

**FINDING: the caption delta does not reach retrieval.** Hit@1 (the top
scene holds the queried object and action) is 0.418 with the Gemini
stand-in's captions, 0.433 with the lower-rated Qwen stand-in's, 0.432 with
gold captions, and 0.395 with every caption blank. The caption is one of
three RRF streams, and "how many <object>" never contains the count the
captioner got wrong.

**FINDING: `fake_embed` is a bag of hashed words, so action-order errors are
invisible.** For all 6 scenes where the Qwen stand-in swapped the actions and
made no other error, its caption embedding is exactly the gold one. "Pours
then stirs" and "stirs then pours" are the same vector, though the lesson's
own demo asks "what happened first pour or stir".

Structure: `fixture` draws the gold scenes; `caption` renders one with a
captioner's errors; `ratings` is the rubric and the two raters; `hit_rate`
runs the lesson's index.
"""

from __future__ import annotations

import hashlib
import random
import statistics

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "12-video-understanding-pipeline"
SETTINGS, WORDS = "kitchen street beach office stadium garage park market classroom lab".split(), "one two three four five six".split()
OBJECTS, ACTIONS = "cars people dogs chairs cups bikes plates boxes birds lamps balls trays".split(), "pour stir chop lift open close throw catch wave point push pull".split()
PROFILES = {"gemini": {"count": 0.20, "order": 0.10, "drop": 0.05},  # placeholder error rates,
            "qwen": {"count": 0.35, "order": 0.25, "drop": 0.15}}  # not measured on either model


def fixture():
    rng = random.Random(0)
    return [{"setting": rng.choice(SETTINGS), "objects": rng.sample(OBJECTS, 2), "counts": [rng.randint(1, 5)
             for _ in range(2)], "actions": rng.sample(ACTIONS, 2)} for _ in range(50)]


def caption(g, errors):
    counts = [c + 1 if i in errors.get("count", ()) else c for i, c in enumerate(g["counts"])]
    objs = [f"{WORDS[c - 1]} {o}" for c, o in zip(counts, g["objects"])][: 1 if errors.get("drop") else 2]
    acts = g["actions"][::-1] if errors.get("order") else g["actions"]
    return f"{g['setting']} with {' and '.join(objs)}; someone {acts[0]}s then {acts[1]}s"


def draw_errors(rng, p):
    return {"count": [i for i in range(2) if rng.random() < p["count"]], "order": rng.random() < p["order"], "drop": rng.random() < p["drop"]}


def ratings(rng, e):  # the rubric (5, minus 1 per error) seen by two noisy raters
    return [min(5, max(1, round(max(1, 5 - len(e["count"]) - e["order"] - e["drop"]) + rng.gauss(0, 0.6)))) for _ in range(2)]


def bootstrap(diffs, rng):
    means = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(2000))
    return round(means[49], 2), round(means[1949], 2)


def hit_rate(ref, gold, captions, salts=10):
    hits = 0
    for salt in range(salts):
        ref.hash = lambda s, salt=salt: int.from_bytes(hashlib.blake2b(f"{salt}|{s}".encode(), digest_size=8).digest(), "big")
        scenes = [ref.Scene("v", i, i * 30_000, (i + 1) * 30_000, c, f"{g['setting']} talk", " ".join(g["objects"]))
                  for i, (g, c) in enumerate(zip(gold, captions))]
        for s in scenes:
            s.embed()
        for g in gold:
            for q, need in ((f"{g['objects'][0]} {g['actions'][0]}", (g["objects"][0], g["actions"][0])),
                            (f"how many {g['objects'][1]}", (g["objects"][1],))):
                top = gold[ref.multi_vector_search(q, scenes, k=1)[0][0].scene_id]
                hits += all(w in top["objects"] + top["actions"] for w in need)
    return round(hits / (salts * 2 * len(gold)), 3)


def power(rng, trials=200):
    """Share of fresh 50-scene studies whose normal 95% interval for the delta excludes 0."""
    found = 0
    for _ in range(trials):
        d = [statistics.fmean(ratings(rng, draw_errors(rng, PROFILES["gemini"])))
             - statistics.fmean(ratings(rng, draw_errors(rng, PROFILES["qwen"]))) for _ in range(50)]
        found += statistics.fmean(d) - 1.96 * statistics.stdev(d) / 50**0.5 > 0
    return found / trials


def rating_study(rng, errs):
    """The 50-scene human-rating sample: mean ratings, paired delta, bootstrap interval, rater agreement."""
    rated = {m: [ratings(rng, e) for e in errs[m]] for m in PROFILES}
    per_scene = {m: list(map(statistics.fmean, r)) for m, r in rated.items()}
    diffs = [a - b for a, b in zip(per_scene["gemini"], per_scene["qwen"])]
    agree = sum(a == b for r in rated.values() for a, b in r) / (2 * len(diffs))
    return {"mean": {m: round(statistics.fmean(v), 2) for m, v in per_scene.items()},
            "delta": round(statistics.fmean(diffs), 2), "ci": bootstrap(diffs, rng), "agree": round(agree, 2)}


def order_blind(ref, gold, errs):
    """Scenes whose only error is swapped actions, and how many of them embed exactly like the gold caption."""
    only = [(g, e) for g, e in zip(gold, errs) if e["order"] and not (e["count"] or e["drop"])]
    return len(only), sum(ref.fake_embed(caption(g, e)) == ref.fake_embed(caption(g, {})) for g, e in only)


def solve():
    ref, gold, rng = parity.load_reference(PHASE, LESSON, "main"), fixture(), random.Random(7)
    errs = {m: [draw_errors(rng, p) for _ in gold] for m, p in PROFILES.items()}
    study = rating_study(rng, errs)
    caps = {m: list(map(caption, gold, e)) for m, e in errs.items()}
    caps |= {"gold": [caption(g, {}) for g in gold], "blank": [""] * len(gold)}
    n_only, same = order_blind(ref, gold, errs["qwen"])
    return {**study, "hit": {m: hit_rate(ref, gold, c) for m, c in caps.items()},
            "order_only": n_only, "order_invisible": same, "power": power(rng)}


def verify(r):
    return [
        practice.Check(
            "ANSWER: delta +0.34 on the 50-scene sample, and its interval includes 0",
            (r["mean"], r["delta"], r["ci"], r["agree"], r["power"])
            == ({"gemini": 4.17, "qwen": 3.83}, 0.34, (-0.02, 0.71), 0.56, 0.89),
            f"mean rating {r['mean']}, delta {r['delta']:+}, 95% CI {r['ci']}, rater agreement {r['agree']}; {r['power']:.0%} of 200 fresh studies exclude 0",
        ),
        practice.Check(
            "FINDING: the caption delta does not reach retrieval; the lower-rated set retrieves better",
            r["hit"] == {"gemini": 0.418, "qwen": 0.433, "gold": 0.432, "blank": 0.395},
            f"hit@1 on 100 queries x 10 salts by caption set: {r['hit']}",
        ),
        practice.Check(
            "FINDING: fake_embed cannot see action order -- a swapped caption embeds identically",
            r["order_invisible"] == r["order_only"] == 6,
            f"{r['order_invisible']}/{r['order_only']} order-only errors give the gold caption's exact vector",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
