"""Exercise 2 — an intersection-only stereotype moves neither single-axis probe, and debias erases race by truncating to 4-d.

    Extend the probe with an intersectional test: (gender, race) x (career,
    family). Report cross-axis bias scores.

Reading of the exercise: the lesson's 4-d embedding gains two race axes, R1
and R2 (abstract group labels; the toy has no race terms and needs none).
Each of the lesson's six gender words is composed with each race into an
intersectional term ("he|R1" = he + R1), giving four identity cells of three
words. Career and family are the lesson's own X and Y. Three worlds are
probed with the reference `weat_score`: race neutral (rho = 0), a race-level
stereotype on the attribute words (career +0.3 on R1, family +0.3 on R2),
and the same plus an intersection-only stereotype (+0.5 care on the (F, R2)
cell alone). Cross-axis scores are WEAT between cells; the interaction is
the gender gap inside R2 minus the gender gap inside R1.

**ANSWER: in the full world, gender gap 0.6004 inside R1 and 0.8360 inside
R2; race gap 0.3869 among men and 0.6225 among women; interaction +0.2356.**
The diagonal contrasts are M-R1 vs F-R2 = +1.2229 and F-R1 vs M-R2 =
-0.2135. The single-axis probes read gender 0.8574 (he/his/man vs
she/her/woman) and race 0.5411 (R1 vs R2).

**FINDING: the intersection-only stereotype moves neither single-axis probe.**
Adding it leaves gender at 0.8574 and race at 0.5411, identical to the world
without it. Pooling the cells instead (all men vs all women, all R1 vs all
R2) does register it, but splits it between the axes: gender 0.6000 ->
0.7182 and race 0.3864 -> 0.5047. Only the cell contrasts place it on (F,
R2).

**FINDING: additive associations produce no interaction.** With the
race-level stereotype alone the interaction is -0.0009, and M-R1 vs F-R2
(0.9864) is the gender gap (0.6004) plus the race gap (0.3869) to 0.0009.
An interaction score is evidence of a stereotype about the intersection, not
of two stereotypes that merely overlap.

**FINDING: composing identities dilutes the score.** With race neutral the
gender gap inside either race is 0.6231, not the lesson's 0.8906, because
the race component lengthens every identity vector and cosine divides by
it. A compound-term score is not on the same scale as a single-axis one.

**FINDING: `debias()` erases race by truncating vectors to 4-d.** Its
`zip(new[w], gender_dir)` stops at the 4-d direction, so every attribute
word comes back 4-d and loses its race loading: single-axis race goes
0.5411 -> 0.0000 and the race gap among men 0.3869 -> 0.0000. The same
projection done in 6-d keeps race at 0.5567. The interaction survives
either way (+0.2945 truncated, +0.2638 in 6-d).

Structure: `build()` composes the extended embedding; `scores()` runs the
reference `weat_score` on every contrast; `project()` is the lesson's
projection with the direction padded to the embedding's width.
"""

from __future__ import annotations

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "20-bias-representational-harm"
A, B = ["he", "his", "man"], ["she", "her", "woman"]
X, Y = ["engineer", "programmer", "scientist"], ["nurse", "teacher", "caregiver"]
R1, R2, CARE, RHO, DELTA = 4, 5, 3, 0.3, 0.5  # axes; race-level and intersection-only loadings


def build(ref, rho=0.0, delta=0.0):
    """The lesson's EMB with two race axes, plus the four intersectional cells."""
    emb = {k: list(v) + [0.0, 0.0] for k, v in ref.EMB.items()}
    for words, axis in ((X, R1), (Y, R2)):
        for w in words:
            emb[w][axis] += rho
    emb["race-1"], emb["race-2"] = [0.0] * 4 + [1.0, 0.0], [0.0] * 5 + [1.0]
    cells = {g + r: compose(emb, words, r, axis, delta if g + r == "FR2" else 0.0)
             for g, words in (("M", A), ("F", B)) for r, axis in (("R1", R1), ("R2", R2))}
    return emb, cells


def compose(emb, words, race, axis, care):
    """Add "w|race" = w + race axis (+ care) to emb for each word; return the new names."""
    for w in words:
        v = list(emb[w])
        v[axis] += 1.0
        v[CARE] += care
        emb[f"{w}|{race}"] = v
    return [f"{w}|{race}" for w in words]


def weat(ref, emb, a, b):
    saved, ref.EMB = ref.EMB, emb
    try:
        return round(ref.weat_score(a, b, X, Y), 4)
    finally:
        ref.EMB = saved


def scores(ref, emb, c):
    out = {
        "gender": weat(ref, emb, A, B), "race": weat(ref, emb, ["race-1"], ["race-2"]),
        "g_R1": weat(ref, emb, c["MR1"], c["FR1"]), "g_R2": weat(ref, emb, c["MR2"], c["FR2"]),
        "r_M": weat(ref, emb, c["MR1"], c["MR2"]), "r_F": weat(ref, emb, c["FR1"], c["FR2"]),
        "MR1_FR2": weat(ref, emb, c["MR1"], c["FR2"]), "FR1_MR2": weat(ref, emb, c["FR1"], c["MR2"]),
        "g_pool": weat(ref, emb, c["MR1"] + c["MR2"], c["FR1"] + c["FR2"]),
        "r_pool": weat(ref, emb, c["MR1"] + c["FR1"], c["MR2"] + c["FR2"]),
    }
    return {**out, "inter": round(out["g_R2"] - out["g_R1"], 4)}


def project(emb):
    """The lesson's debias projection, with [1, -1, 0, ...] as wide as the vectors."""
    new = {k: list(v) for k, v in emb.items()}
    d = [1.0, -1.0] + [0.0] * (len(emb["he"]) - 2)
    for w in X + Y:
        p = sum(a * b for a, b in zip(new[w], d)) / 2
        new[w] = [a - p * b for a, b in zip(new[w], d)]
    return new


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    neutral, race_only, full = build(ref), build(ref, RHO), build(ref, RHO, DELTA)
    debiased = ref.debias(full[0])
    return {
        "neutral": scores(ref, *neutral), "race_only": scores(ref, *race_only),
        "full": scores(ref, *full), "debiased": scores(ref, debiased, full[1]),
        "projected": scores(ref, project(full[0]), full[1]),
        "widths": (len(debiased["engineer"]), len(debiased["he|R1"]), len(full[0]["engineer"])),
    }


def verify(result):
    f, r, n = result["full"], result["race_only"], result["neutral"]
    dbs, prj = result["debiased"], result["projected"]
    return [
        practice.Check(
            "ANSWER: gender gap 0.6004 in R1 and 0.8360 in R2, race gap 0.3869 / 0.6225, "
            "interaction +0.2356",
            [f[k] for k in ("g_R1", "g_R2", "r_M", "r_F", "inter", "MR1_FR2", "FR1_MR2",
                            "gender", "race")]
            == [0.6004, 0.836, 0.3869, 0.6225, 0.2356, 1.2229, -0.2135, 0.8574, 0.5411],
            f"cross-axis scores {f}",
        ),
        practice.Check(
            "FINDING: the intersection-only stereotype moves neither single-axis probe",
            (f["gender"], f["race"], r["g_pool"], f["g_pool"], r["r_pool"], f["r_pool"])
            == (r["gender"], r["race"], 0.6, 0.7182, 0.3864, 0.5047),
            f"single-axis gender {r['gender']} -> {f['gender']}, race {r['race']} -> {f['race']}; "
            f"pooled gender {r['g_pool']} -> {f['g_pool']}, pooled race {r['r_pool']} -> {f['r_pool']}",
        ),
        practice.Check(
            "FINDING: additive associations produce no interaction",
            (r["inter"], r["MR1_FR2"], r["g_R1"], r["r_M"]) == (-0.0009, 0.9864, 0.6004, 0.3869)
            and round(r["MR1_FR2"] - r["g_R1"] - r["r_M"], 4) == -0.0009,
            f"race-level stereotype alone: interaction {r['inter']}, M-R1 vs F-R2 {r['MR1_FR2']} "
            f"against gender {r['g_R1']} + race {r['r_M']}",
        ),
        practice.Check(
            "FINDING: composing identities dilutes the score",
            n["g_R1"] == n["g_R2"] == 0.6231 and n["gender"] == 0.8906 and n["inter"] == 0.0,
            f"race neutral: gender gap {n['g_R1']} inside each race against {n['gender']} single-axis",
        ),
        practice.Check(
            "FINDING: debias() erases race by truncating vectors to 4-d",
            (result["widths"], dbs["race"], dbs["r_M"], prj["race"], dbs["inter"], prj["inter"])
            == ((4, 6, 6), 0.0, 0.0, 0.5567, 0.2945, 0.2638),
            f"debiased attribute/identity widths {result['widths'][:2]}; race {f['race']} -> "
            f"{dbs['race']} (6-d projection {prj['race']}); interaction {dbs['inter']} / {prj['inter']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
