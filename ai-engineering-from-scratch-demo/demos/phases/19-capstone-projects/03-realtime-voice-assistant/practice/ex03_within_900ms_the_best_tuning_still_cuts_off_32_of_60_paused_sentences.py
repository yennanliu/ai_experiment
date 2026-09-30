"""Exercise 3 — within 900 ms the best tuning still cuts off 32 of 60 paused sentences.

    Run an adversarial turn-detector test: give the user long pauses mid-sentence. Tune the VAD silence threshold and the turn-detector score threshold for lowest false-cutoff without blowing past 900ms.

Reading of the exercise: 60 seeded utterances of 3-10 words, each with one
pause of 300-1600 ms after a random word. They are fed as 20 ms frames with
oracle partials, so ex01's dropped last word does not muddy the result.
Everything runs through the reference `run_session`. Both thresholds are
literals inside it (500 ms, 0.6). They are tuned by swapping the module's
`turn_completion_score` for a wrapper that commits only after S ms of silence
at score threshold T. The grid is S = 500-1500 ms by T in {0.2, 0.55, 0.75,
0.95}; T is only meaningful at the scorer's levels. "900 ms" is read the
way the lesson's Key Terms define first-audio-out: user stop to first audio.
A turn that never commits has blown the budget.

**ANSWER: S = 500 ms, T = 0.55, and it still cuts off 32 of 60 (53%).** User
stop to first audio is exactly S + 300 ms, so only S <= 600 fits in 900 ms.
Of those, T <= 0.55 is the only setting that answers every utterance. The
scorer reads nothing but word count, and a pause after 3 or more words that
is longer than S is cut off. S = 600 gives the same 32. Getting under the
lesson's own 3% bar takes S above 1500 ms, which puts first audio past
1800 ms.

**FINDING: the shipped thresholds trade cutoffs for silence.** At (500 ms,
0.6) only 7 of 60 are cut off, but 21 of 60 are never answered. Every
3-5-word turn scores 0.55 forever, because the synthetic partials carry no
punctuation. At T = 0.95 all 60 go unanswered; that is the only setting with
zero cutoffs.

**FINDING: the lesson's latency figures start the clock too late.** Its clean
weather call reports 740 ms. Measured from user stop, it is 1220 ms. With a
tool, the best tuning's worst turn is 1220 ms, so no setting meets 900 ms.
Without a tool the floor is 800 ms, and 800 is not "under 800ms".

Structure: `corpus()` draws the utterances, `frames_for()` builds frames,
`tuned_scorer()` stands in for the two literals, `grid_cell()` scores one
setting.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "03-realtime-voice-assistant"
VOCAB = "can you check whether the flight to osaka on friday still has an aisle seat left".split()
SILENCES = range(500, 1600, 100)       # run_session calls the scorer from 500 ms of silence on
SCORES = (0.2, 0.55, 0.75, 0.95)       # the reference scorer only returns 0, .2, .55, .75, .95
BUDGET = 900


def corpus(n=60, seed=3):
    """Utterances of 3-10 words with one mid-sentence pause of 300-1600 ms."""
    rng = random.Random(seed)

    def draw(words):
        return {"words": VOCAB[:words], "after": rng.randint(1, words - 1), "pause": rng.randrange(300, 1700, 100)}
    return [draw(rng.randint(3, 10)) for _ in range(n)]


def frames_for(ref, u):
    """Oracle-partial 20 ms frames: 120 ms lead-in, 320 ms per word, the pause, 2.6 s tail."""
    t, said = 0, []
    silent = lambda ms: [ref.Frame(t + 20 * i, False, " ".join(said)) for i in range(ms // 20)]
    frames, t = silent(120), 120
    for i, w in enumerate(u["words"]):
        said.append(w)
        frames += [ref.Frame(t + 20 * j, True, " ".join(said)) for j in range(16)]
        t += 320
        if i + 1 == u["after"]:
            frames += silent(u["pause"])
            t += u["pause"]
    return frames + silent(2600), t


def tuned_scorer(orig, silence, threshold):
    """Stand-in for the two literals in run_session: commit only after `silence` ms at `threshold`."""
    seen = {"partial": None, "calls": 0}

    def score(partial):
        seen["calls"] = seen["calls"] + 1 if partial == seen["partial"] else 1
        seen["partial"] = partial
        waited = 480 + 20 * seen["calls"]      # the first call happens at silence_run_ms == 500
        return 1.0 if waited >= silence and orig(partial) >= threshold else 0.0
    return score


def grid_cell(ref, utts, silence, threshold, tool=False):
    orig, runs = ref.turn_completion_score, []
    try:
        for u in utts:     # a fresh scorer per call: utterances share word prefixes
            ref.turn_completion_score = tuned_scorer(orig, silence, threshold)
            frames, stop = frames_for(ref, u)
            runs.append((ref.run_session(frames, use_tool=tool), stop))
    finally:
        ref.turn_completion_score = orig
    cut = sum(0 < m.turn_complete_ms < stop for m, stop in runs)
    mute = sum(m.turn_complete_ms == 0 for m, _ in runs)
    lat = [m.first_audio_out_ms - stop for m, stop in runs if m.turn_complete_ms >= stop]
    return {"false_cutoff": cut, "no_response": mute, "max_latency": max(lat, default=None)}


def feasible(grid):
    return {k: v for k, v in grid.items() if v["no_response"] == 0 and (v["max_latency"] or 0) <= BUDGET}


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    utts = corpus()
    grid = {(s, th): grid_cell(ref, utts, s, th) for s in SILENCES for th in SCORES}
    ok = feasible(grid)
    best = min(ok, key=lambda k: (ok[k]["false_cutoff"], k[0], k[1]))
    clean = ref.run_session(ref.synth_call("what is the weather in tokyo tomorrow"), use_tool=True)
    at_055 = {s: grid[(s, 0.55)] for s in SILENCES}
    return {
        "short": sum(len(u["words"]) < 6 for u in utts),
        "shipped": grid_cell(ref, utts, 500, 0.6), "best": best, "best_cell": grid[best],
        "feasible": sorted(ok), "zero_cutoff": sorted({k[1] for k, v in grid.items() if v["false_cutoff"] == 0}),
        "slope": {s - v["max_latency"] for s, v in at_055.items()},
        "under_3pct": [s for s, v in at_055.items() if v["false_cutoff"] <= 0.03 * len(utts)],
        "with_tool": grid_cell(ref, utts, best[0], best[1], tool=True),
        "clean_reported": clean.latency_ms(), "clean_from_stop": clean.first_audio_out_ms - (120 + 7 * 320),
        "floor": grid[(500, 0.2)]["max_latency"],
    }


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: S = 500 ms, T = 0.55 -- and it still cuts off 32 of 60 paused sentences",
            (r["best"], r["best_cell"], r["feasible"], r["slope"], r["under_3pct"])
            == ((500, 0.55), {"false_cutoff": 32, "no_response": 0, "max_latency": 800},
                [(500, 0.2), (500, 0.55), (600, 0.2), (600, 0.55)], {-300}, []),
            f"feasible (S, T): {r['feasible']}; best {r['best']} -> {r['best_cell']}; first audio = S + "
            f"{-min(r['slope'])} ms; S reaching <= 3% cutoff at T=0.55: {r['under_3pct']}",
        ),
        practice.Check(
            "FINDING: the shipped thresholds trade cutoffs for silence",
            (r["shipped"]["false_cutoff"], r["shipped"]["no_response"], r["short"], r["zero_cutoff"]) == (7, 21, 21, [0.95]),
            f"(500 ms, 0.6): {r['shipped']}; {r['short']}/60 utterances have 3-5 words; zero cutoffs only at T "
            f"{r['zero_cutoff']}, which answers none",
        ),
        practice.Check(
            "FINDING: the lesson's latency figures start the clock too late",
            (r["clean_reported"], r["clean_from_stop"], r["with_tool"]["max_latency"], r["floor"]) == (740, 1220, 1220, 800),
            f"clean call latency_ms {r['clean_reported']} vs {r['clean_from_stop']} ms from user stop; with the tool "
            f"worst turn {r['with_tool']['max_latency']} ms; no-tool floor {r['floor']} ms",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
