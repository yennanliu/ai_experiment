"""Exercise 5 — a 1% per-frame false trigger cancels 58% of turns, and Japanese turns never commit.

    Add voice activity detection for non-English languages (Japanese, Spanish). Measure the Silero VAD v5 false-trigger rate versus language-specific fine-tunes.

Reading of the exercise: Silero v5 weights and Japanese or Spanish audio are
not available offline, so the frame-level false-trigger rate of a real model
cannot be measured here. What can be run is the lesson's pipeline with a VAD
of known false-trigger rate. That shows what a given rate costs, and whether
adding Japanese and Spanish needs anything beyond the VAD. Seven en/es/ja
turns go through `run_session`, 320 ms per token. The false triggers are
i.i.d. spurious speech frames at 0.1%, 1% and 5% per frame. They fall in
the agent's 1.7 s turn, with barge-in armed, over 400 seeded sessions each.
Facts from the Silero README and `utils_vad.py` were read 2026-09-29.

**ANSWER: there is no fine-tune to compare against, and the rate that
matters is per turn, not per frame.** Silero says it "was trained on huge
corpora that include over 6000 languages". Its README lists no
language-specific fine-tunes, so v5 is already the multilingual baseline.
In this pipeline a per-frame rate p cancels the agent's reply with
probability 1 - (1 - p)^85. Measured: 9.8%, 58.0% and 99.8% of turns at
0.1%, 1% and 5%. Predicted: 8.2%, 57.4% and 98.7%. Silero's own
`min_speech_duration_ms=250` (13 frames) takes all three to 0. That holds
only for scattered triggers; background speech from a TV lasts longer than
250 ms. A language-specific VAD can only help through p.

**FINDING: the part that needs adding for Japanese is the turn detector, not
the VAD.** English and Spanish turns commit 4/4; `¿qué hora es?` ends in an
ASCII "?" and scores 0.95. All 3 Japanese turns score 0.2 and never commit.
With no spaces, `split()` counts one word, and the full-width `？` and `。`
are not among the `?.!` endings the scorer checks.

**FINDING: `synth_call`'s noise knob cannot model a false trigger during the
agent's turn.** At noise = 0.5 over 200 calls, 622 of 1,200 lead-in frames
become speech, and 0 of 22,000 trailing frames do. Noise is applied only
before the user speaks.

Structure: `frames_for()` builds a turn; `false_barge_ins()` injects
triggers after the commit and counts `barge_ins`; `debounce()` is the
minimum-speech filter.
"""

from __future__ import annotations

import random

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "03-realtime-voice-assistant"
# (language, tokens, joiner): one token is 320 ms of speech, as in synth_call
TURNS = [
    ("en", "what is the weather in tokyo tomorrow".split(), " "),
    ("en", "what time is it?".split(), " "),
    ("es", "qué tiempo hará mañana en tokio".split(), " "),
    ("es", "¿qué hora es?".split(), " "),
    ("ja", ["明日", "の", "東京", "の", "天気", "は"], ""),
    ("ja", ["明日", "の", "東京", "の", "天気", "は", "？"], ""),
    ("ja", ["今", "何", "時", "です", "か", "。"], ""),
]
RATES = (0.001, 0.01, 0.05)     # per-frame false-trigger probability during the agent's turn
SESSIONS = 400
MIN_SPEECH = 13                  # Silero get_speech_timestamps min_speech_duration_ms=250 -> 13 frames


def frames_for(ref, tokens, joiner):
    frames, t, said = [ref.Frame(20 * i, False) for i in range(6)], 120, []
    for tok in tokens:
        said.append(tok)
        frames += [ref.Frame(t + 20 * j, True, joiner.join(said)) for j in range(16)]
        t += 320
    return frames + [ref.Frame(t + 20 * i, False, joiner.join(said)) for i in range(110)]


def debounce(flags, k):
    """Keep a speech run only if it lasts k frames (a minimum speech duration)."""
    out, run = [False] * len(flags), 0
    for i, f in enumerate(flags):
        run = run + 1 if f else 0
        if run >= k:
            out[i - k + 1: i + 1] = [True] * k
    return out


def false_barge_ins(ref, rate, k, seed):
    rng, hits = random.Random(seed), 0
    base = frames_for(ref, *TURNS[0][1:])
    commit = ref.run_session(base, use_tool=False).turn_complete_ms
    tail = [i for i, f in enumerate(base) if f.t_ms > commit]
    for _ in range(SESSIONS):
        flags = debounce([rng.random() < rate for _ in tail], k)
        frames = list(base)
        for i, flag in zip(tail, flags):
            frames[i] = ref.Frame(base[i].t_ms, flag, base[i].partial)
        hits += ref.run_session(frames, use_tool=False, barge_in_at_ms=commit).barge_ins > 0
    return hits / SESSIONS, len(tail)


def language_turns(ref):
    turns = []
    for lang, toks, joiner in TURNS:
        m = ref.run_session(frames_for(ref, toks, joiner), use_tool=False)
        turns.append((lang, joiner.join(toks), round(ref.turn_completion_score(joiner.join(toks)), 2),
                      bool(m.turn_complete_ms)))
    return turns


def rate_row(ref, p):
    raw, n = false_barge_ins(ref, p, 1, seed=7)
    return {"raw": raw, "debounced": false_barge_ins(ref, p, MIN_SPEECH, seed=7)[0],
            "predicted": round(1 - (1 - p) ** n, 3), "frames": n}


def noise_knob(ref):
    random.seed(5)
    noisy = [ref.synth_call(" ".join(TURNS[0][1]), noise=0.5) for _ in range(200)]
    return sum(f.is_speech for c in noisy for f in c[:6]), sum(f.is_speech for c in noisy for f in c[-110:])


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    turns = language_turns(ref)
    lead, tail = noise_knob(ref)
    return {"turns": turns, "rates": {p: rate_row(ref, p) for p in RATES},
            "commits": {lang: sum(c for x, _, _, c in turns if x == lang) for lang in ("en", "es", "ja")},
            "ja_scores": sorted({sc for x, _, sc, _ in turns if x == "ja"}), "noise_lead": lead, "noise_tail": tail}


def verify(result):
    r = result["rates"]
    commits, ja_scores = result["commits"], set(result["ja_scores"])
    raw = [r[p]["raw"] for p in RATES]
    gap = max(abs(r[p]["raw"] - r[p]["predicted"]) for p in RATES)
    return [
        practice.Check(
            "ANSWER: a per-frame rate p cancels 1 - (1 - p)^85 of turns; a 250 ms minimum speech removes them",
            (raw, [r[p]["debounced"] for p in RATES], gap < 0.02, r[0.01]["frames"])
            == ([0.0975, 0.58, 0.9975], [0.0] * 3, True, 85),
            f"false barge-in per turn at p = {list(RATES)}: measured {raw}, predicted "
            f"{[r[p]['predicted'] for p in RATES]}, debounced {[r[p]['debounced'] for p in RATES]}",
        ),
        practice.Check(
            "FINDING: the part that needs adding for Japanese is the turn detector, not the VAD",
            (commits, ja_scores) == ({"en": 2, "es": 2, "ja": 0}, {0.2}),
            f"turns committed per language {commits}; scores {[(x, t, sc) for x, t, sc, _ in result['turns']]}",
        ),
        practice.Check(
            "FINDING: synth_call's noise knob cannot model a false trigger during the agent's turn",
            (result["noise_lead"], result["noise_tail"]) == (622, 0),
            f"noise=0.5 over 200 calls: {result['noise_lead']}/1200 lead-in frames vs "
            f"{result['noise_tail']}/22000 trailing frames flagged as speech",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
