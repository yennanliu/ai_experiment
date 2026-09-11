<!-- generated:start -->
# 06-speech-and-audio / 14-voice-activity-detection-turn-taking

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/06-speech-and-audio/14-voice-activity-detection-turn-taking/) · upstream spec
`phases/06-speech-and-audio/14-voice-activity-detection-turn-taking/docs/en.md`

```bash
uv run demo practice run 14-voice-activity-detection-turn-taking --ex 1
uv run demo explain 14-voice-activity-detection-turn-taking --ex 1
uv run pytest demos/phases/06-speech-and-audio/14-voice-activity-detection-turn-taking
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run `code/main.py`. It simulates a speech + silence + speech + coughs sequence and test… | code | T0 | `ex01_both_tiers_agree_on_every_chunk.py` |
| 2 | Medium. Install `silero-vad`, process a 5-min recording, tune threshold to minimize both firs… | code | T0 | `ex02_the_threshold_has_three_settings.py` |
| 3 | Hard. Build a mini turn-detector: Silero VAD + a 3-layer MLP on the last 10 words' embeddings… | code | T0 | `ex03_you_cannot_beat_one_by_ten_percent.py` |
<!-- generated:end -->

## Answers

`code/main.py` prints two event lists under headings that say they differ —
"many false positives on cough" and "rejects cough" — and the two lists are the
same four events. That is the shape of all three exercises here: the lesson
attributes to its model a behaviour that belongs to the state machine underneath
it, and the state machine has a counter that does the opposite of what the doc
says it does.

All three run at **T0**; the lesson's own code imports `math` and `random`.

### 1 — both tiers agree on every chunk

```text
t= 440 ms START    t=1480 ms END    t=2040 ms START    t=2800 ms END
```

**ANSWER: that list is printed twice.** The energy gate and the Silero-style VAD
agree on **151 of 151 chunks**. There are no false positives in the list headed
"many false positives on cough", and nothing was rejected in the list headed
"rejects cough".

**FINDING: the Silero-style tier calls the cough speech.** It scores **0.55**,
above the 0.5 the caller compares against; the energy gate reads it at **−9.1
dBFS**, above the −40 dBFS threshold. What stops a turn from starting is
`TurnDetector`'s `min_speech_ms = 250` against one 20 ms chunk — and **both arms
run the same state machine**.

**MECHANISM: the 0.92 branch is unreachable.**

| test | value | verdict |
|---|---|---|
| `duration < 0.03` | every chunk is **0.0200 s** | always true |
| `max − min > 0.6` | narrowest speech **0.954**, cough **3.545** | always true when loud |

So `not transient` is false wherever `rms > 0.08` succeeds, and the whole stream
takes exactly two probabilities: **{0.02, 0.55}**.

**FINDING: two tiers, not three.** `main()` compares `energy_vad` and
`fake_silero_vad`; the doc's third tier is a semantic turn detector and no such
name exists in the module. The "coughs" are `("cough", 1)` — 20 ms of a 3020 ms
stream.

**CONTROL:** `fake_silero_vad(chunk, prev_state, threshold=0.5)` never reads
`prev_state` or `threshold`; the caller re-implements the comparison.

### 2 — the threshold has three settings, and the doc recommends two of them

`silero_vad`, `torch`, `torchaudio` and `onnxruntime` are all absent, and the
reference tree ships no recording of any length. Sweeping the threshold in
hundredths over `[0, 1]` — **101 settings** — gives **3** distinct event lists:

| threshold band | turn events | precision | recall | F1 |
|---|---|---:|---:|---:|
| 0.00 – 0.02 | one START, never an END | 0.000 | 0.000 | **0.000** |
| **0.03 – 0.55** | START, END, START, END | 1.000 | 1.000 | **1.000** |
| 0.56 – 1.00 | none at all | 0.000 | 0.000 | **0.000** |

**FINDING: the doc's two thresholds are one setting.** `0.5` (default) and `0.3`
("sensitive") both land in the middle band — identical events, identical
clipping. The dial has one usable position and two ways to break, so "minimize
both" has no interior to search. `threshold` is inert in `fake_silero_vad`
regardless.

**FINDING: first-word clipping is not on this dial.** Speech begins at **200 ms**
and START fires at **440 ms** — **240 ms clipped**, the same at 0.3 and 0.5,
because the delay is `min_speech_ms = 250`. The parameter that exists to fix it,
`pre_roll_ms = 300`, is assigned in `__init__` and **never read in `update()`**.
The demo ships the pitfall its own doc names: *"No pre-roll buffer. First
200-300 ms of user audio lost."*

**CONTROL: the end-pointing delay is one chunk short of its setting.**
`silence_hangover_ms` is 500; the measured gap from speech offset to END is
**480 ms**, because `silence_ms` is compared after being incremented.

### 3 — you cannot beat 1.000 by ten percent

**ANSWER: on the stream the lesson ships, Silero-only already scores F1 1.000** —
two gold turn ends, two predicted, no false positives. "Beat Silero-only by 10%
F1" has no solution on the exercise's own fixture.

A fixture that contains the failures the doc names does have one: a clatter of
**15** isolated 20 ms transients between turns ("coughs or chair noise"), and a
turn split by a **640 ms** pause ("Hmm, let me think..."). 10880 ms, three
hand-labelled turn ends:

| arm | F1 | true pos | false pos |
|---|---:|---:|---:|
| Silero-only, as shipped | **0.750** | 3 | 2 |
| + count *consecutive* speech | **0.857** | 3 | 1 |

**ANSWER: +10.7 points, from one line.** Clearing `speech_ms` on a silent chunk
while idle is the whole change. The target the exercise sets for a 3-layer MLP is
met by a state-machine fix.

**MECHANISM: `min_speech_ms` is cumulative, not consecutive.** `update` never
clears `speech_ms` in the idle state, so 20 ms transients accumulate across any
amount of silence — **the 13th fires a START** whatever the spacing (3600 ms at
280 ms gaps, 9840 ms at 800 ms gaps). The doc describes this guard as rejecting
"speech shorter than 250 ms"; it rejects the first twelve coughs and accepts the
thirteenth. With the fix, 40 isolated transients fire nothing.

**CONTROL: the remaining error is the one an MLP is actually for.** The 640 ms
mid-turn pause exceeds the 500 ms hangover, so one turn is reported as two, and
no counter fixes it. A 700 ms hangover reaches F1 **1.000** here — but the window
is narrow: at 900 ms every predicted end falls outside the ±700 ms scoring collar
and F1 is **0.000**. And it cannot be built anyway:
`sentence_transformers`, `torch` and `silero_vad` are absent, and `synth_chunk`
returns Gaussian noise — there is no text in the module, so "the last 10 words'
embeddings" has no input.
