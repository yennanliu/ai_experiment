<!-- generated:start -->
# 06-speech-and-audio / 07-text-to-speech

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/06-speech-and-audio/07-text-to-speech/) · upstream spec
`phases/06-speech-and-audio/07-text-to-speech/docs/en.md`

```bash
uv run demo practice run 07-text-to-speech --ex 1
uv run demo explain 07-text-to-speech --ex 1
uv run pytest demos/phases/06-speech-and-audio/07-text-to-speech
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run `code/main.py`. Builds a phoneme dictionary from a toy vocab, estimates duration pe… | code | T0 | `ex01_the_number_is_never_spoken.py` |
| 2 | Medium. Install Kokoro, synthesize the same sentence at voice `af_bella` and `am_adam`. Compa… | code | T0 | `ex02_the_duration_model_has_no_voice_input.py` |
| 3 | Hard. Record a 5-second reference clip of yourself. Use F5-TTS to clone it. Report SECS betwe… | code | T0 | `ex03_secs_is_a_property_of_the_encoder.py` |
<!-- generated:end -->

## Answers

`code/main.py` is a front end without a back end: a grapheme-to-phoneme table, a
duration table, and a frame schedule. All three exercises ask it for something it
does not have — a text normaliser, a voice, and a speaker encoder — and in each
case the missing piece is the one the exercise is actually about.

All three run at **T0**; the lesson's own code imports `math` and `random`.

### 1 — the number is never spoken

42 phonemes → 210 mel frames → **2.625 s** at a 12.5 ms hop, 63,000 samples at
24 kHz, a rate of 16.0 phonemes per second. Three things in that output disagree
with the line above it.

**FINDING: the one character it cannot phonemize is the number.** `phonemize`
tries 3-, 2- and 1-character keys and, on a miss, does `i += 1` with no output.
The only miss in "Please remind me to water the plants at 6 pm." is **`6`** — so
the demo synthesises "at pm". Text normalisation is the first stage of every real
TTS front end; here its absence is a skip.

**FINDING: six of the 42 duration entries can never be reached.**
`AA, AE, AY, OW, OY, ZH`. Two of them because `G2P` sends their own graphemes
elsewhere: `"ay" → EY`, `"ow" → AW`. The table reads as a phone inventory and is
a superset of one.

**FINDING: the jitter does not jitter.**
`int(round(base * uniform(-0.1, 0.1)))` needs `base · 0.1` to *exceed* 0.5, since
Python rounds a half to even — so every entry of 5 frames or fewer (**18 of 42**,
including every stop and every glide) is exactly deterministic. Only **14** of
this sentence's 42 phonemes can move at all, by at most one frame.

**ANSWER: at the shipped seed, two durations move and they cancel.** Seed 42
moves 2 of 42 phonemes, `+1` and `−1`, so the total lands on 210 — the no-jitter
number. Across 200 seeds the total spans **205–215**, sd **1.89 frames**, 0.9% of
the sentence.

### 2 — the duration model has no voice input

`kokoro`, `torch`, `soundfile` and `sounddevice` are all absent, so nothing here
synthesises. What can still be settled is whether the comparison is one this
lesson can express.

```text
duration(phones, jitter, seed)
mel_schedule(phones, durs, hop_ms)
```

**ANSWER: no voice argument, and no voice-shaped name in either body.** `af_bella`
and `am_adam` produce the same 210 frames, byte for byte, so the duration
comparison returns exactly **0 ms**.

**MECHANISM: that is the wrong half to omit.** In a FastSpeech-style model the
acoustic decoder carries timbre and the duration predictor carries rhythm — and
rhythm is the half a stopwatch measures. This lesson models exactly the component
the exercise asks about, and leaves out its only input.

| knob | total for the sentence |
|---|---|
| `jitter = 0` | 210 frames, fixed |
| `jitter = 0.1`, 200 seeds | 205–215 frames, **4.8%** end to end |
| `jitter = 0.3`, 200 seeds | 191–228 frames |

**FINDING: the hop the schedule assumes is 17% longer than a 24 kHz vocoder's.**
Step 4 converts at 300 samples per frame (12.5 ms at 24 kHz). Kokoro, F5-TTS and
HiFi-GAN at 24 kHz hop by **256** samples — 10.67 ms. The same 210 frames are
**2.240 s** of audio, not the **2.625 s** printed: **17.2% long**, larger than
any plausible gap between two voices, and it would swamp the comparison even if
the comparison could be made.

### 3 — SECS is a property of the encoder, not of the clone

`f5_tts`, `kokoro`, `torch`, `soundfile` and `sounddevice` are absent, there is no
microphone, and the reference tree ships no audio file. But SECS is
*speaker-encoder cosine similarity*, and this phase already has a speaker encoder
— Lesson 06's `embed_mfcc_stats` and `cosine`. Running it over a reference voice,
four other voices, and a graded family of imperfect clones is the `DESIGN D11`
scaled-down version.

**ANSWER: SECS has no zero.** Two utterances of the same voice average **0.9958**;
two of different voices average **0.6468**. The whole meaningful range is
`[0.65, 1.00]`, so a bare "SECS = 0.78" is not 78% of anything — it sits at the
top of the different-speaker distribution.

**ANSWER: a high-looking SECS is a rejection.** The equal-error threshold on this
encoder is **0.9909**.

| clone error | SECS | accepted at the EER threshold |
|---:|---:|---|
| +2% | 0.9939 | yes |
| **+5%** | **0.9856** | **no** |
| +10% | 0.9412 | no |
| +25% | 0.7732 | no |
| +50% | 0.6238 | no — the different-speaker mean |

A clone 5% off in its formants is reported at 98.6% similarity and rejected as a
different speaker by the same encoder. Without the threshold, and without the two
distributions the threshold comes from, the number cannot be read in either
direction.

**FINDING: one coefficient of the encoder moves the floor by 0.16.** Dropping MFCC
`c0` takes the different-speaker mean from **0.6468 to 0.4854** while the
same-speaker mean barely moves (0.9958 → 0.9944). The same +10% clone scores
**0.9412 or 0.9123** depending on a choice inside the measuring instrument.
Published SECS numbers use Resemblyzer, WavLM or ECAPA embeddings — three
different scales, not comparable to each other either.

A complete SECS report is four numbers, not one: the encoder, the same-speaker
distribution, the different-speaker distribution, and the score.
