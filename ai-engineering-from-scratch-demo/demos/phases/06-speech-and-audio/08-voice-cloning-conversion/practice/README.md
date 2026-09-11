<!-- generated:start -->
# 06-speech-and-audio / 08-voice-cloning-conversion

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/06-speech-and-audio/08-voice-cloning-conversion/) · upstream spec
`phases/06-speech-and-audio/08-voice-cloning-conversion/docs/en.md`

```bash
uv run demo practice run 08-voice-cloning-conversion --ex 1
uv run demo explain 08-voice-cloning-conversion --ex 1
uv run pytest demos/phases/06-speech-and-audio/08-voice-cloning-conversion
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run `code/main.py`. Demonstrates the speaker-embedding swap by computing the cosine bet… | code | T0 | `ex01_step_three_recomputes_step_two.py` |
| 2 | Medium. Use OpenVoice v2 to clone your own voice. Measure SECS between reference and clone. M… | code | T0 | `ex02_one_dial_and_two_metrics_that_pull_apart.py` |
| 3 | Hard. Apply SilentCipher watermark to 20 clones, run them through 128 kbps MP3 encode+decode,… | code | T0 | `ex03_there_is_no_audio_to_encode.py` |
<!-- generated:end -->

## Answers

`code/main.py` calls itself "the lego-model of a cloning pipeline", and the
lego-model is honest about being one. What the three exercises turn up is that
the two *numbers* it prints — SECS and bit accuracy — are not measurements of
anything the lesson built: one is a re-reading of the mixing constant, and the
other is a subtraction.

All three run at **T0**; the lesson imports `hashlib`, `math` and `random`.

### 1 — Step 3 recomputes Step 2, byte for byte

**ANSWER: `wav_converted == wav_cloned`.** Step 3 calls
`extract_content(wav_bob_orig, contents)`, gets back the *exact dictionary key*
it started from, looks the original vector up by that key, and passes it to
`fake_tts(content, alice, 0.5)` — the call Step 2 already made. Nothing is
estimated and nothing is converted; the cosine Step 3 prints is Step 2's cosine.

**MECHANISM: the SECS is a function of `mix` and two norms.** `fake_tts` returns
`(1−mix)·content + mix·speaker`, with `‖speaker‖ = 1.000` and `‖content‖ = 2.189`:

| `mix` | SECS to the target | cosine to the content |
|---:|---:|---:|
| 0.10 | 0.1133 | 0.9987 |
| **0.50** (shipped) | **0.4620** | 0.9142 |
| 0.721 | **0.7800** | — |
| 1.00 | 1.0000 | 0.0631 |

**FINDING: the demo's own number is below the worst row of its own leaderboard.**
It prints `SECS = 0.462` and then quotes "production ECAPA-TDNN on real clones
lands SECS in 0.65 – 0.78". Matching the *top* row takes no model: `mix = 0.721`
gives exactly 0.78.

**CONTROL: the speaker probe has no threshold and no reject option.**
`extract_speaker` sorts three cosines and returns the largest, so "should stay
alice" holds for any `mix > 0` — bob scores **−0.0126**.

**CONTROL: the content channel is a hash.** `content_vector` is SHA-256 of the
text, so two different sentences score **−0.107** and deleting one character
scores **0.087**. `extract_content` is exact-match retrieval under another name,
which is why Step 3 could recover its input exactly.

### 2 — one dial, and the two metrics pull against each other on it

`openvoice`, `torch`, `whisper`, `transformers`, `soundfile` and `sounddevice`
are all absent, and there is no microphone and no audio file. Against *this*
module both numbers are computable exactly — and they are one number.

| `mix` | SECS | content cosine | right text retrieved from 51 candidates |
|---:|---:|---:|---|
| 0.10 | 0.1133 | 0.9987 | yes |
| 0.50 | 0.4620 | 0.9142 | yes |
| 0.70 | 0.7493 | 0.7082 | yes |
| **0.90** | **0.9726** | **0.2934** | **no** |
| 1.00 | 1.0000 | 0.0631 | no |

**ANSWER: there is no trade-off curve to explore, because there is no second
parameter.** Every row is `mix`, and a "better clone" is a larger number typed
into one call. At `mix = 0.9` this clone beats every row of the lesson's own
leaderboard while losing its content.

**FINDING: CER has no definition here.** `content_vector` is SHA-256 — one-way —
so there is no pre-image to decode and no characters to score. Retrieval is the
nearest measurable thing, and it is a **Bernoulli, not a rate**: correct up to
`mix ≈ 0.855`, wrong above, never partly right. The failure a CER exists to catch
— a clone that says something *slightly* different — is exactly the failure this
representation cannot have.

### 3 — there is no audio to encode

`silentcipher`, `audioseal`, `torch`, `lameenc`, `pydub` and `soundfile` are all
absent, and that is the smaller obstacle.

**ANSWER: a "clone" here is a 64-element vector.** `fake_tts` mixes two
64-dimensional embeddings. The result has no sample rate, no duration, and is
never converted to samples anywhere in the module — an MP3 encoder has nothing to
consume. The 32-bit payload is written into 64 numbers: **one bit per two
"samples"**.

**MECHANISM: 100% bit accuracy is structural.**
`detect_watermark(wave_original, wave_wm)` takes the original and subtracts it —
which is exactly what a distributor does not have. It is a difference, not a
detection, and it recovers the payload from an arbitrarily small perturbation:

| strength | bit accuracy over 20 clones |
|---:|---:|
| 0.003 (shipped) | 1.0000 |
| 1e-06 | 1.0000 |
| **1e-15** | **1.0000** |

**FINDING: the real margin is the quantisation step.** Once the difference is
taken away and real degradation applied:

| degradation | bit accuracy |
|---|---:|
| 16-bit PCM, strength 0.003 | 1.0000 |
| 16-bit PCM, strength 1e-05 | 0.9406 |
| 16-bit PCM, strength 1e-06 | 0.5766 |
| noise at 40 dB SNR | 0.9969 |
| noise at 20 dB SNR | 0.5875 |
| **mean removed from each residue class** | **0.5203** |

**ANSWER to the MP3 question: the payload is a DC offset.** `watermark` adds a
constant `±0.003` to every index sharing a residue mod 32. Removing the mean of
each residue class — the minimum any transform codec does to a constant — takes
the accuracy to **0.5203**, chance. A per-sample DC shift is not a signal a
perceptual codec preserves, so the scheme could not survive 128 kbps MP3 even if
there were audio to encode. The doc says as much in `watermark`'s own docstring —
"this demo just proves the encode/decode contract holds" — and the exercise asks
for the property the docstring disclaims.
