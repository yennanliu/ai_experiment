<!-- generated:start -->
# 06-speech-and-audio / 11-real-time-audio-processing

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/06-speech-and-audio/11-real-time-audio-processing/) · upstream spec
`phases/06-speech-and-audio/11-real-time-audio-processing/docs/en.md`

```bash
uv run demo practice run 11-real-time-audio-processing --ex 1
uv run demo explain 11-real-time-audio-processing --ex 1
uv run pytest demos/phases/06-speech-and-audio/11-real-time-audio-processing
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run `code/main.py`. Simulates a ring buffer + energy VAD; prints stage latencies for a… | code | T1 | `ex01_there_is_no_ring_buffer.py` |
| 2 | Medium. Using `sounddevice`, build a passthrough loop that processes your mic in 20 ms frames… | code | T1 | `ex02_the_gate_is_a_statement_about_microphone_gain.py` |
| 3 | Hard. Build a full duplex echo test with `aiortc`: browser → WebRTC → Python → WebRTC → brows… | code | T0 | `ex03_the_probe_repeats_every_millisecond.py` |
<!-- generated:end -->

## Answers

`code/main.py` is a latency simulator, and the number it exists to print —
**420 ms, `target: < 500 ms`** — is the one number in the file that does not
survive being checked against the table four lines below it. The other two
exercises take the two components the pipeline is built from, the VAD gate and
the timing probe, and ask what each can actually resolve.

Exercises 1 and 2 run at **T1** because they measure wall-clock time. Exercise 3
is **T0**.

### 1 — there is no ring buffer, and the stream is 1.9 seconds

The exercise makes three claims about the file. Only the VAD is there.

| claim | what the file does |
|---|---|
| "a ring buffer" | the substring `ring` never occurs; `buffered` is an unbounded list |
| "an energy VAD" | ✓ `rms_dbfs > −40` |
| "a fake 10-second stream" | **95 chunks × 20 ms = 1900 ms** |

**ANSWER: the stream is 5.3× short.** 10 s would be 500 chunks. Step 1's caption
says "1.5 s of user speech" over a 1.9 s stream, and the gate keeps **24,000
samples — exactly 1.500 s**.

**FINDING: nothing in the module is a ring buffer.** The doc's own Step 1 defines
a `RingBuffer` and sizes it at "32,000 samples at 16 kHz = 2 s"; no line imports
or reimplements it. At the 1.5 s this demo keeps, an unbounded list and a 2 s ring
behave identically — which is exactly why the omission is invisible here. At the
**10 s stream the exercise names**, a speech-only buffer is 160,000 samples,
**5.0× that capacity**, and a ring would have been dropping samples for eight of
those ten seconds.

**FINDING: three of the six stages are never timed, and they cost the target.**

| stage | Step 4 says | `main()` times it |
|---|---:|---|
| network in | 50–100 ms | no |
| VAD | 20–80 ms | no |
| STT stream | 100–300 ms | **yes** |
| LLM stream | 100–500 ms | **yes** |
| TTS TTFA | 100–300 ms | **yes** |
| network out | 50–100 ms | no |

Adding the table's own **minima** to the 405 ms measured gives **525 ms** —
already past the `target: < 500 ms` printed on the same line. Its maxima give
685 ms.

**FINDING: the TOTAL row is not the sum of its own column.** The six stages sum
to **420–1380 ms**; the row printed underneath them says **400–1400**. Both ends
are off by 20 ms, in opposite directions.

**CONTROL: the latency measured is the sleep.** Measured total is within a few percent of
the closed-form 405 ms. And `main()` calls `random.seed(0)` while every draw comes
from its own `random.Random(0)`, so the module-level seed changes no output.

### 2 — the gate is a statement about microphone gain

`sounddevice`, `pyaudio` and `soundfile` are all absent and there is no
microphone, so the mic is synthesised: four seconds of speech-like bursts of
0.12–0.35 s separated by the 0.03–0.22 s gaps ordinary speech has. Everything
else is the lesson's own `rms_dbfs` and `vad`, called once per 20 ms frame.

**ANSWER: printing the state at every frame prints a flicker.** Over 200 frames
the gate changes state **24 times, 6.0 per second** — against 25 in the reference
labelling, so it tracks the truth to within one transition. The display is
accurate and unusable.

**MECHANISM: the standard fix costs 40% of the lesson's own budget.**

| hangover | state changes |
|---:|---:|
| 0 ms | 24 |
| 100 ms | 4 |
| **200 ms** | **0** |

200 ms is **40%** of the `target: < 500 ms` the same lesson prints, and it is
added to every end-of-turn decision.

**FINDING: the threshold is absolute, so it describes the microphone.** The gate
fires on `rms_dbfs > −40` — an RMS of exactly **0.01**. With speech at amplitude
0.15 the noise floor reaches that at `20·log10(0.15/0.01)` = **23.52 dB SNR**:

| SNR | false accepts on silent frames |
|---:|---:|
| 40 dB | 0.2623 |
| 25 dB | 0.2623 |
| **20 dB** | **1.0000** |
| 10 dB | 1.0000 |

**MECHANISM: one and a half samples of 320 decide the frame.** Crossing 0.01 RMS
needs `320·(0.01/0.15)²` = **1.42** loud samples: measured, **1** loud sample
trips the gate on **22%** of draws and **4** on **83%**. That is why silent frames
false-accept at 0.26 even at 40 dB — every frame straddling a syllable edge is
99% silence and reads as speech.

### 3 — the probe repeats every millisecond

`aiortc`, `av`, `sounddevice` and `torch` are all absent and there is no browser,
so the transport cannot be built. The measurement can: a known delay of **187
samples (11.688 ms)** injected into a noisy return and recovered by fixed-window
cross-correlation.

**ANSWER: a clean return recovers it exactly, with either probe.** At 0 dB SNR
both a 20 ms 1 kHz burst and a 20 ms 300–3000 Hz sweep return 187 on **20/20**
trials, to a resolution of one sample — **0.0625 ms**.

**FINDING: every error the 1 kHz probe makes is a whole millisecond.**

| SNR | 1 kHz burst exact | sweep exact |
|---:|---:|---:|
| 0 dB | 20/20 | 20/20 |
| −6 dB | **15/20** | 20/20 |
| −10 dB | **9/20** | 20/20 |

All **16** of the tone's failures land within one sample of a multiple of 16
samples: `−16, −16, −1, +1, +16` at −6 dB and
`−48, −47, −16, −16, −16, +1, +1, +16, +16, +16, +31` at −10 dB. The sweep fails
**0 of 40** over the same returns.

**MECHANISM: the cause is arithmetic, not noise.** 1000 Hz at 16 kHz is exactly
**16 samples per cycle**, so the tone's autocorrelation one period out is
**0.9500** of its own peak — and the 5% it loses is the shortened overlap, not
the waveform. The sweep's is **−0.0598**. The argmax is choosing between
candidates that are identical by construction.

**FINDING: the probe resolves 3000× finer than the path it is measuring.** The
lesson's own figures put an echo path at `network in` 50–100 ms plus
`network out` 50–100 ms (Step 4) plus a 60–80 ms jitter buffer (The Concept):
**160–280 ms**, which is **2560× to 4480×** the probe's 0.0625 ms, on a transport
whose Opus frames quantise arrivals to 20 ms. A glass-to-glass figure quoted to
sub-millisecond precision is reporting the probe, not the path.
