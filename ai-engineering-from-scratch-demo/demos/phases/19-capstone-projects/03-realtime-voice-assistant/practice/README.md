<!-- generated:start -->
# 19-capstone-projects / 03-realtime-voice-assistant

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/19-capstone-projects/03-realtime-voice-assistant/) · upstream spec
`phases/19-capstone-projects/03-realtime-voice-assistant/docs/en.md`

```bash
uv run demo practice run 03-realtime-voice-assistant --ex 1
uv run demo explain 03-realtime-voice-assistant --ex 1
uv run pytest demos/phases/19-capstone-projects/03-realtime-voice-assistant
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Swap Deepgram Nova-3 for faster-whisper v3 turbo on a g5.xlarge. Measure the latency and WER… | code | T0 | `ex01_any_real_asr_lag_drops_the_last_word_for_14pct_wer_and_1s_whisper_chunks_never_commit.py` |
| 2 | Add an interruption-arbitration policy: what does the agent do when the user barges in during… | code | T0 | `ex02_the_scheduler_cannot_hear_a_barge_in_during_a_tool_call_and_finish_tool_then_stop_saves_420ms.py` |
| 3 | Run an adversarial turn-detector test: give the user long pauses mid-sentence. Tune the VAD s… | code | T0 | `ex03_within_900ms_the_best_tuning_still_cuts_off_32_of_60_paused_sentences.py` |
| 4 | Deploy the same agent on PSTN via Twilio. Compare PSTN first-audio-out to WebRTC. Explain the… | code | T0 | `ex04_the_lesson_reports_320ms_on_both_paths_while_pstn_is_heard_at_1426ms_and_webrtc_at_928ms.py` |
| 5 | Add voice activity detection for non-English languages (Japanese, Spanish). Measure the Siler… | code | T0 | `ex05_a_1pct_per_frame_false_trigger_cancels_58pct_of_turns_and_japanese_turns_never_commit.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's `code/main.py`. It is a frame-by-frame
scheduler over simulated 20 ms frames. A VAD flag and an ASR partial are
given on each frame. A word-count "turn detector" commits after 500 ms of
silence at score 0.6. Fixed delays stand in for the LLM (140 ms), the TTS
(180 ms) and the weather tool (420 ms). No real ASR, network, Twilio number
or Silero weights are reachable offline. Each exercise therefore changes the
input the scheduler sees and measures what it does. External facts were read
on 2026-09-29 from the faster-whisper README, Twilio's Media Streams docs,
and Silero's README and `utils_vad.py`.

### 1 — any real ASR lag drops the last word (14% WER), and 1 s whisper chunks never commit a turn

**Measured at the lesson's scheduler, the latency gap is 520 ms and the WER
gap is total.** If the transcript is read at commit, Nova-3's 280 ms partials
are complete by the 480 ms commit point: 800 ms from user stop to first
audio. A 1 s faster-whisper chunk loop cannot commit before 1000 ms: 1320 ms.
The shipped scheduler commits 0 of 3 turns on whisper.

| ASR partial lag | 7 words | 6 words | 9 words |
|---|---|---|---|
| oracle (`synth_call`, -320 ms) | WER 0 | WER 0 | WER 0 |
| ideal streaming (0 ms) | WER 14.3% | never commits | WER 11.1% |
| Nova-3 (280 ms) | WER 14.3% | never commits | WER 11.1% |
| whisper 1 s chunks | never commits | never commits | never commits |

CPU-vs-GPU matters in two places. The first is the final decode, which must
fit inside the 480 ms silence window. At the README's speeds, re-decoding
the utterance hides up to 5.94 s of speech on the GPU (large-v2 fp16) and
2.38 s on the CPU (small fp32). The second is load. The GPU with batch 8
sustains 45.9 real-time streams and 8 CPU threads sustain 5.0, both short of
the lesson's 50 calls. Those figures are for an RTX 3070 Ti and an i7, not
a g5.xlarge.

**The scheduler drops the last word of any ASR that is not an oracle.** It
reads `f.partial` only on speech frames. A real ASR reports the last word
after it is spoken, on a silence frame, so the word is never read. The same
runs read at the commit frame score WER 0. `latency_ms()` is 320 ms on every
committed run whatever the ASR, because its clock starts at the VAD commit.

### 2 — the scheduler cannot hear a barge-in during a tool call, and finish-tool-then-stop saves 420 ms

**Finish-tool-then-stop wins.** The test is 40 barge-ins, 20-400 ms into the
420 ms tool call, lasting 300 ms or 600 ms. The timings are measured from
the reference; filler audio (1 s) and answer audio (3 s) are assumed.

| policy | talk-over | follow-up latency | correction latency | stale answer |
|---|---:|---:|---:|---|
| hard cancel | 0 ms | 1220 ms | 1220 ms | no |
| finish-tool-then-stop | 0 ms | 800 ms | 1220 ms | no |
| queue next turn | 390 ms | 3880 ms | 4300 ms | yes |

Hard cancel throws the tool result away, so a follow-up has to re-run the
tool. It is the right choice only for cheap, idempotent tools.

**The shipped scheduler cannot hear a barge-in during a tool call at all.**
It checks for barge-in only in SPEAKING and THINKING, and the tool runs in
`State.TOOL`. It hears 33 of the 40, each exactly when the tool returns. The
other 7 are short "no, wait"s that end before the tool returns, and for
those the stale answer is spoken anyway. The filler plays over the user in
30 of 40. The lesson's TypeScript port runs the tool inside THINKING, so it
would hear the barge-in at once.

### 3 — within 900 ms the best tuning still cuts off 32 of 60 paused sentences

**The best setting is S = 500 ms silence with score threshold 0.55, and it
still cuts off 32 of 60 (53%).** The test set is 60 seeded utterances of
3-10 words, each with one 300-1600 ms pause. Both thresholds are literals
inside `run_session`, so they were tuned by wrapping the scorer. User stop
to first audio is exactly S + 300 ms, so only S <= 600 fits in 900 ms. Only
T <= 0.55 answers every utterance. The scorer counts words and nothing else,
so any pause longer than S after 3 or more words cuts the user off.

| (S, T) | false cutoffs | never answered | worst latency |
|---|---:|---:|---:|
| (500, 0.55), best | 32 | 0 | 800 ms |
| (500, 0.6), shipped | 7 | 21 | 800 ms |
| (1500, 0.55) | 5 | 0 | 1800 ms |
| any S, 0.95 | 0 | 60 | none |

The shipped thresholds look better only because they never answer 21
turns. Every 3-5-word turn scores 0.55, since the synthetic partials carry
no punctuation. The lesson's clean call reports 740 ms, but measured from
user stop it is 1220 ms. With a tool no setting meets 900 ms. Without one
the floor is 800 ms, which is not "under 800ms".

### 4 — the lesson reports 320 ms on both paths while PSTN is heard at 1426 ms and WebRTC at 928 ms

**Under stated network assumptions, PSTN first audio is 1426 ms and WebRTC is
928 ms. Of the 498 ms gap, 378 ms is the jitter buffer (both ways) and 120 ms
is the longer path.** Twilio Media Streams carry 8 kHz mu-law as base64 over
a WebSocket, which runs on TCP. At 3% loss, UDP simply loses 3.1% of packets
(concealed) and needs a 24 ms p99 buffer. TCP loses nothing, but its
retransmits stall 17.5% of packets by more than 100 ms, and the buffer must
hold 213 ms.

On the codec: G.711 mu-law is 160 bytes (216 base64 characters) per 20 ms
frame with 37.5 dB SNR. It drops everything above 4 kHz. That is only 2.8%
of the test signal's energy, but it is the whole 4-8 kHz fricative band.
Opus on WebRTC keeps it. The lesson's `latency_ms()` reads 320 ms on both
paths: codec, network and buffers are all outside it.

### 5 — a 1% per-frame false trigger cancels 58% of turns, and Japanese turns never commit

**There is no language-specific Silero fine-tune to compare against.** Silero
says v5 was "trained on huge corpora that include over 6000 languages" and
lists no fine-tunes. Without the weights and ja/es audio offline, what was
measured is what a given false-trigger rate costs. With barge-in armed over
the agent's 85-frame turn:

| per-frame false trigger | turns falsely cancelled | 1 - (1 - p)^85 | with 250 ms minimum speech |
|---:|---:|---:|---:|
| 0.1% | 9.75% | 8.2% | 0 |
| 1% | 58.0% | 57.4% | 0 |
| 5% | 99.75% | 98.7% | 0 |

The 250 ms minimum is Silero's own `min_speech_duration_ms` default. It
removes scattered triggers but not sustained background speech.

**What Japanese needs is a turn detector, not a VAD.** English and Spanish
turns commit 4/4. All 3 Japanese turns score 0.2 and never commit.
`split()` sees one word, and the full-width `？` and `。` are not among the
`?.!` endings the scorer checks. The lesson's `synth_call(noise=...)` cannot
test any of this: at noise 0.5 it flags 622 of 1,200 lead-in frames and 0 of
22,000 trailing frames.
