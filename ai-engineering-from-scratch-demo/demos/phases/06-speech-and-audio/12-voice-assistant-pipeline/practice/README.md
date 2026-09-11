<!-- generated:start -->
# 06-speech-and-audio / 12-voice-assistant-pipeline

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/06-speech-and-audio/12-voice-assistant-pipeline/) · upstream spec
`phases/06-speech-and-audio/12-voice-assistant-pipeline/docs/en.md`

```bash
uv run demo practice run 12-voice-assistant-pipeline --ex 1
uv run demo explain 12-voice-assistant-pipeline --ex 1
uv run pytest demos/phases/06-speech-and-audio/12-voice-assistant-pipeline
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run `code/main.py`. It simulates one full turn end-to-end with stub modules and prints… | code | T1 | `ex01_the_largest_row_of_the_budget_is_a_constant.py` |
| 2 | Medium. Replace the STT stub with a real Whisper model on a pre-recorded `.wav`. Measure WER… | code | T0 | `ex02_the_stub_returns_one_sentence_for_every_input.py` |
| 3 | Hard. Add tool calling: implement `get_weather` (any API) and `set_timer`. Route the LLM thro… | code | T0 | `ex03_the_reply_is_written_before_the_tool_runs.py` |
<!-- generated:end -->

## Answers

`code/main.py` is honest about being stubs — "No real models — replace each stub
with Silero VAD / Whisper / GPT-4o / Kokoro". What the three exercises turn up is
that the *numbers* the stubs produce are not stub versions of real numbers. The
latency budget is four constants, the WER is a literal compared with itself, and
the tool call is a substring match whose reply was written before the tool ran.

Exercise 1 runs at **T1** because it times the stubs; 2 and 3 are **T0**.

### 1 — the largest row of the budget is a constant

| row | source | nominal |
|---|---|---:|
| VAD + capture | `silent_ms`, the hang-over counter | **400.0 ms** |
| STT | `sleep(0.08 + samples/sr * 0.05)` | 166.0 ms |
| LLM + tool | `sleep(0.12)` | 120.0 ms |
| TTS TTFA | `sleep(0.10)` | 100.0 ms |
| | **total** | **786.0 ms** |

**ANSWER: the margin to the stated `< 800 ms` target is 14.0 ms — 1.8% of the
budget.** And `time.sleep` never undershoots: over five runs the same four rows
time at ~798 ms, an overshoot of ~12 ms that consumes about **88%** of the
margin. Whether the demo meets its own target is a scheduler question.

**FINDING: the biggest row, 50.9% of the total, is read off the fixture.**
`silent_ms` counts 20 ms per non-speech chunk after the trigger and breaks at 400.
`mic_generator`'s default mask ends in exactly **20** silent chunks, so it breaks
on the very last one. Trim the tail to 19 and the generator runs out first:

| trailing silent chunks | `silent_ms` at exit | printed budget |
|---:|---:|---:|
| 20 (the default) | **400** (break) | **786.0 ms** |
| 19 | **380** (exhausted) | **765.0 ms** |

**FINDING: the capture writes one more chunk than the mic yielded.** 85 chunks in,
**86** out. On the trigger the pre-roll ring already holds the current chunk — it
was appended at the top of the same iteration — and `buffered.extend(chunk)` adds
it again. 1700 ms of audio becomes 1720 ms, and the duplicated 20 ms is the speech
onset.

**CONTROL: the row named "LLM + tool" excludes the tool.** `t_llm` stops before
the `for call in response["tool_calls"]` loop, so `dispatch_tool`'s `sleep(0.01)`
sits outside every stage. Two more timings are dead: `t_capture` is printed in
Step 1 and never enters the budget, and `t_play` appears exactly **once** in
`main()` — its own assignment.

### 2 — the stub returns one sentence for every input, including silence

`whisper`, `torch`, `transformers`, `soundfile`, `sounddevice`, `silero_vad` and
`openai` are all absent, and scanning the whole reference checkout for `.wav`,
`.flac`, `.mp3`, `.ogg`, `.m4a` finds **0** files. Both measurements are decided
before a model is chosen.

**ANSWER: WER is 0.000 for every input.** `streaming_stt` uses `len(utterance)`
only to pick a sleep, then returns the literal `'set a timer for five minutes'`:

| input | transcript | WER |
|---|---|---:|
| the captured turn | `set a timer for five minutes` | 0.000 |
| pure silence | `set a timer for five minutes` | 0.000 |
| white noise | `set a timer for five minutes` | 0.000 |
| an empty buffer | `set a timer for five minutes` | 0.000 |

Swapping in a real Whisper can only make this worse — the shipped value is the
floor.

**FINDING: silence produces a full sentence.** The doc's own third failure mode is
*"Silence hallucination. Whisper outputs 'Thanks for watching' on the silent
warm-up frames. Always VAD-gate."* The stub does exactly that, on an all-zero
buffer, unconditionally — it is the only behaviour it has.

**FINDING: `streaming_stt` does not stream.** `main()` calls it once, after the
capture loop returns, on the whole buffer, and gets one final string — no
partials, which is what the doc's Step 3 says streaming STT is for.

| arrangement | VAD | STT | LLM | TTS | total |
|---|---:|---:|---:|---:|---:|
| as shipped, one call after capture | 400.0 | **166.0** | 120.0 | 100.0 | **786.0** |
| streaming, only the tail left to pay | 400.0 | **81.0** | 120.0 | 100.0 | **701.0** |

**85.0 ms, 10.8% of the budget**, is the cost of the architecture the function is
named after and does not have.

**MECHANISM: the fixed 80 ms dominates only at this length** — **48.2%** of the
STT row at the demo's 1.72 s turn, 13.8% at 10 s, 5.1% at 30 s. The figure the
demo prints is mostly its own constant.

### 3 — the reply is written before the tool runs

Eight probes, each with an intended tool and an intended duration. `openai`,
`anthropic`, `torch` and `requests` are all absent, so "any API" is a local
station table.

| router | right tool | right arguments | distinct replies |
|---|---:|---:|---:|
| shipped `llm_with_tools` | **5 / 8** | **2 / 8** | **2** |
| routed through real tools | **8 / 8** | **8 / 8** | **7** |

**ANSWER: the utterance the exercise names is the one the router was fitted to.**
`"set a 5 minute timer"` fires `set_timer({'seconds': 300})` and answers "Sure,
setting a 5 minute timer." One probe out of eight.

**FINDING: no weather branch, a constant duration, and a call on a
cancellation.** `llm_with_tools` is `if "timer" in transcript` and nothing else:

| utterance | shipped | intended |
|---|---|---|
| `set a ten minute timer` | `set_timer(300)` | `set_timer(600)` |
| `give me a two hour timer` | `set_timer(300)` | `set_timer(7200)` |
| `what is the weather in taipei` | *no tool*, `'OK.'` | `get_weather('taipei')` |
| `cancel the timer` | **`set_timer(300)`** | *no tool* |

`dispatch_tool('get_weather')` answers with the single key `['ok']` — no error
string for a model to read and no retry, which is the failure the doc's own
Pitfalls section asks you to handle.

**MECHANISM: the reply is one of two literals, chosen before the tool runs.**
`llm_with_tools` picks `response["text"]`; `main()` dispatches afterwards and
never feeds the result back. The doc's Step 4 pseudocode has
`continue_streaming(result)` for exactly this. As shipped, a dispatch returning
`{'ok': False}` still leaves the assistant saying "Sure, setting a 5 minute
timer."

**ANSWER: routing through real tools scores 8/8 on both** — a duration parser, a
`get_weather` over a station table, and a reply built from the result. It draws
**7** distinct replies against the shipped 2, and it makes a failure audible: an
unknown location answers *"Sorry, that did not work: no station for 'mars'."*
