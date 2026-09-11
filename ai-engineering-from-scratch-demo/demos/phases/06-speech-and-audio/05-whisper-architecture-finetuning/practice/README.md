<!-- generated:start -->
# 06-speech-and-audio / 05-whisper-architecture-finetuning

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/06-speech-and-audio/05-whisper-architecture-finetuning/) · upstream spec
`phases/06-speech-and-audio/05-whisper-architecture-finetuning/docs/en.md`

```bash
uv run demo practice run 05-whisper-architecture-finetuning --ex 1
uv run demo explain 05-whisper-architecture-finetuning --ex 1
uv run pytest demos/phases/06-speech-and-audio/05-whisper-architecture-finetuning
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run `code/main.py`. It tokenizes a Whisper-style prompt, computes decoded shape budgets… | code | T0 | `ex01_the_frame_count_contradicts_the_line_below_it.py` |
| 2 | Medium. Install `faster-whisper`, transcribe a 10-minute podcast, compare WER against a human… | code | T0 | `ex02_the_schedule_costs_nineteen_points_of_wer.py` |
| 3 | Hard. Using HF `datasets`, pick a language Whisper struggles with (e.g., Urdu), fine-tune Med… | code | T0 | `ex03_two_hours_at_two_epochs_is_sixty_steps.py` |
<!-- generated:end -->

## Answers

This lesson is arithmetic, not audio — `code/main.py` imports `math` and uses it
once. So every exercise is checkable in full, and three of the numbers it prints
are wrong in ways the file itself contradicts four lines later.

All three run at **T0**; nothing here needs a GPU to be wrong.

### 1 — the frame count contradicts the line printed below it

**FINDING: `encoder_frames` is two frames short, at every duration.**

| clip | `encoder_frames` | centred front end |
|---:|---:|---:|
| 1 s | 98 | **100** |
| 10 s | 998 | **1000** |
| 30 s | **2998** | **3000** |

Step 2 prints 2998 and the next line says "Whisper zero-pads all inputs to 30 s →
**3000** frames". The line is right — Whisper's mel front end centres its frames,
so the count is `seconds · sr / hop` exactly. The function uses the un-centred
`1 + (samples − 400)//hop`. Two frames is 20 ms, and both numbers are on screen
at once.

**FINDING: Step 4's Turbo row overwrites the decoder with an encoder.**

| | encoder | decoder | embed | total | published |
|---|---:|---:|---:|---:|---:|
| as printed | 629.3 | **78.7** | 70.2 | **778.2 M** | 809 M |
| decoder counted as a decoder | 629.3 | **104.9** | 70.2 | **804.4 M** | 809 M |

`if name == "Turbo": dec = enc` charges four *encoder* blocks — self-attention and
MLP — for a stack that in a decoder also carries cross-attention. The hack costs
26.2 M parameters and moves the estimate from **3.8% low to 0.6% low**. The loop
computes the right value one line earlier and throws it away; the unhacked
Large-v3 row sits at 1538.7 M against 1550 M.

**CONTROL: two more things in the same function.** `transformer_params` takes
`n_heads` and never reads it. And its embedding term charges **3000** rows of
audio positional embedding (3.84 M at `d_model=1280`) where Step 2's own closing
line says the encoder sees **1500** tokens after the stride-2 convolution.

### 2 — the schedule costs nineteen points of WER before the model runs

`faster_whisper`, `ctranslate2`, `torch`, `transformers` and `soundfile` are all
absent, and there is no podcast and no human transcript anywhere in the reference
tree. What the transcription would be *wrapped in* is entirely present, and that
is where the answer is.

600 s of reference transcript, 1,500 timed words at 150 wpm, chunked on
`chunk_schedule(600, 30, 5)`, every chunk transcribed **exactly right**, scored
with Lesson 04's `wer` (this lesson ships no metric):

| merge | hypothesis words | WER |
|---|---:|---:|
| concatenate the 24 chunks | **1787** | **0.1913** |
| drop the first 5 s of every chunk after the first | 1500 | **0.0000** |

**MECHANISM: 715 s of audio for a 600 s clip.** 24 chunks of 30 s stepping by
`chunk_s − stride_s = 25` decode **19% more audio than exists**, and the extra
**115 s** is speech transcribed twice. The overlap is deliberate and right — it is
what stops words being cut at a boundary. The duplication is what nothing removes.

**MECHANISM: the schedule cannot undo itself.** `chunk_schedule` returns
`(start, end)` pairs and nothing else, so no caller can tell which part of a chunk
is the overlap without re-deriving `stride_s`. The fix is one number the return
value omits.

**FINDING: the `language="auto"` arm cannot be posed against this code.**
`build_prompt("auto")` raises `KeyError: 'auto'` — `LANG` holds `en, fr, ja` and
the builder indexes it directly. The doc's own Pitfalls section already states the
verdict the comparison is meant to reach: *"Whisper's auto LID mis-routes noisy
clips to Japanese or Welsh; force `language='en'` when you know."*

### 3 — two hours at two epochs is sixty optimizer steps

`datasets`, `transformers`, `peft` and `torch` are all absent, so the WER delta
cannot be measured — but the budget computes exactly, and the budget is the
finding.

Whisper pads every clip to one 30 s window, so **2 hours is 240 examples**. Two
epochs is 480 forward passes; at batch 8 that is **60 gradient updates** for a
3.1 M-parameter adapter. Published Whisper LoRA recipes run thousands.

**FINDING: `lora_params` undercounts by exactly one third.**

| Medium, `r=16`, `q_proj`+`v_proj` | |
|---|---:|
| `lora_params` as written | **3.146 M** |
| counting the decoder's cross-attention | **4.719 M** |
| ratio | exactly 3/2 |

It charges `n_layers * 2 * per_block` — one attention block per encoder layer and
one per decoder layer — but a Whisper decoder layer carries **two**, self and
cross.

**FINDING: Step 5 claims "100x+" and never prints the 242× it is.** Against
Medium's own printed total of 761.1 M, the adapter is **242×** smaller and
**0.41%** of the model; with the cross-attention adapters counted, 161×. Neither
number is in the output.

**MECHANISM: the same ratio is what the method buys, in bytes.** fp32 Adam keeps a
master copy and two moments — 12 bytes per trainable parameter:

| | optimizer state | shipped weights |
|---|---:|---:|
| full fine-tune of Medium | **9.1 GB** | 3.0 GB |
| LoRA `r=16` on `q_proj`,`v_proj` | **37.7 MB** | **12.6 MB** |

So the exercise's three numbers pull against each other. Two hours and two epochs
is a sample size chosen for a full fine-tune's *cost*, applied to a method whose
whole point is that it does not have that cost.

The full run is `peft.LoraConfig(r=16, target_modules=["q_proj","v_proj"])` over
`openai/whisper-medium` on `mozilla-foundation/common_voice_17_0` — about 3 GB of
weights, 2 hours of audio, and under ten minutes of A100 time for the 60 steps as
specified. Budget for 2,000 steps instead, and for a test set large enough to
resolve the delta (Lesson 04, Exercise 3).
