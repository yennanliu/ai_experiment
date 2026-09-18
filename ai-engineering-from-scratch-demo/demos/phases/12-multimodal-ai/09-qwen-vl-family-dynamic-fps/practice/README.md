<!-- generated:start -->
# 12-multimodal-ai / 09-qwen-vl-family-dynamic-fps

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/12-multimodal-ai/09-qwen-vl-family-dynamic-fps/) · upstream spec
`phases/12-multimodal-ai/09-qwen-vl-family-dynamic-fps/docs/en.md`

```bash
uv run demo practice run 09-qwen-vl-family-dynamic-fps --ex 1
uv run demo explain 09-qwen-vl-family-dynamic-fps --ex 1
uv run pytest demos/phases/12-multimodal-ai/09-qwen-vl-family-dynamic-fps
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Compute M-RoPE rotations for a patch at (t=3, h=5, w=7) with hidden 48 (16 per band, base the… | code | T0 | `ex01_three_bands_one_schedule_so_the_axes_are_interchangeable.py` |
| 2 | A 10-minute security-camera recording at 1 FPS produces how many frames? At 384 resolution wi… | code | T0 | `ex02_the_sampler_would_never_have_chosen_one_fps.py` |
| 3 | Pick FPS for a 30-second tennis rally vs a 30-second recipe demo vs a 30-second UI-agent reco… | code | T0 | `ex03_two_of_the_three_get_the_same_answer.py` |
| 4 | Qwen2.5-VL drops the Q-Former entirely. Why does a simple MLP work in 2025 but not in 2023? (… | explain | T0 | prose, below |
| 5 | Parse three Qwen2.5-VL JSON tool-call outputs into Python dicts. What fails for malformed JSO… | code | T0 | `ex05_the_fallback_recovers_none_of_the_failure_it_is_for.py` |
<!-- generated:end -->

## Answers

Four code solutions and one prose one. Each code solution runs the lesson's own
`main.py` rather than reimplementing it, and three of the four find the same
thing from different angles: the sampler's budget arithmetic is real, and it
almost never decides anything.

### 1 — three bands, one schedule, so the axes are interchangeable

**ANSWER: the first three pairs of each band at (t, h, w) = (3, 5, 7):**

| band | pair 0 | pair 1 | pair 2 |
|---|---:|---:|---:|
| temporal (t = 3) | 3.0 | 0.948683 | 0.3 |
| height (h = 5) | 5.0 | 1.581139 | 0.5 |
| width (w = 7) | 7.0 | 2.213594 | 0.7 |

Rotating `e0` through `mrope_rotate` and reading the angle back with `atan2`
gives **3.0**, matching the temporal band's first angle — the two functions
recompute theta independently and agree.

**FINDING: all three bands share one theta schedule.** Every band is
16-dimensional, so θ_i = 10000^(−i/8) in all three and an angle is just
`position × θ_i`. A patch **three frames later** and a patch **three rows down**
produce the identical triple (3.0, 0.948683, 0.3); only which slice of the
hidden state they land in distinguishes them.

**FINDING: the eight pairs span wavelengths from 6.28 to 19,869 positions** — a
**3,162×** range. Pair 0 turns fully every 2π positions, so a 27-wide patch grid
wraps it **4.3** times; pair 7's wavelength is longer than any sequence. Three
pairs is the fast end of a range whose other end never turns.

**FINDING: `MRoPEConfig.hidden` is never read.** `hidden=999` with the same
16+16+16 bands gives byte-identical angles, so a config whose bands do not sum to
its hidden size is accepted silently.

### 2 — the sampler would never have chosen one FPS

**ANSWER: 600 frames, 48,600 tokens, and no — 1.48× a 32,768 context.**
384/14 = 27, pooled by 3 → a 9×9 grid → 81 tokens a frame.

**FINDING: the lesson's own sampler never offers 1 FPS here.** `fps_max` for
this clip is **0.674**, so the low-motion ladder returns **0.5**: 300 frames,
**24,300** tokens, 74.2% of the context. The configuration the exercise asks
about is one the lesson's logic rejects.

**FINDING: the ladder discards a quarter of the budget it just computed.**

| demo clip | budget used |
|---|---:|
| tennis 30 s | 59.3% |
| recipe 30 s | **29.7%** |
| security 600 s | 74.2% |
| UI replay 60 s | 59.3% |

Never above three quarters, never the same twice.

**FINDING: for short clips the budget does not bind at all.** The top rung stops
fitting only beyond **50.6 s** (high), **101.1 s** (medium), **404.5 s** (low).

### 3 — two of the three get the same answer

| clip | motion | FPS | frames | tokens |
|---|---|---:|---:|---:|
| tennis rally | high | **8** | 240 | 19,440 |
| recipe demo | medium | **4** | 120 | 9,720 |
| UI replay | medium | **4** | 120 | 9,720 |

**ANSWER: 8 / 4 / 4** — and the lesson's logic cannot separate the second from
the third.

**FINDING: at 30 seconds the budget binds for none of them.** `fps_max` is
**13.48**, above every ladder top, so all three answers are ladder tops. The
motion label is the entire decision and nothing in the lesson computes it —
relabelling the UI clip "high" is a **2.0×** cost swing from a string.

**FINDING: `frame_times` is exactly uniform, which is the wrong shape for the UI
case.** One distinct inter-frame gap, 0.25 s. On a stated model where the
interesting events occupy 5 of the 30 seconds, uniform sampling puts **20 of
120** frames on them — **16.7%**, exactly the time share, because uniform
sampling knows nothing about the content. The rate is not the lever; *where* the
frames go is, and the sampler has no way to say it.

### 4 — why an MLP works in 2025 and not in 2023

Drawing on **Qwen2-VL (September 2024) — M-RoPE and native resolution**, which is
where the lesson records the Q-Former being dropped for a plain MLP.

**The short answer is that the MLP's job got smaller.** A projector does one
thing: move vectors from the encoder's space into the LLM's. In 2023 it was
being asked to do considerably more than that, and a Q-Former was the thing
doing the rest.

**What the Q-Former was actually for — three jobs, not one:**

1. **Compression.** 256 patch tokens into 32. That mattered when contexts were
   2–4k and an image at full patch count was a quarter of the window. Lesson
   12.03 measures the trade; the compression was the point.
2. **A trained alignment objective.** Stage 1 of BLIP-2 trained the bridge with
   ITC / ITM / ITG *before the LLM was ever attached* — a supervised signal that
   the visual tokens should be comparable with text. An MLP has no such stage.
3. **Absorbing a weak encoder.** CLIP ViT-L/14 at 224 or 336 produced features
   that needed a transformer's worth of reshaping to be linguistically useful.

**Each of the three stopped being necessary, and for a different reason.**

*Compression* stopped mattering because contexts went from 4k to 32k–128k while
pooling and merging got cheaper — the same 2,592 video tokens this lesson's
sampler budgets would have been the entire 2023 context. Lesson 12.06 makes the
point precisely: token count is now a knob (`min_pixels` / `max_pixels`), not a
constraint the architecture has to solve.

*The alignment objective* stopped mattering because the data arrived. The
Prismatic ablation (Lesson 12.05, exercise 4) found the projector-alignment stage
removable with no measured loss once the instruction mixture was large enough to
carry the alignment signal itself. An objective you no longer need is an
architecture you no longer need.

*The weak encoder* is the one that changed most. SigLIP SO400m trained with a
sigmoid loss at native aspect ratio produces patch features that are already
semantically organised — Lesson 12.07's table puts it +2.5 MMMU and +5.0 DocVQA
over CLIP L/14 at the *same token count*, which is a statement about feature
quality and nothing else. A better encoder means less reshaping, and a linear map
plus a GELU is enough reshaping.

**The part worth keeping.** Not "MLPs are better" — Lesson 12.07 rates connector
architecture at **5%** of variance, the smallest of its six axes, which is a
finding that the connector does not matter rather than that this one wins. The
Q-Former was never beaten; it was made redundant by three separate things moving
at once. That is the usual shape of an architecture disappearing, and it is why
"why did X win" is nearly always the wrong question to ask of a design choice.

### 5 — the fallback recovers none of the failure it is for

**ANSWER: three of the four parse; the truncated one does not.** Two parse
directly, the prose-wrapped one is recovered by slicing from the first `{` to the
last `}`, and `{"tool": "type_text", "text": "hello"` has no closing brace.

**FINDING: the brace slice fails on any prose that contains a brace.**
`Clicking {here} at {"tool": "mouse_click", ...} now` holds a valid tool call and
returns `PARSE_ERROR`, because the slice starts at `{here}` and spans both.

**FINDING: it recovers 0 of 63 truncations of a valid call, and 0 of 32 of a
nested one.** A prefix of a JSON object almost never ends at a closing brace. So
the fallback covers the failure that constrained decoding already solves (chatty
preamble) and *none* of the one that survives it (hitting `max_tokens`
mid-object).

**FINDING: the sentinel lives in the tool namespace.** `PARSE_ERROR` goes in the
`tool` field — the field a dispatcher switches on — and is distinguishable from a
model that legitimately emitted that string only by the presence of `raw`.
Meanwhile **brace depth** separates the cases outright, 1 for the truncated
example against 0 for a complete one, and the fallback never computes it.

**ANSWER: the recovery worth having is not string surgery.** Four steps, in
order of how much they save:

1. **Constrain the decode** to the tool schema, so malformed output cannot be
   produced. This removes cases 1, 2 and 4 outright.
2. **Detect truncation as truncation** — unbalanced brace depth, or a finish
   reason of `length` — and retry with a higher limit. A failed parse is the
   wrong signal; it conflates "ran out of room" with "wrote nonsense".
3. **One repair turn** for anything else: feed back the offending text and the
   parser's own error message. Once, not in a loop.
4. **Validate against the schema before dispatch.** "Parsed successfully" and
   "safe to execute" are different questions, and a tool-calling agent that
   treats them as one has no gap between a JSON bug and an action.
