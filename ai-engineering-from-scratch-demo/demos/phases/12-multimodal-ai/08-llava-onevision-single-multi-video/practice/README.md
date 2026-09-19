<!-- generated:start -->
# 12-multimodal-ai / 08-llava-onevision-single-multi-video

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/12-multimodal-ai/08-llava-onevision-single-multi-video/) · upstream spec
`phases/12-multimodal-ai/08-llava-onevision-single-multi-video/docs/en.md`

```bash
uv run demo practice run 08-llava-onevision-single-multi-video --ex 1
uv run demo explain 08-llava-onevision-single-multi-video --ex 1
uv run pytest demos/phases/12-multimodal-ai/08-llava-onevision-single-multi-video
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Your product supports 80% single-image, 10% multi-image (2-4 images), 10% video (8-16 frames)… | code | T0 | `ex01_the_saving_the_exercise_asks_about_does_not_exist.py` |
| 2 | Read LLaVA-OneVision Section 4.3 (emergent capabilities). Propose a fourth emergent skill the… | explain | T0 | prose, below |
| 3 | Swap the curriculum order — train multi-image first, then single-image, then video. Predict w… | code | T0 | `ex03_the_function_the_exercise_wants_to_perturb_has_no_order.py` |
| 4 | The paper reports video benchmarks trained on only 8 frames per sample. Does that generalize… | code | T0 | `ex04_temporal_reasoning_breaks_six_point_two_five_times_earlier.py` |
| 5 | Bilinear pooling of 24x24 patches to 12x12 is a 4x reduction per dim. Implement the pooling i… | code | T0 | `ex05_the_grid_the_lesson_actually_pools_is_odd.py` |
<!-- generated:end -->

## Answers

Four code solutions and one prose one. Three of the four find the lesson's own
planner doing something other than what its surrounding prose says: it leaves
57% of the budget unspent (exercise 1), it defaults to 4× the trained frame
count (exercise 4), and it pools an odd grid with an even factor (exercise 5).

### 1 — the saving the exercise asks about does not exist

| scenario | planner's answer | % of 4,096 | best fit in its own search space |
|---|---:|---:|---:|
| single-image | 1,690 | **41.3%** | **3,645** (89.0%) |
| multi-image | 1,352 | 33.0% | 2,916 (71.2%) |
| video | **2,592** | 63.3% | 2,704 (66.0%) |

**ANSWER: 1,746 tokens per sample expected, 57.4% of the budget unused.**
0.8 × 1,690 + 0.1 × 1,352 + 0.1 × 2,592. There is nowhere to put a saving
because the budget was never spent.

**ANSWER: and there is no saving from skipping heavy multi-image.** It is the
**cheapest** of the three. Dropping it moves the expected cost by **+43.8**
tokens, because the 10% it frees is reweighted onto scenarios that cost more.

**FINDING: the planner maximises tiles, not tokens.** Its loops run count outer
and resolution-and-pooling inner, so it returns the *first* fit: nine pooled
tiles at 1,690 when four unpooled tiles at 3,645 also fit.

**FINDING: the lesson misses its own spread target by 0.3 points.** The three
plans span **1,240** tokens — **30.3%** of the budget — printed directly above a
line reading "keep the spread under 30% for predictable LLM cost".

**ANSWER: so the budget goes where the mix is.** At 80% of samples, moving
single-image to the 3,645 the planner passed over takes the expected cost to
**3,310** — still inside 4,096. The same move on video is weighted by 0.1.

### 2 — a fourth emergent skill

Drawing on **Emergent cross-scenario skills**, which is where the lesson lists
the three the paper reports and notes that "none of the three appears in
stage-SI data".

**Proposed fourth skill: visual state-diffing — given two images of the same
scene taken at different times, say what changed.** "Here is the dashboard
before and after the deploy; what moved?" "Here are two photos of the same
shelf; what is missing?"

**Why the curriculum would unlock it.** Look at what the three reported skills
have in common rather than at what they do. Multi-camera reasoning, set-of-mark
prompting and screenshot agency all require the same primitive: **binding one
entity across two images that were never seen together in training**. That
primitive is not in stage-SI data — a single image has nothing to bind to — and
it is not separately supervised anywhere; it falls out of stage OV putting
multi-image and video in the same budget, so the model learns that two visual
token blocks in one sequence can refer to the same thing.

State-diffing is that primitive plus a difference operation, and the difference
operation is the part the frozen LLM already has: comparing two descriptions and
reporting what is not in both is ordinary text reasoning. So the skill decomposes
into one capability the curriculum demonstrably creates and one the LLM brought
with it, which is exactly the shape of the three that were reported.

**Why it plausibly was not reported.** There is no standard benchmark for it.
Multi-camera reasoning and set-of-mark have suites; "what changed between these
two screenshots" is evaluated, when at all, inside product-specific harnesses.
An emergent capability with no benchmark is an emergent capability nobody
measures.

**How to check it cheaply, and what would falsify the claim.** Take 200 image
pairs where exactly one object was added, removed or moved, and ask for the
change in one sentence. Score against the known edit. Run the same 200 through
the stage-SI checkpoint and the stage-OV checkpoint. The claim predicts a large
gap between the two, since stage SI has no mechanism for it at all. It is
falsified if stage-SI does nearly as well — which would mean the model is
describing each image independently and letting the LLM difference the
descriptions, in which case nothing emerged and the capability was always
available through captioning.

That falsification test is worth stating explicitly, because it is the one that
separates a genuine cross-scenario skill from a pipeline the LLM can fake. Two
of the three reported skills would arguably fail it.

### 3 — the function the exercise wants to perturb has no order

**FINDING: `curriculum_stages` has no order parameter.** Its signature is
`curriculum_stages(mix)` and its body holds three literal stage rows, only the
last of which reads the argument. The permutation cannot be expressed in the
planner the lesson ships.

What the swap *does* change is cumulative exposure — the sum of each scenario's
share across the three stages:

| ordering | single | multi | video |
|---|---:|---:|---:|
| shipped (SI first) | **1.9** | 0.6 | 0.5 |
| multi-image first | **0.9** | 1.6 | 0.5 |
| change | **−53%** | +167% | **0%** |

**ANSWER: single-image degrades, by the largest relative move of the three** —
so MMMU and DocVQA, the benchmarks that ride on single-image.

**FINDING: the lesson has a number for one of six orderings, and it is not this
one.** The printed note covers video-first at "2–4 MMMU". That ordering leaves
single-image at **0.9** stage-units — the same figure as this swap — which is
why its penalty is the best available estimate for a permutation nobody
measured.

**FINDING: video is invariant under both swaps.** It stays at 0.5 stage-units
whichever of the other two goes first. The scenario the curriculum exists to
reach is the one the order does not move.

### 4 — temporal reasoning breaks 6.25× earlier

**ANSWER: temporal reasoning, by 6.25×.** A frame costs **81** tokens at the
lesson's own video plan, so 4,096 holds **50** frames. Training saw **8**.
Extrapolation begins at frame 9; the budget is untouched until frame 51.

**FINDING: "30 seconds" is not a frame count.**

| rate | frames | tokens | |
|---|---:|---:|---|
| 1 FPS | 30 | 2,430 | 59% of budget |
| 2 FPS | 60 | **4,860** | over |
| 30 FPS | 900 | **72,900** | **17.8×** over |

The two failure modes swap places between 1 and 2 FPS, and the exercise names a
duration without a rate.

**FINDING: the lesson's own default is already outside the training
distribution.** `plan_video(4096)` returns **32** frames — 4× the trained 8 —
chosen by a loop that tries 32 first and stops at the first fit.

**FINDING: the budget is a hyperbola and training pinned one point on it.** Of
the 7 (frames, tokens-per-frame) pairs that fit, three sit at 8 frames and the
best spends **1,352** tokens — **33%** of the budget. Staying inside the
training distribution means leaving two thirds unspent; spending it means
leaving the distribution.

### 5 — the grid the lesson actually pools is odd

**ANSWER: at 24 → 12 the two agree to 0.0 on all 144 outputs.** At an integer
factor of 2 each output's bilinear sample point lands exactly on its 2×2 block's
centre and the four corner weights come out 0.25 each. It is the same arithmetic
written twice.

**FINDING: "4× reduction per dim" is 2× per dim.** 24 → 12 is a factor of 2 on
each axis and 4× in token count. The phrase names the area reduction and
attaches it to the wrong noun — which matters because the lesson's *other*
pooling factor, 3, is 9× in tokens and not 3×.

**FINDING: the lesson's own default grid is 27, and 2×2 blocks do not tile it.**

| | |
|---|---:|
| `384 // 14` | 27 |
| `27 // 2` | 13 |
| patches covered | 676 of **729** |
| dropped | **53** (7.3%) |
| actual reduction | **4.31×** |

The clean 24×24 case the exercise names is `(336, 14, 2)`, which the planner
lists third and never reaches because `(384, 14, 2)` fits first.

**FINDING: at 27 → 13 the two methods stop agreeing** — up to **0.439** apart on
a unit-scale grid, against 0.0 at 24. Bilinear samples at non-integer positions
and interpolates across block boundaries; the crop simply discards the last row
and column. The verification the exercise asks for passes on a grid the lesson
does not use and fails on the one it does.
