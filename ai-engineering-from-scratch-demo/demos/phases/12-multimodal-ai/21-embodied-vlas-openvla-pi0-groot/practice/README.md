<!-- generated:start -->
# 12-multimodal-ai / 21-embodied-vlas-openvla-pi0-groot

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/12-multimodal-ai/21-embodied-vlas-openvla-pi0-groot/) · upstream spec
`phases/12-multimodal-ai/21-embodied-vlas-openvla-pi0-groot/docs/en.md`

```bash
uv run demo practice run 21-embodied-vlas-openvla-pi0-groot --ex 1
uv run demo explain 21-embodied-vlas-openvla-pi0-groot --ex 1
uv run pytest demos/phases/12-multimodal-ai/21-embodied-vlas-openvla-pi0-groot
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | A 10-DOF arm at 30 Hz control rate. Discrete-bin tokenization at 256 bins emits how many toke… | code | T0 | `ex01_the_bin_count_does_not_enter_the_rate_at_all.py` |
| 2 | FAST tokenization compresses 30-step trajectories to ~10 tokens. What does the user lose if t… | code | T0 | `ex02_the_cutoff_is_one_and_a_half_hertz_and_drumming_is_gone.py` |
| 3 | π0's flow-matching head denoises in ~5 steps. Compare throughput to OpenVLA's autoregressive… | code | T0 | `ex03_sixty_model_invocations_against_five.py` |
| 4 | GR00T's System 1 / System 2 split maps to Kahneman. Propose a different split (System 3?) tha… | explain | T0 | prose, below |
| 5 | Read Open X-Embodiment Section 4 on dataset curation. Name the three curation rules that prev… | explain | T0 | prose, below |
<!-- generated:end -->

## Answers

Three code solutions and two prose ones. The three code ones are all the same
arithmetic seen from three angles — tokens per second of motion — and each finds
the lesson's own numbers pointing somewhere its prose does not.

### 1 — the bin count does not enter the rate at all

**ANSWER: 300 tokens a second, and no.** Ten DOF is ten tokens a step and 30 Hz
makes 300; at **30–80** tok/s a 7B is **3.75× to 10×** too slow.

**FINDING: `bins` changes the precision and not the rate.**

| bins | tokens/step | round-trip error |
|---:|---:|---:|
| 2 | 10 | 1.9 |
| 16 | 10 | 0.13 |
| 256 | **10** | **0.0069** |
| 1024 | 10 | 0.0017 |

Rate is `DOF × control_rate`; the vocabulary does not appear in it.

**FINDING: control rate is throughput over DOF, and it brackets the lesson's own
figure.** 30–80 tok/s at 10 DOF is **3.0–8.0 Hz**, and the lesson reports OpenVLA
at **4–5 Hz** — inside the bracket. Its architecture section and its token
arithmetic agree, and neither reaches 30.

**FINDING: so DOF is the lever nobody names.** At 80 tok/s: 7-DOF arm **11.4 Hz**,
10-DOF **8.0**, 26-DOF humanoid **3.1**. The identical model is **3.7×** slower on
the humanoid for reasons that have nothing to do with the humanoid being harder.

### 2 — the cutoff is 1.5 Hz, and drumming is gone

**ANSWER: everything above 1.5 Hz.** `keep_coeff=4` of 30 steps keeps DCT bins
0–3, and at 30 Hz bin *k* is *k*/2 Hz. The tokenizer is a low-pass filter.

| signal | energy retained by 4 coefficients |
|---|---:|
| 0.5 Hz | **90.5%** |
| 2 Hz | 38.5% |
| 5 Hz | **1.5%** |
| 8 Hz | **0.4%** |

A drumming trajectory does not come back degraded. It comes back as a straight
line.

**FINDING: the lesson's own FAST output is 40 tokens, not the ~10 the exercise
names** — four coefficients *per dimension*, for **7.5×** against 300. The
exercise's figure is 4× smaller than the function's.

**FINDING: there is no inverse in the module, so nothing checks it.**
`fast_compress` has no `fast_decompress`; the lesson prints a compression ratio
and never reconstructs. The inverse DCT above had to be written to measure the
loss at all — which is the point: a tokenizer whose forward pass is a low-pass
filter and whose round trip is untested reports only the half of the trade that
looks good.

### 3 — sixty model invocations against five

**ANSWER: 5 against 300, a factor of 60.** One second of 30 Hz, 10-DOF motion is
**300** discrete action tokens, each an autoregressive step; the same second is
one 30-step chunk denoised in **5** passes.

**FINDING: at its own reported decode rate, OpenVLA takes 7.5 seconds to produce
one second of motion.** 4 Hz of 10-token steps is 40 tokens a second against a
requirement of 300 — RTF **7.5**, or **6.0** at 5 Hz. It is not slow at 30 Hz; it
cannot run at 30 Hz.

**FINDING: the flow head needs 5 passes a second and has 6×–16× of headroom** at
the same throughput budget.

**FINDING: and the advantage is the chunk, not the flow.** The 60× is
`chunk_steps × DOF / denoise_steps` = 30 × 10 / 5 — the same for a diffusion
head, a regression head, or anything else that emits thirty steps at once. Flow
matching supplies the 5; the 300 comes from emitting one token per DOF per step,
and any architecture that stops doing that gets the same win.

### 4 — a third system for bipedal walking

Drawing on **GR00T N1 — dual-system for humanoids**, which puts System 2 at ~1 Hz
and System 1 at 50–100 Hz and maps the split to Kahneman.

**The two-system split has a hole, and bipedal walking falls through it.** The
rates are the giveaway: 1 Hz and 50–100 Hz, with nothing between. Balance lives
in that gap. A push recovery is a **5–20 Hz** phenomenon — too fast for a
subgoal, too slow and too *stateful* for a reactive joint controller, and
critically it is the one thing that cannot be interrupted for replanning.

**Proposal — System 1.5: a balance controller with its own clock and its own
authority.**

| | System 2 | **System 1.5** | System 1 |
|---|---|---|---|
| rate | ~1 Hz | **~10 Hz** | 50–100 Hz |
| horizon | task | **one or two footsteps** | one command |
| state | none needed | **centre of mass, contact, momentum** | none needed |
| can it override? | no | **yes — it can refuse a subgoal** | no |

**The three properties that make it a separate system rather than a bigger
System 1:**

1. **It owns different state.** System 1 is conditioned on a subgoal and the
   current observation; it is Markov by construction. Balance is not — it depends
   on momentum, on which foot is loaded, and on where the centre of mass has been
   for the last half-second. Bolting that state onto System 1 makes System 1 big
   and slow, which is the thing the split exists to prevent.
2. **It has to be able to say no.** System 2 emits "walk to the door"; if the
   robot is mid-fall, the correct action is a recovery step in the wrong
   direction. Neither existing system can generate that: System 2 is too slow to
   notice and System 1 has no authority to disobey. A veto is a different kind of
   component from a translator.
3. **Its failure is catastrophic and immediate.** A bad subgoal wastes a second;
   a bad joint command is filtered by the actuator; a bad balance decision puts a
   humanoid on the floor. Different consequences justify different verification,
   different update rates, and — in practice — different engineering.

**Where the Kahneman analogy stops helping.** Fast and slow is a two-term
metaphor and bipedal locomotion is a three-timescale problem: intent (seconds),
balance (tenths), actuation (hundredths). Reaching for a third Kahneman system is
the wrong move; the right one is to notice that the analogy was borrowed for
exposition and that the rates are the real constraint. Exercise 1 makes the same
point about DOF — the humanoid's difficulty is in its numbers, not in its
metaphor.

### 5 — three curation rules against domain leakage

Drawing on **Open X-Embodiment**, which lists the corpus's training hygiene —
"unify action space, normalize joint ranges, resize cameras" — and the LoRA
fine-tuning step that closes the remaining gap.

**What leakage means here, first, because it is not the usual thing.** In a
22-robot, 22-dataset corpus the risk is not that test examples appear in training
— it is that a *held-out robot* is not actually held out, because something about
it survives in the corpus in another form. Three rules, each closing a different
route:

**1. Split by robot and by scene, never by trajectory.** The unit of
independence is the embodiment, not the episode. Two trajectories from the same
robot in the same kitchen share camera pose, lighting, table height and gripper
geometry, so a random trajectory split lets the model memorise the *scene* and
report it as generalisation. The rule is that every robot-scene pair is entirely
in train or entirely in test.

**2. Normalise per-robot, with statistics computed on the training split only.**
The corpus unifies action spaces and normalises joint ranges — and a global
normaliser computed over all 22 datasets leaks the held-out robot's joint limits
into the shared scale. Every robot's mean and range has to come from the training
partition, which also means a held-out robot's normaliser has to be estimated at
adaptation time rather than looked up.

**3. Deduplicate across datasets before splitting, not after.** The 22 datasets
were assembled independently and overlap: the same public benchmark appears
re-recorded, the same demonstrations are republished, and near-duplicate
trajectories cross dataset boundaries. Deduplicating within each dataset and then
splitting leaves cross-dataset twins on both sides of the line.

**And the rule that matters most is the one about what the split is *for*.** The
corpus exists so that a model transfers to a robot nobody trained on, and the
honest test is exactly that: hold out an entire embodiment, then measure after
the 100–1,000-demo LoRA the lesson describes. Any evaluation that reports
zero-shot numbers on a robot whose siblings were in training is measuring the
corpus's redundancy, not the model's transfer.
