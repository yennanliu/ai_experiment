<!-- generated:start -->
# 08-generative-ai / 01-generative-models-taxonomy-history

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/08-generative-ai/01-generative-models-taxonomy-history/) · upstream spec
`phases/08-generative-ai/01-generative-models-taxonomy-history/docs/en.md`

```bash
uv run demo practice run 01-generative-models-taxonomy-history --ex 1
uv run demo explain 01-generative-models-taxonomy-history --ex 1
uv run pytest demos/phases/08-generative-ai/01-generative-models-taxonomy-history
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. For each of these five products, identify the family and backbone: ChatGPT image, Midjo… | explain | T0 | prose, below |
| 2 | Medium. The paper you are about to read tomorrow claims 100x faster sampling than diffusion.… | explain | T0 | prose, below |
| 3 | Hard. Take one domain you care about (e.g. protein structure, CAD, molecules, trajectories).… | explain | T0 | prose, below |
<!-- generated:end -->

## Prose answers

All three exercises are prose only, so they ship these answers and no file
(`DESIGN D11`). Each names the lesson section it rests on, and that citation is
the gate: it must be a real heading in `docs/en.md`, which
`scripts/audit_practice.py` checks.

### 1 — only two of the five published anything, and neither is a technical report

Drawing on **The Concept** (the five families) and **Use It**.

The exercise sets an evidence standard — "evidence should be from public
technical reports" — and then names five products. Applying the standard before
answering the question:

| Product | Family | Backbone | What the public record actually is |
|---|---|---|---|
| ChatGPT image | **5** — AR over discrete codes | GPT-4o itself | An *addendum to a safety system card* (25 Mar 2025). Says the model "is autoregressive" and uses "the same architecture as the GPT‑4o LLM" — one sentence of architecture inside a risk document. |
| Midjourney v7 | 2 or 4, presumed | **unknown** | **Nothing.** A "ground-up" rework described in office hours. No paper, no model card. |
| Sora | **2/4** + a 5-style tokenizer | DiT over spacetime patches of latent codes | A capabilities blog post, *Video generation models as world simulators*. Names the architecture in one line, then says it withholds training and architecture detail. |
| Runway Gen-3 Alpha | 2 or 4, presumed | **unknown** | A launch blog post. No model card, no architecture, no training sources. |
| ElevenLabs | **unknown** | **unknown** | Product docs: model names, latency figures, language counts. No architecture at all. |

**Two of five name an architecture. Zero of five published a technical report in
the sense the exercise means** — a document with a method section, parameter
counts, training data and ablations. The two that say anything say it in a
safety addendum and a capabilities blog, both written to announce a product.

**The gap is not neutral: it gets filled.** Because Runway published nothing,
the secondary write-ups disagree about its own question. Third-party pages
describe Gen-3 as "a world model foundation… rather than frame-by-frame
diffusion" and, elsewhere, as "a transformer-based diffusion approach". Those
are different answers to *identify the backbone*, neither is sourced, and a
reader following the exercise's instruction ends up citing a blogger while
believing they cited a technical report.

**The lesson already knows the answer it cannot evidence.** **Use It** maps
text-to-video to "Diffusion Transformer + flow matching" and names Sora in the
same row. So the family is available from the taxonomy by category — what is
missing, for three of the five, is any way to *check* it. That is the real
finding: for closed products the family is inferable from behaviour and release
timing, and the backbone is not inferable at all.

Sources: [4o image generation system card
addendum](https://openai.com/index/gpt-4o-image-generation-system-card-addendum/) ·
[Video generation models as world
simulators](https://openai.com/index/video-generation-models-as-world-simulators/) ·
[Introducing Gen-3
Alpha](https://runway.com/research/introducing-gen-3-alpha) ·
[ElevenLabs model docs](https://elevenlabs.io/docs/overview/models)

### 2 — every "100x faster" is a question about the denominator

Drawing on **Production note: five families, five inference shapes**, which
gives the translation rule: read "faster than diffusion" as either "fewer steps
× same step cost" or "same steps × cheaper step cost".

**Q1. What is the denominator — which sampler, at how many steps?** 100× against
1,000-step DDPM is the flattering comparison, and nobody has sampled that way
since DDIM. Against a 20-step DPM-Solver baseline the same model is 2×. Ask for
the baseline's *sampler name and step count* before reading the method section;
if the paper reports only a ratio, the ratio is the finding it is hiding.

**Q2. Is guidance switched on in both arms?** Classifier-free guidance runs the
network twice per step — conditional and unconditional — so it is a 2× multiplier
on `step_cost` that applies to whichever arm uses it. A number measured with the
baseline guided and the new model unguided is inflated 2× before any of the
method matters. This is the "survives conditioning" half of the exercise:
conditioning is not free, and it is not automatically constant across the two
arms.

**Q3. At what resolution, and which of the two factors carries the 100×?** Step
*count* is resolution-independent; step *cost* is not. For a diffusion
transformer the token count grows with the square of the side length and
attention grows with the square of the token count, so `step_cost` at 1024² is
far more than 4× its value at 512². A method that removes steps keeps its
speedup at high resolution. A method that makes each step cheaper may have been
exploiting something that only holds at 256². Ask which factor moved, then ask
for the same measurement at the resolution you actually ship.

**The three are one question.** Write the claim as `steps × step_cost` for both
arms and ask which of the four numbers moved. The Production note already says
this; what the exercise adds is *where* the claim breaks — guidance and
resolution are both multipliers on `step_cost`, so they are precisely the two
things that can erase a step-cost win while leaving a step-count win untouched.
That asymmetry is the answer to "does it survive".

### 3 — the sampler cannot say it does not know

Drawing on **The five-question triage**, for **protein structure** and
**AlphaFold 3**.

1. **What is being modeled?** All-atom coordinates directly — proteins, nucleic
   acids, ligands, ions and modified residues in one model. Not AF2's torsion
   angles over a residue frame. Changing the represented object is what let the
   model accept ligands at all.
2. **Explicit or implicit density?** Neither, in practice. It is a conditional
   diffusion model, so it optimises a denoising objective corresponding to a
   weighted ELBO — bucket 2 — but no `log p(structure | sequence)` is ever
   exposed. Confidence comes from a **separately trained head** (pLDDT, PAE),
   not from the density the family nominally has.
3. **Sampling: one-shot or iterative?** Iterative. A diffusion module replaced
   AF2's deterministic structure module, so a prediction is now a *sample* and
   two runs of the same sequence differ.
4. **Conditioning?** Sequence, MSA and templates as before, plus ligands and
   ions. The Evoformer became the Pairformer and the MSA path was thinned.
5. **Evaluation?** LDDT and TM-score against experimental structures, plus
   PoseBusters for whether a ligand pose is physically legal.

**What a better model would change.** Going diffusion cost the model its
uncertainty *display*. In AF2 a disordered region came out as an extended
spaghetti ribbon — the shape of the output was the confidence signal, legible at
a glance and hard to ignore. A diffusion sampler always returns a well-formed
sample, so AF3 draws compact, confident-looking structure exactly where AF2 drew
visible doubt. DeepMind knew: they patched it with cross-distillation against
AlphaFold-Multimer v2.3, which teaches the new model to imitate the old model's
ribbons — a fix applied to the *output* rather than to the objective. Measured
afterwards on intrinsically disordered proteins, roughly a fifth of residues are
still predicted ordered where experiment says disordered.

This is **The Concept**'s explicit/implicit distinction arriving with
consequences. The stated con of bucket 3 is "only a sampler — no way to evaluate
`p(x)`". AF3 is bucket 2, so a bound exists in principle; it is simply not
surfaced, and what replaced it is a second learned model that can be confidently
wrong in the same places as the first.

So a better model changes the **output type, not the backbone**: return a
calibrated distribution — or an ensemble with a spread that means something —
instead of one sample, and derive confidence from the model's own bound rather
than from a head trained to predict its accuracy. Concretely, report per-residue
variance across N draws next to pLDDT and require it to be calibrated against
experimental disorder annotations. Drawing N samples is already possible today.
The reason nobody ships it is question 5: no number on the leaderboard rewards a
model for admitting it does not know, so **evaluation is the question that has
to change first** — which is Lesson 14's subject, and why the triage puts it
last but the fix puts it first.

Sources: [Hallucinations in AlphaFold3 for intrinsically disordered
proteins](https://arxiv.org/abs/2510.15939) · [Modeling intrinsically disordered
regions from AlphaFold2 to AlphaFold3](https://onlinelibrary.wiley.com/doi/full/10.1002/pro.70426)
