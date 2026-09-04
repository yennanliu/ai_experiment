<!-- generated:start -->
# 01-math-foundations / 09-information-theory

Solutions to all 4 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/01-math-foundations/09-information-theory/) · upstream spec
`phases/01-math-foundations/09-information-theory/docs/en.md`

```bash
uv run demo practice run 09-information-theory --ex 1
uv run demo explain 09-information-theory --ex 1
uv run pytest demos/phases/01-math-foundations/09-information-theory
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Compute the entropy of the English alphabet assuming uniform distribution (26 letters). Then… | code | T0 | `ex01_alphabet_entropy.py` |
| 2 | A model outputs logits [5.0, 2.0, 0.5] for a sample with true class 1. Compute the cross-entr… | code | T0 | `ex02_cross_entropy_by_hand.py` |
| 3 | Show that KL divergence is not symmetric. Pick two distributions P and Q and compute D_KL(P \… | code | T0 | `ex03_kl_asymmetry.py` |
| 4 | Build a function that computes perplexity for a sequence of token predictions. Given a list o… | code | T0 | `ex04_sequence_perplexity.py` |
<!-- generated:end -->

## Answers to the questions the exercises ask

**1 — which entropy is higher, and why?** Uniform, at **4.700440** bits against
**4.181386** for real English letter frequencies. The "why" is a theorem, not an
observation: the uniform distribution maximises entropy over a fixed support, so
uniform is higher for *any* non-uniform frequency table. The 0.519054-bit gap is
not merely related to the divergence — it **is** D_KL(real ‖ uniform), exactly,
because for uniform q the divergence reduces to log₂|X| − H(p). That is the
0.52 bits per letter a compressor can recover for free.

**2 — what logits would give zero loss?** **None.** The loss is −log softmax(z)[1],
which is 0 only when p₁ = 1, and softmax cannot reach 1 while the other logits are
finite. The sweep makes it concrete — margin 1 → 0.551, 5 → 0.0134, 10 → 9.1e-05,
20 → 4.1e-09 — approaching zero without arriving. The first *exact* 0.0 appears at
margin 40, which is a float64 fact rather than a mathematical one. The honest
answer to the exercise is that the question has no finite solution, and knowing
where the arithmetic pretends otherwise is the useful part.

(By hand: e⁻³ = 0.049787, e⁻⁴·⁵ = 0.011109, logsumexp = 5 + ln(1.060896) =
5.059114, loss = 3.059114.)

**3 — why do the two directions differ?** Because each term of
Σ p·log(p/q) is weighted by **p**, the first argument. So q being tiny where p has
mass is catastrophic, and the reverse is nearly free: with q₀ = 1e-9 against
p₀ = 0.7, D_KL(P‖Q) = 20.27 while D_KL(Q‖P) = 2.40, 8× apart.

One trap worth recording. The obvious choice of P = [0.7, 0.2, 0.1] and
Q = [0.1, 0.2, 0.7] is a *palindrome*, and comes out **symmetric** — 1.684413 in
both directions. A single example cannot demonstrate asymmetry; you have to pick
the pair deliberately, which is why the solution runs four.

**4 — the perplexity definition that matters.** exp of the **mean** cross-entropy,
not the sum. Using the sum makes perplexity grow with sequence length, and the
check that catches it is invariance under repeating the sequence: 12 tokens → 8.0,
24 tokens → 8.0. The scale is pinned by construction — a uniform model over V
tokens has perplexity exactly V — which is what licenses reading the number as an
effective branching factor. A confidently wrong model reaches 2988 against 8 for a
uniform guess: perplexity is unbounded above, so misplaced confidence costs more
than ignorance.
