<!-- generated:start -->
# 18-ethics-safety-alignment / 23-watermarking-synthid-stable-signature-c2pa

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/18-ethics-safety-alignment/23-watermarking-synthid-stable-signature-c2pa/) · upstream spec
`phases/18-ethics-safety-alignment/23-watermarking-synthid-stable-signature-c2pa/docs/en.md`

```bash
uv run demo practice run 23-watermarking-synthid-stable-signature-c2pa --ex 1
uv run demo explain 23-watermarking-synthid-stable-signature-c2pa --ex 1
uv run pytest demos/phases/18-ethics-safety-alignment/23-watermarking-synthid-stable-signature-c2pa
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py`. Report z-scores for watermarked 1000-token generation vs human-authored t… | code | T0 | `ex01_watermark_z_25_30_vs_human_minus_1_64_and_the_95pct_threshold_flags_4_7pct_of_iid_text_but_52pct_of_repeated_text.py` |
| 2 | Implement a paraphrase attack that replaces 30% of tokens with synonyms. Re-measure the z-score. | code | T0 | `ex02_a_30pct_synonym_swap_leaves_z_at_4_4_still_flagged_at_95pct_and_a_parity_flipping_synonym_drops_it_to_2_7.py` |
| 3 | Read Kirchenbauer et al. 2023 Section 6 on robustness. Why do text watermarks fail under para… | code | T0 | `ex03_a_10pct_text_excerpt_keeps_z_near_8_at_every_context_width_while_each_scattered_edit_spoils_k_plus_1_positions.py` |
| 4 | Design a deployment that uses SynthID-text + C2PA metadata. Describe the provenance chain a c… | code | T0 | `ex04_a_one_token_typo_voids_the_c2pa_manifest_and_paraphrased_copied_text_reads_exactly_like_human_text.py` |
| 5 | The 2024 "Stable Signature is Unstable" result shows fine-tuning removes the image watermark.… | code | T0 | `ex05_signed_releases_reject_every_tuned_checkpoint_but_a_1000_token_audit_passes_a_bias_0_56_decoder_41pct.py` |
<!-- generated:end -->

## Answers

Every exercise runs the lesson's `code/main.py`, a toy Kirchenbauer-style text
watermark: 200-token vocabulary, green list keyed by the hash of the previous
K = 4 tokens, green sampled with probability 0.9, detection by green-token
z-score. All randomness is seeded by swapping a `random.Random` into the module.

### 1 — watermark z = 25.30 vs human -1.64; the 95% threshold flags 4.7% of iid text but 52% of repeated text

**The shipped run scores the watermarked 1000 tokens at z = 25.30 and the
human text at -1.64, and the false-positive rate at 95% is 4.7%.** 25.30 is
exactly the expected 2 × 0.4 × √1000. At the one-sided cut z ≥ 1.645 a human
text needs at least 527 of 1000 green tokens: 4.68% exactly (the green count is
Binomial(1000, ½)), and 10 of 200 seeded draws (5.0%) measured.

**The printed "0.000" does not support the takeaway's "<1% FPR at z=4".** It is
z ≥ 4 over 100 draws, and zero hits in 100 only bounds the rate below 2.95%.
The exact rate at z ≥ 4 is 0.0029%.

**The 5% holds only because "human text" is iid uniform tokens.** The detector
scores repeated contexts again, so a text that is one 5-token phrase repeated
has only 5 distinct green tests: |z| is 6.32, 18.97 or 31.62, and 52% of 100
seeded phrases are flagged at 95%.

### 2 — a 30% synonym swap leaves z at 4.4, still flagged at 95%; a parity-flipping synonym drops it to 2.7

**A 30% swap takes z from 24.98 (the first text, unattacked) to a mean of 4.38
over 40 seeded texts, and all 40 remain above 1.645.** A position keeps its
signal only if it and its 4 context tokens are untouched, 0.7⁵ = 16.8% of
positions, which predicts 4.25. 27 of 40 stay above the reference's own z ≥ 4
line, so the shipped run's 3.86 is a low draw.

**Which synonym you pick matters, because the reference's green set is all the
even or all the odd tokens.** The hash contributes one bit.

| 30% swap | mean z | closed form | over 1.645 | over 4 |
|---|---:|---:|---:|---:|
| random replacement (reference `paraphrase`) | 4.38 | 4.25 | 40/40 | 27/40 |
| same-parity synonym, t → t+100 | 5.95 | 6.07 | 40/40 | 36/40 |
| parity-flipping synonym, t → t xor 1 | 2.66 | 2.43 | 34/40 | 5/40 |

**It takes about 43% to defeat the 95% detector, not 30%.** Mean z is 4.38 /
1.90 / 0.65 at 30 / 40 / 50%, and the closed form crosses 1.645 at 43%.

### 3 — a 10% text excerpt keeps z near 8 at every context width, while each scattered edit spoils K+1 positions

**Text watermarks fail under paraphrase because each position's evidence is
keyed to its K preceding tokens.** One substitution spoils its own position
and the next K. Cropping keeps every window inside the crop intact. The text
analogue of a crop, a contiguous 10% excerpt, also survives: mean z is 7.82 /
7.92 / 8.10 / 7.96 at K = 1 / 2 / 4 / 8, close to 0.8 × √100 = 8. The split is
between contiguous and scattered edits, not between image and text.

**Widening the context makes the watermark more fragile** (20 seeded texts per setting):

| context | mean z after 30% | closed form 0.8·√1000·0.7^(K+1) | 10% crop |
|---|---:|---:|---:|
| K = 1 | 12.46 | 12.40 | 7.82 |
| K = 2 | 8.93 | 8.68 | 7.92 |
| K = 4 (shipped) | 4.68 | 4.25 | 8.10 |
| K = 8 | 1.14 | 1.02 | 7.96 |
| fixed list (no context) | 17.72 | 17.71 | — |

**The robust end gives the key away.** With one fixed list, the 100 most
frequent output tokens are 100% green. At K = 4 they are 48% green, which is
chance.

**`K = 0` is not "no context" in the reference.** `green_set` slices
`prev_tokens[-K:]`, and `[-0:]` is the whole list, so one edit at position 100
drops z from 24.94 to 4.65.

### 4 — a one-token typo voids the C2PA manifest, and paraphrased, copied text reads exactly like human text

**The consumer sees a signed chain while the manifest survives, a
watermark-only verdict once it is stripped, and "unknown origin" when both are
gone.** The manifest has C2PA's shape: `c2pa.created` with
`trainedAlgorithmicMedia`, a SHA-256 hard binding, and an HMAC standing in for
the COSE signature. The watermark is the reference's detector at z ≥ 4.

| scenario | C2PA | watermark z | consumer verdict |
|---|---|---:|---|
| as published | valid | 25.49 | AI-generated, signed by the provider |
| copied as plain text | absent | 25.49 | AI-generated (watermark only, no chain) |
| one-token typo fix | hash mismatch | 25.23 | edited after signing, watermark present |
| 60% paraphrase, manifest kept | hash mismatch | -0.25 | edited after signing, no watermark |
| 60% paraphrase, copied | absent | -0.25 | unknown origin |
| human text | absent | -1.14 | unknown origin |
| human text, forged manifest | bad signature | -1.14 | untrusted manifest |

**C2PA's failure mode: text has no container.** Copying drops the manifest,
and when it does travel, one corrected token breaks the hard binding.

**The watermark's failure modes: paraphrase and length.** A 60% paraphrase
leaves z = -0.25. At z ≥ 4 the reference detects 19.5% / 54.5% / 97.5% of
16 / 25 / 50-token generations (200 seeds each). Paraphrased, copied text gets
the same verdict as human text, which is the skill file's own hard reject,
"model-not-watermarked ≠ authentic".

### 5 — signed releases reject every tuned checkpoint, but a 1000-token audit passes a bias-0.56 decoder 41% of the time

**The control: the loader serves only checkpoints whose digest the provider
signed, and a customer fine-tune is re-signed only if 4000 audit tokens are at
least 58% green.** A checkpoint here is the reference sampler's `bias` plus a
seeded weight vector. The removal fine-tune lowers `bias` towards 0.5.

| checkpoint | bias | loader | mean z (10 × 1000 tok) | 1000-tok z≥4 audit | 4000-tok 58% audit |
|---|---:|---|---:|---:|---:|
| released | 0.9 | accept | 25.06 | 100% | 100% |
| one weight nudged by 1e-9 | 0.9 | reject | 25.06 | 100% | 100% |
| partial removal | 0.6 | reject | 6.47 | 99.1% | 99.5% |
| near removal | 0.56 | reject | 4.12 | 41.2% | 0.6% |
| removal | 0.5 | reject | 0.26 | 0.0% | 0.0% |

Audit rates are exact binomial tails, since each output token is green with
probability `bias`. The measured mean z sits within 0.4 of 2(bias − ½)√1000
for every row.

**The attacker only needs to remove 84.2% of the watermark.** z ≥ 4 on 1000
tokens needs 564 green tokens, i.e. bias > 0.5632. A presence audit (one
1000-token sample at z ≥ 4) passes the bias-0.56 decoder 41.2% of the time,
so the audit has to measure strength over many tokens.

**The control protects the serving path, not the weights.**
`watermarked_sample` takes `bias` as a caller argument (default 0.9). Setting
it to 0.5 is the removal row: mean z 0.26, indistinguishable from human text.
