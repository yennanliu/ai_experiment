<!-- generated:start -->
# 07-transformers-deep-dive / 04-positional-encoding

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/07-transformers-deep-dive/04-positional-encoding/) · upstream spec
`phases/07-transformers-deep-dive/04-positional-encoding/docs/en.md`

```bash
uv run demo practice run 04-positional-encoding --ex 1
uv run demo explain 04-positional-encoding --ex 1
uv run pytest demos/phases/07-transformers-deep-dive/04-positional-encoding
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Plot the sinusoidal `PE` matrix as a heatmap for `max_len=512, d=128`. Confirm the "str… | code | T0 | `ex01_half_the_matrix_has_no_stripes.py` |
| 2 | Medium. Implement NTK-aware RoPE scaling. Train a tiny LM on sequences of length 256, then te… | code | T0 | `ex02_interpolation_relabels_gap_four_as_gap_one.py` |
| 3 | Hard. Implement ALiBi and RoPE in the same attention module. Train a 4-layer transformer on a… | code | T0 | `ex03_alibi_degrades_by_becoming_local.py` |
<!-- generated:end -->

## Answers

The lesson is four print-only demos over three encodings. Exercise 1 asks you to
confirm a visual pattern, and the pattern is half absent. Exercises 2 and 3 both
ask for a trained language model and a perplexity, and the lesson ships no model,
no tokenizer, no data, no loss and no optimizer — `torch` is absent too. So both
measure the thing perplexity would have been a noisy estimate of: the attention
score as a function of gap, through the lesson's own `apply_rope`, `dot` and
`alibi_slopes`. All three exercises are **T0** — pure stdlib, as the lesson is.

### 1 — half the matrix has no stripes to get wider

`matplotlib` is absent, so `imshow` is a raster of the same matrix — 16 sampled
dimensions down, 96 sampled positions right:

```text
   0 + :-@%#. .*@@=  =@@#. .#%@-: =*@*+ .-%@%:  *@@=  -@@#. .#%@=: =+@#+ .:%@%:  *@@+  -@@#. .*%@=- -
   8 +*.#.@.@ @ @ @.@.*:*+++:*:@.@ @ @ @.@.#.*=*+-+:%:% @ @ @ @ %.#-#=-=-#-% % @ @ @ % %-#-===#-#.% @
  16 +@= *@..%# :@+ +@= #@..%* -@+ *@-.%% :@+ =@= #@:.%% -@+ =@- %%.:@# -@= +@: %%.-@* =@- *%..@# -@*
  24 +%@*: :*@%=. -#@#- .+@@*. :#@%=  =%@#: .+@@+. :#@%=  =@@#. .*@@=  -%@#- .+@@*. :*@%=  -%@#: .*@@
  32 +#@@@#=:   :+#@@%#=.  .-*%@@%*-.  .-*%@@%+-.  :=#@@@#+:   :=#@@@#=:  .-+%@@%*=.  .-*%@@%*-.  .=*
  40 +*#%@@@@%#+=:.    .:-+*#%@@@@%#+=:.    .:-+*#@@@@@%#+=:.    .:-+*#@@@@@%*+=:.    .:-+*#@@@@@%*+=
  48 ++*#%%@@@@@@@@%##*+=--:..       ..:--=+*##%@@@@@@@@%%#*++=-:..        .::-=+**#%%@@@@@@@%%#**+=-
  56 +++**##%%%@@@@@@@@@@@@@@%%%##***++==--::...              ...:::--==++**###%%%@@@@@@@@@@@@@%%%###
  64 ++++****#####%%%%%@@@@@@@@@@@@@@@@@@@@@@@@%%%%%#####****++++===----:::::.....
  72 +++++++*******########%%%%%%%%%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%%%%%%%%%%########****
  80 ++++++++++++*************##############%%%%%%%%%%%%%%%%%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
  88 ++++++++++++++++++++++**********************########################%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  96 ++++++++++++++++++++++++++++++++++++++****************************************##################
 104 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++****************************
 112 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
 120 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
```

**ANSWER: confirmed, over a factor of 8,660.** Sign changes per sine column fall
monotonically **162 → 0**; wavelengths run 6.28 positions at band 0 to 54,410 at
band 63.

**FINDING: 33 of the 64 bands never complete a cycle.** A stripe needs
`2π·base^(2i/d) ≤ 512`, true only up to band 30. **Half the picture has no
stripes** — the bottom four rows above are flat, and the last band traverses
**0.94%** of one cycle across the whole matrix. The pattern the exercise asks to
confirm lives in the top half only.

**FINDING: every row has exactly the same norm.** `sin² + cos² = 1` per band, so
`|PE[pos]| = √(d/2) = 8.000000000000` for all 512 positions — min and max agree
to the last printed digit. A heatmap of this matrix *redistributes* brightness
along a row and never adds any, which is the one thing the picture cannot show.

**FINDING: the sinusoidal dot product is already purely relative.**
`PE[p]·PE[p+g]` moves by at most **3e-14** as `p` ranges over 0, 17, 100, 300.
The lesson's closing line credits RoPE with "encoding relative position in the
dot product itself" — sinusoidal PE does that too. What RoPE actually has is
survival of `Wq` and `Wk`: sinusoidal PE is *added to x* before those
projections, and the relative structure does not survive them.

### 2 — interpolation relabels gap 4 as gap 1, exactly

**SETUP: extrapolation alone breaks nothing.** The score at a fixed gap is
identical at base positions 0, 50, 500 and 4,000 to **3e-14**. Testing at
position 1024 after training at 256 changes no score the model has seen; only
*gaps* beyond the trained range are new.

**FINDING: position interpolation is a relabelling, and an exact one.** Calling
`apply_rope` at `pos/4` makes gap 4 return **the same float** as gap 1 does
unscaled — difference `0.0`, not 1e-16. A model trained on gaps 1–256 has its
whole learned profile compressed into the first quarter of its range.

**ANSWER: NTK-aware scaling stretches the slowest band by exactly `s` and the
fastest by exactly 1.** `base' = 10000 · 4^(64/62) = 41,829.4`:

| | band 0 | band 31 |
|---|---:|---:|
| wavelength ratio, NTK | **1.000000** | **4.000000** |
| wavelength ratio, interpolation | 4 | 4 |

That *is* the construction: interpolate in frequency, not in position.

**ANSWER: over gaps 1–16, interpolation distorts 7× more.** Normalised by the
unscaled profile's own magnitude, interpolation moves the score by **47.6%** and
NTK-aware by **6.8%**. Short gaps carry most of a language model's predictive
mass, so the *sign* of the perplexity comparison is settled by this even though
the perplexity is not buildable here.

**CONTROL: 13 of 32 bands wrap inside the trained 256 positions.** At test length
1024 the unscaled base has 18 wrapping — 5 bands newly in untrained territory —
and NTK returns 2 of those 5 to the regime the model saw. That is the quantity
perplexity would have been estimating.

### 3 — ALiBi degrades by becoming local; RoPE degrades into untrained

The module is built: `q` and `k` rotated by the lesson's `apply_rope`, the
lesson's ALiBi slope subtracted from the score afterwards.

**ANSWER: they compose by addition, exactly.** Combined score minus bias equals
the RoPE score to **6e-14** at gaps 1, 99 and 2048. RoPE acts before the dot
product and ALiBi after it; "in the same attention module" is a `+`.

**FINDING: RoPE alone has no distance decay whatever — and not approximately.**
The rotation is orthogonal, so it moves `k` without changing its norm, and over
isotropic `q, k` the score's distribution *cannot* depend on the gap. Measured:
sd **2.137** at gap 1 and **2.137** at gap 2048, 1e-15 apart, mean zero
throughout. The "long-term decay" RoPE is credited with belongs to trained `Wq`
and `Wk`. Extrapolated, RoPE gives an **untrained** score, not a decayed one.

**ANSWER: ALiBi's degradation is bounded in kind, not in size.**

| head | slope | penalty at 512 | at 2048 | at 2048, in content σ |
|---|---:|---:|---:|---:|
| 0 | 0.2500 | −128 | **−512** | 240 |
| 1 | 0.0625 | −32 | −128 | 60 |
| 2 | 0.0156 | −8 | −32 | 15 |
| 3 | 0.0039 | **−2** | −8 | **4** |

At 512 the flattest head's penalty is **0.9σ** and content still competes. At
2048 it is 4σ, so every head is distance-dominated. Going 512 → 2048 turns ALiBi
from "content competes with distance" into "distance decides" — a failure you can
name. RoPE's is a region nobody trained.

**CONTROL: the slopes span 64×, so the heads do not degrade together.** Head 0 is
purely local past gap ≈9; head 3 is still global past 500. "Compare degradation"
has a per-head answer, not one answer.

**FINDING: the lesson's own `alibi_bias` cannot be called at 2048.** It
materialises `n_heads · L²` Python floats — **1,048,576** at L=512, measured
**32 MiB** — so **16,777,216** floats and about **516 MiB** at the length the
exercise sets, quadratically. The sequence length the exercise names is out of
reach of the code it names.
