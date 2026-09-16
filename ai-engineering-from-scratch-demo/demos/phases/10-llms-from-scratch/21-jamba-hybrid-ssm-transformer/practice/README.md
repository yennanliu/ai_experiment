<!-- generated:start -->
# 10-llms-from-scratch / 21-jamba-hybrid-ssm-transformer

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/10-llms-from-scratch/21-jamba-hybrid-ssm-transformer/) · upstream spec
`phases/10-llms-from-scratch/21-jamba-hybrid-ssm-transformer/docs/en.md`

```bash
uv run demo practice run 21-jamba-hybrid-ssm-transformer --ex 1
uv run demo explain 21-jamba-hybrid-ssm-transformer --ex 1
uv run pytest demos/phases/10-llms-from-scratch/21-jamba-hybrid-ssm-transformer
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Run `code/main.py` to compute KV cache at 256k context for a 32-layer pure Transformer (hidde… | code | T0 | `ex01_the_eight_times_is_thirty_two_over_four.py` |
| 2 | Modify the calculator to model a 1:3 hybrid (4 Mamba : 1 Attention) and a 1:15 hybrid (14 Mam… | code | T0 | `ex02_the_crossover_is_a_context_not_a_ratio.py` |
| 3 | Read Section 3 of the Jamba paper (arXiv:2403.19887). Explain why AI21 uses Mamba-1 rather th… | code | T0 | `ex03_one_integer_separates_the_two_mambas.py` |
| 4 | Compute the parameter overhead of MoE-every-other-layer in Jamba 1.5 Large (398B total, 94B a… | code | T0 | `ex04_the_dense_half_is_always_active.py` |
| 5 | Read Section 3 of the Mamba-3 paper (arXiv:2603.15569). Explain in three sentences why a comp… | code | T0 | `ex05_a_real_diagonal_reaches_two_points_of_the_circle.py` |
<!-- generated:end -->

## Answers

`code/main.py` is 152 lines holding one dataclass and two functions:
`kv_cache_bytes` and `ssm_state_bytes`. Three of the five exercises ask about
quantities that are not in it — Mamba versions, parameter counts, state matrices
— and the two that are in it turn out to be one division and one term that
rounds away.

All five are **T0** on **no** dependency group (stdlib only).

> Exercises 3 and 5 are scaffolded as prose items. They are built here as
> **measurements**: one an audit of what the calculator can express about the two
> Mambas, the other a direct check of the complex-rotation identity against
> RoPE's own 2×2 matrix.

### 1 — the eight times is thirty-two over four

| config | KV at 256k | SSM state | total |
|---|---:|---:|---:|
| pure Transformer 32L | **128.00 GB** | — | 128.00 GB |
| Jamba 1:7 hybrid 32L | **16.00 GB** | 3.50 MB | 16.00 GB |
| Jamba 1:7 + GQA-8 | 4.00 GB | 3.50 MB | 4.00 GB |

**ANSWER: exactly 8.00× on the KV cache, and the 8 is `32 / 4`.**
`kv_cache_bytes` is linear in `attn_layers`, and the two configs differ in that
field and no other.

**FINDING: the SSM state is 0.02% of the hybrid's cache** and constant in
context.

**FINDING: every Jamba config in the file sets `n_kv_heads=32`** where Jamba uses
GQA. With 8 the reduction is **32×** — the configuration understates the
architecture by 4×.

**MECHANISM: the two savings multiply.** 8× from layers, 4× from heads.

### 2 — the crossover is a context, not a ratio

| attention layers | KV at 256k | SSM state | KV / SSM |
|---:|---:|---:|---:|
| 16 (1:1) | 64.000 GB | 2.00 MB | 32,768 |
| 4 (1:7) | 16.000 GB | 3.50 MB | 4,681 |
| 1 (1:31) | 4.000 GB | 3.88 MB | **1,057** |

**ANSWER: at no integer ratio.** Solving for equality gives **A = 0.00098**
attention layers.

**MECHANISM: the two scale in different variables.** Sweeping the ratio moves KV
16× and the state 1.94×; only the context — which the exercise holds fixed —
can close three orders of magnitude.

**FINDING: the crossover is **248 tokens** of context**, with 1 attention layer
and 31 Mamba layers.

**FINDING: the SSM state *grows* as attention layers are removed**, in the
opposite direction from the term that decides the answer.

### 3 — one integer separates the two Mambas

| `ssm_state_size` | SSM at 256k | share of the KV cache |
|---:|---:|---:|
| 16 (Mamba-1) | 3.50 MB | 0.0214% |
| 256 | 56.00 MB | **0.3418%** |

**ANSWER: raising the state 16× takes it from 0.02% to 0.34% of the cache.**

**MECHANISM: `ssm_state_bytes` takes no context argument** — `(cfg,
bytes_per_elem)` against `kv_cache_bytes`' `(cfg, ctx, bytes_per_elem)`.

**FINDING: `HybridConfig`'s eight fields contain no Mamba version.** The
discretisation, the state's realness, the SSM head structure and the outer
convolution have no representation.

**FINDING: the state would need to be 4,681 wide to reach 1 GB** — and even there
it is 6.2% of the hybrid's cache.

### 4 — the dense half is always active

| model | total | active | active ratio | layers that can route |
|---|---:|---:|---:|---:|
| Jamba 1.5 Large | 398B | 94B | **23.6%** | 50.0% |
| DeepSeek-V3 | 671B | 37B | **5.5%** | 95.1% |

**ANSWER: a factor of 4.3.**

**MECHANISM: a Mamba block is not an MLP with a router in front of it.** Half of
Jamba's layers have no experts to skip, so every parameter in them is active on
every token.

**FINDING: Jamba's floor is the architecture** — 50% against DeepSeek's 4.9%.

**FINDING: the calculator cannot price any of this.** Its callables are
`fmt_bytes`, `kv_cache_bytes`, `ssm_state_bytes`, `main`; both computing ones
return bytes of cache.

### 5 — a real diagonal reaches two points of the circle

```text
e^(i·0.7) × (0.3 − 1.2i)                  = (1.0025138809, −0.7245453186)
[[cos, −sin], [sin, cos]] · (0.3, −1.2)   = (1.0025138809, −0.7245453186)
worst disagreement across 5 angles: 0.0e+00
```

**ANSWER: a complex diagonal state matrix *is* RoPE's rotation.** "Data-dependent"
means the angle comes from the input rather than the position index.

**MECHANISM: `|e^(iθ)| = 1`.** The norm is preserved and rotations compose —
`e^(i0.5)·e^(i1.1) = e^(i1.6)` to 1.1e-16 — so the state can record *where it is
on the circle*, which a magnitude cannot carry.

**FINDING: a real multiplier reaches exactly `{0, π}`** — 2 phase shifts of
infinitely many. Mamba-2's scaled identity is the extreme case.

**FINDING: the calculator has no state matrix.** A complex state of the same
width would take exactly **2×** the bytes — the one consequence of Mamba-3's
change this module could have shown.
