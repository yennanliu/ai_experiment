<!-- generated:start -->
# 12-multimodal-ai / 03-blip2-qformer-bridge

Solutions to all 5 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/12-multimodal-ai/03-blip2-qformer-bridge/) · upstream spec
`phases/12-multimodal-ai/03-blip2-qformer-bridge/docs/en.md`

```bash
uv run demo practice run 03-blip2-qformer-bridge --ex 1
uv run demo explain 03-blip2-qformer-bridge --ex 1
uv run pytest demos/phases/12-multimodal-ai/03-blip2-qformer-bridge
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Implement the cross-attention block in PyTorch. Verify that with 32 queries and 256 keys/valu… | code | T0 | `ex01_the_row_sum_test_cannot_fail.py` |
| 2 | In BLIP-2 stage 1 the Q-Former runs three losses simultaneously: ITC, ITM, ITG. Write the for… | explain | T0 | prose, below |
| 3 | Compare parameter counts: Q-Former (12 layers, 768 hidden) vs a 2-layer MLP projector (1408 →… | code | T0 | `ex03_the_bridge_pays_back_at_a_twenty_million_parameter_llm.py` |
| 4 | Read Section 3.2 of the BLIP-2 paper (arXiv:2301.12597) on how the Q-Former is initialized. E… | explain | T0 | prose, below |
| 5 | For a 10-minute video at 1 FPS sampled to 60 frames, compute the per-frame token cost at (Q-F… | code | T0 | `ex05_both_of_them_fit_so_the_question_starts_at_227_frames.py` |
<!-- generated:end -->

## Answers

Three code solutions and two prose ones, all T0 and stdlib. The code ones
measure the lesson's own `main.py` and its stated shapes; nothing here is
quoted that could be computed.

### 1 — the row-sum test cannot fail

**ANSWER: 32 × 256, and every row sums to 1 to 2.2e-16.** That is the whole of
the check the exercise asks for — and it is a property of `softmax`, so it
would pass on an untrained block, a randomly permuted one, or a constant one.

**FINDING: untrained, this attention is already almost one-hot.**

| | measured | uniform baseline |
|---|---:|---:|
| mean row entropy | **0.321** nats | 5.545 |
| mean top weight | **0.876** | 0.0039 |

The lesson's own `summarize_attention` prints entropy against that uniform
baseline as though concentration were evidence of training. At initialisation it
is already there.

**FINDING: the 1/√d scale is defeated by the lesson's own init.** `mat()` draws
N(0, 1), so `Q = W_q q` has entries of std **3.65** — √d = 4, not 1 — and the
logits *after* the 1/√d division have std **14.53**, which is d = 16.

| weight scale | mean entropy |
|---|---:|
| lesson, 1.0 | 0.321 |
| 1/√d | 5.066 |
| 1/d | **5.543** (uniform is 5.545) |

The saturation is an initialisation bug, not a property of cross-attention.

**FINDING: the lesson ships 8 queries over 64 patches.** `NUM_QUERY = 8`,
`NUM_PATCH = 64` — the 32 × 256 shape the exercise names is **16×** the
attention entries the module builds, and has to be constructed by the reader.

### 2 — which loss needs the text encoder path

Drawing on **Two-stage training**, which is where the lesson lists the three
stage-1 objectives and says the text path shares the self-attention and FFN
weights with the query path.

The three forward signatures, in pseudo-code:

```text
ITC  itc(image_embeds, text_ids) -> loss
     Q = qformer(queries, cross_attend=image_embeds, mask=UNIMODAL)   # 32 x 768
     T = qformer(text_ids,          cross_attend=None, mask=UNIMODAL)  # CLS  x 768
     sim = max_i cosine(Q[i], T[CLS])        # max over the 32 queries
     loss = symmetric_infonce(sim_matrix_over_batch)

ITM  itm(image_embeds, text_ids) -> loss
     Z = qformer([queries; text_ids], cross_attend=image_embeds, mask=BIDIRECTIONAL)
     logit = mean_i binary_head(Z[i])        # over the 32 query positions only
     loss  = bce(logit, is_matching_pair)

ITG  itg(image_embeds, text_ids) -> loss
     Z = qformer([queries; text_ids], cross_attend=image_embeds, mask=CAUSAL_ON_TEXT)
     loss = cross_entropy(lm_head(Z[text]), text_ids[1:])
```

**ANSWER: all three require it, so the question's presupposition does not
hold.** There is no stage-1 objective that the query path can produce alone —
ITC needs a text CLS to be contrastive against, ITM needs text tokens in the
sequence to classify, and ITG needs text tokens to predict.

What actually distinguishes the three is the **attention mask**, and that is the
answer worth having:

| loss | mask | do queries see text? | does text see queries? |
|---|---|---|---|
| ITC | unimodal | no | no |
| ITM | bidirectional | **yes** | **yes** |
| ITG | causal on text | no | **yes** |

If the question is read strictly — which one needs the text path as an
*encoder*, jointly with the image — the answer is **ITM**, the only objective
where the two paths are coupled in a single forward. ITC runs the text path in a
separate, masked-apart pass, and ITG uses it as a *decoder* rather than an
encoder. The shared weights are what make one module serve all three roles;
the mask is what keeps the roles apart.

### 3 — the bridge pays back at a 20M-parameter LLM

**ANSWER: 18.5M parameters.** Per training image the LLM is spared 224 tokens of
forward-and-backward, and the bridge charges its own 129.7M extra parameters
over the 32 queries it runs. Setting `6·L·224 = 6·ΔP·32` gives L = ΔP/7.

| LLM | past break-even by |
|---|---:|
| OPT-2.7B | **146×** |
| OPT-6.7B | 362× |
| Flan-T5-XXL 11B | 594× |

There is no LLM scale at which the MLP projector is cheaper to *train*. The
Q-Former's real cost is stage 1 — a pretraining run the projector does not need
— and that is not a FLOPs-per-image quantity at all.

**FINDING: the lesson's own shapes do not reconstruct 188M.**

| assumption | total |
|---|---:|
| cross-attention every layer + BERT embeddings + 32 queries + 768→4096 proj | **152.2M** |
| cross-attention every *other* layer | 132.2M |
| the paper | **188M** |

**19.0% short**, and the other reading of "12 layers" is further away.

**FINDING: the comparison is between a projector and a language encoder.** The
2-layer MLP is **22.55M**; the Q-Former is 6.75× that, of which **23.8M** is the
BERT embedding table and **125.2M** the blocks.

**FINDING: the break-even moves the wrong way with LLaVA's shape.** At 576
visual tokens the saving is 544 per image and break-even drops to **7.6M** — the
bridge is easiest to justify against the configuration that abandoned it.

### 4 — why BERT-base initialisation converges faster

Drawing on **Architecture**, which states that the Q-Former's text path "shares
the self-attention and FFN weights with the query path".

Two sentences, as asked:

> Initialising from BERT-base means only the cross-attention sublayers start
> random — **26%** of the bridge's parameters by the count in exercise 3, 40.1M
> of 152.2M — so 74% of the module begins as a working language encoder rather
> than as noise.
> That accelerates convergence because the weights shared between the two paths
> are exactly the ones BERT already trained: from step one the text path emits
> usable sentence representations for ITC to contrast and usable token
> predictions for ITG to score, so the image-text pairs are spent learning the
> *bridge* instead of re-learning English from 129M captions.

The structural point underneath is that sharing is what makes the initialisation
transferable. If the query path had its own self-attention and FFN, a BERT
checkpoint would only warm-start the text path, and the query path — the half
the bridge exists for — would still start from noise. Because the weights are
shared, the queries begin life operating inside a representation space that
already has linguistic structure, which is also why ITC's comparison between a
query output and a text `[CLS]` is meaningful before any training at all.

The gradient argument is the same one in different clothes: random
cross-attention on top of a trained backbone is a small perturbation that the
backbone can absorb, whereas random everything means the cross-attention
receives gradients from a text path that is itself still noise.

### 5 — both of them fit, so the question starts at 227 frames

| | 60 frames | share of 128k |
|---|---:|---:|
| Q-Former, 32/frame | **1,920** | 1.5% |
| MLP @ 256 | 15,360 | 11.7% |
| MLP @ 576 | **34,560** | 26.4% |

**ANSWER: both fit, with room to spare.** The question the exercise asks does
not discriminate between the two configurations it names.

**FINDING: the decimation happens before the question.** 10 minutes at 1 FPS is
**600** frames, so "sampled to 60 frames" is a **10×** drop applied before
either bridge is chosen. Unsampled, the projector needs **345,600** tokens —
**2.64×** the window — and only the Q-Former survives.

**FINDING: the question begins at 227 frames.**

| | frames that fit 128k | at 1 FPS |
|---|---:|---|
| MLP @ 576 | **227** | 3m47s |
| MLP @ 256 | 512 | 8m32s |
| Q-Former | **4,096** | 68m16s |

An **18.04×** span — 576/32 plus the floor at 227 — because both arms are linear
in frames and the ratio never moves.

**FINDING: 576 is not this lesson's own projector.** The lesson states a ViT
producing **256** patch tokens; 576 is LLaVA's 24×24 grid from Lesson 12.05. The
exercise's two numbers come from two different vision towers.
