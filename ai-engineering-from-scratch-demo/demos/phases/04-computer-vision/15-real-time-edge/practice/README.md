<!-- generated:start -->
# 04-computer-vision / 15-real-time-edge

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/04-computer-vision/15-real-time-edge/) · upstream spec
`phases/04-computer-vision/15-real-time-edge/docs/en.md`

```bash
uv run demo practice run 15-real-time-edge --ex 1
uv run demo explain 15-real-time-edge --ex 1
uv run pytest demos/phases/04-computer-vision/15-real-time-edge
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | (Easy) Measure p50 latency for `resnet18`, `mobilenet_v3_small`, `efficientnet_v2_s`, and `co… | code | T1 | `ex01_flops_latency_divergence.py` |
| 2 | (Medium) Apply post-training static quantisation to `mobilenet_v3_small`. Report FP32 vs INT8… | code | T1 | `ex02_int8_ptq_signal_floor.py` |
| 3 | (Hard) Export `convnext_tiny` to ONNX, run it through `onnxruntime` with the `CPUExecutionPro… | code | T1 | `ex03_graph_fusion_without_onnx.py` |
<!-- generated:end -->

## Answers

The lesson's own rule — "use FLOPs for architecture search, use on-device latency
for deployment decisions" — turns out to be understated. All three exercises here
run into the same wall: **the quantity the exercise asks you to report is not the
quantity that decides anything**, and in two of the three the lesson's own helper
computes it wrongly or its own recipe will not run. Wall-clock figures below come
from one run on the development host (arm64 macOS, `torch.set_num_threads(2)`) and
are machine-dependent; every other number is exact and reproduces on any host.

### 1 — flops latency divergence

At 224×224 on CPU, with `weights=None` so every model is randomly initialised:

| model | params | lesson GFLOPs | corrected GFLOPs | p50 ms | p95 ms | published top-1 | top-1/ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| mobilenet_v3_small | 2.54M | 0.113 | 0.113 | 21.5 | 21.6 | 67.67 | 3.14 |
| resnet18 | 11.69M | 3.628 | 3.628 | 11.1 | 11.4 | 69.76 | 6.28 |
| efficientnet_v2_s | 21.46M | 5.700 | 5.700 | 34.1 | 34.4 | 84.23 | 2.47 |
| convnext_tiny | 28.59M | **0.639** | **8.911** | 268.9 | 272.0 | 82.52 | 0.31 |

**ANSWER: only last place is stable.** On this host `resnet18` wins
accuracy-per-ms at 6.28 points/ms and `convnext_tiny` loses it at 0.31, a 20×
gap. The top-1 column is torchvision's published `acc@1` for the *pretrained*
weights, read out of the `Weights` enum's `.meta` dict with no download — it is
not measured here, because there is nothing to measure.

**ANSWER: no accuracy is available locally.** Over one fixed batch of 32 probe
images the four random-init models emit **1, 1, 2 and 32** distinct argmax classes
out of 1,000. Two of the four predict the same class for every input. The
exercise's "best accuracy-per-ms" cannot be answered without either a download or
borrowed metadata.

**FINDING: FLOPs does not rank latency.** Dividing corrected FLOPs by measured
p50 gives 0.005, 0.326, 0.167 and 0.033 GFLOP/ms — a **62× spread** across four
models on one machine. `resnet18` converts FLOPs into time best and
`mobilenet_v3_small` worst, despite the latter having 32× fewer of them.

**MECHANISM: the lesson's `flops_estimate` is 13.94× low on ConvNeXt.** Its
`linear_hook` adds `2 * in_features * out_features` once per `nn.Linear` call with
no factor for the spatial positions the layer runs at. ConvNeXt does all of its
pointwise mixing in `nn.Linear` over a permuted (N, H, W, C) tensor, so up to
3,136 positions per block are missed. Restore the factor, halve to MACs, and
evaluate at each weight entry's own crop size, and the count reproduces
torchvision's published `_ops` exactly:

| model | corrected GMACs | torchvision `_ops` @ crop |
|---|---:|---|
| mobilenet_v3_small | 0.057 | 0.057 @ 224 |
| resnet18 | 1.814 | 1.814 @ 224 |
| efficientnet_v2_s | 8.366 | 8.366 @ **384** |
| convnext_tiny | 4.456 | 4.456 @ 224 |

Worst relative error **8.6e-03** (mobilenet, where the published figure is rounded
to three decimals). `parameter_count` matches `num_params` 4/4.

**CONTROL: the reference's p95 and p99 are the same number.** `measure_latency`
defaults to `iters=20` and indexes the sorted list at `int(20*0.95) = 19` — the
last element — while `p99_ms` is `times[-1]`, also 19. One run reports p50 12.85,
p95 14.01, p99 14.01 ms. Every tail figure the helper prints is a single sample of
p100 until `iters` reaches **21**, the first count at which the two indices differ.

**CONTROL: the ranking is a property of the host.** Re-timing the two cheapest
models at 1, 2 and 4 CPU threads:

| threads | mobilenet_v3_small | resnet18 | ratio |
|---:|---:|---:|---:|
| 1 | 10.1 ms | 13.2 ms | 1.30× |
| 2 | 22.7 ms | 12.8 ms | 0.56× |
| 4 | 27.9 ms | 10.7 ms | 0.38× |

The winner flips with the thread count alone. Nothing about the 32× FLOP
advantage survives into a 10× latency advantage at any setting.

### 2 — int8 ptq signal floor

CIFAR-10 and pretrained weights are both unreachable offline, so the held-out set
is 32 fixed Gaussian images and the score is **top-1 agreement** between the FP32
model and its INT8 copy — the quantity accuracy loss proxies. The pipeline is
`torch.ao.quantization.quantize_fx` on the `qnnpack` backend (chosen from
`supported_engines`, since the docs' hardcoded `x86` is not offered on this host).

| arm | state dict | vs FP32 | p50 | agreement | SQNR |
|---|---:|---:|---:|---:|---:|
| FP32 | 6,242,327 B | 1.00× | 20.7 ms | — | — |
| static INT8, BatchNorm re-estimated | 1,645,059 B | **3.79×** | 2.7 ms | **1.000** | **25.7 dB** |
| static INT8, model as built | 1,645,059 B | 3.79× | — | **0.000** | **0.00 dB** |
| dynamic INT8 (`nn.Linear` only) | 4,443,599 B | 1.40× | 20.5 ms | — | — |

**ANSWER: static INT8 is 3.79× smaller and costs nothing in agreement** — once
the model has a signal to quantise.

**FINDING: quantise `mobilenet_v3_small` as built and the INT8 output is pure
noise.** Agreement 0.000 at **0.00 dB SQNR**, because the FP32 logits themselves
have standard deviation **4.23e-10**: the quantisation step is wider than the
signal it encodes. Nothing in calibration or conversion failed. A PTQ report that
listed only compression would have shown a clean 3.8× and no warning at all.

**MECHANISM: untrained BatchNorm is the identity in eval, so 28 halving gates go
unopposed.** All 34 `BatchNorm2d` layers still hold `running_mean=0,
running_var=1`, which in `.eval()` is an identity map, while the network has 19
`Hardswish` and 9 `Hardsigmoid` gates whose measured small-signal gain is
**0.500** each. Mean |activation| across the 13 feature stages:

| stage | 0 | 1 | 2 | 3 | 4-6 | 7-8 | 9-11 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mean abs | 2.5e-01 | 4.0e-02 | 7.7e-03 | 7.9e-03 | 1.8e-04 | 4.8e-06 | 6.6e-08 | 1.9e-08 |

A surviving fraction of **7.8e-08**, within 21× of the 3.7e-09 that 28 halvings
alone predict. The flat runs are residual blocks whose branch output has already
decayed to nothing.

**CONTROL: eight training-mode batches and no gradient step repair it.**
Re-estimating the BatchNorm running statistics over 8 forwards of 16 images lifts
the FP32 logit spread from 4.23e-10 to **8.25e-05**, the INT8 copy from 0.00 to
**25.7 dB**, and agreement from 0.000 to **1.000**. The quantiser was always
correct; the dynamic range it had to observe is what changed — which is exactly
why PTQ calibration data has to come from the deployed distribution.

**MECHANISM: the INT8 weights sit exactly on their scale lattice.** The first
packed weight is `torch.qint8` under `per_tensor_affine` with scale **2.643e-03**.
Rebuilding it as `(int_repr - zero_point) * scale` reproduces `dequantize()` with
max error **0.0** in float32 and 1.16e-08 in float64 — the latter being float32
rounding, 4.4e-06 of one step. Its integers span [-128, 124] over **169** of the
256 values 8 bits allow. The 4× size drop is a change of alphabet, not of
arithmetic.

**CONTROL: dynamic quantisation, the cheap option, never touches a convolution.**
It compresses 1.40× against static's 3.79× and leaves p50 at 1.01×, because it can
only reach the **2** `nn.Linear` layers — 39.3% of the parameters and almost none
of the work — while all **52** convolutions stay in FP32. The lesson's "small
speedup" for dynamic PTQ is, on a conv net, no speedup.

Two Python warnings are recorded rather than silenced, both from torch:
`torch.ao.quantization is deprecated and will be removed in 2.10`
(DeprecationWarning) and a UserWarning that `torch.quantize_per_tensor ... and
other quantized tensor creation functions` are deprecated. On a qnnpack host the
dynamic arm also emits a C++ log line, `qnnpack incorrectly ignores
reduce_range`, straight to stderr.

### 3 — graph fusion without onnx

**ANSWER: the exercise cannot be run here.** `torch.onnx.export` refuses on both
paths and the runtime is absent:

| path | outcome |
|---|---|
| default (dynamo) | `ModuleNotFoundError: No module named 'onnxscript'` |
| `dynamo=False` (legacy) | `OnnxExporterError: Module onnx is not installed!` |
| `importlib.util.find_spec` | no `onnx`, no `onnxscript`, no `onnxruntime` |

None may be fetched, so there is no ONNX Runtime latency on this host. Everything
below uses the one graph optimiser that is available offline, `torch.jit.freeze`
plus `torch.jit.optimize_for_inference`.

**ANSWER: "the first layer where ONNX Runtime is faster" is not well posed.** The
question presumes the two graphs have the same layers, which is exactly what a
fusing runtime destroys. Freezing takes convnext_tiny's traced graph from **749**
nodes to **390**, and `aten::linear` does not survive it: all **37** become 37
`aten::matmul` plus 37 `aten::add`. A per-layer comparison against eager has
nothing left to line up; ORT's own profiler is the only instrument for that
question.

**FINDING: the available graph optimiser is faithful and buys no real speed.** It
reproduces the eager logits to **2.68e-07** and runs at p50 171.7 ms against
eager's 177.2 ms — **1.03×**, well inside the noise of a 7-iteration timing on a
loaded laptop. "Compile it and it gets faster" is a hypothesis to measure, not a
property of graph runtimes.

**MECHANISM: the fusion ORT is usually credited for cannot fire on this model.**
Module census of `convnext_tiny`:

| Conv2d | BatchNorm2d | LayerNorm | LayerNorm2d | Permute | Linear |
|---:|---:|---:|---:|---:|---:|
| 22 | **0** | 18 | 5 | 36 | 37 |

Conv+BN folding has nothing to fold. What the graph holds instead is **46**
`aten::permute` and **23** `aten::layer_norm` calls against 22 convolutions — the
(N,C,H,W) → (N,H,W,C) shuffles ConvNeXt needs so that LayerNorm and Linear can
work on the last axis.

**CONTROL: on resnet18 the same fusion is exact.** After re-estimating its running
statistics over 4 training-mode batches, all **20** Conv2d→BatchNorm2d pairs fold
through `torch.nn.utils.fusion.fuse_conv_bn_eval` into single convolutions: 20
folded, 0 BatchNorm left, 11,181,642 → 11,176,842 parameters (**4,800** gone), and
the folded network reproduces the original's logits to **7.75e-07**. The fusion is
real; convnext_tiny is simply not built for it.

**CONTROL: freezing deletes only bookkeeping.** Of the 359 nodes removed, not one
is arithmetic:

| op | traced | frozen |
|---|---:|---:|
| `aten::permute` | 46 | 46 |
| `aten::layer_norm` | 23 | 23 |
| `aten::_convolution` | 22 | 22 |
| `aten::gelu` | 18 | 18 |
| `prim::GetAttr` | 384 | **0** |
| `prim::ListConstruct` | 158 | **0** |

Constant-folding the frozen weights is the whole win, and it buys no arithmetic —
while `aten::linear` is *split* into two nodes, so the optimised graph carries
more compute ops than the traced one.

Six warnings are recorded rather than silenced, all raised inside torch: that
`torch.jit.trace` and `torch.jit.trace_method` are not supported on Python 3.14+,
that `torch.jit.freeze` and `torch.jit.optimize_for_inference` are deprecated in
favour of `torch.compile`, and two from the legacy ONNX exporter before it gives
up.
