<!-- generated:start -->
# 06-speech-and-audio / 16-anti-spoofing-audio-watermarking

Solutions to all 3 exercises. Source: [lesson page](https://yennj12.js.org/ai-engineering-from-scratch/phases/06-speech-and-audio/16-anti-spoofing-audio-watermarking/) · upstream spec
`phases/06-speech-and-audio/16-anti-spoofing-audio-watermarking/docs/en.md`

```bash
uv run demo practice run 16-anti-spoofing-audio-watermarking --ex 1
uv run demo explain 16-anti-spoofing-audio-watermarking --ex 1
uv run pytest demos/phases/06-speech-and-audio/16-anti-spoofing-audio-watermarking
```

Solutions import the lesson's own `code/` rather than copying it, so every check
compares against the reference implementation and not a fork of it (`DESIGN D5`).

| # | Exercise | Kind | Tier | Ships |
|---|---|---|---|---|
| 1 | Easy. Run `code/main.py`. Toy detector + toy watermark embed/detect on synthetic audio. | code | T1 | `ex01_both_printed_numbers_are_artifacts.py` |
| 2 | Medium. Install `audioseal`, embed a 16-bit payload in a TTS output, re-decode. Corrupt the a… | code | T0 | `ex02_noise_improves_the_bit_accuracy.py` |
| 3 | Hard. Fine-tune a RawNet2 or AASIST on ASVspoof 2019 LA. Measure EER. Test on a held-out set… | code | T1 | `ex03_a_margin_that_is_the_residue_of_two_cancellations.py` |
<!-- generated:end -->

## Answers

`code/main.py` prints two measurements and labels each with the production figure
it should be compared against: an **EER of 100.00%** beside AASIST's 0.42%, and a
**bit accuracy of 31.2%** beside AudioSeal's >99%. Neither number is a
measurement of what its label says. The detector is perfect and reported
backwards; the watermark detector never reads the payload at all. The third
exercise asks what happens off-distribution, and the answer is that the margin
this detector lives on is the residue of two effects ten times its size.

Exercises 1 and 3 run at **T1** — each pushes tens of clips through the lesson's
O(N²) DFT. Exercise 2 is **T0**.

### 1 — both numbers it prints as results are artifacts

**ANSWER: `EER ~ 100.00%` is `1 − EER`.** A coin scores 50%, and any detector
above that improves by flipping its sign, so 100% means perfect separation read
backwards. It is:

| | range | mean |
|---|---|---:|
| 20 real clips | 0.190020 – 0.197743 | 0.194239 |
| 20 fake clips | 0.210054 – 0.211017 | 0.210544 |
| gap | **0.012311** | — |
| **correct-polarity EER** | — | **0.0000** |

The sweep counts `far` as the fakes scoring *above* the threshold — for a score
that rises with fakeness that is the detection rate, not the false-accept rate.

**ANSWER: the watermark detector never sees the payload.**
`toy_watermark_detect` returns `1 if audio[idx] > 0 else 0` — the sign of the
carrier at 16 fixed indices. Run on the **unwatermarked** clip it returns the
same 16 bits byte for byte, and inverting all 16 payload bits changes nothing.

**MECHANISM: the step is 467× too small.** `toy_watermark_embed` adds `±0.0005`
where the carrier runs to **0.2335** at the probe indices, and **0 of 16** probes
sit close enough to zero for it to change a sign.

**CONTROL: over 500 random payloads the mean accuracy is 0.5000** exactly. The
printed 31.25% is 5 of 16 — one draw of that coin.

**CONTROL: the EER grid cannot express the number quoted beside it.** At 20 and
20, both error rates move in steps of `1/20`, so the reachable EERs are multiples
of **2.5 pp**; 0.42% is not on the grid. The doc's Build It Step 1 also defines
`spectral_rolloff` and `is_suspicious`, and `code/main.py` defines neither.

### 2 — adding noise improves the bit accuracy

`audioseal`, `silentcipher`, `torch`, `torchaudio` and `lameenc` are all absent
and there is no TTS output to embed in, so the corruption sweep runs against the
scheme the lesson does ship.

| noise SNR | Bit Recovery Accuracy |
|---:|---:|
| 100 dB (effectively clean) | **0.3125** |
| 40 dB | 0.3219 |
| 20 dB | 0.4000 |
| 10 dB | 0.4297 |
| 0 dB | **0.4594** |

**ANSWER: the curve rises toward chance instead of falling.** The readout is
independent of the payload, so the clean value is a fixed wrong pattern — 5 of 16
— and noise walks it toward the 0.5000 a coin scores. Corrupting the audio makes
the reported number *better*. There is nothing in the signal to degrade.

**MECHANISM: what a blind sign read would need.** The step must outvote the
carrier at every probe — above **0.2335**, which is **467×** the shipped `0.0005`
and **0.79** of the clip's own peak of 0.2937. A watermark that loud is not a
watermark.

| step | BRA, no noise |
|---:|---:|
| 0.0005 (shipped) | 0.3125 |
| 0.05 | 0.6875 |
| **0.2335** | **1.0000** |

**ANSWER: at a step that works, the curve the exercise asks for exists** —
**1.0000** from clean down to 20 dB SNR, **0.9891** at 10 dB, **0.9234** at 0 dB.

**FINDING: even repaired it is 16 pinpricks whose addresses are the message.**
The payload occupies **16 of 16,000 samples (0.10%)** at indices fixed by
`len(audio) // n_bits`, so dropping one leading sample takes BRA to **0.5000**.
AudioSeal adds a per-sample field across the whole clip, which is why a crop or a
resample leaves it readable.

### 3 — the margin is the residue of two effects ten times its size

`torch`, `torchaudio`, `transformers` and `soundfile` are absent and ASVspoof
2019 LA is not here, so nothing is fine-tuned. The off-distribution families
below come from a generator that reproduces `synth_fake_speech` **element for
element** at the shipped settings, so only the parameter named in each row
differs.

| family | mean high-band ratio | vs real | EER |
|---|---:|---:|---:|
| real (noise 0.02, no artifact) | 0.193926 | — | — |
| fake generator, **artifact removed** | **0.038557** | **−0.155369** | 1.0000 |
| fake **as shipped** (6000 Hz, noise 0.002) | 0.210493 | **+0.016567** | **0.0000** |
| artifact moved to **3900 Hz** | — | — | **1.0000** |
| 6000 Hz kept, **real noise floor** | **0.317614** | +0.123688 | 0.0000 |

**ANSWER: in distribution the EER is 0.0000, on a margin of 0.016567.**

**MECHANISM: that margin is what two much larger effects leave behind.** The
fake's noise floor is 10× lower, which pulls its high-band ratio down by
**0.155**; its 6 kHz artifact pushes it back up by **0.172**. Each term is
**9.4×** and **10.4×** the margin the classifier lives on.

**ANSWER: 100 Hz off distribution the detector inverts rather than degrades.**
`toy_detector_score` sums `spec[len(spec)//2:]`, which at `n_fft=256` and 16 kHz
begins at **4000 Hz**. Move the artifact from 6000 Hz to 3900 Hz, change nothing
else, and the EER goes **0.0000 → 1.0000** — every spoof now ranks as more real
than every real clip. A hundred hertz, on a boundary that is a hard-coded `//2`.

**FINDING: a parameter that is not an artifact moves the score further than the
artifact does.** Keeping the 6 kHz tone and restoring the real noise floor gives
**0.317614** — **6.5×** the decision margin further from real than the shipped
fake — with nothing about the spoof changed.

**CONTROL:** at 10 and 10 the EER grid is multiples of **5.0 pp**, so neither the
0.42% the doc quotes for AASIST nor the 7.23% it quotes for ASVspoof 5 is a
number this fixture could produce.

The run the exercise asks for is `ASVspoof2019` LA (~25 GB) through AASIST for a
few epochs on one GPU, then the same detector on F5-TTS output. Budget for the
OOD set to be the measurement, not the footnote: the in-distribution EER here is
0.0000 and the off-distribution EER is 1.0000, and that is the whole lesson.
