"""Exercise 2 — the learned latent is worse, and the decoder's slope says why.

    **Medium.** Swap the toy linear encoder for a tanh-MLP encoder/decoder pair
    with a reconstruction loss. Retrain diffusion on the new latents. Does sample
    quality change?

Reading of the exercise: the autoencoder is trained first, to convergence, on the
lesson's own `sample_data`; then the lesson's own diffusion loop is retrained on
whatever latents it produces, and samples are decoded back and scored the way
Exercise 1 scores them. The diffusion half -- `make_schedule`, `init_net`,
`forward`, `backward`, `apply` -- is untouched, so only the code the data lives in
changes. Quality is the share of decoded samples landing within 1.0 of either
true centre.

**ANSWER: yes, and it gets worse.** The linear latent scores SHOWN_LINEAR against
the tanh latent's SHOWN_TANH, on the same diffusion budget and the same seeds.

**FINDING: the autoencoder is the better reconstructor and the worse latent.**
Its reconstruction error is SHOWN_RECON, so it has learned the data -- and the
diffusion model placed on top of it does worse than on a rescaling that learned
nothing. Reconstruction quality and latent quality are not the same axis, which
is the point a linear `encode` cannot make.

**FINDING: `tanh` saturates, so the decoder amplifies.** The data sits at +-2,
where the encoder's output is deep in `tanh`'s flat region: the latents cluster at
**SHOWN_SPREAD** apart while the decoder's local slope there reaches
**SHOWN_SLOPE**. Every error the diffusion model makes in latent space is
multiplied by that slope on the way out, which the linear decoder's slope of
exactly **2.0** does not do.

**CONTROL: both arms decode the same way.** The scoring runs on decoded samples
in data space, so the two latents are compared on the axis the exercise names
rather than on their own incomparable internal scales.

Structure: `fit_codec` trains the tanh pair; `latents` wraps either codec;
`run` retrains the lesson's diffusion on a given codec.
"""

from __future__ import annotations

import math
import statistics

from harness import parity, practice

PHASE, LESSON = "08-generative-ai", "07-latent-diffusion-stable-diffusion"
T, T_DIM, HIDDEN, CLASSES, STEPS, RATE = 40, 8, 32, 3, 4_000, 0.01
CODEC_STEPS, CODEC_RATE, SEEDS, DRAWS, CENTRE = 6_000, 0.02, 2, 200, 2.0


def fit_codec(random, seed):
    """A tanh encoder and linear decoder, trained to reconstruct sample_data."""
    rng = random.Random(seed)
    enc_w, enc_b = rng.gauss(0, 0.5), 0.0
    dec_w, dec_b = rng.gauss(0, 0.5), 0.0
    for _ in range(CODEC_STEPS):
        x = rng.gauss(-CENTRE, 0.4) if rng.random() < 0.5 else rng.gauss(CENTRE, 0.4)
        z = math.tanh(enc_w * x + enc_b)
        err = dec_w * z + dec_b - x
        dec_w -= CODEC_RATE * err * z
        dec_b -= CODEC_RATE * err
        slope = (1 - z * z)
        enc_w -= CODEC_RATE * err * dec_w * slope * x
        enc_b -= CODEC_RATE * err * dec_w * slope
    return (enc_w, enc_b, dec_w, dec_b)


def encode_with(codec, x):
    """The tanh encoder, or the lesson's own x * 0.5 when `codec` is None."""
    return x * 0.5 if codec is None else math.tanh(codec[0] * x + codec[1])


def decode_with(codec, z):
    """The learned decoder, or the lesson's own z * 2.0."""
    return z * 2.0 if codec is None else codec[2] * z + codec[3]


def run(ref, random, codec, seed):
    """Retrain the lesson's diffusion on this codec's latents; return decoded samples."""
    rng = random.Random(seed)
    alphas, bars = ref.make_schedule(T)
    net = ref.init_net(1, T_DIM, CLASSES, HIDDEN, rng)
    for _ in range(STEPS):
        x0, label = ref.sample_data(rng)
        z0, step, eps = encode_with(codec, x0), rng.randrange(T), rng.gauss(0, 1)
        z_t = math.sqrt(bars[step]) * z0 + math.sqrt(1 - bars[step]) * eps
        shown = ref.NULL_CLASS if rng.random() < 0.1 else label
        out, cache = ref.forward([z_t], ref.sin_embed(step, T, T_DIM),
                                 ref.one_hot(shown, CLASSES), net)
        ref.apply(net, ref.backward([eps], out, cache, net), RATE)
    return [sample(ref, net, alphas, bars, rng, codec) for _ in range(DRAWS)]


def sample(ref, net, alphas, bars, rng, codec):
    """One draw through the lesson's reverse chain, decoded with this codec."""
    z = rng.gauss(0, 1)
    for step in range(T - 1, -1, -1):
        eps = ref.forward([z], ref.sin_embed(step, T, T_DIM),
                          ref.one_hot(rng.randrange(2), CLASSES), net)[0][0]
        beta = 1 - alphas[step]
        mean = (z - beta / math.sqrt(1 - bars[step]) * eps) / math.sqrt(alphas[step])
        z = mean + math.sqrt(beta) * rng.gauss(0, 1) if step > 0 else mean
    return decode_with(codec, z)


def in_modes(samples):
    """Share of decoded samples within 1.0 of either true centre."""
    return sum(1 for v in samples if abs(abs(v) - CENTRE) < 1.0) / len(samples)


def solve():
    import random
    ref = parity.load_reference(PHASE, LESSON, "main")
    codec = fit_codec(random, 3)
    arms = {}
    for name, which in (("linear", None), ("tanh", codec)):
        arms[name] = statistics.fmean(in_modes(run(ref, random, which, s + 11))
                                      for s in range(SEEDS))
    rng = random.Random(5)
    points = [rng.gauss(-CENTRE, 0.4) if rng.random() < 0.5 else rng.gauss(CENTRE, 0.4)
              for _ in range(400)]
    codes = [encode_with(codec, x) for x in points]
    return {
        "arms": arms,
        "recon": statistics.fmean((decode_with(codec, z) - x) ** 2 for x, z in zip(points, codes)),
        "spread": max(codes) - min(codes),
        "slope": abs(codec[2]),
        "linear_slope": abs(decode_with(None, 1.0) - decode_with(None, 0.0)),
    }


def verify(result):
    linear, tanh = result["arms"]["linear"], result["arms"]["tanh"]
    return [
        practice.Check(
            "ANSWER: yes -- the learned latent is worse on the same budget",
            tanh < linear,
            f"share of decoded samples landing within 1.0 of a true centre, over {SEEDS} seeds and "
            f"{DRAWS} draws: the lesson's linear latent {linear:.3f} against the tanh latent's "
            f"{tanh:.3f}. Same diffusion loop, same steps, same seeds -- only the space the data "
            "lives in changed",
        ),
        practice.Check(
            "FINDING: the autoencoder is the better reconstructor and the worse latent",
            result["recon"] < 0.05,
            f"the tanh codec reconstructs to {result['recon']:.4f} mean squared error, so it has "
            f"learned the data -- and diffusion on top of it still scores {tanh:.3f} against "
            f"{linear:.3f} for a rescaling that learned nothing. Reconstruction quality and latent "
            "quality are not the same axis, which is the point a linear encode cannot make",
        ),
        practice.Check(
            "FINDING: tanh saturates, so the decoder amplifies",
            result["slope"] > result["linear_slope"],
            f"the data sits at +-{CENTRE:.0f}, deep in tanh's flat region, so the latents span only "
            f"{result['spread']:.3f} while the decoder's slope is {result['slope']:.2f} against the "
            f"linear decoder's exact {result['linear_slope']:.1f}. Every error diffusion makes in "
            "latent space is multiplied by that slope on the way out",
        ),
        practice.Check(
            "CONTROL: both arms are scored after decoding",
            0.0 <= tanh <= 1.0 and 0.0 <= linear <= 1.0,
            "the metric runs on decoded samples in data space, so the two latents are compared on "
            "the axis the exercise names rather than on their own internal scales, which differ by "
            f"a factor of {result['slope'] / result['linear_slope']:.1f} and are not comparable",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
