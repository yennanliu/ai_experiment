"""Exercise 4 — the "comparable" is exact, and what is left is the parameter count.

    Compute training FLOPs for Emu3-7B at 300B tokens and compare to Stable
    Diffusion 3. Which was more expensive to train?

Reading of the exercise: 6ND is applied to both arms, which for the diffusion
model is a stated assumption -- a DiT's forward pass is the same order as a
transformer's -- and the comparison is made in tokens first, because the lesson
pairs "~300B tokens" with "~300M image-steps" and those are only comparable
after a conversion the lesson does not write down.

**ANSWER: 1.26e22 FLOPs for Emu3-7B, and Stable Diffusion 3 is 17% more.**
6 x 7e9 x 300e9 against 6 x 8e9 x 307.2e9. SD3 is the more expensive of the two,
and only just.

**FINDING: the lesson's "comparable" is exact once you convert.** 300M
image-steps at 1,024 latent tokens each is **307.2B tokens** -- **1.024x** Emu3's
300B. The two figures in the lesson's parenthetical are the same number in
different units, and it never says so.

**FINDING: so the entire remaining difference is the parameter count.** 8B
against 7B is **1.143x**, and 1.143 x 1.024 = **1.170**, which is the FLOPs
ratio to three decimals. Nothing about diffusion versus autoregression appears
in the answer; the arithmetic is two model sizes and one unit conversion.

**FINDING: per image token, the unified model spends far less.** Emu3's 300B
tokens are a mixture -- at Lesson 12.10's 40/35/20/5 split, taken here as a
stated reference for a native-multimodal corpus, the image-bearing share is
**60%**, or 180B tokens, **176M** images at 1,024 each. SD3 spends all 300M of
its steps on images. Equal total compute, **1.71x** the image exposure for the
specialist.

Structure: `flops` is 6ND, `as_tokens` converts image-steps at a stated latent
length, and `image_exposure` applies a stated corpus mix to the unified arm.
"""

from __future__ import annotations

from harness import practice

EMU3_PARAMS, EMU3_TOKENS = 7e9, 300e9
SD3_PARAMS, SD3_STEPS, LATENT_TOKENS = 8e9, 300e6, 1024
FLOPS_PER_PARAM_TOKEN = 6
CORPUS_MIX = {"text": 0.40, "interleaved": 0.35, "caption": 0.20, "video": 0.05}
IMAGE_BEARING = ("interleaved", "caption", "video")


def flops(params, tokens):
    return FLOPS_PER_PARAM_TOKEN * params * tokens


def as_tokens(steps, latent=LATENT_TOKENS):
    return steps * latent


def image_exposure(tokens, mix=CORPUS_MIX, latent=LATENT_TOKENS):
    share = sum(mix[name] for name in IMAGE_BEARING)
    return share, tokens * share / latent


def solve():
    sd3_tokens = as_tokens(SD3_STEPS)
    emu3, sd3 = flops(EMU3_PARAMS, EMU3_TOKENS), flops(SD3_PARAMS, sd3_tokens)
    share, images = image_exposure(EMU3_TOKENS)
    return {
        "emu3_flops": emu3, "sd3_flops": sd3,
        "ratio": round(sd3 / emu3, 3),
        "dearer": "sd3" if sd3 > emu3 else "emu3",
        "sd3_tokens": sd3_tokens,
        "token_ratio": round(sd3_tokens / EMU3_TOKENS, 3),
        "param_ratio": round(SD3_PARAMS / EMU3_PARAMS, 3),
        "product": round((SD3_PARAMS / EMU3_PARAMS) * (sd3_tokens / EMU3_TOKENS), 3),
        "image_share": round(share, 2),
        "emu3_images": round(images / 1e6),
        "sd3_images": round(SD3_STEPS / 1e6),
        "exposure_ratio": round(SD3_STEPS / images, 2),
    }


def verify(result):
    return [
        practice.Check(
            "ANSWER: 1.26e22 FLOPs for Emu3-7B, and SD3 is 17% more",
            all([result["emu3_flops"] == 1.26e22,
                 round(result["sd3_flops"] / 1e22, 4) == 1.4746,
                 result["ratio"] == 1.17, result["dearer"] == "sd3"]),
            f"6ND gives {result['emu3_flops']:.2e} for 7B at 300B tokens and "
            f"{result['sd3_flops']:.3e} for 8B at {result['sd3_tokens'] / 1e9:.1f}B -- a "
            f"ratio of {result['ratio']}, so {result['dearer'].upper()} is the more "
            "expensive of the two, and only just. 6ND on a DiT is a stated assumption: its "
            "forward pass is the same order as a transformer's",
        ),
        practice.Check(
            "FINDING: the lesson's 'comparable' is exact once you convert",
            all([result["sd3_tokens"] == 307.2e9, result["token_ratio"] == 1.024]),
            f"{SD3_STEPS / 1e6:.0f}M image-steps at {LATENT_TOKENS:,} latent tokens each is "
            f"{result['sd3_tokens'] / 1e9:.1f}B tokens -- {result['token_ratio']}x Emu3's "
            f"{EMU3_TOKENS / 1e9:.0f}B. The two figures in the lesson's parenthetical are the "
            "same number in different units, and it never says so",
        ),
        practice.Check(
            "FINDING: so the entire remaining difference is the parameter count",
            all([result["param_ratio"] == 1.143, result["product"] == result["ratio"]]),
            f"{SD3_PARAMS / 1e9:.0f}B against {EMU3_PARAMS / 1e9:.0f}B is "
            f"{result['param_ratio']}x, and {result['param_ratio']} x "
            f"{result['token_ratio']} = {result['product']}, which is the FLOPs ratio "
            "exactly. Nothing about diffusion against autoregression appears in the answer: "
            "two model sizes and one unit conversion",
        ),
        practice.Check(
            "FINDING: per image token, the unified model spends far less",
            all([result["image_share"] == 0.6, result["emu3_images"] == 176,
                 result["sd3_images"] == 300, result["exposure_ratio"] == 1.71]),
            f"at Lesson 12.10's corpus split, taken as a stated reference, the image-bearing "
            f"share of Emu3's tokens is {result['image_share']:.0%} -- "
            f"{result['emu3_images']}M images at {LATENT_TOKENS:,} tokens each -- against "
            f"SD3 spending all {result['sd3_images']}M of its steps on images. Equal total "
            f"compute, {result['exposure_ratio']}x the image exposure for the specialist",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
