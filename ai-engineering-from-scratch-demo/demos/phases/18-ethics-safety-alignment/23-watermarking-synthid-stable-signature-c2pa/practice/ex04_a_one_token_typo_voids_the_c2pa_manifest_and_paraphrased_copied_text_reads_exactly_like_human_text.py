"""Exercise 4 — a one-token typo fix voids the C2PA manifest, a 60% paraphrase erases the watermark, and stripped+paraphrased text reads exactly like human text.

    Design a deployment that uses SynthID-text + C2PA metadata. Describe the
    provenance chain a consumer sees. Identify one failure mode of each
    component.

Reading of the exercise: the deployment is built and every scenario is run
through it. The watermark is the reference's (`watermarked_sample` at
generation, `detect` at the consumer, z >= 4 as the reference prints). The C2PA
layer is a minimal manifest in C2PA's shape: a `c2pa.created` action with
digitalSourceType `trainedAlgorithmicMedia`, a hard binding (SHA-256 of the
content) and a signature. HMAC-SHA256 stands in for the COSE certificate
signature, since the stdlib has no public-key signing; what matters here is
that it fails on any change to content, claims or key. The consumer view
checks the manifest first, then the watermark, and prints one verdict.

**ANSWER: the consumer sees a signed chain when the manifest survives, a
watermark-only verdict when it does not, and "unknown origin" when both are
gone.** Seven seeded scenarios:

| scenario | C2PA | watermark z | consumer verdict |
|---|---|---:|---|
| as published | valid | 25.49 | AI-generated, signed by the provider |
| copied as plain text | absent | 25.49 | AI-generated (watermark only, no chain) |
| one-token typo fix | hash mismatch | 25.23 | edited after signing, watermark present |
| 60% paraphrase, manifest kept | hash mismatch | -0.25 | edited after signing, no watermark |
| 60% paraphrase, copied | absent | -0.25 | unknown origin |
| human text | absent | -1.14 | unknown origin |
| human text, forged manifest | bad signature | -1.14 | untrusted manifest |

**FINDING: C2PA's failure mode is that text has no container.** Copying the
text drops the manifest, and when the manifest does travel, its hard binding
breaks on any edit: a single corrected token turns "signed by the provider"
into "edited after signing". The watermark is what still says 25.23.

**FINDING: the watermark's failure mode is paraphrase and length.** A 60%
paraphrase leaves z = -0.25, and short text is not detectable: at z >= 4 the
reference catches 19.5% / 54.5% / 97.5% of 16 / 25 / 50-token generations
(200 seeds each). Once the text is paraphrased and copied, its verdict is identical to
real human text, which is the skill file's own hard reject:
"model-not-watermarked ≠ authentic".

Structure: `sign()` and `consumer_view()` are the two ends of the chain;
`scenarios()` builds the seven inputs; `short_rate()` sweeps length.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import random

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "23-watermarking-synthid-stable-signature-c2pa"
N, ZREF, PROVIDER_KEY, LENGTHS = 1000, 4.0, b"provider-signing-key", (16, 25, 50)
VERDICTS = {  # (C2PA status, watermark z >= 4) -> what the consumer is told
    ("valid", True): "AI-generated, signed by the provider",
    ("absent", True): "AI-generated (watermark only, no chain)",
    ("hash mismatch", True): "edited after signing, watermark present",
    ("hash mismatch", False): "edited after signing, no watermark", ("absent", False): "unknown origin"}


@contextlib.contextmanager
def seeded(ref, seed):
    saved, ref.random = ref.random, random.Random(seed)
    try:
        yield
    finally:
        ref.random = saved


def digest(tokens):
    return hashlib.sha256(",".join(map(str, tokens)).encode()).hexdigest()


def mac(key, claims):
    return hmac.new(key, json.dumps(claims, sort_keys=True).encode(), "sha256").hexdigest()


def sign(tokens, key=PROVIDER_KEY):
    claims = {"claim_generator": "provider-model", "assertions": [{
        "label": "c2pa.actions", "action": "c2pa.created",
        "digitalSourceType": "trainedAlgorithmicMedia"}], "hard_binding": digest(tokens)}
    return {"claims": claims, "signature": mac(key, claims)}


def c2pa_status(tokens, manifest):
    if manifest is None:
        return "absent"
    if not hmac.compare_digest(manifest["signature"], mac(PROVIDER_KEY, manifest["claims"])):
        return "bad signature"
    return "valid" if manifest["claims"]["hard_binding"] == digest(tokens) else "hash mismatch"


def consumer_view(ref, tokens, manifest):
    status, z = c2pa_status(tokens, manifest), ref.detect(tokens)
    return status, round(z, 2), VERDICTS.get((status, z >= ZREF), "untrusted manifest")


def generate(ref, n):
    return ref.watermarked_sample(n, [ref.random.randrange(ref.VOCAB) for _ in range(ref.K)])


def scenarios(ref):
    with seeded(ref, 23):
        text = generate(ref, N)
        human, para = ref.unwatermarked_sample(N, text[:ref.K]), ref.paraphrase(text, 0.6)
    typo = text[:500] + [(text[500] + 1) % ref.VOCAB] + text[501:]
    return {
        "as published": (text, sign(text)), "copied as plain text": (text, None),
        "one-token typo fix": (typo, sign(text)), "60% paraphrase, manifest kept": (para, sign(text)),
        "60% paraphrase, copied": (para, None), "human text": (human, None),
        "human text, forged manifest": (human, sign(human, b"attacker-key")),
    }


def short_rate(ref, n, seeds=200):
    with seeded(ref, n):
        zs = [ref.detect(generate(ref, n)) for _ in range(seeds)]
    return sum(z >= ZREF for z in zs) / seeds


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    skill = (parity.lesson_dir(PHASE, LESSON) / "outputs" / "skill-provenance-audit.md").read_text()
    return {
        "view": {k: consumer_view(ref, *v) for k, v in scenarios(ref).items()},
        "short": {n: short_rate(ref, n) for n in LENGTHS},
        "skill_reject": "model-not-watermarked ≠ authentic" in skill,
    }


def verify(result):
    v = result["view"]
    status, zs, verdicts = (list(col) for col in zip(*v.values()))
    return [
        practice.Check(
            "ANSWER: signed chain, watermark-only, or unknown origin, by what survived",
            status == ["valid", "absent", "hash mismatch", "hash mismatch", "absent", "absent", "bad signature"]
            and zs == [25.49, 25.49, 25.23, -0.25, -0.25, -1.14, -1.14]
            and verdicts == [*VERDICTS.values(), "unknown origin", "untrusted manifest"],
            "; ".join(f"{k}: {s}, z {z}, '{d}'" for k, (s, z, d) in v.items()),
        ),
        practice.Check(
            "FINDING: C2PA's failure mode is that text has no container",
            v["one-token typo fix"] == ("hash mismatch", 25.23, VERDICTS[("hash mismatch", True)]),
            f"one corrected token: {v['one-token typo fix']}",
        ),
        practice.Check(
            "FINDING: the watermark's failure mode is paraphrase and length",
            v["60% paraphrase, copied"][2] == v["human text"][2] == "unknown origin"
            and result["short"] == {16: 0.195, 25: 0.545, 50: 0.975} and result["skill_reject"],
            f"paraphrased+copied {v['60% paraphrase, copied']} vs human {v['human text']}; "
            f"detection rate at z >= 4 by length {result['short']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
