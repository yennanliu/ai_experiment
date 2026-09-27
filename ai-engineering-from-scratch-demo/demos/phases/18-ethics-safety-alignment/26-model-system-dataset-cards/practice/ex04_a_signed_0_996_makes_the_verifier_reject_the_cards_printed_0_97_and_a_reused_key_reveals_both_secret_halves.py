"""Exercise 4 — a signed 0.996 makes the verifier reject the card's printed 0.97 and a reused key reveals both secret halves.

    Laminator (Duddu et al. 2024) uses TEEs for verifiable attestations.
    Design a model-card field that carries a cryptographic attestation of an
    evaluation result and describe the verifier's role.

Reading of the exercise: the field is built and exercised, not just drawn.
The "enclave" is a function that holds the only copy of a signing key, runs
the evaluation itself (Lesson 21's `predict` on Exercise 2's model and the
datasheet's test split) and signs what it measured. The stdlib has no
public-key signature, so the signature is a Lamport one-time signature over
SHA-256 -- genuinely public-key, a few lines, and its one-time property turns
out to be part of the verifier's job. A real TEE would add a hardware quote
chaining the key to the vendor; here that is a trusted-key registry.

**ANSWER: an `Attestations` section with one row per attested claim:
`metric | value | payload | signer`, where the payload binds the metric and
value to the SHA-256 of the model weights, the evaluation dataset and the
evaluation code, and the Lamport signature over it (8,192 bytes) travels as a
sidecar.** The verifier re-derives everything it can and trusts only the
signature: signer key in the registry, signature valid over the payload,
model hash equal to the artifact it downloaded, dataset hash equal to the
datasheet's dataset, the card's printed value equal to the signed one, and
the key never seen before. The honest card is accepted; each of six
tampering cases is rejected, each at its own step.

**FINDING: attached to the lesson's card, the attestation rejects the card's
own number.** The generated card prints accuracy 0.97; the enclave measures
and signs 0.996 on the datasheet's data, so the verifier rejects the card at
the value step. Today none of the three cards carries an attestation, so all
three Quantitative Analysis figures are self-report.

**FINDING: the verifier must enforce one-time keys.** Two attestations
signed with one Lamport key reveal both secret halves at exactly the
positions where the two payload digests differ -- about half of the 256
bits, the exact count depending on the digests -- and every revealed half
verifies against the public key, which lets anyone forge signatures on
digests that agree with the signed ones elsewhere. The registry has to retire a key after one use.

Structure: `keypair`/`bits`/`check_sig` are the Lamport scheme; `enclave()`
evaluates and signs; `verifier()` returns the first failing step or
"accept".
"""

from __future__ import annotations

import hashlib
import inspect
import json
import pathlib
import random

from harness import parity, practice

PHASE, LESSON = "18-ethics-safety-alignment", "26-model-system-dataset-cards"
HERE = pathlib.Path(__file__).resolve().parent
EX01 = practice.load_module(next(HERE.glob("ex01_*.py")))
EX02 = practice.load_module(next(HERE.glob("ex02_*.py")))


def sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else json.dumps(data).encode()).digest()


def fingerprint(pk):
    return sha(b"".join(h for pair in pk for h in pair)).hex()[:16]


def keypair(rng):
    sk = [[rng.randbytes(32), rng.randbytes(32)] for _ in range(256)]
    return sk, [[sha(x) for x in pair] for pair in sk]


def bits(msg):
    d = sha(msg)
    return [(d[i // 8] >> (7 - i % 8)) & 1 for i in range(256)]


def check_sig(pk, msg, sig):
    return all(sha(s) == pk[i][b] for i, (s, b) in enumerate(zip(sig, bits(msg))))


def enclave(sk, pk, l21, model, test):
    """Evaluate inside the 'TEE' and sign the result; the key never leaves."""
    preds = l21.predict(model, test)
    payload = {"metric": "accuracy", "value": EX02.accuracy(preds), "model": sha(model).hex(),
               "dataset": sha(test).hex(), "code": sha(inspect.getsource(l21.predict).encode()).hex(),
               "signer": fingerprint(pk)}
    return payload, [sk[i][b] for i, b in enumerate(bits(payload))]      # Lamport signature


def verifier(card_value, payload, sig, registry, model, dataset, used):
    """The verifier's role: re-derive what it can, trust only the signature."""
    pk = registry.get(payload["signer"])
    steps = [("signer", pk is not None), ("signature", pk is not None and check_sig(pk, payload, sig)),
             ("model hash", payload["model"] == sha(model).hex()),
             ("dataset hash", payload["dataset"] == sha(dataset).hex()),
             ("value", card_value == payload["value"]), ("one-time key", payload["signer"] not in used)]
    return next((name for name, ok in steps if not ok), "accept")


def scenarios(ref, l21, model, test, rng):
    sk, pk = keypair(rng)
    registry, (payload, sig) = {fingerprint(pk): pk}, enclave(sk, pk, l21, model, test)
    rogue = enclave(*keypair(rng), l21, model, test)
    card = float(EX01.sections(ref.model_card())["Quantitative Analysis"][0].split()[1])
    value = payload["value"]
    run = lambda v=value, p=payload, s=sig, m=model, d=test, u=(): verifier(v, p, s, registry, m, d, u)
    return {
        "honest": run(), "reference card": run(v=card),
        "edited payload": run(v=card, p=dict(payload, value=card)),
        "swapped weights": run(m=[model[0] + 1e-9] + model[1:]),
        "other dataset": run(d=EX02.datasheet_data(len(test), random.Random(0))),
        "unregistered key": run(p=rogue[0], s=rogue[1]),
        "reused key": run(u=(payload["signer"],)),
    }, card, value, (sk, pk, payload, sig)


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    l21, model, test, _ = EX02.measured()
    rng = random.Random(4)
    outcome, card, value, (sk, pk, first, sig) = scenarios(ref, l21, model, test, rng)
    again, second = enclave(sk, pk, l21, model, EX02.datasheet_data(len(test), rng))   # key reused
    differ = [i for i, (a, b) in enumerate(zip(bits(first), bits(again))) if a != b]
    return {
        "outcome": outcome, "card": card, "value": value, "sig_bytes": sum(map(len, sig)),
        "attest_fields": sum("attest" in c.lower() for c in EX01.cards(ref).values()),
        "leaked": [i for i, (a, b) in enumerate(zip(sig, second)) if a != b],
        "differ": differ,   # the leaked halves must open both public-key hashes at each position
        "opened": all({sha(sig[i]), sha(second[i])} == set(pk[i]) for i in differ),
    }


def verify(result):
    out = result["outcome"]
    expected = {"honest": "accept", "reference card": "value", "edited payload": "signature",
                "swapped weights": "model hash", "other dataset": "dataset hash",
                "unregistered key": "signer", "reused key": "one-time key"}
    return [
        practice.Check(
            "ANSWER: metric | value | payload | signer, checked in six verifier steps",
            out == expected and result["sig_bytes"] == 8192,
            f"verifier outcome per case {out}; Lamport signature {result['sig_bytes']} bytes",
        ),
        practice.Check(
            "FINDING: attached to the lesson's card, the attestation rejects the card's own number",
            result["card"] == 0.97 and result["value"] == 0.996 and result["attest_fields"] == 0,
            f"card prints {result['card']}, enclave signs {result['value']}; cards carrying an "
            f"attestation: {result['attest_fields']} of 3",
        ),
        practice.Check(
            "FINDING: the verifier must enforce one-time keys",
            result["leaked"] == result["differ"] and 0 < len(result["differ"]) < 256
            and result["opened"],
            f"two signatures under one key reveal both halves at {len(result['leaked'])} of 256 "
            f"positions, exactly where the payload digests differ: {result['leaked'] == result['differ']}; "
            f"every revealed pair opens both public-key hashes: {result['opened']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
