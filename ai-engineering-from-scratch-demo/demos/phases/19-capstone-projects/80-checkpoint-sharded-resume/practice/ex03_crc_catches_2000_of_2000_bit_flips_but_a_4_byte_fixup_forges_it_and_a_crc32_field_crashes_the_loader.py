"""Exercise 3 — the CRC path catches 2,000 of 2,000 bit flips, but a crc32 field in the manifest crashes the lesson's loader with TypeError.

    Add a CRC-only fast verification path for the inner-loop reload (rotation rolls a checkpoint into being the new active one without full sha256).

Reading of the exercise: after the lesson's `save_sharded`, `write_crc`
records each shard's `zlib.crc32` and size; `fast_verify` rereads the shards
and checks only those, with no sha256 and no deserialisation. It is tested
on the lesson's 4 demo ranks against the damage an inner-loop reload has to
catch (bit flips, truncation, and the lesson's own append-"corruption"
tamper), timed against sha256 over 4 MB shards, and tried against a
deliberate edit that keeps the CRC.

**ANSWER: `fast_verify` catches every accidental corruption tried, and is
several times cheaper than sha256.** It flags 2,000 of 2,000 seeded
single-bit flips in `rank0.bin`, a shard truncated by one byte, and the
lesson's `+ b"corruption"` tamper, and passes the untouched checkpoint.
Over 16 MB of shards CRC32 is at least 2x faster than sha256 (best of 5;
about 13x on the machine this was written on). The CRCs live in a sidecar
`crc32.json`, not in the manifest -- see the finding.

**FINDING: the manifest cannot carry the new field.** Add `crc32` to each
shard entry and the lesson's `load_sharded` raises `TypeError`
(`ShardEntry.__init__() got an unexpected keyword argument 'crc32'`), not
`CheckpointError`. Bumping `schema_version` to 2 changes nothing: the
version check runs after `ShardEntry(**s)` has already crashed, so the
schema-version defence cannot fire for any schema that adds a field.

**FINDING: CRC is an accident check, not a tamper check.** CRC32 is affine
over GF(2), so rewriting one weight and solving for the 4 bytes of a
neighbouring weight keeps the CRC: `fast_verify` accepts the edited shard,
`torch.load` returns a tensor with 2 changed values, and only the lesson's
sha256 path rejects it. The fast path is only safe for files this job wrote
itself; a checkpoint from anywhere else needs the full sha256.

Expected output: three PASS checks.
"""

from __future__ import annotations

import hashlib
import json
import random
import struct
import tempfile
import timeit
import zlib
from pathlib import Path

from harness import parity, practice

try:
    import torch
except ImportError as exc:  # pragma: no cover - env guard
    raise practice.Skip(f"needs torch: uv sync --extra llm ({exc})") from None

PHASE, LESSON = "19-capstone-projects", "80-checkpoint-sharded-resume"


def write_crc(ckpt):
    table = {p.name: [zlib.crc32(b := p.read_bytes()), len(b)] for p in sorted(Path(ckpt).glob("rank*.bin"))}
    Path(ckpt, "crc32.json").write_text(json.dumps(table))


def fast_verify(ckpt):
    table = json.loads(Path(ckpt, "crc32.json").read_text())
    return all([zlib.crc32(b := Path(ckpt, n).read_bytes()), len(b)] == want for n, want in table.items())


def damaged(path, data, mutate):
    path.write_bytes(mutate(data))
    try:
        return fast_verify(path.parent)
    finally:
        path.write_bytes(data)


def gf2_solve(effects, target):
    """XOR-basis over GF(2): pick the rows whose effects XOR to target; return (mask, residue)."""
    basis = {}
    for e, m in effects:
        while e and e.bit_length() in basis:
            e, m = e ^ basis[e.bit_length()][0], m ^ basis[e.bit_length()][1]
        if e:
            basis[e.bit_length()] = (e, m)
    pick = 0
    while target and target.bit_length() in basis:
        target, pick = target ^ basis[target.bit_length()][0], pick ^ basis[target.bit_length()][1]
    return pick, target


def forge(data, start, at):
    """Set the float at `start` to 99.0, then rewrite the 4 bytes at `at` so the CRC is unchanged."""
    zero = zlib.crc32(bytes(len(data)))  # CRC is affine: crc(a ^ d) = crc(a) ^ crc(d) ^ crc(0...0)
    want = bytearray(len(data))
    want[start : start + 4] = bytes(a ^ b for a, b in zip(data[start : start + 4], struct.pack("<f", 99.0)))
    effects = []
    for bit in range(32):
        d = bytearray(len(data))
        d[at + bit // 8] = 1 << (bit % 8)
        effects.append((zlib.crc32(d) ^ zero, 1 << bit))
    pick, residue = gf2_solve(effects, zlib.crc32(want) ^ zero)
    want[at : at + 4] = pick.to_bytes(4, "little")
    return bytes(a ^ b for a, b in zip(data, want)), residue


def speed(ref):
    torch.manual_seed(0)
    blobs = [ref._serialize_state({"param_shard": torch.randn(1 << 20)}) for _ in range(4)]
    best = [min(timeit.repeat(lambda f=f: [f(b) for b in blobs], number=1, repeat=5))
            for f in (zlib.crc32, lambda b: hashlib.sha256(b).digest())]
    return best[1] / best[0]


def flip(pos):
    return lambda b: b[: pos // 8] + bytes([b[pos // 8] ^ 1 << pos % 8]) + b[pos // 8 + 1 :]


def load_error(ref, tmp):
    try:
        ref.load_sharded(tmp, 4)
    except Exception as exc:  # noqa: BLE001 - the point is which type escapes
        return f"{type(exc).__name__}: {exc}"
    return "loaded"


def cases(ref, tmp, shard, data, param):
    picks = random.Random(0).sample(range(8 * len(data)), 2000)
    out = {"clean": fast_verify(tmp), "flips": sum(not damaged(shard, data, flip(p)) for p in picks),
           "trunc": damaged(shard, data, lambda b: b[:-1]), "append": damaged(shard, data, lambda b: b + b"corruption")}
    start = data.find(param.numpy().tobytes())
    forged, residue = forge(data, start, start + 4)
    shard.write_bytes(forged)
    changed = int((ref._deserialize_state(forged)["param_shard"] != param).sum())
    out["forged"], out["forged_sha"] = (residue, fast_verify(tmp), changed), load_error(ref, tmp)
    shard.write_bytes(data)
    return out


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    states = [ref.make_demo_state(r, 4) for r in range(4)]
    with tempfile.TemporaryDirectory(prefix="ex03-") as tmp:
        ref.save_sharded(states, tmp, step=100)
        write_crc(tmp)
        shard = Path(tmp, "rank0.bin")
        out = cases(ref, tmp, shard, shard.read_bytes(), states[0]["param_shard"])
        manifest = json.loads(Path(tmp, "manifest.json").read_text())
        manifest["schema_version"] = 2
        for s in manifest["shards"]:
            s["crc32"] = zlib.crc32(Path(tmp, s["path"]).read_bytes())
        Path(tmp, "manifest.json").write_text(json.dumps(manifest))
        out["field"] = load_error(ref, tmp)
    return {**out, "ratio": speed(ref)}


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: the CRC path catches every accidental corruption and beats sha256 by at least 2x",
            r["clean"] and r["flips"] == 2000 and not r["trunc"] and not r["append"] and r["ratio"] > 2,
            f"clean passes {r['clean']}; flips caught {r['flips']}/2000; truncation passes {r['trunc']}; "
            f"append passes {r['append']}; sha256/crc32 time {r['ratio']:.1f}x",
        ),
        practice.Check(
            "FINDING: a crc32 field in the manifest crashes load_sharded with TypeError, schema_version 2 or not",
            r["field"] == "TypeError: ShardEntry.__init__() got an unexpected keyword argument 'crc32'", r["field"],
        ),
        practice.Check(
            "FINDING: a 4-byte fix-up keeps the CRC, so only sha256 catches a deliberate edit",
            r["forged"] == (0, True, 2) and r["forged_sha"].startswith("CheckpointError: sha256 mismatch on rank 0"),
            f"residue, fast_verify, changed values {r['forged']}; lesson loader: {r['forged_sha'][:42]}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
