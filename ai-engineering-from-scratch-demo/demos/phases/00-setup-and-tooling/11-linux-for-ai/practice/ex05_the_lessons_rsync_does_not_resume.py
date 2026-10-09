"""Exercise 5 — rsync resends one block, not one byte, and the lesson's command does not resume.

    Transfer a file from your local machine to a remote one using `scp`, then do the
    same transfer with `rsync` and compare the experience.

Reading of the exercise: there is no remote host in CI, and the "experience" of
the two tools differs only on the second transfer, after the file has changed --
the first copy of a new file is the whole file for both. So both are measured by
the bytes they put on the wire for a 1 MiB checkpoint (float32 weights from a
fixed seed). `scp` sends the file. rsync is rsync's own delta algorithm, written
out: the receiver sends a weak rolling checksum and an MD5 per block, block size
from rsync's rule (`max(700, sqrt(size))`, here 1024); the sender rolls the weak
sum byte by byte, and sends block references plus literal bytes for what
matches nothing.

**ANSWER: after a 1-byte edit, scp sends 1,048,576 bytes and rsync 25,596 --
41x less.** 1,024 of those are the one changed block; 20,480 are the checksum
list for 1024 blocks (counted at 4 + 16 bytes each), which rsync pays even when
nothing changed, and 4,092 are block references (real rsync run-length codes
them, so this is an upper bound). The first transfer is the full file for both.

**FINDING: inserting one byte at the front is cheaper than changing one.** Every
block after the insertion has moved by one byte, so a fixed-offset block compare
would match none of them and resend everything. The rolling checksum finds all
1024 of them and sends 1 literal byte, against 1,024 for the in-place edit,
which breaks a block where the insertion only shifts them.

**FINDING: the lesson's command does not resume.** It says rsync "resumes on
failure" and "handles interrupted connections", and every rsync command it gives
is `rsync -avz --progress`. Without `--partial` (or `-P`, which is `--partial
--progress`) rsync deletes a partially transferred file when interrupted, so a
dropped copy of a 50 GB checkpoint starts again from zero.

**FINDING: `-z` buys little on weights.** zlib compresses the float32 checkpoint
to ~93% of its size: only the exponent bytes are predictable.

**CONTROL:** the reconstructed file equals the edited one byte for byte in both
cases.

Structure: `signature` is the receiver's side, `delta` the sender's rolling
scan, `patch` the reconstruction.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
import struct
import zlib
from itertools import accumulate

from harness import parity, practice

PHASE, LESSON = "00-setup-and-tooling", "11-linux-for-ai"
SIZE, MOD = 1 << 20, 1 << 16
SIG_BYTES, REF_BYTES = 4 + 16, 4  # weak + strong sum per block; a block reference


def block_size(n):
    return max(700, int(math.sqrt(n)) // 8 * 8)


def weak(block):
    """rsync's checksum: a = sum of bytes, b = sum of running prefix sums."""
    return sum(block) % MOD, sum(accumulate(block)) % MOD


def signature(basis, size):
    blocks = [basis[i:i + size] for i in range(0, len(basis), size)]
    return {weak(b): (i, hashlib.md5(b).digest()) for i, b in enumerate(blocks)}, len(blocks)


def delta(target, sig, size):
    """[(block index) or (literal bytes)] by rolling the weak sum one byte at a time."""
    ops, literal, pos = [], bytearray(), 0
    a, b = weak(target[:size])
    while pos + size <= len(target):
        hit = sig.get((a, b))
        if hit and hashlib.md5(target[pos:pos + size]).digest() == hit[1]:
            ops += [bytes(literal), hit[0]] if literal else [hit[0]]
            literal, pos = bytearray(), pos + size
            a, b = weak(target[pos:pos + size])
            continue
        out = target[pos]
        literal.append(out)
        pos += 1
        nxt = target[pos + size - 1] if pos + size <= len(target) else 0
        a, b = (a - out + nxt) % MOD, (b - size * out + a - out + nxt) % MOD
    literal += target[pos:]
    return ops + ([bytes(literal)] if literal else [])


def patch(basis, ops, size):
    return b"".join(basis[op * size:(op + 1) * size] if isinstance(op, int) else op
                    for op in ops)


def rsync(basis, target):
    size = block_size(len(basis))
    sig, blocks = signature(basis, size)
    ops = delta(target, sig, size)
    literal = sum(len(op) for op in ops if isinstance(op, bytes))
    refs = sum(isinstance(op, int) for op in ops)
    wire = blocks * SIG_BYTES + refs * REF_BYTES + literal
    return {"wire": wire, "literal": literal, "refs": refs, "blocks": blocks,
            "exact": patch(basis, ops, size) == target}


def checkpoint():
    rng = random.Random(0)
    return struct.pack(f"<{SIZE // 4}f", *(rng.gauss(0, 0.02) for _ in range(SIZE // 4)))


def solve():
    basis = checkpoint()
    edited = bytearray(basis)
    edited[SIZE // 2] ^= 0xFF
    doc = parity.doc_text(PHASE, LESSON)
    return {
        "edit": rsync(basis, bytes(edited)),
        "insert": rsync(basis, b"\x00" + basis),
        "commands": re.findall(r"^rsync .*?(?=\s+(?:\S+:)?[./~])", doc, re.M),
        "claims": "resumes on failure" in doc and "handles interrupted connections" in doc,
        "zlib": len(zlib.compress(basis, 6)) / SIZE,
    }


def verify(result):
    edit, insert, cmds = result["edit"], result["insert"], result["commands"]
    return [
        practice.Check(
            "ANSWER: after a 1-byte edit scp sends the whole file, rsync about 2%",
            edit["literal"] <= 2 * block_size(SIZE) and SIZE / edit["wire"] > 20,
            f"scp {SIZE} bytes; rsync {edit['wire']} bytes ({SIZE / edit['wire']:.0f}x less): "
            f"{edit['literal']} literal + {edit['blocks']} x {SIG_BYTES} signature + "
            f"{edit['refs']} x {REF_BYTES} references",
        ),
        practice.Check(
            "FINDING: a 1-byte insertion at the front costs 1 literal byte, not the file",
            insert["refs"] == insert["blocks"] and insert["literal"] == 1,
            f"{insert['refs']} of {insert['blocks']} blocks matched after the shift; "
            f"{insert['literal']} literal bytes",
        ),
        practice.Check(
            "FINDING: the lesson's rsync claims to resume, its command has no --partial",
            result["claims"] and cmds and not any("--partial" in c or " -P" in c for c in cmds),
            f"commands: {sorted(set(cmds))}; without --partial an interrupted rsync deletes "
            "the partial file",
        ),
        practice.Check(
            "FINDING: -z compresses float32 weights only to ~93%",
            0.85 < result["zlib"] < 0.99,
            f"zlib level 6 on the checkpoint: {result['zlib']:.1%} of its size",
        ),
        practice.Check(
            "CONTROL: both reconstructions are byte-exact",
            edit["exact"] and insert["exact"],
            f"edit: {edit['exact']}; insertion: {insert['exact']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
