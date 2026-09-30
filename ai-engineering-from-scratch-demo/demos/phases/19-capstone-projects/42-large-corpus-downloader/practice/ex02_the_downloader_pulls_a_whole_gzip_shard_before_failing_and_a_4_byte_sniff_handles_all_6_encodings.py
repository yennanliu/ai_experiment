"""Exercise 2 — the downloader pulls a whole gzip shard before failing on it; a 4-byte sniff handles all 6 encodings.

    Add gzip support next to zstd by sniffing the magic bytes. The downloader should not require the caller to specify the codec.

Reading of the exercise: "the downloader" is the reference
`StreamingDownloader` plus `ZstdDocIterator`, called exactly as the lesson
calls them, with an unchanged `ShardPlan(shard_id, url)` -- no codec field
anywhere. Both hardwire the module-level name `zstd`, so gzip support goes in
at that one seam: `sniffing_codec()` replaces it with a decompressor whose
`stream_reader` peeks the first four bytes and hands back either
`gzip.GzipFile` or the real zstd reader. One 200-document JSONL payload
(16,624 bytes) is served over `file://` in six encodings, before and after.

Magic bytes, read 2026-09-29: zstd frames start 0xFD2FB528 little-endian and
skippable frames 0x184D2A50-0x184D2A5F (RFC 8878 sections 3.1.1 and 3.1.2,
https://datatracker.ietf.org/doc/html/rfc8878); gzip members start 1f 8b
(RFC 1952 section 2.3.1, https://datatracker.ietf.org/doc/html/rfc1952).

**ANSWER: sniff the first 4 bytes; zstd and gzip then decode to identical
documents with no codec argument.**

| encoding | shipped | with sniffing |
|---|---|---|
| zstd | 200 docs | 200 docs |
| gzip | ZstdError | 200 docs |
| gzip, 2 members | ZstdError | 200 docs |
| zstd, 2 frames | 200 docs | 200 docs |
| zstd, skippable frame first | 200 docs | 200 docs |
| plain JSONL | ZstdError | refused: `unknown shard codec, magic bytes 7b226964` |

Every decoded case matches the source line for line and reports 16,624
decompressed bytes. Unknown bytes fail closed with the magic in the message.

**FINDING: the shipped downloader fetches the whole shard before it looks at
the codec.** `download()` writes every byte, then decompresses to count
documents. A gzip shard is fully on disk (1,364 of 1,364 bytes) when
`ZstdError: zstd decompress error: Unknown frame descriptor` is raised, and
the message never says "gzip". The sniff inherits that ordering: the plain
JSONL shard is refused only after all 16,624 bytes landed. The cache also
names every shard `<id>.zst` whatever it holds.

Structure: `sniff()` is the codec decision; `sniffing_codec()` is the seam;
`fetch()` runs the reference download and document iterator on one shard.
"""

from __future__ import annotations

import gzip
import io
import pathlib
import struct
import tempfile
import types

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "42-large-corpus-downloader"
ZSTD_MAGIC = struct.pack("<I", 0xFD2FB528)  # RFC 8878 section 3.1.1
GZIP_MAGIC = b"\x1f\x8b"  # RFC 1952 section 2.3.1


def sniff(head):
    """Codec name from the first four bytes; anything unknown fails closed."""
    if head[:4] == ZSTD_MAGIC or (len(head) >= 4 and struct.unpack("<I", head[:4])[0] >> 4 == 0x184D2A5):
        return "zstd"
    if head[:2] == GZIP_MAGIC:
        return "gzip"
    raise ValueError(f"unknown shard codec, magic bytes {head[:4].hex() or 'empty'}")


def sniffing_codec(real):
    """A drop-in for the module's `zstd` name: every stream_reader sniffs before decoding."""

    def stream_reader(fh):
        fh = fh if hasattr(fh, "peek") else io.BufferedReader(fh)
        if sniff(fh.peek(4)[:4]) == "gzip":
            return gzip.GzipFile(fileobj=fh, mode="rb")
        return real.ZstdDecompressor().stream_reader(fh)

    return types.SimpleNamespace(ZstdDecompressor=lambda: types.SimpleNamespace(stream_reader=stream_reader),
                                 ZstdCompressor=real.ZstdCompressor, ZstdError=real.ZstdError)


def make_fixture():
    lines = [f'{{"id": {i}, "text": "document {i} ' + "token " * (i % 17) + '"}' for i in range(200)]
    return ("\n".join(lines) + "\n").encode()


def encodings(ref, payload):
    half = payload.index(b"\n", len(payload) // 2) + 1
    skip = struct.pack("<II", 0x184D2A50, 8) + b"metadata"
    zc = ref.zstd.ZstdCompressor(level=3)
    return {
        "zstd": zc.compress(payload),
        "gzip": gzip.compress(payload, mtime=0),
        "gzip_two_members": gzip.compress(payload[:half], mtime=0) + gzip.compress(payload[half:], mtime=0),
        "zstd_two_frames": zc.compress(payload[:half]) + zc.compress(payload[half:]),
        "zstd_skippable_first": skip + zc.compress(payload),
        "plain_jsonl": payload,
    }


def fetch(ref, blob, name):
    """Download one shard through the unmodified downloader; docs via ZstdDocIterator."""
    with tempfile.TemporaryDirectory() as d:
        src = pathlib.Path(d) / name
        src.write_bytes(blob)
        loader = ref.StreamingDownloader(pathlib.Path(d) / "cache")
        try:
            res = loader.download(ref.ShardPlan("s", src.as_uri()))
            with (loader.cache_dir / "s.zst").open("rb") as fh:
                docs = list(ref.ZstdDocIterator(fh))
            return {"docs": len(docs), "bytes": res.decompressed_bytes, "same": docs}
        except Exception as exc:
            on_disk = (loader.cache_dir / "s.zst").stat().st_size if (loader.cache_dir / "s.zst").exists() else 0
            return {"error": f"{type(exc).__name__}: {exc}", "on_disk": on_disk}


def solve():
    payload = make_fixture()
    ref = parity.load_reference(PHASE, LESSON, "main")
    blobs = encodings(ref, payload)
    before = {k: fetch(ref, b, f"{k}.bin") for k, b in blobs.items()}
    ref.zstd = sniffing_codec(ref.zstd)
    after = {k: fetch(ref, b, f"{k}.bin") for k, b in blobs.items()}
    want = payload.decode().splitlines()
    for r in (*before.values(), *after.values()):
        if "same" in r:
            r["same"] = r["same"] == want
    return {"before": before, "after": after, "sizes": {k: len(b) for k, b in blobs.items()}, "n": len(want),
            "nbytes": len(payload)}


def verify(result):
    before, after, n, nbytes = result["before"], result["after"], result["n"], result["nbytes"]
    ok, zstd_err = {"docs": n, "bytes": nbytes, "same": True}, "ZstdError: zstd decompress error: Unknown frame descriptor"
    decoded = [k for k in after if after[k] == ok]
    shipped = [k for k in before if before[k] == ok]
    failed = [(before[k]["error"], before[k]["on_disk"]) for k in ("gzip", "gzip_two_members", "plain_jsonl")]
    sizes = [result["sizes"][k] for k in ("gzip", "gzip_two_members", "plain_jsonl")]
    return [
        practice.Check(
            "ANSWER: a 4-byte sniff decodes zstd and gzip to identical documents with no codec argument",
            (decoded, after["plain_jsonl"]["error"]) == (
                ["zstd", "gzip", "gzip_two_members", "zstd_two_frames", "zstd_skippable_first"],
                "ValueError: unknown shard codec, magic bytes 7b226964"),
            f"decoded {decoded} ({n} docs, {nbytes} bytes each); plain JSONL -> {after['plain_jsonl']['error']}",
        ),
        practice.Check(
            "FINDING: the shipped downloader fetches the whole shard before it looks at the codec",
            (shipped, failed, after["plain_jsonl"]["on_disk"]) == (
                ["zstd", "zstd_two_frames", "zstd_skippable_first"], [(zstd_err, b) for b in sizes], nbytes),
            f"shipped: {failed} (error, bytes on disk) for gzip, 2-member gzip, plain JSONL of sizes {sizes}; "
            f"with sniffing plain JSONL is refused after {after['plain_jsonl']['on_disk']} bytes",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
