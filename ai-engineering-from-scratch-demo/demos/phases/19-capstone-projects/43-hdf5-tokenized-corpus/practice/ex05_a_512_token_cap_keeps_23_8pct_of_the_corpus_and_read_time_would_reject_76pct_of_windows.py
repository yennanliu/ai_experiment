"""Exercise 5 — a 512-token cap keeps 23.8% of the demo corpus; at read time the same cap would reject 76% of windows.

    Add a `--max-document-tokens` flag that truncates very long documents at write time. Defend the trade-off against deciding at read time.

Reading of the exercise: `--max-document-tokens N` is parsed by argparse and
applied by giving the reference pipeline a `Tokenizer` whose `encode` returns
at most N ids. The writer, the boundary injection and the index are
untouched. The shipped demo corpus (6 documents of 2,151 byte-tokens each) is
written with N = 512 and without a cap. "Deciding at read time" is modelled
on the uncapped corpus. The reference store keeps no per-document offsets, so
a reader can apply the cap only by locating each token's position in its
document (the distance back to the last boundary token) and rejecting windows
past N. That is measured over 10,000 seed-7 window starts.

**ANSWER: write-time truncation, with the cost stated plainly.** With N =
512, the shards hold 6 x (512 + 1) = 3,078 tokens instead of 12,912. That
keeps 23.8% of the corpus, and the document count stays 6. Deciding at read
time instead would reject 76.4% of uniformly drawn windows, because their
start lies more than 512 tokens into a document. That is 4.2 reads per
accepted window, and it needs a boundary scan the store does not provide.
Write time wins on read cost. Its price is that it is irreversible. Every
shard's sha256 changes (5c811f78961b to d3102859baae), so a different N is
a new corpus version and a full re-tokenization.

**FINDING: a byte-level cap cuts characters, not just documents.** Tokens are
UTF-8 bytes. A 3-byte-per-character Chinese document capped at 512 tokens
keeps 170 whole characters plus 2 bytes of the 171st, and the reference
`decode` turns that into U+FFFD. The cap has to back off to a character
boundary, or be counted in characters.

**FINDING: the lesson ships a no-loss alternative it never wires in.**
`pack_documents` splits long documents into 512-token groups instead of
dropping the tail. It keeps all 12,911 tokens (6 x 2,151 plus 5 boundaries)
in 26 groups. `ShardedTokenizationPipeline` never calls it.

Structure: `capped()` is the flag's tokenizer; `positions()` gives each
token's offset within its document for the read-time model.
"""

from __future__ import annotations

import argparse
import pathlib
import random
import tempfile

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "43-hdf5-tokenized-corpus"
PARSER = argparse.ArgumentParser(prog="ex05")
PARSER.add_argument("--max-document-tokens", type=int, default=None)
CHINESE = "替滑動窗口加上一個確定性種子" * 20


def capped(ref, limit):
    tokenizer = ref.Tokenizer()
    if limit is not None:
        encode = tokenizer.encode
        tokenizer.encode = lambda text: encode(text)[:limit]
    return tokenizer


def write(ref, root, limit):
    pipe = ref.ShardedTokenizationPipeline(capped(ref, limit), root, chunk_size=512)
    entries = pipe.write_corpus(ref.build_demo_corpus())
    with ref.MmapTokenStore(entries) as store:
        return entries, store.get_slice(0, store.total_tokens)


def positions(tokens, boundary):
    """Offset of every token within its document (boundaries restart the count)."""
    pos, cur = [], 0
    for token in tokens.tolist():
        pos.append(cur)
        cur = 0 if token == boundary else cur + 1
    return pos


def solve(argv=("--max-document-tokens", "512")):
    ref = parity.load_reference(PHASE, LESSON, "main")
    limit = PARSER.parse_args(list(argv)).max_document_tokens
    with tempfile.TemporaryDirectory() as tmp:
        full, stream = write(ref, pathlib.Path(tmp) / "full", None)
        cut, cut_stream = write(ref, pathlib.Path(tmp) / "cut", limit)
    pos = positions(stream, ref.BOUNDARY_TOKEN_ID)
    rng = random.Random(7)
    starts = [rng.randint(0, len(stream) - 65) for _ in range(10000)]
    rejected = sum(pos[s] >= limit for s in starts) / len(starts)
    groups = list(ref.pack_documents(ref.Tokenizer(), [d for v in ref.build_demo_corpus().values() for d in v], 512))
    text = ref.Tokenizer().decode(capped(ref, limit).encode(CHINESE))
    return {
        "limit": limit, "full": len(stream), "cut": len(cut_stream),
        "docs": sum(e.document_count for e in cut), "rejected": round(rejected, 3),
        "sha": (full[0].sha256[:12], cut[0].sha256[:12]),
        "whole_chars": len(text.rstrip("�")), "tail": text[-1],
        "packed": (sum(len(g) for g in groups), len(groups)),
    }


def verify(result):
    r = result
    kept = r["cut"] / r["full"]
    return [
        practice.Check(
            "ANSWER: a 512-token cap keeps 23.8% at write time; at read time it rejects 76% of windows",
            (r["limit"], r["full"], r["cut"], r["docs"]) == (512, 12912, 3078, 6)
            and round(kept, 3) == 0.238 and r["rejected"] == 0.764
            and r["sha"] == ("5c811f78961b", "d3102859baae"),
            f"{r['full']} -> {r['cut']} tokens ({kept:.1%} kept), {r['docs']} docs; read-time "
            f"cap rejects {r['rejected']:.1%} of windows ({1 / (1 - r['rejected']):.1f} reads per "
            f"sample); sha256 {r['sha'][0]} -> {r['sha'][1]}",
        ),
        practice.Check(
            "FINDING: a byte-level cap cuts characters, not just documents",
            r["whole_chars"] == 170 and r["tail"] == "�",
            f"512 bytes of 3-byte characters: {r['whole_chars']} whole, then {r['tail']!r}",
        ),
        practice.Check(
            "FINDING: the lesson ships a no-loss alternative it never wires in",
            r["packed"] == (12911, 26),
            f"pack_documents keeps {r['packed'][0]} tokens in {r['packed'][1]} groups of <= 512",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
