"""Exercise 3 — resume-only refuses the 4 of 6 cache states in which the shipped downloader restarts from byte zero.

    Add a `--resume-only` mode that refuses to start a fresh download if no checkpoint is found. Useful in CI to keep one run from accidentally re-pulling 200 GB.

Reading of the exercise: "a fresh download" is any transfer that starts at
byte zero, not only the one with no checkpoint file -- a checkpoint the
reference rejects, or a server that ignores `Range`, both send the reference
back to byte zero as well. `resume_only()` therefore demands the reference's
own verified checkpoint (`_read_checkpoint` + `_verify_partial`), and its
opener refuses any request without `Range` and any answer that is not 206.
Both modes are run against six cache states a CI job can find, over an
in-process HTTP server that honours RFC 7233 byte ranges (and `file://` for
the last), for one zstd shard with 4 KiB chunks cut after 3 chunks (12,288
bytes).

**ANSWER: `resume_only()` below; it resumes a verified partial and refuses
everything that would start from zero.**

| cache state | shipped downloader | `--resume-only` |
|---|---|---|
| empty | full GET | refused, 0 requests |
| interrupted at 12,288 | `Range: bytes=12288-`, rest only | same, sha256 matches |
| partial with one byte flipped | full GET, no warning | refused, partial kept |
| server ignores `Range` | deletes partial, 2nd full GET | refused after one 200, partial kept |
| shard already complete | `HTTP Error 416` | "already complete" |
| `file://` URL | re-reads the whole file | refused, partial kept |

**FINDING: re-running the shipped downloader on a finished shard crashes.**
The checkpoint is never removed on completion, so the next run sends
`Range: bytes=<size>-` and urllib raises `HTTPError 416 Range Not
Satisfiable`. `resume_only()` reads the 416's `Content-Range: bytes */<size>`
as done.

**FINDING: the lesson's demo transport cannot resume at all.** A `file://`
response has no status, the 206 check fails, and the reference re-reads all
bytes from zero. The doc's resume story is only true over HTTP.

Structure: `serve()` is the local range server; `tap()` logs each request,
can drop the connection, and in strict mode is the mode's network guard;
`resume_only()` is the mode; `run_case()` runs one mode on one cache state.
"""

from __future__ import annotations

import contextlib
import hashlib
import http.server
import pathlib
import random
import tempfile
import threading
import urllib.request

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "42-large-corpus-downloader"
CHUNK, CASES = 4096, ("empty", "interrupted", "corrupted", "ignores_range", "completed", "file")


@contextlib.contextmanager
def serve(blob, honour_range=True):
    """An in-process HTTP server for one shard with RFC 7233 byte ranges (or one that ignores them)."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            rng, n = self.headers.get("Range") if honour_range else None, len(blob)
            start = int(rng[6:].rstrip("-")) if rng else 0
            self.send_response(416 if start >= n else 206 if rng else 200)
            if rng:
                self.send_header("Content-Range", f"bytes */{n}" if start >= n else f"bytes {start}-{n - 1}/{n}")
            self.send_header("Content-Length", str(max(n - start, 0)))
            self.end_headers()
            self.wfile.write(blob[start:])

        log_message = staticmethod(lambda *args: None)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/shard.zst"
    server.shutdown()
    server.server_close()


def tap(wire, drop_after=-1, strict=False):
    """urlopen that logs each Range header and can drop after n chunks; strict is --resume-only's guard."""

    def opener(request):
        wire.append(request.get_header("Range"))
        if strict and wire[-1] is None:
            raise PermissionError("refusing a request without Range")
        response, left = urllib.request.urlopen(request), [drop_after]
        if strict and response.status != 206:
            response.close()
            raise PermissionError(f"server answered {response.status} to a Range request")
        real = response.read

        def read(size):
            left[0] -= 1
            if left[0] == -1:
                raise ConnectionResetError("simulated drop")
            return real(size)

        response.read = read
        return response

    return opener


def resume_only(ref, cache, plan, wire):
    """--resume-only: the reference's own verified checkpoint must exist before anything is fetched."""
    loader = ref.StreamingDownloader(cache, opener=tap(wire, strict=True), chunk_bytes=CHUNK)
    shard, ckpt = loader._paths_for(plan.shard_id)
    state = loader._read_checkpoint(ckpt)
    if state is None or state.url != plan.url or not loader._verify_partial(shard, state):
        raise PermissionError(f"no verified checkpoint for {plan.shard_id}")
    try:
        return loader.download(plan).sha256
    except urllib.request.HTTPError as exc:  # 416 on a finished shard: nothing left to fetch
        if exc.code == 416 and exc.headers.get("Content-Range") == f"bytes */{state.verified_bytes}":
            return "complete"
        raise


def run_case(ref, blob, case, mode):
    """One mode against one cache state, set up as a CI job would find it: (outcome, Range headers, bytes on disk)."""
    with tempfile.TemporaryDirectory() as d, serve(blob, case != "ignores_range") as url:
        cache, wire, src = pathlib.Path(d), [], pathlib.Path(d) / "src.zst"
        src.write_bytes(blob)
        plan = ref.ShardPlan("s", src.as_uri() if case == "file" else url)
        with contextlib.suppress(ConnectionResetError):
            if case != "empty":
                ref.StreamingDownloader(cache, tap([], 1 << 30 if case == "completed" else 3), CHUNK).download(plan)
        if case == "corrupted":
            (cache / "s.zst").write_bytes(b"\x00" + (cache / "s.zst").read_bytes()[1:])
        try:
            out = resume_only(ref, cache, plan, wire) if mode == "gate" else (
                ref.StreamingDownloader(cache, tap(wire), CHUNK).download(plan).sha256)
        except (PermissionError, urllib.request.HTTPError) as exc:
            out = f"{type(exc).__name__}: {exc}"
        out = {hashlib.sha256(blob).hexdigest(): "ok"}.get(out, out)
        return out, wire, sum(f.stat().st_size for f in cache.glob("s.zst"))


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    rng = random.Random(7)
    text = "\n".join(" ".join(f"t{rng.randrange(50000)}" for _ in range(12)) for _ in range(3000)) + "\n"
    blob = ref.zstd.ZstdCompressor(level=1).compress(text.encode())
    return {"size": len(blob), **{m: {c: run_case(ref, blob, c, m) for c in CASES} for m in ("reference", "gate")}}


def verify(result):
    s, p, ref, gate = result["size"], 3 * CHUNK, result["reference"], result["gate"]
    rng, no_ckpt, refused = f"bytes={p}-", "PermissionError: no verified checkpoint for s", "PermissionError: server"
    return [
        practice.Check(
            "ANSWER: resume-only resumes a verified partial and refuses every start from byte zero",
            [gate[c] for c in CASES] == [(no_ckpt, [], 0), ("ok", [rng], s), (no_ckpt, [], p),
                                         (f"{refused} answered 200 to a Range request", [rng], p),
                                         ("complete", [f"bytes={s}-"], s),
                                         (f"{refused} answered None to a Range request", [rng], p)],
            "; ".join(f"{c}: {gate[c][0]}, requests {gate[c][1]}, {gate[c][2]} B on disk" for c in CASES),
        ),
        practice.Check(
            "FINDING: the shipped downloader restarts from byte zero in 4 of 6 cache states",
            [ref[c][:2] for c in CASES if c != "completed"] == [("ok", [None]), ("ok", [rng]), ("ok", [None]),
                                                                  ("ok", [rng, None]), ("ok", [rng, None])],
            "; ".join(f"{c}: requests {ref[c][1]}" for c in CASES if c != "completed"),
        ),
        practice.Check(
            "FINDING: re-running the shipped downloader on a finished shard crashes",
            ref["completed"] == ("HTTPError: HTTP Error 416: Range Not Satisfiable", [f"bytes={s}-"], s),
            f"finished shard, second run sends {ref['completed'][1]} -> {ref['completed'][0]}",
        ),
        practice.Check(
            "FINDING: the lesson's demo transport cannot resume at all",
            ref["file"][1] == [rng, None] and gate["file"][0].endswith("answered None to a Range request"),
            f"file:// with a {p}-byte partial sends {ref['file'][1]}; resume-only: {gate['file'][0]}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
