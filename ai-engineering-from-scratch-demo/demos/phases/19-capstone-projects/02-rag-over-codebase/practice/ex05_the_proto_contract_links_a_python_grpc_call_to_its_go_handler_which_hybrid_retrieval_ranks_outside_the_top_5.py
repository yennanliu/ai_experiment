"""Exercise 5 -- the proto contract links a Python gRPC call to its Go handler, which hybrid retrieval ranks outside the top 5.

    Extend to cross-language symbol resolution: a Python function that calls a Go service over gRPC. Use the symbol graph to link them.

Reading of the exercise: the lesson's code has no symbol graph, so one is
built here over chunks in the lesson's `Chunk` shape. The fixture adds seven
chunks to `SAMPLE_CORPUS`. A Python job, `cancel_stale_uploads`, calls
`UploaderStub(channel).Abort(...)`. Two .proto files each declare an `Abort`
rpc: `Uploader` and a look-alike `Exporter`. Two Go servers implement them
and register their structs with `pb.Register<Service>Server`. The uploader's
handler calls the lesson's own `AbortMultipartOnFail`. The graph resolves a
call in three steps. The Python stub gives the service and the called method.
The proto confirms that `service.rpc` exists. The Go registration maps the
receiver struct to its service. Plain same-name calls add the within-language
edges. The question "what runs on the server when cancel_stale_uploads aborts
an upload" is asked of the lesson's `answer`, then the top hit is expanded
along graph edges.

**ANSWER: the graph links `cancel_stale_uploads` to `uploaderServer.Abort`
and on to `AbortMultipartOnFail`, a Python -> gRPC -> Go -> Go chain.**
Expanding the top hit two hops recovers both server-side chunks, 2 of 2.

**FINDING: hybrid retrieval alone does not find the server side.** On the
question above, `answer` puts the Python caller first. Its top 5 hold 1 of
the 2 server chunks (`AbortMultipartOnFail`, on the word "abort"), and the
Go handler that actually runs is not among them. The caller never names the
Go repo or file. The only link is the pair of names `Uploader` and `Abort`.

**FINDING: linking by method name alone is wrong half the time.** Matching
the call `Abort` to any Go method named `Abort` gives 2 edges, one of them to
the export job's handler. Qualifying by the service in the proto and the Go
registration gives 1 edge, the right one.

Structure: `FIXTURE` holds the seven chunks; `edges` extracts the graph with
regexes (a stand-in for tree-sitter queries); `expand` walks it breadth-first.
"""

from __future__ import annotations

import hashlib
import re

from harness import parity, practice

PHASE, LESSON = "19-capstone-projects", "02-rag-over-codebase"
FIXTURE = [  # (repo, path, symbol, body, summary)
    ("billing", "jobs/cleanup.py", "cancel_stale_uploads",
     "stub = uploader_pb2_grpc.UploaderStub(channel)\nfor up in stale: stub.Abort(uploader_pb2.AbortRequest(upload_id=up.id))",
     "cancels uploads idle for more than a day"),
    ("protos", "uploader/v1/uploader.proto", "Uploader",
     "service Uploader { rpc Abort(AbortRequest) returns (AbortReply); }", "gRPC contract of the uploader service"),
    ("protos", "exporter/v1/exporter.proto", "Exporter",
     "service Exporter { rpc Abort(ExportAbortRequest) returns (Empty); }", "gRPC contract of the export service"),
    ("uploader", "services/grpc/server.go", "uploaderServer.Abort",
     "func (s *uploaderServer) Abort(ctx context.Context, req *pb.AbortRequest) (*pb.AbortReply, error) "
     "{ return s.retry.AbortMultipartOnFail(ctx, req.UploadId) }", "gRPC handler for the uploader service"),
    ("uploader", "cmd/uploader/main.go", "main", "pb.RegisterUploaderServer(srv, &uploaderServer{})",
     "starts the uploader gRPC server"),
    ("exporter", "services/grpc/export.go", "exportServer.Abort",
     "func (s *exportServer) Abort(ctx context.Context, req *pb.ExportAbortRequest) (*pb.Empty, error) "
     "{ return s.jobs.Cancel(req.JobId) }", "gRPC handler that stops a running export job"),
    ("exporter", "cmd/exporter/main.go", "main", "pb.RegisterExporterServer(srv, &exportServer{})",
     "starts the exporter gRPC server"),
]
QUESTION = "what runs on the server when cancel_stale_uploads aborts an upload"


def rpc_tables(chunks):
    declared = {(svc, m) for c in chunks for svc, body in re.findall(r"service (\w+) \{([^}]*)\}", c.body)
                for m in re.findall(r"rpc (\w+)\(", body)}
    served_by = dict((s, v) for c in chunks for v, s in re.findall(r"Register(\w+)Server\(\w+, &(\w+)\{", c.body))
    handlers = {}
    for c in chunks:
        for struct, method in re.findall(r"func \(\w+ \*(\w+)\) (\w+)\(ctx context\.Context", c.body):
            handlers.setdefault(method, []).append((served_by.get(struct), c))
    return declared, handlers


def edges(chunks, qualified=True):
    """{caller anchor: [callee chunks]}; gRPC edges need the service to match when `qualified`."""
    declared, handlers = rpc_tables(chunks)
    by_symbol = {c.symbol: c for c in chunks}
    graph = {}
    for c in chunks:
        out = [by_symbol[n] for n in re.findall(r"\.(\w+)\(", c.body) if n in by_symbol and by_symbol[n] is not c]
        graph[c.anchor()] = out + grpc_callees(c, declared, handlers, qualified)
    return graph


def grpc_callees(c, declared, handlers, qualified):
    """Go handlers behind every `<Service>Stub(...).<method>(...)` call in chunk `c`."""
    calls = [(svc, m) for svc in re.findall(r"(\w+)Stub\(", c.body) for m in re.findall(r"stub\.(\w+)\(", c.body)]
    if qualified:
        return [h for svc, m in calls if (svc, m) in declared for s, h in handlers.get(m, []) if s == svc]
    return [h for _, m in calls for _, h in handlers.get(m, [])]


def expand(graph, start, hops=2):
    seen, frontier = [], [start]
    for _ in range(hops):
        frontier = [c for f in frontier for c in graph.get(f.anchor(), []) if c not in seen]
        seen += frontier
    return seen


def server_hits(chunks, top):
    server = ["uploaderServer.Abort", "AbortMultipartOnFail"]
    return [c.symbol for c in chunks if c.symbol in server and c.anchor() in top]


def solve():
    ref = parity.load_reference(PHASE, LESSON, "main")
    ref.hash = lambda s: int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")
    chunks = list(ref.SAMPLE_CORPUS) + [ref.Chunk(r, p, 1, 30, s, b, m) for r, p, s, b, m in FIXTURE]
    dense, bm25 = ref.DenseIndex(), ref.BM25Index()
    for c in chunks:
        dense.add(c)
        bm25.add(c)
    top = ref.answer(QUESTION, dense, bm25)["rerank_top"]
    caller = next(c for c in chunks if c.anchor() == top[0])
    loose = edges(chunks, qualified=False)[chunks[6].anchor()]
    return {
        "top1": caller.symbol, "server_in_top5": server_hits(chunks, top),
        "chain": [c.symbol for c in expand(edges(chunks), caller)],
        "loose": [c.repo for c in loose], "qualified": [c.repo for c in edges(chunks)[chunks[6].anchor()]],
        "lesson_graph_names": sorted(n for n in dir(ref) if re.search("graph|edge|symbol", n, re.I)),
    }


def verify(result):
    r = result
    return [
        practice.Check(
            "ANSWER: the graph links cancel_stale_uploads -> uploaderServer.Abort -> AbortMultipartOnFail",
            r["chain"] == ["uploaderServer.Abort", "AbortMultipartOnFail"] and r["lesson_graph_names"] == [],
            f"two hops from the top hit: {r['chain']}; graph-like names in the lesson module: {r['lesson_graph_names']}",
        ),
        practice.Check(
            "FINDING: hybrid retrieval alone does not find the server side",
            (r["top1"], r["server_in_top5"]) == ("cancel_stale_uploads", ["AbortMultipartOnFail"]),
            f"top hit {r['top1']}; server-side chunks in the top 5: {r['server_in_top5']} of 2",
        ),
        practice.Check(
            "FINDING: linking by method name alone is wrong half the time",
            (sorted(r["loose"]), r["qualified"]) == (["exporter", "uploader"], ["uploader"]),
            f"name-only edges to {r['loose']}; service-qualified edges to {r['qualified']}",
        ),
    ]


PRACTICE_IMPL = {"solve": solve, "verify": verify}

if __name__ == "__main__":
    raise SystemExit(practice.selfcheck(globals()))
