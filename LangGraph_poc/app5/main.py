import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from rag.ingest import ingest_file, list_collections, delete_collection
from rag.query_transform import hyde_embedding
from rag.ingest import _client as chroma_client
from rag.chunkers import STRATEGIES
from langchain_openai import OpenAIEmbeddings
from agent.graph import run

_embedder = None
def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"))
    return _embedder


def pca2d(matrix: np.ndarray) -> np.ndarray:
    """Project N×D matrix to N×2 via PCA."""
    m = matrix - matrix.mean(axis=0)
    _, _, Vt = np.linalg.svd(m, full_matrices=False)
    return (m @ Vt[:2].T).tolist()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")).mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder="static", static_url_path="")


@app.get("/")
def ui():
    return send_from_directory("static", "index.html")


@app.get("/collections")
def get_collections():
    return jsonify(list_collections())


@app.delete("/collections/<name>")
def remove_collection(name):
    delete_collection(name)
    return jsonify({"deleted": name})


@app.post("/load-samples")
def load_samples():
    strategy = request.get_json(force=True, silent=True) or {}
    chunk_strategy = strategy.get("chunk_strategy", "paragraph")
    data_dir = Path(__file__).parent / "data"
    results = []
    for f in sorted(data_dir.glob("*.txt")):
        n = ingest_file(str(f), "samples", f.name, strategy=chunk_strategy)
        results.append({"file": f.name, "chunks": n})
    return jsonify({"collection": "samples", "chunk_strategy": chunk_strategy, "files": results})


@app.post("/upload")
def upload():
    file = request.files.get("file")
    collection = request.form.get("collection", "default")
    chunk_strategy = request.form.get("chunk_strategy", "char")
    if not file:
        return jsonify({"error": "no file"}), 400

    allowed = {".pdf", ".txt", ".md"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        return jsonify({"error": f"unsupported type: {ext}"}), 400

    dest = Path(UPLOAD_DIR) / file.filename
    file.save(dest)
    n = ingest_file(str(dest), collection, file.filename, strategy=chunk_strategy)
    return jsonify({"file": file.filename, "collection": collection, "chunks": n, "chunk_strategy": chunk_strategy})


@app.post("/chat")
def chat():
    body = request.get_json(force=True)
    question = body.get("question", "").strip()
    collection = body.get("collection", "default")
    k = int(body.get("k", 5))
    query_transform = body.get("query_transform", "none")
    rerank = bool(body.get("rerank", False))
    evaluate = bool(body.get("evaluate", False))
    reference_answer = body.get("reference_answer", "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400
    result = run(
        question, collection, k=k,
        query_transform=query_transform, rerank=rerank,
        evaluate=evaluate, reference_answer=reference_answer,
    )
    return jsonify(result)


@app.post("/api/chunk-preview")
def chunk_preview():
    """
    Given raw text, return what each chunking strategy produces.
    Useful for understanding chunk boundaries before ingestion.

    Request body: { "text": "...", "strategies": ["char", "sentence", "paragraph"] }
    Response: { "char": [...], "sentence": [...], "paragraph": [...] }
    """
    body = request.get_json(force=True)
    text = body.get("text", "").strip()
    if not text:
        return jsonify({"error": "text is required"}), 400

    requested = body.get("strategies", list(STRATEGIES.keys()))
    unknown = [s for s in requested if s not in STRATEGIES]
    if unknown:
        return jsonify({"error": f"unknown strategies: {unknown}", "valid": list(STRATEGIES.keys())}), 400

    result = {}
    for name in requested:
        chunks = STRATEGIES[name](text)
        result[name] = {
            "chunks": chunks,
            "count": len(chunks),
            "avg_chars": round(sum(len(c) for c in chunks) / len(chunks)) if chunks else 0,
        }
    return jsonify(result)


@app.get("/viz")
def viz_page():
    return send_from_directory("static", "viz.html")


@app.get("/api/viz/<collection>")
def viz_data(collection):
    col = chroma_client.get_or_create_collection(collection)
    if col.count() == 0:
        return jsonify({"points": [], "stats": {"total": 0, "sources": {}}})

    result = col.get(include=["embeddings", "documents", "metadatas"])
    embs = np.array(result["embeddings"], dtype=float)
    docs = result["documents"]
    metas = result["metadatas"]

    coords = pca2d(embs)
    sources = list(dict.fromkeys(m.get("source", "?") for m in metas))
    source_idx = {s: i for i, s in enumerate(sources)}

    points = [
        {
            "id": i,
            "x": coords[i][0],
            "y": coords[i][1],
            "text": docs[i],
            "source": metas[i].get("source", "?"),
            "chunk": metas[i].get("chunk", i),
            "strategy": metas[i].get("strategy", "char"),
            "color_idx": source_idx[metas[i].get("source", "?")],
        }
        for i in range(len(docs))
    ]

    source_counts = {}
    for m in metas:
        s = m.get("source", "?")
        source_counts[s] = source_counts.get(s, 0) + 1

    return jsonify({
        "points": points,
        "sources": sources,
        "stats": {"total": len(docs), "sources": source_counts},
    })


@app.post("/api/viz/query")
def viz_query():
    body = request.get_json(force=True)
    question = body.get("question", "").strip()
    collection = body.get("collection", "default")
    mode = body.get("mode", "none")  # none | hyde
    if not question:
        return jsonify({"error": "question required"}), 400

    col = chroma_client.get_or_create_collection(collection)
    if col.count() == 0:
        return jsonify({"error": "collection is empty"}), 400

    result = col.get(include=["embeddings"])
    all_embs = np.array(result["embeddings"], dtype=float)

    if mode == "hyde":
        q_emb, hypo_doc = hyde_embedding(question)
    else:
        q_emb = _get_embedder().embed_query(question)
        hypo_doc = ""

    q_arr = np.array(q_emb, dtype=float)
    combined = np.vstack([all_embs, q_arr])
    coords = pca2d(combined)
    query_coord = coords[-1]

    return jsonify({
        "x": query_coord[0],
        "y": query_coord[1],
        "hypo_doc": hypo_doc,
        "mode": mode,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5001)
