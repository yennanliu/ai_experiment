import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, request, jsonify, send_from_directory
from rag.ingest import ingest_file, list_collections, delete_collection
from agent.graph import run

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
    if not question:
        return jsonify({"error": "question is required"}), 400
    result = run(question, collection, k=k, query_transform=query_transform, rerank=rerank)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
