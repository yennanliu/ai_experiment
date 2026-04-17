import os
import json
import asyncio
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")).mkdir(parents=True, exist_ok=True)

from models.ollama_client import list_models, health_check, chat_stream
from rag.ingest import ingest_file, list_collections, delete_collection
from rag.retriever import retrieve

app = FastAPI(title="PrivateAssist")
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Health & Models ──────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    ok = await health_check()
    return {"ollama": "ok" if ok else "unreachable"}


@app.get("/models")
async def models():
    return {"models": await list_models()}


# ── Collections ───────────────────────────────────────────────────────────────

@app.get("/collections")
async def collections():
    return {"collections": list_collections()}


@app.delete("/collections/{name}")
async def remove_collection(name: str):
    delete_collection(name)
    return {"deleted": name}


# ── Upload & Ingest ───────────────────────────────────────────────────────────

@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    collection: str = Form(...),
):
    allowed = {".pdf", ".txt", ".md"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    dest = Path(UPLOAD_DIR) / file.filename
    dest.write_bytes(await file.read())

    chunks = await ingest_file(str(dest), collection, file.filename)
    return {"file": file.filename, "collection": collection, "chunks": chunks}


# ── Chat (SSE streaming) ──────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    collection: str


@app.post("/chat")
async def chat(req: ChatRequest):
    async def event_stream():
        # Step 1: retrieve
        yield f"data: {json.dumps({'step': 'retrieving'})}\n\n"
        chunks = await retrieve(req.question, req.collection)

        if not chunks:
            yield f"data: {json.dumps({'step': 'done', 'answer': 'No documents found in this collection.', 'sources': []})}\n\n"
            return

        sources = list(dict.fromkeys(src for _, src in chunks))
        yield f"data: {json.dumps({'step': 'retrieved', 'count': len(chunks), 'sources': sources})}\n\n"

        # Step 2: stream generation
        context = "\n\n---\n\n".join(text for text, _ in chunks)
        prompt = f"Context:\n{context}\n\nQuestion: {req.question}\n\nAnswer:"
        system = (
            "You are a helpful assistant. Answer using ONLY the provided context. "
            "If the answer is not in the context, say so."
        )

        yield f"data: {json.dumps({'step': 'generating'})}\n\n"

        full_answer = []
        async for token in chat_stream(prompt, system=system):
            full_answer.append(token)
            yield f"data: {json.dumps({'step': 'token', 'token': token})}\n\n"

        yield f"data: {json.dumps({'step': 'done', 'answer': ''.join(full_answer), 'sources': sources})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── UI ────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def ui():
    return Path("static/index.html").read_text()
