"""FastAPI server for the Ticket Processing System"""

import asyncio
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from graph import process_ticket

app = FastAPI(title="Ticket Processing System")

# In-memory store: ticket_id → record
store: dict[str, dict] = {}

# ── Models ────────────────────────────────────────────────────────────────────

class SubmitRequest(BaseModel):
    message: str

class TicketRecord(BaseModel):
    ticket_id: str
    status: Literal["pending", "processing", "done", "error"]
    submitted_at: str
    message: str
    result: dict | None = None
    error: str | None = None

# ── Background task ───────────────────────────────────────────────────────────

async def _run_pipeline(ticket_id: str, message: str) -> None:
    store[ticket_id]["status"] = "processing"
    try:
        state = await asyncio.to_thread(process_ticket, ticket_id, message)
        store[ticket_id]["status"] = "done"
        store[ticket_id]["result"] = {
            "category": state.category,
            "confidence": round(state.confidence, 2),
            "priority": state.priority,
            "sla_hours": state.sla_hours,
            "department": state.department,
            "response": state.response,
            "quality_score": round(state.quality_score, 2),
            "retry_count": state.retry_count,
            "history": state.history,
        }
    except Exception as exc:
        store[ticket_id]["status"] = "error"
        store[ticket_id]["error"] = str(exc)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/tickets", response_model=TicketRecord, status_code=202)
async def submit_ticket(body: SubmitRequest):
    ticket_id = f"TKT-{uuid.uuid4().hex[:6].upper()}"
    record = {
        "ticket_id": ticket_id,
        "status": "pending",
        "submitted_at": datetime.now().isoformat(),
        "message": body.message,
        "result": None,
        "error": None,
    }
    store[ticket_id] = record
    asyncio.create_task(_run_pipeline(ticket_id, body.message))
    return record


@app.get("/tickets", response_model=list[TicketRecord])
async def list_tickets():
    return list(store.values())


@app.get("/tickets/{ticket_id}", response_model=TicketRecord)
async def get_ticket(ticket_id: str):
    if ticket_id not in store:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return store[ticket_id]


# ── UI ────────────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_ui():
    return FileResponse("ui/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
