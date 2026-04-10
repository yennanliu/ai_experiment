# LangChain API

FastAPI app backed by LangChain / LangGraph.

## Run

```bash
uv run uvicorn app.main:app --reload --port 8000
```

## Endpoints

- `GET /` — Web UI
- `GET /health` — Health check
- `POST /chat` — Chat (`{"message": "..."}`)
- `POST /research` — Research → summary (`{"topic": "..."}`)
- `GET /docs` — Swagger UI
