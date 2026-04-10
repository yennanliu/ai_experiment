# Music Generation App

AI-powered music composer — describe music in natural language, get playable sheet music.

## How it works

1. You describe the music you want (genre, mood, key, tempo...)
2. LangChain + OpenAI generates music in **ABC notation**
3. The browser renders sheet music and plays it using **abcjs**

## Run

```bash
uv run uvicorn app2.main:app --reload --port 8001
```

Open http://localhost:8001

## Endpoints

- `GET /` — Web UI with sheet music renderer + audio player
- `POST /generate` — Generate music (`{"prompt": "..."}` → ABC notation)
- `GET /health` — Health check
- `GET /docs` — Swagger UI
