# Mini ML Profiler

Lightweight ML model profiling tool with SQLite storage and FastAPI dashboard.

## Features

- Profile ML models (latency, parameters)
- SQLite storage (PostgreSQL optional)
- FastAPI dashboard with JSON API
- Demo mode (no torch required)

## Quick Start

### Using Docker

```bash
docker-compose up -d
# Dashboard at http://localhost:8000
```

### Local Development

```bash
# Install
uv sync

# Profile (demo mode - no torch needed)
uv run ml-profiler profile --model my_model --demo

# Profile with torch (optional)
uv pip install ml-profiler[torch]
uv run ml-profiler profile --model resnet18

# View history
uv run ml-profiler history

# Start dashboard
uv run ml-profiler serve
```

## CLI Commands

```bash
ml-profiler profile --model NAME --version 1.0.0 --demo
ml-profiler history
ml-profiler models
ml-profiler serve   # Start web dashboard
```

## API Endpoints

- `GET /` - Dashboard
- `GET /api/results` - All results (JSON)
- `GET /api/models` - List models
- `GET /api/compare/{model}` - Compare versions

## Optional Dependencies

```bash
# For real PyTorch profiling
uv pip install ml-profiler[torch]

# For PostgreSQL
uv pip install ml-profiler[postgres]
export DATABASE_URL=postgresql://user:pass@host:5432/db
```
