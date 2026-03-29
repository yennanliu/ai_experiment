# Mini ML Profiler

Lightweight ML model profiling tool with SQLite storage, FastAPI dashboard, and regression detection.

## Features

- **Profile ML models** - Latency, parameters, kernel-level metrics
- **Version bisection** - Detect performance regressions between versions
- **Interactive dashboard** - Charts for trends and model comparison
- **Real profiling** - `torch.profiler` integration when available
- **Demo mode** - Works without torch for quick testing

## Quick Start

```bash
# Install
uv sync

# Profile models (demo mode)
uv run ml-profiler profile --model resnet --version 1.0.0 --demo
uv run ml-profiler profile --model resnet --version 1.1.0 --demo

# Detect regressions
uv run ml-profiler bisect resnet

# Start dashboard
uv run ml-profiler serve
# → http://localhost:8000
```

### Using Docker

```bash
docker-compose up -d
# Dashboard at http://localhost:8000
```

## CLI Commands

```bash
# Profile a model
ml-profiler profile --model NAME --version 1.0.0 --demo

# Detect regressions between versions
ml-profiler bisect MODEL_NAME
ml-profiler bisect resnet --baseline 1.0.0 --target 2.0.0 --threshold 10

# View history
ml-profiler history
ml-profiler history --model resnet

# List models
ml-profiler models

# Start dashboard
ml-profiler serve
```

## Dashboard Pages

- `/` - Main dashboard with performance charts
- `/bisect` - Version comparison with regression detection
- `/api/results` - JSON API for all results
- `/api/compare/{model}` - Version comparison API

## Kernel-Level Profiling

When torch is installed, the profiler captures detailed kernel metrics:

```
       Top Kernels (by CPU time)
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━┓
┃ Operation        ┃ CPU (ms) ┃ Calls ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━┩
│ aten::conv2d     │ 0.834    │ 10    │
│ aten::relu       │ 0.208    │ 20    │
│ aten::batch_norm │ 0.313    │ 10    │
└──────────────────┴──────────┴───────┘
```

## Optional Dependencies

```bash
# For real PyTorch profiling
uv pip install .[torch]

# For PostgreSQL
uv pip install .[postgres]
export DATABASE_URL=postgresql://user:pass@host:5432/db
```

## Architecture

```
┌─────────────┐     ┌─────────────────┐     ┌──────────┐
│   CLI       │────▶│  Profiler       │────▶│  SQLite  │
│  (profile)  │     │  (torch.profiler)│     │          │
└─────────────┘     └─────────────────┘     └────┬─────┘
                                                 │
┌─────────────┐     ┌─────────────────┐          │
│   CLI       │────▶│  Bisection      │◀─────────┤
│  (bisect)   │     │  (regression)   │          │
└─────────────┘     └─────────────────┘          │
                                                 │
                    ┌─────────────────┐          │
                    │  Dashboard      │◀─────────┘
                    │  (FastAPI+Chart.js)
                    └─────────────────┘
```
