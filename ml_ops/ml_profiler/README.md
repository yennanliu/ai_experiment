# Mini ML Profiler

A lightweight ML model profiling tool with PostgreSQL storage and Streamlit dashboard.

## Features

- Profile PyTorch models (latency, memory, FLOPs, parameters)
- Store metrics in PostgreSQL for historical analysis
- Streamlit dashboard for visualization and version bisection
- CLI for quick profiling and querying

## Quick Start

### Using Docker (Recommended)

```bash
# Start PostgreSQL and Dashboard
docker-compose up -d

# Open dashboard at http://localhost:8501
```

### Local Development

```bash
# Install uv if not installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Start PostgreSQL (required)
docker-compose up -d db

# Run profiler CLI
uv run ml-profiler profile --model resnet18 --version 1.0.0

# View history
uv run ml-profiler history

# Start dashboard
uv run streamlit run src/ml_profiler/dashboard.py
```

## CLI Commands

```bash
# Profile a model
ml-profiler profile --model resnet18 --version 1.0.0 --batch-size 1

# Available models: resnet18, resnet50, vit

# View profiling history
ml-profiler history
ml-profiler history --model resnet18 --limit 10

# List profiled models
ml-profiler models

# Compare versions
ml-profiler compare resnet18
```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   CLI       │────▶│  Profiler   │────▶│  PostgreSQL │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
                    ┌─────────────┐             │
                    │  Dashboard  │◀────────────┘
                    │  (Streamlit)│
                    └─────────────┘
```

## Configuration

Set `DATABASE_URL` environment variable:

```bash
export DATABASE_URL=postgresql://user:pass@host:5432/dbname
```
