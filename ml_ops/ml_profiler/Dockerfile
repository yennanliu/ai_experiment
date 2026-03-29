FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ src/

# Install dependencies (lightweight, no torch)
RUN uv pip install --system .

# Expose FastAPI port
EXPOSE 8000

# Default command runs the dashboard
CMD ["uvicorn", "ml_profiler.dashboard:app", "--host", "0.0.0.0", "--port", "8000"]
