FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ src/

# Install dependencies
RUN uv pip install --system -e .

# Expose Streamlit port
EXPOSE 8501

# Default command runs the dashboard
CMD ["streamlit", "run", "src/ml_profiler/dashboard.py", "--server.address=0.0.0.0"]
