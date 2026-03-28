FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY dvs ./dvs
ENV PYTHONPATH=/app

# Default working directory for user data / repo mounts
WORKDIR /workspace

ENTRYPOINT ["python", "-m", "dvs"]
CMD ["--help"]
