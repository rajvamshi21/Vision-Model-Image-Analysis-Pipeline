FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[db,api]"

COPY db ./db
COPY demo ./demo

EXPOSE 8000
CMD ["uvicorn", "vqa.api:app", "--host", "0.0.0.0", "--port", "8000"]
