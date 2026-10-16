FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv

COPY pyproject.toml ./
COPY airlock ./airlock
COPY server ./server
RUN pip install --no-cache-dir ".[server]"

RUN useradd --create-home app
USER app

# The demo uses the scripted model only: no API key, no cost.
CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
