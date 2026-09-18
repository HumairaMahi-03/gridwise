# Production image for the Gridwise campus energy optimizer.
#
# PuLP ships a bundled CBC binary for linux/amd64 only. The coinor-cbc package
# is installed so the image also works on arm64 (Cloud Run, Fly.io, Apple
# Silicon); PuLP falls back to the system solver when no bundled one applies.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends coinor-cbc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Run as an unprivileged user.
RUN useradd --create-home --uid 1000 gridwise \
    && chown -R gridwise:gridwise /app
USER gridwise

EXPOSE 8000

# /health needs no credentials, which makes it usable as a platform probe.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.getenv('PORT','8000')+'/health', timeout=4)" || exit 1

# Shell form so ${PORT} is expanded by the platform at runtime.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
