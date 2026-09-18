#!/usr/bin/env bash
# Local development server.
set -euo pipefail

if [ ! -f .env ]; then
  echo "No .env found. Copy .env.example to .env and add your LLM credentials." >&2
fi

exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --reload
