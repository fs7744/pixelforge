#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
PY="${PY:-$HOME/.workbuddy/binaries/python/envs/pixelforge/Scripts/python.exe}"
[ -x "$PY" ] || PY="$HOME/.workbuddy/binaries/python/envs/pixelforge/bin/python"
echo "PixelForge → http://127.0.0.1:8787"
exec "$PY" -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8787
