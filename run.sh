#!/usr/bin/env bash
# Launch the Fashion-Aware Context Retrieval demo: FastAPI backend + React (Vite)
# frontend. The React app proxies /api to the backend, so just open the Vite URL.
set -e
cd "$(dirname "$0")"

# faiss + torch share OpenMP; on macOS with duplicate libomp this segfaults
# unless duplicates are allowed and threading is constrained.
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

PY=./venv/bin/python
[ -x "$PY" ] || PY=python3

echo "▶ Starting API on http://127.0.0.1:8000 …"
$PY -m uvicorn server.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!
trap "kill $API_PID 2>/dev/null" EXIT

# wait for the API (first call also warms the models)
for i in $(seq 1 60); do
  curl -s http://127.0.0.1:8000/api/health >/dev/null 2>&1 && break
  sleep 1
done
echo "✔ API ready."

echo "▶ Starting frontend (Vite) …"
cd web
[ -d node_modules ] || npm install
npm run dev
