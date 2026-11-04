#!/bin/bash
# ChronoFact demo web app on http://127.0.0.1:8090 (CPU; needs the LLM server and a finished run in results/).
# On a remote machine: ssh -L 8090:127.0.0.1:8090 <host>, then open http://localhost:8090 locally.
CF=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
. "$CF/scripts/env.sh"
mkdir -p "$CF/logs"
while true; do
  python -m uvicorn demo.server:app --app-dir "$CF" --host 127.0.0.1 --port "${CF_DEMO_PORT:-8090}" >> "$CF/logs/demo.log" 2>&1
  echo "$(date '+%F %T') demo server exited with $?; restarting" >> "$CF/logs/demo.log"
  sleep 3
done
