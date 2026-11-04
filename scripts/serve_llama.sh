#!/bin/bash
# Llama-3.1-8B-Instruct (Meta open weights, bf16, the commit pinned in model_pins.json) behind an OpenAI-compatible
# API on 127.0.0.1:8017, restarted automatically if it exits (the LLM layer waits for it up to 2 h).
#   CF_GPU      GPU index (default 0)          CF_GPU_MEM  share of GPU memory vLLM may use (default 0.85)
#   CF_PORT     port (default 8017; config.yaml llm.providers.local must match)
# One 24 GB GPU is enough (weights 16 GB + KV cache); a larger share of memory gives more concurrent requests.
CF=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
[ -f "$CF/env.local.sh" ] && . "$CF/env.local.sh"
[ -f "$CF/.venv/bin/activate" ] && . "$CF/.venv/bin/activate"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=${CF_GPU:-0}
REPO=unsloth/Llama-3.1-8B-Instruct
REV=$(python -c "import json; print(json.load(open('$CF/model_pins.json'))['$REPO'])")
mkdir -p "$CF/logs"
while true; do
  echo "$(date '+%F %T') starting vLLM ($REPO@$REV, GPU $CUDA_VISIBLE_DEVICES, memory ${CF_GPU_MEM:-0.85})" >> "$CF/logs/vllm_restarts.log"
  vllm serve "$REPO" --revision "$REV" --served-model-name llama-3.1-8b-instruct \
    --host 127.0.0.1 --port "${CF_PORT:-8017}" --dtype bfloat16 --max-model-len 8192 \
    --gpu-memory-utilization "${CF_GPU_MEM:-0.85}" --max-num-seqs 256 --enable-prefix-caching ${CF_VLLM_EXTRA} \
    >> "$CF/logs/vllm.log" 2>&1
  echo "$(date '+%F %T') vLLM exited with $?" >> "$CF/logs/vllm_restarts.log"
  sleep 20
done
