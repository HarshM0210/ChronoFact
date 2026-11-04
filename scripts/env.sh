# source scripts/env.sh   (sets up the environment for every ChronoFact command and cds into the repo)
CF=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[ -f "$CF/env.local.sh" ] && . "$CF/env.local.sh"          # machine-specific settings (not in git), e.g. HF_HOME
[ -f "$CF/.venv/bin/activate" ] && . "$CF/.venv/bin/activate"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}   # models: download_models.py
[ -z "$META_API_KEY" ] && [ -f "$CF/.meta_key" ] && export META_API_KEY=$(cat "$CF/.meta_key")  # only profiles hybrid/spark
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=""          # pipeline processes run on CPU; only the vLLM server uses a GPU
cd "$CF"
