"""Downloads every model ChronoFact uses, at the commit pinned in model_pins.json, into HF_HOME:
Contriever (embeddings), BART-large-MNLI (verification) and Llama-3.1-8B-Instruct (LLM + token counter; the
ungated mirror unsloth/Llama-3.1-8B-Instruct of Meta's weights).

  source scripts/env.sh && HF_HUB_OFFLINE=0 python scripts/download_models.py [--no-llm]
"""
import json
import os
import sys

from huggingface_hub import snapshot_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PINS = json.load(open(os.path.join(ROOT, "model_pins.json")))
LLM = "unsloth/Llama-3.1-8B-Instruct"

for repo, rev in PINS.items():
    if repo == LLM and "--no-llm" in sys.argv:
        patterns = ["*.json", "tokenizer*"]                  # the token counter only
    elif repo == LLM:
        patterns = ["*.json", "*.safetensors", "tokenizer*"]
    else:
        patterns = ["*.json", "*.txt", "*.model", "*.safetensors", "*.bin", "merges.txt", "vocab.*", "tokenizer*"]
    print(f"{repo}@{rev}", flush=True)
    snapshot_download(repo, revision=rev, allow_patterns=patterns)
print("all models downloaded")
