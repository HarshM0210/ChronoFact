"""config.yaml, the selected profile, and the decisions the pre-declared rules wrote into <results>/steps/:
- steps/batch_check.json -> write.extract_chunk (batched extraction on the Meta API, profile spark)
- steps/pilot.json       -> write thresholds actually used (README 13.2)
Every process reads them at import, so a step always sees the decisions of the steps before it.

Environment: CF_PROFILE (local | hybrid | spark), CF_RESULTS (results directory), CF_CACHE_DIR (where the LLM,
embedding and NLI caches live; default the results directory), CF_CONFIG (another config file)."""
import copy
import json
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.environ.get("CF_CONFIG", os.path.join(ROOT, "config.yaml"))) as _f:
    _BASE = yaml.safe_load(_f)


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


PROFILE = os.environ.get("CF_PROFILE") or _BASE["profile"]
if PROFILE not in _BASE["profiles"]:
    raise SystemExit(f"unknown profile {PROFILE!r}; config.yaml defines {sorted(_BASE['profiles'])}")
CFG = _merge({k: v for k, v in _BASE.items() if k != "profiles"}, _BASE["profiles"][PROFILE])
CFG["profile"] = PROFILE

RESULTS = os.environ.get("CF_RESULTS") or os.path.join(ROOT, "results" if PROFILE == "local" else f"results_{PROFILE}")
CACHE_DIR = os.environ.get("CF_CACHE_DIR") or RESULTS
STEPS = os.path.join(RESULTS, "steps")


def path(*parts):
    return os.path.join(ROOT, *parts)


def rpath(*parts):
    return os.path.join(RESULTS, *parts)


def cpath(name):
    """a cache file (LLM, embeddings, NLI, rate limiter) by its config name"""
    return name if os.path.isabs(name) else os.path.join(CACHE_DIR, name)


def step_file(name):
    return os.path.join(STEPS, f"{name}.json")


def load_step(name):
    with open(step_file(name)) as f:
        return json.load(f)


def _overlay():
    if os.path.exists(step_file("batch_check")):
        CFG["write"]["extract_chunk"] = load_step("batch_check")["chosen_chunk"]
    if os.path.exists(step_file("pilot")):
        CFG["write"].update(load_step("pilot")["final_thresholds"])


_overlay()


_PINS = path("model_pins.json")


def pin(repo_id):
    """Pinned commit of a Hugging Face model (model_pins.json; scripts/download_models.py fetches exactly these)."""
    with open(_PINS) as f:
        return json.load(f).get(repo_id)


def embedder_name():
    return CFG["embeddings"]["name"]


def role_model(role):
    """the model entry (dict with its name and provider) a role runs on"""
    return CFG["llm"]["models"][CFG["llm"]["roles"][role]["model"]]


def uses_meta_api():
    return any(role_model(r)["provider"] == "meta" for r in CFG["llm"]["roles"])
