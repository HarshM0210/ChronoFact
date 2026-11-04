"""Profiles and the Meta API policy guard (no network: every check runs before a request would be sent)."""
import asyncio
import json
import os
import sqlite3
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from chronofact import llm as L  # noqa: E402
from chronofact.config import CFG  # noqa: E402


def _profile(name, expr):
    out = subprocess.run([sys.executable, "-c", f"import json; from chronofact.config import CFG, RESULTS, role_model; "
                          f"print(json.dumps({expr}))"], cwd=ROOT, capture_output=True, text=True,
                         env=dict(os.environ, CF_PROFILE=name, CF_RESULTS=""))
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_profiles_resolve():
    loc = _profile("local", '{r: role_model(r)["name"] for r in CFG["llm"]["roles"]} | {"_res": RESULTS}')
    assert set(v for k, v in loc.items() if k != "_res") == {"llama-3.1-8b-instruct"}
    assert loc["_res"].endswith("/results")
    sp = _profile("spark", '{"ex": role_model("extractor")["name"], "judge": role_model("judge")["name"], '
                           '"chunk": CFG["write"]["extract_chunk"], "steps": CFG["steps"], "res": RESULTS, '
                           '"rule_c": CFG["w2"]["rule_c"], "num_sim": CFG["w2"]["num_sim"]}')
    assert sp["ex"] == "muse-spark-1.3-contributor" and sp["judge"] == "muse-spark-1.3"
    assert sp["steps"] == {"batch_check": True, "cost_gate": True} and sp["chunk"] in (2, 3, 5)
    assert sp["rule_c"] is True and sp["num_sim"] == 0.80          # deep merge keeps the untouched keys
    assert sp["res"].endswith("/results_spark")
    hy = _profile("hybrid", '[role_model("extractor")["name"], role_model("judge")["name"]]')
    assert hy == ["llama-3.1-8b-instruct", "muse-spark-1.3"]


def test_unknown_profile_fails():
    out = subprocess.run([sys.executable, "-c", "import chronofact.config"], cwd=ROOT, capture_output=True, text=True,
                         env=dict(os.environ, CF_PROFILE="nope"))
    assert out.returncode != 0 and "unknown profile" in (out.stderr + out.stdout)


@pytest.fixture
def guard(tmp_path):
    g = L.LLM.__new__(L.LLM)
    g.cfg = CFG["llm"]
    g.db = sqlite3.connect(str(tmp_path / "c.sqlite"), check_same_thread=False)
    g.db.executescript(L._SCHEMA)
    import threading
    g.lock = threading.Lock()
    g.stop_path = str(tmp_path / "policy_stop.json")
    return g


def test_guard_stops_after_max_rejections(guard):
    names = ["muse-spark-1.3", "muse-spark-1.3-contributor"]
    guard._check_policy("meta", names)                     # nothing refused yet: passes
    for i in range(int(CFG["llm"]["policy_guard"]["max_rejections"])):
        guard.db.execute("INSERT INTO rejected VALUES (?,?,?,?)", (f"k{i}", "muse-spark-1.3", 0.0, "400 filtered"))
    with pytest.raises(L.PolicyStop):
        guard._check_policy("meta", names)
    assert json.load(open(guard.stop_path))["provider"] == "meta"
    with pytest.raises(L.PolicyStop):                      # the stop file alone keeps stopping
        guard.db.execute("DELETE FROM rejected")
        guard._check_policy("meta", names)
    guard._check_policy("local", ["llama-3.1-8b-instruct"])   # other providers are not affected


def test_refused_request_is_never_resent(guard, monkeypatch):
    guard._flight, guard._logs, guard.n429 = {}, [], 0
    key = L.LLM.key("muse-spark-1.3", [{"role": "user", "content": "x"}],
                    {"temperature": 0.0, "max_tokens": 1152, "extra": {"reasoning_effort": "minimal"}})
    guard.db.execute("INSERT INTO rejected VALUES (?,?,?,?)", (key, "muse-spark-1.3", 0.0, "400 filtered"))

    async def boom(*a, **k):
        raise AssertionError("a refused request was sent again")
    monkeypatch.setattr(guard, "_send", boom)
    monkeypatch.setitem(CFG["llm"]["roles"], "judge", {"model": "spark", "max_tokens": 128})
    with pytest.raises(L.LLMJSONError) as e:
        asyncio.run(guard.acall("judge", [{"role": "user", "content": "x"}]))
    assert e.value.rejected


def test_missing_api_key_never_sends(guard, monkeypatch):
    monkeypatch.delenv("META_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="META_API_KEY"):
        guard._client({"clients": {}}, "meta")
