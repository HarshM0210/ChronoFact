"""The LLM call layer (README section 11). One function, acall(), for every LLM call.

- a role runs on a model (config llm.roles), a model on a provider (llm.models / llm.providers): Llama-3.1-8B-Instruct
  on a local vLLM server and/or Muse Spark on the Meta Model API, both through the OpenAI-compatible chat API
- temperature 0; role max_tokens + the model's reasoning allowance (Spark's hidden reasoning counts against
  max_tokens); reasoning_effort only for models that take it; JSON schema via response_format (vLLM enforces it by
  guided decoding)
- a model with an rpm limit gets one rate limiter SHARED BY ALL PROCESSES (a file-locked schedule), set below the
  limit, so a run never depends on 429 retries
- disk cache (SQLite) keyed by sha256 of {"model", "messages", "params"}; every attempt is cached, including invalid
  JSON, so a re-run replays the same retry path; identical requests in flight are coalesced
- JSON retries: bad reply + nudge, max_tokens doubled, 3 attempts in total, then LLMJSONError
- outages (connection errors, timeouts, 5xx) and 429s: wait 5 s growing to 60 s, retry, up to llm.outage_max_s
- spend cap over ALL paid calls (priced from `usage`; the running total is on disk, shared by all processes)
- policy guard for the Meta API (llm.policy_guard): a request the API refuses (4xx, e.g. its content filter) is
  recorded and never sent again, and callers never re-send its content in another form (LLMJSONError.rejected);
  a 401/403 writes policy_stop.json and every process stops sending to that provider; after max_rejections
  refusals the same happens. Nothing reaches the provider again until the operator deletes policy_stop.json.
- the stats of a call include the tokens of ALL its attempts (Model4 counted only the last one)
"""
import asyncio
import contextvars
import fcntl
import hashlib
import json
import os
import sqlite3
import threading
import time

from openai import (APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, AuthenticationError,
                    BadRequestError, InternalServerError, PermissionDeniedError, RateLimitError)

from chronofact.config import CFG, cpath
from chronofact.interface import CallStats

PHASE = contextvars.ContextVar("phase", default="eval")       # write | read | eval
RUN_TAG = contextvars.ContextVar("run_tag", default="")
JSON_NUDGE = "That was not valid JSON (possibly cut off). Reply again with shorter, valid JSON only."


class LLMJSONError(RuntimeError):
    """No valid reply. rejected=True: the API refused the request itself (4xx, e.g. its content filter); callers must
    NOT re-send the same content in another form."""

    def __init__(self, msg, stats=None, rejected=False):
        super().__init__(msg)
        self.stats = stats or CallStats()
        self.rejected = rejected


class SpendCapError(RuntimeError):
    pass


class PolicyStop(RuntimeError):
    """The policy guard stopped all traffic to a provider (see policy_stop.json). Never caught by the pipeline."""


class _Rejected(RuntimeError):
    """a request the API refused (recorded in the `rejected` table; never sent again)"""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, model TEXT, response TEXT, tokens_in INTEGER,
    tokens_out INTEGER, latency_ms REAL, created REAL, finish TEXT);
CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, role TEXT, phase TEXT,
    run_tag TEXT, model TEXT, tokens_in INTEGER, tokens_out INTEGER, latency_ms REAL, cache_hit INTEGER,
    pid INTEGER, finish TEXT, json_ok INTEGER);
CREATE TABLE IF NOT EXISTS spend (id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, key TEXT, model TEXT,
    tokens_in INTEGER, tokens_cached INTEGER, tokens_out INTEGER, usd REAL);
CREATE TABLE IF NOT EXISTS rejected (key TEXT PRIMARY KEY, model TEXT, ts REAL, error TEXT);
"""


def _is_server_error(e):
    return isinstance(e, (APIConnectionError, APITimeoutError, InternalServerError)) or (
        isinstance(e, APIStatusError) and getattr(e, "status_code", 0) >= 500)


class LLM:
    def __init__(self):
        self.cfg = CFG["llm"]
        p = os.environ.get("CF_LLM_CACHE") or cpath(self.cfg["cache_path"])
        if os.environ.get("CF_FAKE_LLM") == "1" and not os.environ.get("CF_LLM_CACHE"):
            p = p + ".fake"                       # fake answers never reach the real cache
        os.makedirs(os.path.dirname(p), exist_ok=True)
        self.db = sqlite3.connect(p, timeout=120, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(_SCHEMA)
        self.db.commit()
        self.lock = threading.Lock()
        self.stop_path = cpath("policy_stop.json")
        self._per_loop = {}
        self._old_clients = []
        self._flight = {}
        self._logs = []
        self.n429 = 0

    # ------------------------------------------------------------------ models
    def spec(self, model_key):
        """model entry + its provider entry"""
        m = self.cfg["models"][model_key]
        return m, self.cfg["providers"][m["provider"]]

    def _guarded(self, provider):
        return provider in self.cfg.get("policy_guard", {}).get("providers", ["meta"])

    def _loop_state(self):
        loop = asyncio.get_running_loop()
        st = self._per_loop.get(id(loop))
        if st is None or st["loop"] is not loop:
            # a finished loop's clients are kept (never garbage-collected): their __del__ would schedule aclose() on
            # the NEW loop for sockets of the closed one ("Event loop is closed" noise)
            for v in self._per_loop.values():
                if v["loop"].is_closed():
                    self._old_clients += list(v["clients"].values())
            self._per_loop = {k: v for k, v in self._per_loop.items() if not v["loop"].is_closed()}
            st = self._per_loop[id(loop)] = {"loop": loop, "clients": {}, "sem": {}}
        return st

    def _client(self, st, provider):
        if provider not in st["clients"]:
            pv = self.cfg["providers"][provider]
            env = pv.get("api_key_env")
            key = os.environ.get(env, "") if env else "EMPTY"
            if env and not key:      # never send without a key: a 401 would look like a blocked account
                raise RuntimeError(f"provider {provider!r} needs the API key in ${env}")
            st["clients"][provider] = AsyncOpenAI(base_url=pv["base_url"], api_key=key, timeout=600, max_retries=0)
        return st["clients"][provider]

    def params(self, role, model_key, json_schema, max_tokens):
        m, _ = self.spec(model_key)
        p = {"temperature": 0.0,
             "max_tokens": int(max_tokens or self.cfg["roles"][role]["max_tokens"]) + int(m["reasoning_allowance"])}
        if json_schema is not None:
            p["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": json_schema}}
        return p, ({"reasoning_effort": m["reasoning_effort"]} if m.get("reasoning_effort") else None)

    def _slot(self, name, rpm):
        """Seconds to wait for this request's turn under the model's shared rate limit (all processes)."""
        d = cpath("ratelimit")
        os.makedirs(d, exist_ok=True)
        gap = 60.0 / float(rpm)
        with open(os.path.join(d, name), "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.seek(0)
                raw = f.read().strip()
                now = time.time()
                nxt = max(now, float(raw) if raw else 0.0)
                f.seek(0)
                f.truncate()
                f.write(f"{nxt + gap:.6f}")
                f.flush()
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
        return nxt - now

    # ------------------------------------------------------------------ policy guard
    def _stopped(self, provider):
        if os.path.exists(self.stop_path):
            with open(self.stop_path) as f:
                s = json.load(f)
            if s.get("provider") == provider:
                return s
        return None

    def _stop(self, provider, reason):
        tmp = f"{self.stop_path}.tmp{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump({"provider": provider, "reason": reason, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
        os.replace(tmp, self.stop_path)
        print(f"[llm] POLICY STOP for {provider}: {reason}. No request is sent to it until {self.stop_path} is "
              f"deleted.", flush=True)

    def _check_policy(self, provider, names):
        if not self._guarded(provider):
            return
        s = self._stopped(provider)
        if s:
            raise PolicyStop(f"requests to {provider} are stopped ({s['reason']}, {s['ts']}); see {self.stop_path}")
        with self.lock:
            n = self.db.execute(f"SELECT COUNT(*) FROM rejected WHERE model IN ({','.join('?' * len(names))})",
                                names).fetchone()[0]
        limit = int(self.cfg.get("policy_guard", {}).get("max_rejections", 3))
        if n >= limit:
            self._stop(provider, f"{n} requests refused by the provider (limit {limit})")
            raise PolicyStop(f"{n} requests to {provider} were refused; stopped before sending more")

    # ------------------------------------------------------------------ cache + log
    @staticmethod
    def key(model, messages, params):
        blob = json.dumps({"model": model, "messages": messages, "params": params}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _get(self, key):
        with self.lock:
            return self.db.execute("SELECT response, tokens_in, tokens_out, latency_ms, finish FROM cache WHERE key=?",
                                   (key,)).fetchone()

    def _put(self, key, model, text, tin, tout, lat, finish):
        with self.lock:
            self.db.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?,?,?,?)",
                            (key, model, text, tin, tout, lat, time.time(), finish))
            self.db.commit()

    def _log(self, role, model, tin, tout, lat, hit, finish, json_ok):
        with self.lock:
            self._logs.append((time.time(), role, PHASE.get(), RUN_TAG.get(), model, tin, tout, lat, int(hit),
                               os.getpid(), finish, json_ok))
            if len(self._logs) >= 200:
                self._flush_locked()

    def _flush_locked(self):
        if self._logs:
            self.db.executemany("INSERT INTO calls (ts, role, phase, run_tag, model, tokens_in, tokens_out, "
                                "latency_ms, cache_hit, pid, finish, json_ok) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                                self._logs)
            self.db.commit()
            self._logs = []

    def flush(self):
        with self.lock:
            self._flush_locked()

    # ------------------------------------------------------------------ spend
    def spent(self):
        with self.lock:
            return float(self.db.execute("SELECT COALESCE(SUM(usd), 0) FROM spend").fetchone()[0])

    def _price(self, key, m, usage):
        pr = m["price_per_1m"]
        tin = int(getattr(usage, "prompt_tokens", 0) or 0)
        det = getattr(usage, "prompt_tokens_details", None)
        cached = int(getattr(det, "cached_tokens", 0) or 0) if det is not None else 0
        tout = int(getattr(usage, "completion_tokens", 0) or 0)
        usd = ((tin - cached) * pr["input"] + cached * pr["cached_input"] + tout * pr["output"]) / 1e6
        with self.lock:
            self.db.execute("INSERT INTO spend (ts, key, model, tokens_in, tokens_cached, tokens_out, usd) "
                            "VALUES (?,?,?,?,?,?,?)", (time.time(), key, m["name"], tin, cached, tout, usd))
            self.db.commit()

    # ------------------------------------------------------------------ one request
    async def _send(self, role, model_key, msgs, params, extra, key):
        m, _ = self.spec(model_key)
        provider, name = m["provider"], m["name"]
        names = [x["name"] for x in self.cfg["models"].values() if x["provider"] == provider]
        st = self._loop_state()
        client = self._client(st, provider)
        sem = st["sem"].setdefault(name, asyncio.Semaphore(self.cfg["max_inflight"]))
        waited = 0.0
        async with sem:
            while True:
                self._check_policy(provider, names)
                if self.spent() >= self.cfg["spend_cap_usd"]:
                    raise SpendCapError(f"LLM spend reached the ${self.cfg['spend_cap_usd']:.0f} cap "
                                        f"(${self.spent():.2f}); no new calls are sent. Step stopped.")
                if m.get("rpm"):
                    await asyncio.sleep(self._slot(name, m["rpm"]))
                t0 = time.perf_counter()
                try:
                    resp = await client.chat.completions.create(model=name, messages=msgs, extra_body=extra or None,
                                                                **params)
                    break
                except (AuthenticationError, PermissionDeniedError) as e:
                    if self._guarded(provider):
                        self._stop(provider, f"{type(e).__name__}: {str(e)[:200]}")
                        raise PolicyStop(f"{provider} refused access: {str(e)[:200]}") from e
                    raise
                except Exception as e:
                    if not (_is_server_error(e) or isinstance(e, RateLimitError)):
                        raise
                    if isinstance(e, RateLimitError):
                        self.n429 += 1
                    if waited >= float(self.cfg["outage_max_s"]):
                        raise
                    pause = min(60.0, 5.0 + waited / 4)
                    if waited == 0 or int(waited) % 300 < pause:
                        print(f"[llm] {role}: {type(e).__name__}; retrying (waited {waited:.0f}s)", flush=True)
                    await asyncio.sleep(pause)
                    waited += pause
        lat = (time.perf_counter() - t0) * 1000
        text = resp.choices[0].message.content or ""
        finish = resp.choices[0].finish_reason or ""
        tin = int(resp.usage.prompt_tokens or 0) if resp.usage else 0
        tout = int(resp.usage.completion_tokens or 0) if resp.usage else 0
        if resp.usage is not None:
            self._price(key, m, resp.usage)
        return text, tin, tout, lat, finish

    async def _cached_call(self, role, model_key, msgs, params, extra):
        """(text, tin, tout, lat, finish, hit) for one attempt, through the cache and coalescing."""
        name = self.cfg["models"][model_key]["name"]
        kp = dict(params, **({"extra": extra} if extra else {}))
        key = self.key(name, msgs, kp)
        hit = self._get(key)
        if hit:
            return (*hit, True)
        with self.lock:
            rej = self.db.execute("SELECT error FROM rejected WHERE key=?", (key,)).fetchone()
        if rej:                                   # refused once by the API: never sent again
            raise _Rejected(rej[0])
        loop = asyncio.get_running_loop()
        ck = (id(loop), key)
        fut = self._flight.get(ck)
        if fut is not None:
            return (*(await asyncio.shield(fut)), True)
        fut = loop.create_future()
        self._flight[ck] = fut
        try:
            try:
                res = await self._send(role, model_key, msgs, params, extra, key)
            except BadRequestError as e:
                with self.lock:
                    self.db.execute("INSERT OR REPLACE INTO rejected VALUES (?,?,?,?)",
                                    (key, name, time.time(), str(e)[:500]))
                    self.db.commit()
                raise _Rejected(str(e)[:500]) from e
            self._put(key, name, *res)
            fut.set_result(res)
            return (*res, False)
        except BaseException as e:
            if isinstance(e, asyncio.CancelledError):
                fut.cancel()
            else:
                fut.set_exception(e)
                fut.exception()
            raise
        finally:
            self._flight.pop(ck, None)

    async def acall(self, role, messages, json_schema=None, max_tokens=None, model=None):
        """-> (text, CallStats) or (parsed JSON, CallStats). The stats report the original (uncached) tokens
        of every attempt, so a resumed run reports the same costs; the call log records cache hits.
        model: a config model key to use instead of the role's (the answer split)."""
        if os.environ.get("CF_FAKE_LLM") == "1":
            return _fake(role, messages, json_schema)
        model_key = model or self.cfg["roles"][role]["model"]
        name = self.cfg["models"][model_key]["name"]
        params, extra = self.params(role, model_key, json_schema, max_tokens)
        stats = CallStats()
        msgs = messages
        err = None
        for _ in range(3):
            try:
                text, tin, tout, lat, finish, hit = await self._cached_call(role, model_key, msgs, params, extra)
            except _Rejected as e:               # the API refused the request: no reply will ever come
                raise LLMJSONError(f"role={role}: request rejected: {str(e)[:300]}", stats, rejected=True)
            stats.add(CallStats(1, tin, tout, lat))
            if json_schema is None:
                self._log(role, name, tin, tout, lat, hit, finish, None)
                return text, stats
            try:
                parsed = json.loads(text)
                self._log(role, name, tin, tout, lat, hit, finish, 1)
                return parsed, stats
            except json.JSONDecodeError as e:
                self._log(role, name, tin, tout, lat, hit, finish, 0)
                err = e
                msgs = messages + [{"role": "assistant", "content": text},
                                   {"role": "user", "content": JSON_NUDGE}]
                allow = int(self.cfg["models"][model_key]["reasoning_allowance"])
                params = dict(params, max_tokens=(params["max_tokens"] - allow) * 2 + allow)   # README: double it
        raise LLMJSONError(f"role={role}: invalid JSON after 3 attempts: {err}", stats)


_LLM = None


def get_llm() -> LLM:
    global _LLM
    if _LLM is None:
        _LLM = LLM()
    return _LLM


async def acall(role, messages, json_schema=None, max_tokens=None, model=None):
    return await get_llm().acall(role, messages, json_schema, max_tokens, model)


# ============================================================================ fake LLM (smoke test only)
def _fake(role, messages, json_schema):
    """Deterministic stand-in with the right JSON shape for every role (README step 1). Never used for results."""
    import re as _re
    h = int(hashlib.md5(json.dumps(messages, sort_keys=True).encode()).hexdigest(), 16)
    st = CallStats(1, 100, 20, 1.0)
    if role == "extractor" and "### Message 1" in messages[-1]["content"]:
        parts = messages[-1]["content"].split("### Message ")[1:]
        facts = []
        for k, part in enumerate(parts, 1):
            for f in _fake("extractor", [{"role": "user", "content": part}], None)[0]["facts"]:
                facts.append(dict(f, message=k))
        return {"facts": facts}, st
    if role == "labeler" and "### Fact 1" in messages[-1]["content"]:
        labels = []
        for k, part in enumerate(messages[-1]["content"].split("### Fact ")[1:], 1):
            if (h + k) % 9:                   # some labels missing, as a real model might do
                labels.append(dict(_fake("labeler", [{"role": "user", "content": part}], None)[0], id=k))
        return {"labels": labels}, st
    if role == "extractor":
        msg = messages[-1]["content"].split("User message:")[-1].strip()
        sents = [x.strip() for x in _re.split(r"(?<=[.!?])\s+", msg) if len(x.split()) >= 4]
        facts = []
        for x in sents[:3]:
            if _re.search(r"\b(I|my|I'm|I've)\b", x):
                fam = ["purchase", "likes", "count", "visit", "residence", "owns"][h % 6]
                facts.append({"subject": "user", "family": fam,
                              "slot_key": "_".join(_re.findall(r"[a-z]+", x.lower())[:3]),
                              "object": " ".join(x.split()[:6]), "object_type": "other",
                              "assertion_type": "fact", "time_expression": ["", "yesterday", "last year"][h % 3],
                              "statement": "The user said " + x[:80], "quote": x[:120]})
        return {"facts": facts}, st
    if role == "controller":
        return {"op": "NOOP", "target_fact_id": None, "reason": "fake"}, st
    if role == "labeler":
        msg = messages[-1]["content"]
        fam = _re.search(r"Draft family: (\S+)", msg).group(1)
        sl = _re.search(r"Draft slot key: (\S+)", msg).group(1)
        draft = _re.search(r"Draft statement: (.*)", msg).group(1)
        from chronofact.relations import FAMILIES
        kind = FAMILIES.get(fam, "state")
        return {"kind": kind, "family": fam if h % 4 else "other_" + kind, "slot_key": sl,
                "statement": draft + (" in detail" if h % 2 else "")}, st
    if role == "judge":
        if json_schema and "yes" in json_schema.get("properties", {}):       # W2 rule (c) update check
            return {"reason": "fake", "yes": h % 3 == 0}, st
        if json_schema and "stale" in json_schema.get("properties", {}):
            return {"reason": "fake", "stale": bool(h % 2)}, st
        return {"reason": "fake", "label": "CORRECT" if h % 3 else "WRONG"}, st
    text = messages[-1]["content"]
    if "FINAL:" in text:
        return "Looking at the memory.\nFINAL: " + ("NOT IN MEMORY" if h % 5 == 0 else f"answer {h % 7}"), st
    return ("NOT IN MEMORY" if h % 5 == 0 else f"answer {h % 7}"), st
