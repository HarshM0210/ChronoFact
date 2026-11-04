"""ChronoFact demo server: the real pipeline behind a web UI (demo/static).

  Live memory   chat with an assistant whose memory is ChronoFact; every message runs the real write path
                (chronofact.writer, Llama-3.1-8B via the local vLLM server, Contriever, BART-MNLI) and its stages are
                streamed to the page as they happen (Server-Sent Events)
  Ask           one question, the same token budget: ChronoFact (MODEL5) and RAG over raw transcripts (RAG_plain),
                exactly as in the benchmark (chronofact.reader + answer). The page shows these two systems; the
                README reports the third benchmark system (RAG_harness) as well.
  Forget        deletes a fact and the messages it came from (memory and transcript index)
  Benchmark     the finished LongMemEval run (results/): every history, context, answer and verdict, read-only
  Results       the report tables of results/final

Run: scripts/serve_demo.sh  (127.0.0.1:8090; open through an SSH tunnel)
"""
import asyncio
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(ROOT, "demo_data")
os.makedirs(DEMO, exist_ok=True)
os.environ.setdefault("CF_LLM_CACHE", os.path.join(DEMO, "llm_cache.sqlite"))   # the benchmark cache stays untouched
sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from chronofact import pipeline as P  # noqa: E402
from chronofact.answer import answer, is_abstention  # noqa: E402
from chronofact.config import CFG, load_step, rpath  # noqa: E402
from chronofact.interface import Turn  # noqa: E402
from chronofact.llm import acall  # noqa: E402
from chronofact.reader import Retriever  # noqa: E402
from chronofact.store import Store  # noqa: E402
from chronofact.timeparse import question_mode  # noqa: E402
from chronofact.writer import Writer  # noqa: E402

TAU = P.taus()["A"]               # the gate tau the benchmark chose (fold A)
SYSTEMS = ["MODEL5", "RAG_plain"]                  # the systems the page shows
CHAT_SYSTEM = ("You are a friendly personal assistant with long-term memory. Use the memory below only when it is "
               "relevant, never invent facts about the user, and reply in one to three short sentences. Reply to the "
               "user's latest message only; never repeat or restate one of your earlier replies.")

app = FastAPI()
app.mount("/static", StaticFiles(directory=os.path.join(ROOT, "demo", "static")), name="static")


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


def day(s):
    return datetime.strptime(s[:10], "%Y-%m-%d")


# ============================================================================ live memory
class Live:
    UID = "demo_user"

    def __init__(self):
        self.path = os.path.join(DEMO, "live.sqlite")
        self.lock = asyncio.Lock()
        self.open()

    def open(self):
        self.store = Store(self.path)
        self.writer = Writer(self.store)

    def reset(self):
        self.store.close()
        for ext in ("", "-wal", "-shm"):
            if os.path.exists(self.path + ext):
                os.remove(self.path + ext)
        self.open()

    def next_ts(self, date):
        """(session id, next turn index, timestamp) for a message on `date` (one session per day)"""
        sid = f"session_{date}"
        n = self.store.db.execute("SELECT COUNT(*) FROM episodes WHERE user_id=? AND session_id=?",
                                  (self.UID, sid)).fetchone()[0]
        return sid, n, day(date).replace(hour=10) + timedelta(seconds=30 * n)

    def memory(self):
        st, uid = self.store, self.UID
        ents = {k: v["name"] for k, v in st.entity_rows(uid).items()}
        rich = st.rich_map()
        eps = {r[0]: {"id": r[0], "text": r[1], "ts": r[2], "speaker": r[3], "session": r[4]} for r in st.db.execute(
            "SELECT id, text, ts, speaker, session_id FROM episodes WHERE user_id=? ORDER BY id", (uid,))}
        facts = []
        for f in st.facts(uid, statuses=("active", "superseded", "rejected", "forgotten")):
            val = ents.get(f["obj_id"], "") if f["obj_id"] is not None else (f["obj_literal"] or "")
            if f["status"] == "rejected" and " | " in val:
                val = val.split(" | ", 1)[1]
            facts.append({k: f[k] for k in ("id", "status", "kind", "family", "slot_key", "statement", "valid_from",
                                            "valid_to", "recorded_at", "quote", "superseded_by", "conflict_with",
                                            "confidence", "assertion_type", "date_precision")}
                         | {"value": val, "rich": rich.get(f["id"]), "subject": ents.get(f["subj_id"], "user"),
                            "sources": [eps[e] for e in f["source_episode_ids"] if e in eps]})
        ops = [{"id": r[0], "ts": r[1], "op": r[2], "fact_id": r[3], "reason": r[4]} for r in st.db.execute(
            "SELECT id, ts, op, fact_id, reason FROM ops_log WHERE user_id=? ORDER BY id DESC LIMIT 400", (uid,))]
        counts = {s: sum(1 for f in facts if f["status"] == s) for s in ("active", "superseded", "rejected",
                                                                          "forgotten")}
        msgs = [e for e in eps.values()]
        opc = dict(st.db.execute("SELECT op, COUNT(*) FROM ops_log WHERE user_id=? GROUP BY op", (uid,)).fetchall())
        return {"facts": facts, "ops": ops, "counts": counts, "op_counts": opc, "messages": msgs,
                "entities": len(ents)}


LIVE = Live()


def session_history(store, sid, n, k=6):
    """the last k turns of the current session as chat messages (the conversation so far, oldest first). Without them
    the model only saw its own earlier reply inside the memory excerpts and repeated it ("Thanks" -> same reply)."""
    rows = store.db.execute("SELECT speaker, text FROM episodes WHERE user_id=? AND session_id=? AND turn_idx<? AND "
                            "text != '' ORDER BY turn_idx DESC LIMIT ?", (Live.UID, sid, n, k)).fetchall()
    msgs = [{"role": "user" if s == "user" else "assistant", "content": t} for s, t in reversed(rows)]
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    return msgs


async def chat_reply(text, date, context, history=()):
    msgs = [{"role": "system", "content": CHAT_SYSTEM}, *history,
            {"role": "user", "content": f"Today is {day(date):%A, %B %d, %Y}.\nMemory:\n{context or '(empty)'}\n\n"
                                        f"User: {text}"}]
    out, st = await acall("answerer", msgs, max_tokens=160)
    return out.strip(), st


@app.post("/api/chat")
async def chat(req: Request):
    body = await req.json()
    text, date = body["text"].strip(), body["date"]

    async def gen():
        async with LIVE.lock:
            tasks = []
            try:
                t0 = time.perf_counter()
                sid, n, ts = LIVE.next_ts(date)
                r = Retriever(LIVE.store, Live.UID).retrieve(text, ts, 1024, "MODEL5", tau=TAU, visible_until=ts)
                yield sse("context", {"fact_lines": r["fact_lines"], "turns": len(r["turns"]), "tokens": r["tokens"],
                                      "text": r["text"]})
                reply, st = await chat_reply(text, date, r["text"], session_history(LIVE.store, sid, n))
                yield sse("reply", {"text": reply, "ms": round((time.perf_counter() - t0) * 1000)})
                q = asyncio.Queue()
                LIVE.writer.on_stage = lambda s, d: q.put_nowait((s, d))
                t1 = time.perf_counter()
                turns = [Turn(Live.UID, sid, n, "user", text, ts),
                         Turn(Live.UID, sid, n + 1, "assistant", reply, ts + timedelta(seconds=15))]
                task = asyncio.create_task(LIVE.writer.ingest_batch(turns))
                tasks.append(task)
                while True:
                    get = asyncio.create_task(q.get())
                    done, _ = await asyncio.wait({task, get}, return_when=asyncio.FIRST_COMPLETED)
                    if get in done:
                        s, d = get.result()
                        yield sse(s, d)
                        continue
                    get.cancel()
                    break
                while not q.empty():
                    s, d = q.get_nowait()
                    yield sse(s, d)
                stats = task.result()
                yield sse("memory", LIVE.memory())
                yield sse("done", {"write_ms": round((time.perf_counter() - t1) * 1000),
                                   "llm_calls": sum(x.llm_calls for x in stats)})
            except Exception as e:  # shown on the page; the server keeps running
                yield sse("error", {"message": f"{type(e).__name__}: {e}"})
            finally:
                LIVE.writer.on_stage = None
                for t in tasks:          # the page went away mid-message: finish the write before the next message
                    if not t.done():
                        try:
                            await asyncio.shield(t)
                        except BaseException:
                            pass
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/memory")
async def memory():
    return LIVE.memory()


@app.post("/api/reset")
async def reset():
    async with LIVE.lock:
        LIVE.reset()
    return {"ok": True}


async def run_systems(store, uid, question, as_of, budget, tau):
    """the benchmark read path for the systems the page shows, on one store (retrieval + answer)"""
    mode, ts, te = question_mode(question, as_of)
    rt = Retriever(store, uid)
    ctx = {}
    for s in SYSTEMS:
        t0 = time.perf_counter()
        r = rt.retrieve(question, as_of, budget, s, tau=tau if s == "MODEL5" else None, visible_until=as_of)
        ctx[s] = dict(r, ms=round((time.perf_counter() - t0) * 1000, 1))

    async def one(s):
        t0 = time.perf_counter()
        a, st, raw, err, rec = await answer(question, as_of, ctx[s]["text"], P.MODE[s])
        return s, {"answer": a, "raw": raw, "ms": round((time.perf_counter() - t0) * 1000), "abstained": is_abstention(a)}
    res = dict(await asyncio.gather(*[one(s) for s in SYSTEMS]))
    out = {}
    for s in SYSTEMS:
        c = ctx[s]
        out[s] = res[s] | {"context": c["text"], "tokens": c["tokens"], "fact_lines": c["fact_lines"],
                           "turns": len(c["turns"]), "retrieve_ms": c["ms"]}
    return {"mode": mode, "window": [str(ts) if ts else None, str(te) if te else None], "systems": out}


@app.post("/api/ask")
async def ask(req: Request):
    b = await req.json()
    as_of = day(b["date"]).replace(hour=23, minute=59)
    async with LIVE.lock:
        return await run_systems(LIVE.store, Live.UID, b["question"].strip(), as_of, int(b.get("budget", 256)), TAU)


@app.post("/api/forget")
async def forget(req: Request):
    """Deletes a fact, the messages it came from, every other fact taken from those messages, and the assistant's
    reply to each of those messages (a reply can repeat what the user said, so it would leak the deleted content)."""
    b = await req.json()
    async with LIVE.lock:
        try:
            gone, eps = LIVE.store.forget(Live.UID, int(b["fact_id"]), datetime.now())
        except KeyError:
            return JSONResponse({"error": "no such fact"}, status_code=404)
        return {"facts": gone, "messages": eps, "memory": LIVE.memory()}


@app.get("/api/scenario")
async def scenario():
    with open(os.path.join(ROOT, "demo", "scenario.json")) as f:
        return json.load(f)


# ============================================================================ benchmark explorer (read-only)
class Bench:
    def __init__(self):
        self.df = pd.read_parquet(rpath("final", "runs.parquet"))
        from chronofact.data import load_all
        self.qs = load_all()
        g = self.df.groupby(["question_id", "system"]).correct.mean().unstack("system")
        first = self.df.drop_duplicates("question_id").set_index("question_id")
        self.index = [{"qid": q, "type": "abstention" if first.loc[q, "is_abstention"] else first.loc[q, "question_type"],
                       "question": first.loc[q, "question"], "gold": first.loc[q, "gold"],
                       "acc": {s: round(float(g.loc[q, s]), 3) for s in SYSTEMS}} for q in g.index]

    def detail(self, qid, ck, budget):
        q = self.qs[qid]
        rows = self.df[(self.df.question_id == qid) & (self.df.checkpoint == ck) & (self.df.budget == budget)]
        out = {"qid": qid, "question": q.question, "gold": q.answer, "type": q.question_type,
               "question_date": str(q.question_date), "is_abstention": q.is_abstention, "systems": {}}
        for s in SYSTEMS:
            r = rows[rows.system == s].iloc[0]
            rec = P.read_gz(P.ctx_path(s, qid))
            c = next(c for c in rec["checkpoints"] if c["fraction"] == ck)
            cx = c["contexts"][str(budget)]
            out["systems"][s] = {"answer": r.answer, "raw": r.raw_answer, "correct": bool(r.correct),
                                 "judge": r.judge_reason, "rule": r.judge_rule, "context": cx["text"],
                                 "tokens": int(cx["tokens"]), "fact_lines": int(cx["fact_lines"]),
                                 "evidence_in_context": None if pd.isna(r.ev_any) else bool(r.ev_any)}
        ev = q.evidence_turns()
        out["evidence"] = [{"date": f"{s.date:%Y-%m-%d}", "role": t["role"], "text": t["content"][:1200]}
                           for s in q.sessions for i, t in enumerate(s.turns) if (s.session_id, i) in ev]
        m = P.read_json(P.meta_path("v5f", qid))
        i = P.FRACS.index(ck)
        out["history"] = {"sessions": len(q.sessions), "turns": len(q.turns()),
                          "checkpoint": m["checkpoints"][i]["n_sessions"],
                          "storage": m["checkpoints"][i]["storage"], "ops": m["checkpoints"][i]["ops"],
                          "w2": m["w2"][i]}
        d = sqlite3.connect(f"file:{os.path.join(P.snap_dir('v5f', qid), f'ck{i}.sqlite')}?mode=ro", uri=True)
        ents = dict(d.execute("SELECT id, canonical_name FROM entities"))
        out["facts"] = [{"id": r[0], "status": r[1], "kind": r[2], "family": r[3], "statement": r[4],
                         "valid_from": r[5], "valid_to": r[6], "value": ents.get(r[7], "") if r[7] else (r[8] or "")}
                        for r in d.execute("SELECT id, status, kind, family, statement, valid_from, valid_to, obj_id, "
                                           "obj_literal FROM facts WHERE status IN ('active','superseded') ORDER BY "
                                           "valid_from")]
        d.close()
        return out


BENCH = None


@app.get("/api/bench/list")
async def bench_list():
    return BENCH.index


@app.get("/api/bench/q/{qid}")
async def bench_q(qid: str, ck: float = 1.0, budget: int = 512):
    return BENCH.detail(qid, ck, budget)


# ============================================================================ results
def results():
    F = rpath("final")
    t = lambda n: pd.read_csv(os.path.join(F, "tables", n))  # noqa: E731
    keep = lambda d, col="system": d[d[col].isin(SYSTEMS)]  # noqa: E731
    out = {"accuracy": keep(t("accuracy.csv")).to_dict("records"),
           "paired": t("paired_model5_minus.csv").query("vs == 'RAG_plain'").to_dict("records"),
           "by_type": t("accuracy_by_type.csv").rename(columns={"qtype": "type"})[["type", "n_questions"] + SYSTEMS]
           .to_dict("records"),
           "by_checkpoint": keep(t("accuracy_by_checkpoint.csv")).to_dict("records"),
           "behaviour": keep(t("behaviour.csv")).to_dict("records"),
           "read_cost": keep(t("read_cost.csv")).to_dict("records"),
           "summary": json.load(open(os.path.join(F, "summary.json"))),
           "audit": json.load(open(os.path.join(F, "integrity_audit.json"))),
           "judge_pilot": load_step("judge_pilot"), "ingest": load_step("ingest"), "w2": load_step("w2"),
           "relevance": {h: load_step("relevance")[h]["chosen_tau"] for h in ("A", "B")},
           "models": CFG["llm"]["models"], "embedder": load_step("embedder")["chosen"], "nli": CFG["nli"]["model"]}
    return json.loads(json.dumps(out, default=lambda o: o.item() if hasattr(o, "item") else str(o)).replace("NaN", "null"))


RESULTS = None


@app.get("/api/results")
async def get_results():
    return RESULTS


@app.get("/")
async def index():
    return FileResponse(os.path.join(ROOT, "demo", "static", "index.html"))


@app.on_event("startup")
async def startup():
    global BENCH, RESULTS
    RESULTS = results()
    BENCH = await asyncio.to_thread(Bench)
    print("[demo] ready", flush=True)
