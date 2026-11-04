"""ChronoFact pipeline units (README section 12). Every unit writes its output atomically (*.tmp then rename)
and is skipped when its output exists, so a crash costs only the units in flight. LLM calls, embeddings and NLI
results are cached on disk, so a redone unit is cheap.

results/
  snapshots/v5/<qid>/ck0..4.sqlite + meta.json     ingest (meta.json written last = history done)
  snapshots/v5f/<qid>/ck0..4.sqlite + meta.json    W2 copies
  relevance/<qid>.json                              step 7
  contexts/<system>/<qid>.json.gz                   step 8 (5 checkpoints x 4 budgets)
  rows/<system>/<qid>.json.gz                       step 9 (answers)
  judged/<system>/<qid>.json.gz                     step 11 (verdicts)
CPU worker processes (CUDA_VISIBLE_DEVICES="", 2 torch threads each) do ingest, W2, relevance and retrieval;
answers and verdicts are LLM calls made from one asyncio process.
"""
import asyncio
import gzip
import json
import multiprocessing as mp
import os
import queue
import shutil
import time
import traceback

import numpy as np

from chronofact.config import CFG, rpath

SYSTEMS = ["MODEL5", "RAG_plain"]          # ChronoFact and RAG over raw transcripts (the baseline)
MODE = {"MODEL5": "reason", "RAG_plain": "short"}
BUDGETS = CFG["eval"]["budgets"]
FRACS = CFG["eval"]["checkpoint_fractions"]


# ============================================================================ files
def snap_dir(wv, qid):
    return rpath("snapshots", wv, qid)


def meta_path(wv, qid):
    return os.path.join(snap_dir(wv, qid), "meta.json")


def ctx_path(system, qid):
    return rpath("contexts", system, f"{qid}.json.gz")


def rows_path(system, qid):
    return rpath("rows", system, f"{qid}.json.gz")


def judged_path(system, qid):
    return rpath("judged", system, f"{qid}.json.gz")


def write_json(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = f"{p}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, p)


def read_json(p):
    with open(p) as f:
        return json.load(f)


def write_gz(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = f"{p}.tmp{os.getpid()}"
    with gzip.open(tmp, "wt") as f:
        json.dump(obj, f, default=str)
    os.replace(tmp, p)


def read_gz(p):
    with gzip.open(p, "rt") as f:
        return json.load(f)


def clean_tmp():
    """orphan temp files of a crash (every real output is written atomically)"""
    n = 0
    for root, _, files in os.walk(rpath()):
        for f in files:
            if ".tmp" in f or f.endswith(".part"):
                os.remove(os.path.join(root, f))
                n += 1
    return n


def write_settings():
    """the write thresholds + models a store was built with; a snapshot built differently is redone"""
    from chronofact.config import embedder_name, role_model
    return {"thresholds": {k: CFG["write"][k] for k in sorted(CFG["write"])}, "embedder": embedder_name(),
            "llm_models": {r: role_model(r)["name"] for r in sorted(CFG["llm"]["roles"])},
            "nli": CFG["nli"]["model"]}


def ingest_done(qid, wv="v5"):
    p = meta_path(wv, qid)
    return os.path.exists(p) and read_json(p).get("settings") == write_settings()


# ============================================================================ step 5: ingest one history
def ingest_plan(q):
    """[(checkpoint index, fraction, n_sessions, n_turns_after, [batch of Turn, ...])] in ingest order: each
    checkpoint's new turns, in batches of write.batch_turns. Used by ingest_one and by step batch_check."""
    from chronofact.interface import Turn
    uid = q.question_id
    turns = q.turns()
    sess_end, n = [], 0
    for s in q.sessions:
        n += len(s.turns)
        sess_end.append(n)
    out, done = [], 0
    for i, (frac, n_sess) in enumerate(q.checkpoints()):
        upto = sess_end[n_sess - 1] if n_sess > 0 else 0
        seg = [Turn(uid, t["session_id"], t["turn_idx"], t["speaker"], t["text"], t["ts"]) for t in turns[done:upto]]
        bt = CFG["write"]["batch_turns"]
        out.append((i, frac, n_sess, upto, [seg[j:j + bt] for j in range(0, len(seg), bt)]))
        done = upto
    return out


async def ingest_one(q):
    from chronofact.interface import CallStats
    from chronofact.llm import PHASE, get_llm
    from chronofact.store import Store
    from chronofact.writer import Writer
    uid = q.question_id
    if ingest_done(uid):
        return "skip"
    sd = snap_dir("v5", uid)
    if os.path.exists(sd):
        shutil.rmtree(sd)                          # partial or stale (other thresholds): redo from scratch
    db = rpath("stores", f"{uid}.sqlite")
    for ext in ("", "-wal", "-shm"):
        if os.path.exists(db + ext):
            os.remove(db + ext)
    store = Store(db)
    w = Writer(store)
    cum, lat, cks = CallStats(), [], []
    t_start = time.time()
    PHASE.set("write")
    for i, frac, n_sess, done, batches in ingest_plan(q):
        for batch in batches:
            for st in await w.ingest_batch(batch):
                cum.add(st)
                lat.append(st.latency_ms)
        store.snapshot(os.path.join(sd, f"ck{i}.sqlite"))
        ops = dict(store.db.execute("SELECT op, COUNT(*) FROM ops_log WHERE user_id=? GROUP BY op", (uid,)).fetchall())
        cks.append({"fraction": frac, "n_sessions": n_sess, "n_turns": done,
                    "write_cum": {"llm_calls": cum.llm_calls, "tokens_in": cum.tokens_in, "tokens_out": cum.tokens_out},
                    "role_stats": json.loads(json.dumps(w.role_stats)), "storage": store.stats(uid), "ops": ops})
    arr = np.array(lat) if lat else np.zeros(1)
    meta = {"question_id": uid, "wv": "v5", "settings": write_settings(), "checkpoints": cks,
            "wall_seconds": time.time() - t_start, "write_turn_latency_p50": float(np.percentile(arr, 50)),
            "write_turn_latency_p95": float(np.percentile(arr, 95))}
    store.close()
    get_llm().flush()
    write_json(meta_path("v5", uid), meta)        # last: the history is done
    for ext in ("", "-wal", "-shm"):
        if os.path.exists(db + ext):
            os.remove(db + ext)
    return "done"


# ============================================================================ step 6: W2 copies
def w2_settings():
    """what a W2 copy was built with; a copy built differently is redone"""
    from chronofact import prompts as PR
    from chronofact.config import role_model
    return {"w2": CFG["w2"], "rule_c_model": role_model("judge")["name"],
            "rule_c_prompt_sha": __import__("hashlib").sha256(PR.W2_UPDATE_TEMPLATE.encode()).hexdigest()[:16]}


async def judge_updates(pairs):
    """The judge-role LLM on rule (c) pairs -> ({(older id, newer id)} judged updates, number of failed verdicts).
    A failed verdict counts as no update: W2 never hides a fact without a valid yes."""
    from chronofact import prompts as PR
    from chronofact.llm import PHASE, LLMJSONError, acall
    PHASE.set("write")

    async def one(o, n):
        p = PR.W2_UPDATE_TEMPLATE.format(d1=(o["recorded_at"] or "")[:10], a=o["statement"],
                                         d2=(n["recorded_at"] or "")[:10], b=n["statement"])
        try:
            out, _ = await acall("judge", [{"role": "user", "content": p}], PR.W2_UPDATE_SCHEMA)
        except LLMJSONError:
            return None
        return out.get("yes") if isinstance(out, dict) and isinstance(out.get("yes"), bool) else None
    res = await asyncio.gather(*[one(o, n) for o, n in pairs])
    return {(o["id"], n["id"]) for (o, n), y in zip(pairs, res) if y}, sum(1 for y in res if y is None)


async def w2_one(q):
    from chronofact.models import embedder, nli
    from chronofact.reconcile import reconcile, same_slot_pairs
    qid = q.question_id
    src_meta = read_json(meta_path("v5", qid))
    p = meta_path("v5f", qid)
    if os.path.exists(p):
        m = read_json(p)
        if m.get("settings") == src_meta["settings"] and m.get("w2_settings") == w2_settings():
            return "skip"
    if os.path.exists(snap_dir("v5f", qid)):
        shutil.rmtree(snap_dir("v5f", qid))
    emb = embedder()
    stats = []
    for i in range(len(src_meta["checkpoints"])):
        src = os.path.join(snap_dir("v5", qid), f"ck{i}.sqlite")
        pairs = same_slot_pairs(src) if CFG["w2"].get("rule_c", True) else []
        upd, fails = await judge_updates(pairs)
        st = reconcile(src, os.path.join(snap_dir("v5f", qid), f"ck{i}.sqlite"), emb.encode, nli().contradict, upd)
        stats.append(dict(st, rule_c_pairs=len(pairs), rule_c_judged_updates=len(upd), rule_c_failed_verdicts=fails))
    write_json(p, dict(src_meta, wv="v5f", w2=stats, w2_settings=w2_settings()))
    return "done"


# ============================================================================ step 7: relevance (gate tau)
async def relevance_one(q):
    from chronofact.reader import Retriever
    from chronofact.store import Store
    p = rpath("relevance", f"{q.question_id}.json")
    if os.path.exists(p):
        return "skip"
    meta = read_json(meta_path("v5f", q.question_id))
    i = len(meta["checkpoints"]) - 1                        # final checkpoint
    st = Store(os.path.join(snap_dir("v5f", q.question_id), f"ck{i}.sqlite"), readonly=True)
    ev = q.evidence_turns()
    ev_ids = {r[0] for r in st.db.execute("SELECT id, session_id, turn_idx FROM episodes") if (r[1], r[2]) in ev}
    gold = {r[0] for r in st.db.execute("SELECT id, source_episode_ids FROM facts WHERE status IN ('active','superseded')")
            if ev_ids & set(json.loads(r[1] or "[]"))}
    rt = Retriever(st, q.question_id)
    out = {"question_id": q.question_id, "evidence_facts": len(gold), "tau": {}}
    for tau in CFG["read"]["tau_grid"]:
        r = rt.retrieve(q.question, q.question_date, 1024, "MODEL5", tau=tau, visible_until=q.visible_until())
        out["tau"][f"{tau:.2f}"] = {"past_gate": bool(gold & set(r["gate"])), "fact_lines": r["fact_lines"]}
    st.close()
    write_json(p, out)
    return "done"


# ============================================================================ step 8: retrieve
def taus():
    """fold A = tau chosen on half A, scores half B; fold B = tau chosen on half B, scores half A"""
    from chronofact.config import load_step
    r = load_step("relevance")
    return {"A": r["A"]["chosen_tau"], "B": r["B"]["chosen_tau"]}


def fold_for(qid):
    from chronofact.data import half_of
    return "A" if half_of(qid) == "B" else "B"      # the fold whose tau was chosen on the OTHER half


async def retrieve_one(q, tau_by_fold):
    from chronofact.reader import Retriever
    from chronofact.store import Store
    qid = q.question_id
    if all(os.path.exists(ctx_path(s, qid)) for s in SYSTEMS):
        return "skip"
    meta = read_json(meta_path("v5f", qid))
    fold = fold_for(qid)
    tau = tau_by_fold[fold]
    vis = q.visible_until()
    outs = {s: {"question_id": qid, "system": s, "fold": fold, "tau": tau if s == "MODEL5" else None,
                "write_turn_latency_p50": meta["write_turn_latency_p50"],
                "write_turn_latency_p95": meta["write_turn_latency_p95"], "checkpoints": []} for s in SYSTEMS}
    for i, ck in enumerate(meta["checkpoints"]):
        st = Store(os.path.join(snap_dir("v5f", qid), f"ck{i}.sqlite"), readonly=True)
        rt = Retriever(st, qid)
        for s in SYSTEMS:
            ctxs = {}
            for b in BUDGETS:
                t0 = time.perf_counter()
                r = rt.retrieve(q.question, q.question_date, b, s, tau=tau if s == "MODEL5" else None, visible_until=vis)
                r["retrieve_latency_ms"] = (time.perf_counter() - t0) * 1000
                r.pop("gate", None)
                ctxs[str(b)] = r
            outs[s]["checkpoints"].append({k: ck[k] for k in ck} | {"w2": meta["w2"][i], "contexts": ctxs})
        st.close()
    for s in SYSTEMS:
        write_gz(ctx_path(s, qid), outs[s])
    return "done"


def determinism(qids, n, taus):
    """re-run retrieval for n random (system, question, checkpoint, budget) cells; byte-identical?"""
    import random
    from chronofact.data import load_all
    from chronofact.reader import Retriever
    from chronofact.store import Store
    qs = load_all()
    rng = random.Random(7)
    cells = [(rng.choice(SYSTEMS), rng.choice(qids), rng.randrange(5), rng.choice(BUDGETS)) for _ in range(n)]
    same, bad = 0, []
    for s, qid, i, b in cells:
        rec = read_gz(ctx_path(s, qid))
        ck = rec["checkpoints"][i]
        st = Store(os.path.join(snap_dir("v5f", qid), f"ck{i}.sqlite"), readonly=True)
        r = Retriever(st, qid).retrieve(qs[qid].question, qs[qid].question_date, b, s,
                                        tau=rec["tau"] if s == "MODEL5" else None,
                                        visible_until=qs[qid].visible_until())
        st.close()
        if r["text"] == ck["contexts"][str(b)]["text"]:
            same += 1
        else:
            bad.append([s, qid, i, b])
    return {"same": same, "total": len(cells), "differ": bad[:10]}


# ============================================================================ worker pool
def _worker(kind, jobs, payload, nconc, done_q, env):
    os.environ.update(env)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.nice(5)                  # CPU workers yield to the main process and the LLM server
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    nt = str(CFG["resources"]["torch_threads_per_worker"])
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = nt
    import torch
    torch.set_num_threads(int(nt))
    from chronofact.data import load_all
    from chronofact.llm import RUN_TAG
    RUN_TAG.set(payload.get("run_tag", kind))
    qs = load_all()

    async def runner():
        async def one():
            while True:
                try:
                    qid = jobs.get_nowait()
                except queue.Empty:
                    return
                t0 = time.time()
                try:
                    q = qs[qid]
                    if kind == "ingest":
                        r = await ingest_one(q)
                    elif kind == "w2":
                        r = await w2_one(q)
                    elif kind == "relevance":
                        r = await relevance_one(q)
                    elif kind == "retrieve":
                        r = await retrieve_one(q, payload["taus"])
                    else:
                        raise ValueError(kind)
                    done_q.put((qid, r, time.time() - t0, None))
                except Exception as e:
                    done_q.put((qid, "error", time.time() - t0, f"{e}\n{traceback.format_exc()}"))
        await asyncio.gather(*[one() for _ in range(nconc)])

    asyncio.run(runner())


def run_pool(kind, qids, workers, nconc=1, payload=None, env=None):
    """Runs `kind` over qids in CPU worker processes. Returns the number of failed units."""
    if not qids:
        print(f"[{kind}] nothing to do", flush=True)
        return 0
    ctx = mp.get_context("spawn")
    jobs, done_q = ctx.Queue(), ctx.Queue()
    for q in qids:
        jobs.put(q)
    time.sleep(0.5)
    procs = [ctx.Process(target=_worker, args=(kind, jobs, payload or {}, nconc, done_q, env or {}))
             for _ in range(min(workers, len(qids)))]
    for p in procs:
        p.start()
    n = errors = 0
    t0 = time.time()
    while n < len(qids):
        try:
            qid, r, dt, err = done_q.get(timeout=60)
        except queue.Empty:
            if not any(p.is_alive() for p in procs):
                print(f"[{kind}] workers exited early", flush=True)
                break
            continue
        n += 1
        if err:
            errors += 1
            print(f"[{kind}] ERROR {qid}: {err}", flush=True)
        if n % 10 == 0 or n == len(qids) or err:
            el = time.time() - t0
            print(f"[{kind}] {n}/{len(qids)} | {el / 60:.1f} min, eta {el / n * (len(qids) - n) / 60:.1f} min",
                  flush=True)
    for p in procs:
        p.join()
    return errors + (len(qids) - n)


# ============================================================================ step 9: answers
def _row_base(q, system, ck, budget, c, fold, tau):
    rs = ck.get("role_stats", {})
    gold_turns = q.evidence_turns()
    got = {(t["session_id"], t["turn_idx"]) for t in c["turns"]}
    return {"question_id": q.question_id, "question_type": q.question_type, "is_abstention": q.is_abstention,
            "question": q.question, "gold": q.answer, "system": system, "mode": MODE[system], "fold": fold,
            "tau": tau, "budget": int(budget), "checkpoint": ck["fraction"], "n_sessions": ck["n_sessions"],
            "n_turns": ck["n_turns"], "context_tokens": c["tokens"], "fact_lines": c["fact_lines"],
            "time_mode": c["mode"], "retrieve_latency_ms": c["retrieve_latency_ms"],
            "write_llm_calls": ck["write_cum"]["llm_calls"], "write_tokens_in": ck["write_cum"]["tokens_in"],
            "write_tokens_out": ck["write_cum"]["tokens_out"],
            "write_role_stats": json.dumps(rs, sort_keys=True),
            "storage_episodes": ck["storage"]["episodes"], "storage_entities": ck["storage"]["entities"],
            "facts_active": ck["storage"]["facts_active"], "facts_superseded": ck["storage"]["facts_superseded"],
            "facts_rejected": ck["storage"]["facts_rejected"],
            "ev_any": bool(got & gold_turns) if gold_turns else None,
            "ev_all": bool(gold_turns <= got) if gold_turns else None}


def answer_model(qid):
    """The model key that answers a question. config answer.split = {model, share} sends that share of QUESTIONS (by
    sha256 of the id) to another model (profile spark: the 1,200 req/min tier, to fit a day); every system and grid
    cell of a question uses the same model, so the paired comparisons stay within one model."""
    import hashlib
    sp = CFG["answer"].get("split")
    if sp and int(hashlib.sha256(qid.encode()).hexdigest()[:8], 16) / 16 ** 8 < float(sp["share"]):
        return sp["model"]
    return CFG["llm"]["roles"]["answerer"]["model"]


async def _answer_file(q, system, sem):
    from chronofact.answer import answer, has_final
    from chronofact.llm import PHASE
    p = rows_path(system, q.question_id)
    if os.path.exists(p):
        return 0
    rec = read_gz(ctx_path(system, q.question_id))
    rows = {}
    mk = answer_model(q.question_id)

    async def one(ck, b):
        c = ck["contexts"][str(b)]
        async with sem:
            PHASE.set("read")
            ans, st, raw, err, recovered = await answer(q.question, q.question_date, c["text"], MODE[system], model=mk)
        r = _row_base(q, system, ck, b, c, rec["fold"], rec["tau"])
        r.update(answer=ans, raw_answer=raw, answer_error=err,
                 final_recovered=recovered if MODE[system] == "reason" else None,
                 answer_model=CFG["llm"]["models"][mk]["name"],
                 final_found=has_final(raw) if MODE[system] == "reason" else None,
                 answer_tokens_in=st.tokens_in, answer_tokens_out=st.tokens_out, answer_latency_ms=st.latency_ms,
                 answer_calls=st.llm_calls)
        rows[f"{ck['fraction']}|{b}"] = r

    await asyncio.gather(*[one(ck, b) for ck in rec["checkpoints"] for b in BUDGETS])
    write_gz(p, rows)
    return len(rows)


def answer_all(qids):
    from chronofact.data import load_all
    from chronofact.llm import RUN_TAG, get_llm
    RUN_TAG.set("answer")
    qs = load_all()
    jobs = [(s, q) for q in qids for s in SYSTEMS if not os.path.exists(rows_path(s, q))]
    print(f"[answer] {len(jobs)} files of {len(qids) * len(SYSTEMS)} to answer", flush=True)
    errors = 0

    async def main():
        sem = asyncio.Semaphore(256)          # the LLM layer caps in-flight calls and paces them
        fsem = asyncio.Semaphore(48)
        done, t0 = 0, time.time()

        async def wrap(s, q):
            nonlocal done, errors
            async with fsem:
                try:
                    await _answer_file(qs[q], s, sem)
                except Exception as e:
                    errors += 1
                    print(f"[answer] ERROR {s} {q}: {e}\n{traceback.format_exc()}", flush=True)
                    return
            done += 1
            if done % 50 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"[answer] {done}/{len(jobs)} | {el / 60:.1f} min, eta {el / done * (len(jobs) - done) / 60:.1f} min",
                      flush=True)
        await asyncio.gather(*[wrap(s, q) for s, q in jobs])
        get_llm().flush()
    asyncio.run(main())
    return errors


# ============================================================================ steps 10-11: judge
def evidence_lines(q):
    return [f"[{s.date:%Y-%m-%d}] {t['content'][:1500]}" for s in q.sessions if s.is_evidence
            for t in s.turns if t["role"] == "user" and t.get("has_answer")]


async def judge_row(q, r):
    """README 10: rule judge first, then the LLM judge; stale check for knowledge-update. Returns the verdict fields."""
    from chronofact.answer import is_abstention, judge, stale_check
    from chronofact.llm import PHASE, LLMJSONError
    PHASE.set("eval")
    abst = is_abstention(r["answer"])
    if abst and not q.is_abstention:
        ok, reason, rule = False, "abstained on an answerable question", "rule"
    elif abst and q.is_abstention:
        # README judge rule: "If the gold says the information is not available, CORRECT only if the model answer says
        # it is not in memory" -- decided by rule: the Llama 8B judge labeled such rows WRONG while its own reason
        # said they were correct (judge pilot hand check, 5 of 50 sampled verdicts)
        ok, reason, rule = True, "abstained on an abstention question", "rule"
    else:
        try:
            ok, reason, _ = await judge(q.question, q.answer, r["answer"])
            rule = "llm"
        except LLMJSONError as e:      # still no valid verdict after the README's extra call: counted, never dropped
            ok, reason, rule = False, f"judge returned no valid verdict: {str(e)[:200]}", "judge_error"
    stale = None
    if q.question_type == "knowledge-update" and not q.is_abstention:
        stale = False
        if not ok and not abst and rule == "llm":
            try:
                stale, _, _ = await stale_check(q.question, q.answer, r["answer"], evidence_lines(q))
            except LLMJSONError:
                stale = None            # reported as missing; the audit counts these
    return {"correct": bool(ok), "judge_reason": reason, "judge_rule": rule, "abstained": abst, "stale": stale}


async def _judge_file(q, system):
    p = judged_path(system, q.question_id)
    if os.path.exists(p):
        return 0
    rows = read_gz(rows_path(system, q.question_id))
    keys = list(rows)
    res = await asyncio.gather(*[judge_row(q, rows[k]) for k in keys])
    out = {k: dict(rows[k], **v) for k, v in zip(keys, res)}
    write_gz(p, out)
    return len(out)


def judge_all(qids):
    from chronofact.data import load_all
    from chronofact.llm import RUN_TAG, SpendCapError, get_llm
    RUN_TAG.set("judge")
    qs = load_all()
    jobs = [(s, q) for q in qids for s in SYSTEMS if not os.path.exists(judged_path(s, q))]
    print(f"[judge] {len(jobs)} files of {len(qids) * len(SYSTEMS)} to judge; spent so far "
          f"${get_llm().spent():.2f}", flush=True)
    state = {"errors": 0, "cap": None}

    async def main():
        fsem = asyncio.Semaphore(24)
        done, t0 = 0, time.time()

        async def wrap(s, q):
            nonlocal done
            if state["cap"]:
                return
            async with fsem:
                if state["cap"]:
                    return
                try:
                    await _judge_file(qs[q], s)
                except SpendCapError as e:
                    state["cap"] = str(e)
                    return
                except Exception as e:
                    state["errors"] += 1
                    print(f"[judge] ERROR {s} {q}: {e}\n{traceback.format_exc()}", flush=True)
                    return
            done += 1
            if done % 50 == 0 or done == len(jobs):
                el = time.time() - t0
                print(f"[judge] {done}/{len(jobs)} | {el / 60:.1f} min, eta {el / done * (len(jobs) - done) / 60:.1f} min"
                      f" | spent ${get_llm().spent():.2f}", flush=True)
        await asyncio.gather(*[wrap(s, q) for s, q in jobs])
        get_llm().flush()
    asyncio.run(main())
    if state["cap"]:
        raise SpendCapError(state["cap"])
    return state["errors"]
