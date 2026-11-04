"""ChronoFact runner: README section 12, every step in order. Resumable.

  python scripts/chronofact.py all          every step in order; finished steps (<results>/steps/<step>.json) skipped
  python scripts/chronofact.py <step>       one step

Steps: setup smoke serve_check [batch_check] pilot [cost_gate] ingest w2 relevance retrieve answer judge_pilot judge
report audit. The bracketed ones are on in the spark profile (config `steps`): batch_check measures whether batched
extraction loses facts and picks write.extract_chunk; cost_gate stops once to compare the Meta console's bill.
Steps that stop for a human (exit code 3), then continue with `all`:
  serve_check  read <results>/serve_check.json        -> python scripts/chronofact.py serve_ok "<notes>"
  pilot        read <results>/pilot_review.json       -> python scripts/chronofact.py pilot_ok "<notes>"
  cost_gate    compare the console's bill              -> python scripts/chronofact.py cost_ok <billed_usd> "<notes>"
  judge_pilot  hand check <results>/judge_pilot_sample.json -> python scripts/chronofact.py handcheck <wrong_of_50> "<notes>"
Nothing between retrieve and judge may change once they ran (README 17).
"""
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from chronofact import pipeline as P  # noqa: E402
from chronofact.config import CACHE_DIR, CFG, PROFILE, RESULTS, ROOT, STEPS, load_step, path, rpath, step_file  # noqa: E402

# Model4's per-history counts on the same 30 pilot histories. SKIP and episodes do not depend on any model, so they
# must match exactly (a check of loading, checkpoints and the salience filter); the rest is reported for comparison.
REF = {"SKIP": 58.3, "episodes": 488, "MERGE": 15.1, "NOOP": 6.8, "REJECT": 97.7, "rich": 0.69, "ADD": 269.9,
       "SUPERSEDE": 1.5}


def done(step):
    return os.path.exists(step_file(step))


def save(step, obj):
    P.write_json(step_file(step), dict(obj, finished=time.strftime("%Y-%m-%d %H:%M:%S")))
    print(f"=== step {step} saved", flush=True)


def halves():
    from chronofact.data import halves as h
    return h()


def all_qids():
    a, b = halves()
    return a + b


def W():
    return CFG["resources"]


# ================================================================== 0 setup
def step_setup():
    from chronofact.data import RAW, URL, load_all
    if not os.path.exists(RAW):
        os.makedirs(os.path.dirname(RAW), exist_ok=True)
        print("downloading", URL, flush=True)
        urllib.request.urlretrieve(URL, RAW + ".part")
        os.replace(RAW + ".part", RAW)
    qs = load_all()
    a, b = halves()
    assert len(a) == 251 and len(b) == 249 and not set(a) & set(b) and set(a) | set(b) == set(qs), "halves broken"
    pins = json.load(open(path("model_pins.json")))
    need = [CFG["embeddings"]["query"], CFG["embeddings"]["context"], CFG["nli"]["model"], CFG["tokenizer"]["repo"]]
    assert all(pins.get(k) for k in need), f"missing pins: {pins}"
    assert pins[CFG["tokenizer"]["repo"]] == CFG["tokenizer"]["revision"], "tokenizer revision mismatch"
    import openai
    import torch
    import transformers
    no_ev = [q for q in qs.values() if not any(s.is_evidence for s in q.sessions)]
    save("setup", {"dataset_sha256": hashlib.sha256(open(RAW, "rb").read()).hexdigest(), "questions": len(qs),
                   "halves": {"A": len(a), "B": len(b)}, "pins": pins, "profile": PROFILE,
                   "llm_roles": {r: CFG["llm"]["roles"][r] | {"name": CFG["llm"]["models"][v["model"]]["name"]}
                                 for r, v in CFG["llm"]["roles"].items()},
                   "versions": {"openai": openai.__version__, "transformers": transformers.__version__,
                                "torch": torch.__version__},
                   "histories_without_evidence_session": [q.question_id for q in no_ev]})


# ================================================================== 1 smoke
SMOKE_TAU = 0.5


def _smoke_qids():
    a, b = halves()
    return [a[0], b[0], a[1]]


def step__smoke_run():
    """runs inside a subprocess with CF_RESULTS (and CF_FAKE_LLM) set by step_smoke"""
    qids = _smoke_qids()
    fake = os.environ.get("CF_FAKE_LLM") == "1"
    print(f"[smoke] {'fake' if fake else 'real'} LLM, {qids}", flush=True)
    assert not P.run_pool("ingest", qids, 3, 1), "ingest errors"
    assert not P.run_pool("w2", qids, 3, 1), "w2 errors"
    assert not P.run_pool("relevance", qids, 3, 1), "relevance errors"
    assert not P.run_pool("retrieve", qids, 3, 1, payload={"taus": {"A": SMOKE_TAU, "B": SMOKE_TAU}}), "retrieve errors"
    assert not P.answer_all(qids), "answer errors"
    if fake:
        assert not P.judge_all(qids), "judge errors"
    rep = _smoke_audit(qids, judged=fake)
    P.write_json(rpath("smoke_report.json"), rep)
    print(json.dumps(rep, indent=1), flush=True)
    assert not rep["problems"], rep["problems"]


def _smoke_audit(qids, judged):
    import sqlite3
    from chronofact.data import load_all
    qs = load_all()
    prob, info = [], {}
    for qid in qids:
        for wv in ("v5", "v5f"):
            m = P.read_json(P.meta_path(wv, qid))
            if [c["fraction"] for c in m["checkpoints"]] != P.FRACS:
                prob.append(f"{wv}/{qid}: checkpoints")
            for i in range(5):
                d = sqlite3.connect(f"file:{os.path.join(P.snap_dir(wv, qid), f'ck{i}.sqlite')}?mode=ro", uri=True)
                if d.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    prob.append(f"{wv}/{qid}/ck{i}: integrity")
                if wv == "v5":
                    want = {(t["session_id"], t["turn_idx"], t["speaker"], t["text"])
                            for t in qs[qid].turns()[:m["checkpoints"][i]["n_turns"]]}
                    got = set(d.execute("SELECT session_id, turn_idx, speaker, text FROM episodes"))
                    if got != want:
                        prob.append(f"{wv}/{qid}/ck{i}: stored turns differ from the history")
                d.close()
        m = P.read_json(P.meta_path("v5", qid))
        info[qid] = {"ops_final": m["checkpoints"][-1]["ops"], "storage_final": m["checkpoints"][-1]["storage"],
                     "write_calls": m["checkpoints"][-1]["write_cum"], "wall_s": round(m["wall_seconds"], 1)}
        for s in P.SYSTEMS:
            rec = P.read_gz(P.ctx_path(s, qid))
            if any(sorted(map(int, c["contexts"])) != P.BUDGETS for c in rec["checkpoints"]):
                prob.append(f"{s}/{qid}: context grid")
            rows = P.read_gz(P.rows_path(s, qid))
            if len(rows) != 20:
                prob.append(f"{s}/{qid}: {len(rows)} rows")
            if judged and len(P.read_gz(P.judged_path(s, qid))) != 20:
                prob.append(f"{s}/{qid}: judged rows")
    info["determinism"] = P.determinism(qids, 24, {"A": SMOKE_TAU, "B": SMOKE_TAU})
    if info["determinism"]["same"] != info["determinism"]["total"]:
        prob.append(f"retrieval not deterministic: {info['determinism']}")
    return {"problems": prob, "info": info}


def smoke_dir(kind):
    return rpath("smoke", kind)


def step_smoke():
    """the whole pipeline on 3 histories, first with a fake LLM, then with the real one (README step 1); the smoke
    runs share the main run's caches"""
    out = {}
    for kind in ("fake", "real"):
        env = dict(os.environ, CF_RESULTS=smoke_dir(kind), CF_CACHE_DIR=CACHE_DIR, CF_PROFILE=PROFILE)
        if kind == "fake":
            env["CF_FAKE_LLM"] = "1"
        else:
            env.pop("CF_FAKE_LLM", None)
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "_smoke_run"], env=env)
        if r.returncode:
            raise RuntimeError(f"smoke ({kind}) failed; see the log above")
        out[kind] = json.load(open(os.path.join(smoke_dir(kind), "smoke_report.json")))
    save("smoke", out)


# ================================================================== 2 serve check
def step_serve_check():
    """20 extractor + 20 answer calls; writes them for reading. Run `serve_ok` after reading them."""
    import asyncio
    from chronofact import prompts as PR
    from chronofact import rules as RU
    from chronofact.answer import answer, has_final
    from chronofact.data import load_all
    from chronofact.llm import LLMJSONError, acall
    qs = load_all()
    rng = random.Random(11)
    turns = []
    for qid in rng.sample(sorted(qs), 60):
        cand = [t for t in qs[qid].turns() if t["speaker"] == "user" and not RU.salience(t["text"])]
        if cand:
            turns.append(rng.choice(cand))
        if len(turns) == 20:
            break
    ctxs = []
    for kind in ("real",):
        base = os.path.join(smoke_dir(kind), "contexts")
        for s in P.SYSTEMS:
            for f in sorted(os.listdir(os.path.join(base, s))):
                rec = P.read_gz(os.path.join(base, s, f))
                for ck in rec["checkpoints"]:
                    for b in ("512", "1024"):
                        ctxs.append((s, f.split(".")[0], ck["contexts"][b]["text"]))
    rng.shuffle(ctxs)
    ctxs = ctxs[:20]

    async def main():
        ex = []
        for t in turns:
            lines = [f"Message date: {t['ts'].strftime('%Y/%m/%d (%a)')}", "User message:", RU._clip(t["text"], 1200)]
            try:
                out, st = await acall("extractor", [{"role": "system", "content": PR.EXTRACT_SYSTEM},
                                                     {"role": "user", "content": "\n".join(lines)}], PR.EXTRACT_SCHEMA)
                ex.append({"turn": t["text"][:1500], "facts": out, "valid": True, "attempts": st.llm_calls})
            except LLMJSONError as e:
                ex.append({"turn": t["text"][:1500], "facts": None, "valid": False, "error": str(e)})
        an = []
        for s, qid, text in ctxs:
            q = qs[qid]
            a, st, raw, _, rec = await answer(q.question, q.question_date, text, P.MODE[s])
            an.append({"system": s, "question": q.question, "gold": q.answer, "answer": a, "raw": raw,
                       "final_found": has_final(raw) if P.MODE[s] == "reason" else None,
                       "final_recovered": rec if P.MODE[s] == "reason" else None})
        return ex, an
    ex, an = asyncio.run(main())
    rep = {"extractor_valid_json": sum(e["valid"] for e in ex), "extractor_calls": len(ex),
           "extractor_first_attempt_valid": sum(e.get("attempts") == 1 for e in ex),
           "reason_answers": sum(a["final_found"] is not None for a in an),
           "reason_with_final": sum(bool(a["final_found"]) for a in an),
           "reason_final_recovered": sum(bool(a["final_recovered"]) for a in an), "extractions": ex, "answers": an}
    P.write_json(rpath("serve_check.json"), rep)
    print({k: v for k, v in rep.items() if not isinstance(v, list)}, flush=True)
    print(f"READ {rpath('serve_check.json')}, then run: python scripts/chronofact.py serve_ok \"<notes>\"", flush=True)
    sys.exit(3)


def cmd_serve_ok(notes):
    rep = P.read_json(rpath("serve_check.json"))
    save("serve_check", {k: v for k, v in rep.items() if not isinstance(v, list)} | {"reviewed": True, "notes": notes})


# ================================================================== batch check (spark profile; pre-declared rule)
BATCH_SIZES = [5, 3, 2]          # tried in this order; the first that passes is used
BATCH_TURNS = 300                # evidence user turns in the test
BATCH_MIN_FACT_RATIO = 0.97      # batched verified facts >= 97% of per-turn verified facts on evidence turns
BATCH_MAX_COVER_DROP = 0.02      # share of evidence turns with >= 1 verified fact may fall by at most 2 points


class _NoStore:
    def log(self, *a, **k):
        pass


def step_batch_check():
    """Does batched extraction lose facts? On ~300 salient evidence user turns (the turns that hold the answers),
    extraction runs per turn (README prompt) and in the chunk ingest would really send (same neighbours, same
    context, so the calls are reused by ingest). A fact counts if it passes the quote check and NLI >= min_entail.
    The largest chunk size meeting BOTH pre-declared bars is chosen; if none does, the step stops."""
    import asyncio
    from chronofact import rules as RU
    from chronofact.data import load_all
    from chronofact.llm import PHASE, RUN_TAG
    from chronofact.models import nli
    from chronofact.writer import Writer, chunk_jobs, extraction_jobs
    qs = load_all()
    rng = random.Random(13)
    order = all_qids()
    rng.shuffle(order)
    picks = []                   # (qid, batch jobs, index of the evidence job in them)
    for qid in order:
        q = qs[qid]
        ev = q.evidence_turns()
        prev = {}
        for _, _, _, _, batches in P.ingest_plan(q):
            for batch in batches:
                jobs, _ = extraction_jobs(batch, prev)
                for k, (i, t, ctx) in enumerate(jobs):
                    if (t.session_id, t.turn_idx) in ev:
                        picks.append((qid, jobs, k))
        if len(picks) >= BATCH_TURNS:
            break
    picks = picks[:BATCH_TURNS]
    w = Writer(_NoStore())

    def verified(facts, t):
        cs = [c for c in (w._prepare(f, t, 0) for f in facts) if c and RU.quote_ok(c["quote"], t.text)]
        if not cs:
            return 0
        p = nli().entail([(RU.third_person(RU.premise_window(c["quote"], t.text)), c["statement"]) for c in cs])
        return int((p >= CFG["write"]["min_entail"]).sum())

    async def run(n):
        async def one(qid, jobs, k):
            t = jobs[k][1]
            if n == 1:
                facts, _, err = await w._extract(t, jobs[k][2])
            else:
                ch = next(c for c in chunk_jobs(jobs, n, CFG["write"]["extract_chunk_words"]) if jobs[k] in c)
                rows = await w._extract_chunk(qid, ch)
                facts, _, err = rows[ch.index(jobs[k])]
            return facts, t, err
        return await asyncio.gather(*[one(*x) for x in picks])

    PHASE.set("write")
    RUN_TAG.set("batch_check")
    per = asyncio.run(run(1))
    base_f = [verified(f, t) for f, t, _ in per]
    base = {"verified_facts": sum(base_f), "coverage": float(np.mean([x > 0 for x in base_f])),
            "errors": sum(1 for *_, e in per if e)}
    print(f"[batch_check] per-turn: {base}", flush=True)
    tried, chosen = [], None
    for n in BATCH_SIZES:
        res = asyncio.run(run(n))
        f = [verified(x, t) for x, t, _ in res]
        r = {"chunk": n, "verified_facts": sum(f), "coverage": float(np.mean([x > 0 for x in f])),
             "errors": sum(1 for *_, e in res if e)}
        r["fact_ratio"] = r["verified_facts"] / max(1, base["verified_facts"])
        r["coverage_drop"] = base["coverage"] - r["coverage"]
        r["passed"] = r["fact_ratio"] >= BATCH_MIN_FACT_RATIO and r["coverage_drop"] <= BATCH_MAX_COVER_DROP
        print(f"[batch_check] {r}", flush=True)
        tried.append(r)
        if r["passed"]:
            chosen = n
            break
    out = {"turns": len(picks), "histories": len({p[0] for p in picks}), "per_turn": base, "tried": tried,
           "chosen_chunk": chosen,
           "rule": f"largest chunk in {BATCH_SIZES} with verified facts >= {BATCH_MIN_FACT_RATIO:.0%} of per-turn "
                   f"and coverage drop <= {BATCH_MAX_COVER_DROP:.0%} points; none -> stop"}
    if chosen is None:
        P.write_json(rpath("batch_check_failed.json"), out)
        raise RuntimeError(f"batched extraction loses facts at every chunk size: {tried}; stop and report")
    save("batch_check", out)
    CFG["write"]["extract_chunk"] = chosen          # this process too (new processes read it from the step file)


# ================================================================== 4 pilot
def _pilot_measures(qids):
    ms = {k: [] for k in ("SKIP", "episodes", "MERGE", "NOOP", "REJECT", "ADD", "SUPERSEDE", "ERROR", "RICH_DROP", "DROP",
                          "DROP_EMPTY")}
    rich = kept_all = 0
    w2 = {"superseded": 0, "rule_a": 0, "rule_b": 0, "active_before": 0}
    for q in qids:
        m = P.read_json(P.meta_path("v5", q))
        last = m["checkpoints"][-1]
        ops, sto = last["ops"], last["storage"]
        for k in ("SKIP", "MERGE", "NOOP", "REJECT", "ADD", "SUPERSEDE", "ERROR", "RICH_DROP", "DROP", "DROP_EMPTY"):
            ms[k].append(ops.get(k, 0))
        ms["episodes"].append(sto["episodes"])
        rich += sto["facts_rich"]
        kept_all += sto["facts_active"] + sto["facts_superseded"]
        for k, v in P.read_json(P.meta_path("v5f", q))["w2"][-1].items():
            w2[k] = w2.get(k, 0) + v
    out = {k: float(np.mean(v)) for k, v in ms.items()}
    out["rich"] = rich / max(1, kept_all)            # share of stored (non-rejected) facts with a rich statement
    out["w2_final_checkpoint"] = w2
    return out


def _dropped_rich(qids, n=30):
    import sqlite3
    out = []
    for q in qids:
        d = sqlite3.connect(f"file:{os.path.join(P.snap_dir('v5', q), 'ck4.sqlite')}?mode=ro", uri=True)
        out += [r[0] for r in d.execute("SELECT reason FROM ops_log WHERE op='RICH_DROP' ORDER BY id")]
        d.close()
    return random.Random(5).sample(out, min(n, len(out)))


def step_pilot():
    """README 13.2 pilot: the write path on 30 histories (15 per half) as a health check before the full ingest. The
    thresholds stay the configured ones (calibrated for these models, see config.yaml); the counts are compared with
    Model4 for review, and SKIP / episodes must match Model4 exactly."""
    a, b = halves()
    qids = a[:15] + b[:15]
    th = {k: CFG["write"][k] for k in ("entity_merge_sim", "near_dup_high", "min_entail")}
    print(f"[pilot] thresholds {th}", flush=True)
    if P.run_pool("ingest", qids, W()["ingest_workers"], W()["ingest_per_worker"]) or \
            P.run_pool("w2", qids, W()["retrieve_workers"], 1):
        raise RuntimeError("pilot ingest/w2 errors; rerun the step")
    m = _pilot_measures(qids)
    print(f"[pilot] measures: {m}", flush=True)
    extracted = m["ADD"] + m["NOOP"] + m["REJECT"] + m["DROP"] + m["DROP_EMPTY"]
    if m["DROP"] > 0.01 * max(1.0, extracted):
        raise RuntimeError(f"{m['DROP']:.1f} facts per history dropped for missing fields / invalid family: the "
                           f"extraction JSON schema is not being enforced; fix this before the pilot")
    if round(m["SKIP"], 1) != REF["SKIP"] or round(m["episodes"]) != REF["episodes"]:
        raise RuntimeError(f"SKIP {m['SKIP']:.2f} / episodes {m['episodes']:.2f} differ from Model4's exact values "
                           f"{REF['SKIP']} / {REF['episodes']}: a bug in loading, checkpoints or the salience filter")
    w2 = m["w2_final_checkpoint"]
    exp_w2 = {"superseded": 316 * len(qids) / 500, "rule_a": 104 * len(qids) / 500, "rule_b": 212 * len(qids) / 500}
    checks = {"w2_superseded": {"pilot": w2["superseded"], "model4_scaled": exp_w2["superseded"]},
              "w2_rule_a": {"pilot": w2["rule_a"], "model4_scaled": exp_w2["rule_a"]},
              "w2_rule_b": {"pilot": w2["rule_b"], "model4_scaled": exp_w2["rule_b"]},
              "SUPERSEDE_per_history": {"pilot": m["SUPERSEDE"], "model4": REF["SUPERSEDE"]},
              "rich_share": {"pilot": m["rich"], "model4": REF["rich"]},
              "ADD_per_history": {"pilot": m["ADD"], "model4": REF["ADD"]}}
    flags = [k for k, v in checks.items() if not (0.5 <= (v["pilot"] + 1) / (v.get("model4_scaled", v.get("model4")) + 1) <= 2.0)]
    P.write_json(rpath("pilot_review.json"), {"final_thresholds": th, "measures": m, "reference_model4": REF,
                                               "model4_comparison": checks, "flags_for_review": flags,
                                               "dropped_rich_sample": _dropped_rich(qids)})
    print(f"[pilot] comparison with Model4: {checks}; flags {flags}. Review {rpath('pilot_review.json')}, then run: "
          f"python scripts/chronofact.py pilot_ok \"<notes>\"", flush=True)
    sys.exit(3)


def cmd_pilot_ok(notes):
    save("pilot", P.read_json(rpath("pilot_review.json")) | {"reviewed": True, "review_notes": notes})


# ================================================================== cost gate (spark profile; the operator checks the console)
def _spend_by_model():
    from chronofact.llm import get_llm
    rows = get_llm().db.execute("SELECT model, COUNT(*), SUM(tokens_in), SUM(tokens_cached), SUM(tokens_out), SUM(usd) "
                                "FROM spend GROUP BY model").fetchall()
    return {m: {"calls": c, "tokens_in": ti, "tokens_cached": tc, "tokens_out": to, "usd": u}
            for m, c, ti, tc, to, u in rows}


def step_cost_gate():
    """Stops once so the user can compare the console's billed amount with our computed spend (the contributor
    price is not confirmed by Meta's own pages). Continue with: chronofact.py cost_ok <billed_usd>"""
    from chronofact.llm import get_llm
    rep = {"computed_usd": get_llm().spent(), "by_model": _spend_by_model()}
    P.write_json(rpath("cost_gate.json"), rep)
    print(f"[cost_gate] computed spend so far ${rep['computed_usd']:.4f}: {rep['by_model']}", flush=True)
    print("CHECK the Meta console's billed amount, then run: python scripts/chronofact.py cost_ok <billed_usd>", flush=True)
    sys.exit(3)


def cmd_cost_ok(billed, notes=""):
    rep = P.read_json(rpath("cost_gate.json"))
    billed = float(billed)
    ok = billed <= 1.5 * rep["computed_usd"] + 0.50      # contributor pricing confirmed (standard would be ~12x)
    save("cost_gate", rep | {"billed_usd": billed, "passed": ok, "notes": notes})
    if not ok:
        print(f"billed ${billed:.2f} vs computed ${rep['computed_usd']:.2f}: NOT contributor pricing. Stopped.", flush=True)
        sys.exit(1)


# ================================================================== 5 ingest, 6 w2
def step_ingest():
    if CFG["steps"]["cost_gate"] and not load_step("cost_gate")["passed"]:
        raise RuntimeError("cost gate failed: not running the full ingest")
    qids = all_qids()
    todo = [q for q in qids if not P.ingest_done(q)]
    print(f"[ingest] {len(todo)} of {len(qids)} histories to ingest", flush=True)
    if P.run_pool("ingest", todo, W()["ingest_workers"], W()["ingest_per_worker"]):
        raise RuntimeError("ingest errors; rerun the step to retry")
    tot = {}
    for q in qids:
        last = P.read_json(P.meta_path("v5", q))["checkpoints"][-1]
        for k, v in list(last["ops"].items()) + [(f"store_{k}", v) for k, v in last["storage"].items()] + \
                [(f"write_{k}", v) for k, v in last["write_cum"].items()]:
            tot[k] = tot.get(k, 0) + v
    save("ingest", {"histories": len(qids), "per_history_final_checkpoint": {k: v / len(qids) for k, v in tot.items()}})


def step_w2():
    qids = all_qids()
    if P.run_pool("w2", qids, W()["retrieve_workers"], 1):
        raise RuntimeError("w2 errors; rerun the step to retry")
    t = {}
    for q in qids:
        for k, v in P.read_json(P.meta_path("v5f", q))["w2"][-1].items():
            t[k] = t.get(k, 0) + v
    save("w2", {"final_checkpoint_totals": t})


# ================================================================== 7 relevance (tau per fold)
def step_relevance():
    if P.run_pool("relevance", all_qids(), W()["retrieve_workers"], 1):
        raise RuntimeError("relevance errors; rerun the step to retry")
    a, b = halves()
    out = {}
    for h, ids in (("A", a), ("B", b)):
        rows = [P.read_json(rpath("relevance", f"{q}.json")) for q in ids]
        rows = [r for r in rows if r["evidence_facts"]]
        grid = {}
        for tau in CFG["read"]["tau_grid"]:
            k = f"{tau:.2f}"
            grid[k] = {"past_gate_share": float(np.mean([r["tau"][k]["past_gate"] for r in rows])),
                       "mean_fact_lines": float(np.mean([r["tau"][k]["fact_lines"] for r in rows]))}
        ok = [t for t in CFG["read"]["tau_grid"] if grid[f"{t:.2f}"]["past_gate_share"] >= CFG["read"]["tau_target"]]
        out[h] = {"questions_with_evidence_fact": len(rows), "grid": grid,
                  "chosen_tau": max(ok) if ok else min(CFG["read"]["tau_grid"])}
    print(f"[relevance] tau A {out['A']['chosen_tau']}, tau B {out['B']['chosen_tau']}", flush=True)
    save("relevance", out | {"rule": f"largest tau with past-the-gate share >= {CFG['read']['tau_target']}, else the "
                                    f"smallest tau of the grid (README 13.3); "
                                    "fold A = tau of half A scores half B, fold B = tau of half B scores half A"})


# ================================================================== 8 retrieve, 9 answer
def step_retrieve():
    taus = P.taus()
    qids = [q for q in all_qids() if not all(os.path.exists(P.ctx_path(s, q)) for s in P.SYSTEMS)]
    if P.run_pool("retrieve", qids, W()["retrieve_workers"], 1, payload={"taus": taus}):
        raise RuntimeError("retrieve errors; rerun the step to retry")
    save("retrieve", {"taus": taus})


def step_answer():
    if P.answer_all(all_qids()):
        raise RuntimeError("answer errors; rerun the step to retry")
    save("answer", {"files": len(all_qids()) * len(P.SYSTEMS)})


# ================================================================== 10 judge pilot, 11 judge
def _pilot_rows():
    """the first 1,000 rows, all three systems mixed: questions in halves order, then system, checkpoint, budget"""
    rows = []
    for q in all_qids():
        for s in P.SYSTEMS:
            r = P.read_gz(P.rows_path(s, q))
            for f in P.FRACS:
                for b in P.BUDGETS:
                    rows.append(r[f"{f}|{b}"])
            if len(rows) >= CFG["eval"]["judge_pilot_rows"]:
                return rows[: CFG["eval"]["judge_pilot_rows"]]
    return rows


def step_judge_pilot():
    import asyncio
    from chronofact.data import load_all
    from chronofact.llm import RUN_TAG, get_llm
    qs = load_all()
    rows = _pilot_rows()
    llm = get_llm()
    rounds = load_step("judge_pilot_progress")["rounds"] if done("judge_pilot_progress") else []
    finished = bool(rounds) and rounds[-1]["next_max_tokens"] is None
    mt = (rounds[-1]["max_tokens"] if finished else rounds[-1]["next_max_tokens"]) if rounds \
        else CFG["llm"]["roles"]["judge"]["max_tokens"]
    while True:
        CFG["llm"]["roles"]["judge"]["max_tokens"] = mt
        RUN_TAG.set(f"judge_pilot_{mt}")
        t0 = time.time()
        llm.flush()
        n0 = llm.db.execute("SELECT COUNT(*) FROM spend").fetchone()[0]
        c0 = llm.db.execute("SELECT COALESCE(MAX(id), 0) FROM calls").fetchone()[0]

        async def main():
            async def one(r):
                return await P.judge_row(qs[r["question_id"]], r)
            return await asyncio.gather(*[one(r) for r in rows], return_exceptions=True)
        res = asyncio.run(main())
        llm.flush()
        errs = [f"{type(x).__name__}: {x}" for x in res if isinstance(x, Exception)]
        if any("SpendCap" in e for e in errs):
            raise RuntimeError(errs[0])
        paid = llm.db.execute("SELECT COUNT(*), COALESCE(SUM(usd),0), COALESCE(SUM(tokens_out),0) FROM spend WHERE id > ?",
                              (n0,)).fetchone()
        att, cutoff, bad = llm.db.execute(
            "SELECT COUNT(*), COALESCE(SUM(finish='length'),0), COALESCE(SUM(json_ok=0),0) FROM calls "
            "WHERE id > ? AND role='judge'", (c0,)).fetchone()
        verdicts = [dict(r, **v) for r, v in zip(rows, res) if not isinstance(v, Exception)]
        llm_judged = [v for v in verdicts if v["judge_rule"] == "llm"]
        if finished:                      # resumed after the last round: verdicts replayed from the cache
            break
        rnd = {"max_tokens": mt, "rows": len(rows), "errors": errs[:5], "n_errors": len(errs),
               "paid_calls": paid[0], "usd": paid[1], "usd_per_paid_call": paid[1] / max(1, paid[0]),
               "mean_tokens_out": paid[2] / max(1, paid[0]), "judge_attempts": att, "cut_off": cutoff,
               "invalid_json": bad, "parse_failure_rate": bad / max(1, att), "llm_judged": len(llm_judged),
               "rule_judged": len(verdicts) - len(llm_judged), "seconds": time.time() - t0, "n429": llm.n429}
        print(f"[judge_pilot] {rnd}", flush=True)
        nxt = 512 if (rnd["cut_off"] > 0 and mt < 512) else None      # README 11: raise to 512 on cut-off output
        rounds.append(dict(rnd, next_max_tokens=nxt))
        save("judge_pilot_progress", {"rounds": rounds})
        if nxt is None:
            break
        mt = nxt
    last = rounds[-1]
    # projection: cost per row in the pilot x every row of the grid
    n_rows = len(all_qids()) * len(P.SYSTEMS) * len(P.FRACS) * len(P.BUDGETS)
    per_row = last["usd"] / max(1, last["rows"]) if last["paid_calls"] else 0.0
    proj = per_row * n_rows + llm.spent()
    rng = random.Random(3)
    corr = [v for v in verdicts if v["judge_rule"] == "llm" and v["correct"]]
    wrong = [v for v in verdicts if v["judge_rule"] == "llm" and not v["correct"]]
    sample = rng.sample(corr, min(25, len(corr))) + rng.sample(wrong, min(25, len(wrong)))
    keep = ("question_id", "system", "checkpoint", "budget", "question", "gold", "answer", "correct", "judge_reason")
    P.write_json(rpath("judge_pilot_sample.json"), [{k: v[k] for k in keep} for v in sample])
    P.write_json(rpath("judge_pilot_progress_summary.json"), {"rounds": rounds, "projected_total_usd": proj})
    print(f"[judge_pilot] projected total spend ${proj:.2f} (cap ${CFG['llm']['spend_cap_usd']:.0f}); parse-failure "
          f"rate {last['parse_failure_rate']:.4f}. Hand check {rpath('judge_pilot_sample.json')}, then run: "
          f"python scripts/chronofact.py handcheck <wrong_of_50> \"<notes>\"", flush=True)
    sys.exit(3)


def cmd_handcheck(n_wrong, notes):
    s = P.read_json(rpath("judge_pilot_progress_summary.json"))
    last = s["rounds"][-1]
    ok_cost = s["projected_total_usd"] <= CFG["llm"]["spend_cap_usd"]
    ok_parse = last["parse_failure_rate"] < 0.01
    ok_hand = int(n_wrong) <= 4
    passed = ok_cost and ok_hand and ok_parse
    save("judge_pilot", {"rounds": s["rounds"], "projected_total_usd": s["projected_total_usd"],
                         "judge_max_tokens": last["max_tokens"],
                         "judge_max_inflight": 96 if all(r["n429"] == 0 for r in s["rounds"]) else 48,
                         "handcheck_wrong_of_50": int(n_wrong), "handcheck_notes": notes,
                         "checks": {"cost_under_cap": ok_cost, "parse_failures_under_1pct": ok_parse,
                                    "handcheck_at_most_4_wrong": ok_hand}, "passed": passed})
    if not passed:
        print("judge pilot FAILED its checks: stop and report before judging the rest (README 13.4)", flush=True)
        sys.exit(1)


def step_judge():
    jp = load_step("judge_pilot")
    if not jp["passed"]:
        raise RuntimeError("the judge pilot did not pass (README 13.4); not judging")
    CFG["llm"]["roles"]["judge"]["max_tokens"] = jp["judge_max_tokens"]
    CFG["llm"]["max_inflight"] = jp["judge_max_inflight"]        # 48 if the pilot saw a 429
    if P.judge_all(all_qids()):
        raise RuntimeError("judge errors; rerun the step to retry")
    from chronofact.llm import get_llm
    save("judge", {"spent_usd": get_llm().spent()})


# ================================================================== 12 report, 13 audit
def step_report():
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "report.py")], check=True)
    save("report", {})


def step_audit():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "audit.py")])
    if r.returncode:
        raise RuntimeError(f"audit found problems: {rpath('final', 'integrity_audit.json')}")
    save("audit", {})


ORDER = ["setup", "smoke", "serve_check"] + (["batch_check"] if CFG["steps"]["batch_check"] else []) + ["pilot"] + \
    (["cost_gate"] if CFG["steps"]["cost_gate"] else []) + \
    ["ingest", "w2", "relevance", "retrieve", "answer", "judge_pilot", "judge", "report", "audit"]


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    os.makedirs(STEPS, exist_ok=True)
    print(f"[chronofact] profile {PROFILE}, results {RESULTS}", flush=True)
    if arg == "handcheck":
        cmd_handcheck(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
        sys.exit(0)
    if arg == "pilot_ok":
        cmd_pilot_ok(sys.argv[2] if len(sys.argv) > 2 else "")
        sys.exit(0)
    if arg == "cost_ok":
        cmd_cost_ok(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
        sys.exit(0)
    if arg == "serve_ok":
        cmd_serve_ok(sys.argv[2] if len(sys.argv) > 2 else "")
        sys.exit(0)
    if arg == "_smoke_run":
        P.clean_tmp()
        step__smoke_run()
        sys.exit(0)
    if arg != "all" and arg not in ORDER:
        sys.exit(f"unknown step {arg!r}; steps: {' '.join(ORDER)}; commands: serve_ok pilot_ok cost_ok handcheck")
    n = P.clean_tmp()
    if n:
        print(f"[resume] removed {n} orphan temp files", flush=True)
    for step in (ORDER if arg == "all" else [arg]):
        if arg == "all" and done(step):
            print(f"=== step {step}: already done", flush=True)
            continue
        print(f"=== step {step}: start {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
        globals()[f"step_{step}"]()
