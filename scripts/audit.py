"""ChronoFact integrity audit (README 12.1) -> results/final/integrity_audit.json. Exit 1 on any problem."""
import glob
import json
import os
import sqlite3
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd  # noqa: E402

from chronofact import pipeline as P  # noqa: E402
from chronofact.answer import is_abstention, parse_final  # noqa: E402
from chronofact.config import CFG, rpath  # noqa: E402
from chronofact.data import halves, load_all  # noqa: E402

PROB, INFO = [], {}


def bad(m):
    PROB.append(m)
    print("PROBLEM:", m, flush=True)


def main():
    qs = load_all()
    a, b = halves()
    if set(a) & set(b) or len(a) != 251 or len(b) != 249 or set(a) | set(b) != set(qs):
        bad("halves: not disjoint 251 + 249 covering all 500 questions")
    INFO["halves"] = {"A": len(a), "B": len(b)}

    # snapshots
    for wv in ("v5", "v5f"):
        for qid in a + b:
            mp = P.meta_path(wv, qid)
            if not os.path.exists(mp):
                bad(f"{wv}/{qid}: meta.json missing")
                continue
            m = P.read_json(mp)
            if [c["fraction"] for c in m["checkpoints"]] != P.FRACS:
                bad(f"{wv}/{qid}: checkpoints {[c['fraction'] for c in m['checkpoints']]}")
            for i in range(len(P.FRACS)):
                p = os.path.join(P.snap_dir(wv, qid), f"ck{i}.sqlite")
                if not os.path.exists(p):
                    bad(f"{wv}/{qid}: ck{i} missing")
                    continue
                d = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
                if d.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    bad(f"{wv}/{qid}/ck{i}: integrity_check failed")
                if wv == "v5":
                    want = {(t["session_id"], t["turn_idx"], t["speaker"], t["text"])
                            for t in qs[qid].turns()[:m["checkpoints"][i]["n_turns"]]}
                    if set(d.execute("SELECT session_id, turn_idx, speaker, text FROM episodes")) != want:
                        bad(f"v5/{qid}/ck{i}: stored turns differ from the history")
                else:
                    s = sqlite3.connect(f"file:{os.path.join(P.snap_dir('v5', qid), f'ck{i}.sqlite')}?mode=ro", uri=True)
                    fa = {r[0]: r[1:] for r in s.execute("SELECT id, status, kind, statement FROM facts")}
                    fb = {r[0]: r[1:] for r in d.execute("SELECT id, status, kind, statement FROM facts")}
                    if set(fa) != set(fb):
                        bad(f"v5f/{qid}/ck{i}: fact ids differ from v5")
                    ch = [k for k in fa if k in fb and fa[k] != fb[k]]
                    for k in ch:
                        if not (fa[k][0] == "active" and fb[k][0] == "superseded" and fa[k][1] != "event"):
                            bad(f"v5f/{qid}/ck{i}: fact {k} changed illegally")
                    nlog = d.execute("SELECT COUNT(*) FROM ops_log WHERE op='W2_SUPERSEDE'").fetchone()[0]
                    if nlog != len(ch) or nlog != m["w2"][i]["superseded"]:
                        bad(f"v5f/{qid}/ck{i}: W2 {len(ch)} changed, {nlog} logged, {m['w2'][i]['superseded']} in meta")
                    s.close()
                d.close()

    # write-path health: facts dropped for invalid fields (schema not enforced) and extraction errors
    drop = err = ext = 0
    for qid in a + b:
        o = P.read_json(P.meta_path("v5", qid))["checkpoints"][-1]["ops"]
        drop += o.get("DROP", 0)
        err += o.get("ERROR", 0)
        ext += sum(o.get(k, 0) for k in ("ADD", "NOOP", "REJECT", "DROP", "DROP_EMPTY"))
        INFO["empty_placeholder_facts"] = INFO.get("empty_placeholder_facts", 0) + o.get("DROP_EMPTY", 0)
    INFO["dropped_facts_share"] = drop / max(1, ext)
    INFO["write_errors"] = err
    if INFO["dropped_facts_share"] > 0.01:
        bad(f"{drop} extracted facts dropped for invalid fields ({INFO['dropped_facts_share']:.3%})")

    # contexts
    taus = P.taus()
    for s in P.SYSTEMS:
        for qid in a + b:
            p = P.ctx_path(s, qid)
            if not os.path.exists(p):
                bad(f"context {s}/{qid} missing")
                continue
            r = P.read_gz(p)
            if [c["fraction"] for c in r["checkpoints"]] != P.FRACS or any(
                    sorted(map(int, c["contexts"])) != P.BUDGETS for c in r["checkpoints"]):
                bad(f"context {s}/{qid}: grid incomplete")

    # rows
    df = pd.read_parquet(rpath("final", "runs.parquet"))
    k = ["system", "question_id", "checkpoint", "budget"]
    want = len(P.SYSTEMS) * len(a + b) * len(P.FRACS) * len(P.BUDGETS)
    if len(df) != want:
        bad(f"rows: {len(df)}, expected {want}")
    if set(df.system.unique()) != set(P.SYSTEMS):
        bad(f"systems in the rows {sorted(df.system.unique())}, expected {P.SYSTEMS}")
    if df.duplicated(k).any():
        bad(f"rows: {int(df.duplicated(k).sum())} duplicates")
    m5 = df[df.system == "MODEL5"]
    wrong_fold = m5[m5.apply(lambda r: r.fold != P.fold_for(r.question_id) or r.tau != taus[r.fold], axis=1)]
    if len(wrong_fold):
        bad(f"{len(wrong_fold)} MODEL5 rows not retrieved with the tau of the fold that did not use their half")
    rs = df[df["mode"] == "reason"]
    INFO["final_line_share"] = float(rs.final_found.astype(float).mean())      # after the one recovery call
    INFO["final_recovered_share"] = float(rs.final_recovered.astype(float).mean()) if "final_recovered" in rs else 0.0
    if INFO["final_line_share"] < 0.97:
        bad(f"FINAL line in only {INFO['final_line_share']:.3f} of reason-mode answers")
    mism = int((rs.raw_answer.map(parse_final) != rs.answer).sum())
    if mism:
        bad(f"{mism} reason answers do not re-parse to the stored answer")
    if (df.answer.map(is_abstention) != df.abstained).any():
        bad("abstained flag inconsistent with the answers")
    g = df[df.abstained & ~df.is_abstention & df.correct]
    if len(g):
        bad(f"rule judge violated on {len(g)} rows")
    if df.correct.isna().any():
        bad("missing verdicts")
    INFO["answer_errors"] = int(df.answer_error.notna().sum()) if "answer_error" in df else 0
    if INFO["answer_errors"] > 30:                                   # > 0.1% of rows: not a rare event
        bad(f"{INFO['answer_errors']} answer requests rejected by the API (rows kept, judged WRONG)")
    INFO["judge_errors"] = int((df.judge_rule == "judge_error").sum())
    ku = df[(df.question_type == "knowledge-update") & ~df.is_abstention]
    INFO["stale_check_errors"] = int(ku.stale.isna().sum())
    if INFO["judge_errors"] > 30 or INFO["stale_check_errors"] > 30:     # > 0.1% of rows: not a rare event
        bad(f"judge errors {INFO['judge_errors']}, stale-check errors {INFO['stale_check_errors']}")

    # determinism: 160 random cells
    INFO["determinism"] = P.determinism(a + b, 160, taus)
    if INFO["determinism"]["same"] != INFO["determinism"]["total"]:
        bad(f"retrieval not byte-identical: {INFO['determinism']}")

    # spend, temp files
    from chronofact.llm import get_llm
    INFO["llm_spend_usd"] = get_llm().spent()
    if INFO["llm_spend_usd"] > CFG["llm"]["spend_cap_usd"]:
        bad(f"LLM spend ${INFO['llm_spend_usd']:.2f} above the cap")
    tmp = [f for f in glob.glob(rpath("**", "*.tmp*"), recursive=True)]
    if tmp:
        bad(f"{len(tmp)} orphan temp files")
    out = {"problems": PROB, "info": INFO, "ok": not PROB}
    P.write_json(rpath("final", "integrity_audit.json"), out)
    print(json.dumps(out, indent=1))
    sys.exit(0 if not PROB else 1)


if __name__ == "__main__":
    main()
