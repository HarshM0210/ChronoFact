"""W2 threshold calibration for THIS model stack, on the pilot stores (results/snapshots/v5/<pilot qid>/ck4.sqlite)
-> results/calibration_w2.json. W2 runs on finished stores, so this runs during the full ingest.

Candidates: pairs (older, newer) of ACTIVE non-event facts of one subject with different values that rule (a) or
rule (b) could look at (same slot key and both values with a digit, or two states with >= 2 skeleton content words).
Pairs are binned by their rule cosine, up to N_PER_BIN sampled per bin (seed 0) and labeled by Spark (standard tier):
does the newer fact replace the older one (same attribute, changed value, older no longer true)? Each candidate
threshold set of the README's rules is scored by bin-weighted precision and recall on the labels; the set with the
best F0.5 among those with precision >= MIN_PRECISION is chosen (a wrong supersession hides a true fact). If none
qualifies, the README values stay.

Run with a profile whose judge is Muse Spark, after the pilot:  CF_PROFILE=hybrid python scripts/calibration/calibrate_w2.py
"""
import asyncio
import itertools
import os
import random
import re
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chronofact.config import CFG, rpath  # noqa: E402
from chronofact.data import halves  # noqa: E402
from chronofact.llm import RUN_TAG, LLMJSONError, acall, get_llm  # noqa: E402
from chronofact.models import embedder, nli  # noqa: E402
from chronofact.pipeline import snap_dir, write_json  # noqa: E402
from chronofact.reconcile import _content, _key, _skeleton, _value  # noqa: E402
from chronofact.relations import kind_of  # noqa: E402

N_PER_BIN, MIN_PRECISION = 40, 0.80
EDGES = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0001]
GRID = {"num_sim": [0.6, 0.7, 0.8], "skel_sim": [0.7, 0.75, 0.8, 0.85, 0.9], "slot_sim": [0.6, 0.7, 0.8, 0.85, 0.92],
        "contradiction": [0.3, 0.5, 0.7]}
SCHEMA = {"type": "object", "properties": {"reason": {"type": "string"}, "yes": {"type": "boolean"}},
          "required": ["reason", "yes"]}
Q = ('Two facts a memory system stored about a user, the second recorded later.\nOlder ({d1}): {a}\nNewer ({d2}): {b}\n'
     'Does the newer fact REPLACE the older one, i.e. both are about the same single attribute of the user (the same '
     'thing, not just a similar topic) and the value changed, so the older fact is no longer true? Two facts that can '
     'both be true at once do not replace each other.\nReturn JSON {{"reason": "<one short sentence>", "yes": true | false}}')


def main():
    emb, rng = embedder(), random.Random(0)
    a, b = halves()
    cands = []
    for q in a[:15] + b[:15]:
        p = os.path.join(snap_dir("v5", q), "ck4.sqlite")
        d = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        cols = [r[1] for r in d.execute("PRAGMA table_info(facts)")]
        facts = [dict(zip(cols, r)) for r in d.execute("SELECT * FROM facts WHERE status='active' ORDER BY id")]
        facts = [f for f in facts if kind_of(f) != "event"]
        ents = {r[0]: r[1] for r in d.execute("SELECT id, canonical_name FROM entities")}
        d.close()
        by = {}
        for f in facts:
            by.setdefault(f["subj_id"], []).append(f)
        for grp in by.values():
            for x, y in itertools.combinations(grp, 2):
                old, new = (x, y) if _key(x) < _key(y) else (y, x)
                vo, vn = _value(old, ents), _value(new, ents)
                if vo.strip().lower() == vn.strip().lower():
                    continue
                ra = bool(old.get("slot_key")) and old["slot_key"] == new["slot_key"] and bool(re.search(r"\d", vo)) \
                    and bool(re.search(r"\d", vn))
                so, sn = _skeleton(old["statement"], vo), _skeleton(new["statement"], vn)
                rb = kind_of(old) == "state" and kind_of(new) == "state" and len(_content(so)) >= 2 and len(_content(sn)) >= 2
                if ra or rb:
                    cands.append({"q": q, "old": old, "new": new, "ra": ra, "rb": rb, "skb": (so, sn)})
    if not cands:
        print("[calibrate_w2] no candidate pairs")
        return
    def cos(pairs):
        v1 = emb.encode([x or "none" for x, _ in pairs])
        v2 = emb.encode([y or "none" for _, y in pairs])
        return (v1 * v2).sum(1)
    sa = cos([(_skeleton(c["old"]["statement"], "") , _skeleton(c["new"]["statement"], "")) for c in cands])
    sb = cos([c["skb"] for c in cands])
    sl = cos([((c["old"].get("slot_key") or "").replace("_", " "), (c["new"].get("slot_key") or "").replace("_", " "))
              for c in cands])
    c1 = nli().contradict([(c["old"]["statement"] or "", c["new"]["statement"] or "") for c in cands])
    c2 = nli().contradict([(c["new"]["statement"] or "", c["old"]["statement"] or "") for c in cands])
    for c, x, y, z, u, w in zip(cands, sa, sb, sl, c1, c2):
        c.update(sa=float(x), sb=float(y), sl=float(z), con=float(max(u, w)))
        c["score"] = max(c["sa"] if c["ra"] else -1, c["sb"] if c["rb"] else -1)
    bins = {}
    for c in cands:
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            if lo <= c["score"] < hi:
                bins.setdefault(lo, []).append(c)
    sample = []
    for lo, cs in sorted(bins.items()):
        s = rng.sample(cs, min(N_PER_BIN, len(cs)))
        for c in s:
            c["w"] = len(cs) / len(s)
        sample += s
    print(f"[calibrate_w2] {len(cands)} candidate pairs; bins {[(lo, len(cs)) for lo, cs in sorted(bins.items())]}; "
          f"labeling {len(sample)}", flush=True)

    async def lab():
        async def one(c):
            p = Q.format(d1=(c["old"]["recorded_at"] or "")[:10], a=c["old"]["statement"],
                         d2=(c["new"]["recorded_at"] or "")[:10], b=c["new"]["statement"])
            try:
                out, _ = await acall("judge", [{"role": "user", "content": p}], SCHEMA)
                return out.get("yes") if isinstance(out, dict) and isinstance(out.get("yes"), bool) else None
            except LLMJSONError:
                return None
        return await asyncio.gather(*[one(c) for c in sample])
    RUN_TAG.set("calibrate_w2")
    ys = asyncio.run(lab())
    get_llm().flush()
    S = [(c, y) for c, y in zip(sample, ys) if y is not None]
    pos_w = sum(c["w"] for c, y in S if y)

    def fires(c, t):
        a_ = c["ra"] and c["sa"] >= t["num_sim"] and c["con"] >= t["contradiction"]
        b_ = c["rb"] and c["sb"] >= t["skel_sim"] and c["sl"] >= t["slot_sim"] and c["con"] >= t["contradiction"]
        return a_ or b_

    def score(t):
        tp = sum(c["w"] for c, y in S if y and fires(c, t))
        fp = sum(c["w"] for c, y in S if not y and fires(c, t))
        prec = tp / (tp + fp) if tp + fp else None
        rec = tp / pos_w if pos_w else 0.0
        f = (1.25 * prec * rec / (0.25 * prec + rec)) if prec and rec else 0.0
        n = sum(1 for c, _ in S if fires(c, t))
        return {"precision": prec, "recall": rec, "f05": f, "fires_in_sample": n}
    readme = {k: CFG["w2"][k] for k in GRID}
    rows = []
    for vals in itertools.product(*GRID.values()):
        t = dict(zip(GRID, vals))
        rows.append(dict(t, **score(t)))
    ok = [r for r in rows if r["precision"] is not None and r["precision"] >= MIN_PRECISION and r["fires_in_sample"] >= 5]
    best = max(ok, key=lambda r: (r["f05"], r["num_sim"] + r["skel_sim"] + r["slot_sim"] + r["contradiction"])) if ok else None
    chosen = {k: best[k] for k in GRID} if best else readme
    out = {"candidates": len(cands), "labeled": len(S), "positives_weighted": pos_w,
           "readme": dict(readme, **score(readme)), "best": best, "chosen": chosen,
           "rule": f"max F0.5 with bin-weighted precision >= {MIN_PRECISION} (and >= 5 firing labeled pairs); else README",
           "examples": [{"old": c["old"]["statement"], "new": c["new"]["statement"], "yes": y, "sa": round(c["sa"], 3),
                         "sb": round(c["sb"], 3), "sl": round(c["sl"], 3), "con": round(c["con"], 3), "ra": c["ra"]}
                        for c, y in S][:300],
           "spend_usd_after": get_llm().spent()}
    write_json(rpath("calibration_w2.json"), out)
    print(f"[calibrate_w2] README {out['readme']}\n[calibrate_w2] best {best}\n[calibrate_w2] chosen {chosen}", flush=True)


if __name__ == "__main__":
    main()
