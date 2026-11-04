"""Calibration of W2 rule (c), same-slot updates, on the finished stores -> results/calibration_w2c.json.

Candidates: within one subject, ACTIVE non-event facts that share an exact slot key (not a generic one) and have
different values; the newest of the group vs each older one (as rule a). Up to N pairs (seed 0) are labeled by Spark
(standard tier): does the newer fact replace the older one? Rule variants (which kinds, NLI contradiction floor)
are scored on the labels; the variant with the best F0.5 among those with precision >= MIN_PRECISION and at least
MIN_FIRES firing labeled pairs is chosen. None qualifies -> rule (c) stays off.

Run with a profile whose judge is Muse Spark, after the ingest:  CF_PROFILE=hybrid python scripts/calibration/calibrate_w2c.py
"""
import asyncio
import glob
import os
import random
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chronofact.config import rpath  # noqa: E402
from chronofact.llm import RUN_TAG, LLMJSONError, acall, get_llm  # noqa: E402
from chronofact.models import nli  # noqa: E402
from chronofact.pipeline import write_json  # noqa: E402
from chronofact.reconcile import _key, _value  # noqa: E402
from chronofact.relations import kind_of  # noqa: E402
from chronofact.rules import GENERIC_SLOTS  # noqa: E402

N, MIN_PRECISION, MIN_FIRES = 600, 0.80, 10
SCHEMA = {"type": "object", "properties": {"reason": {"type": "string"}, "yes": {"type": "boolean"}},
          "required": ["reason", "yes"]}
Q = ('Two facts a memory system stored about a user, the second recorded later.\nOlder ({d1}): {a}\nNewer ({d2}): {b}\n'
     'Does the newer fact REPLACE the older one, i.e. both are about the same single attribute of the user (the same '
     'thing, not just a similar topic) and the value changed, so the older fact is no longer true? Two facts that can '
     'both be true at once do not replace each other.\nReturn JSON {{"reason": "<one short sentence>", "yes": true | false}}')
VARIANTS = [(kinds, con) for kinds in (("state",), ("state", "attribute")) for con in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)]


def candidates(paths):
    out = []
    for p in paths:
        d = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        cols = [r[1] for r in d.execute("PRAGMA table_info(facts)")]
        facts = [dict(zip(cols, r)) for r in d.execute("SELECT * FROM facts WHERE status='active' ORDER BY id")]
        facts = [f for f in facts if kind_of(f) != "event" and f.get("slot_key") and f["slot_key"] not in GENERIC_SLOTS]
        ents = {r[0]: r[1] for r in d.execute("SELECT id, canonical_name FROM entities")}
        d.close()
        by = {}
        for f in facts:
            by.setdefault((f["subj_id"], f["slot_key"]), []).append(f)
        for grp in by.values():
            if len(grp) < 2:
                continue
            grp = sorted(grp, key=_key)
            new = grp[-1]
            for old in grp[:-1]:
                if _value(old, ents).strip().lower() != _value(new, ents).strip().lower():
                    out.append({"store": p, "old": old, "new": new})
    return out


def main():
    paths = sorted(glob.glob(rpath("snapshots", "v5", "*", "ck4.sqlite")))
    paths = [p for p in paths if os.path.exists(os.path.join(os.path.dirname(p), "meta.json"))]
    cands = candidates(paths)
    rng = random.Random(0)
    sample = rng.sample(cands, min(N, len(cands)))
    c1 = nli().contradict([(c["old"]["statement"] or "", c["new"]["statement"] or "") for c in sample])
    c2 = nli().contradict([(c["new"]["statement"] or "", c["old"]["statement"] or "") for c in sample])
    for c, x, y in zip(sample, c1, c2):
        c["con"] = float(max(x, y))
        c["kinds"] = (kind_of(c["old"]), kind_of(c["new"]))
    print(f"[calibrate_w2c] {len(paths)} stores, {len(cands)} same-slot candidate pairs; labeling {len(sample)}", flush=True)

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
    RUN_TAG.set("calibrate_w2c")
    ys = asyncio.run(lab())
    get_llm().flush()
    S = [(c, y) for c, y in zip(sample, ys) if y is not None]
    pos = sum(1 for _, y in S if y)
    rows = []
    for kinds, con in VARIANTS:
        f = [(c, y) for c, y in S if c["kinds"][0] in kinds and c["kinds"][1] in kinds and c["con"] >= con]
        tp = sum(1 for _, y in f if y)
        prec = tp / len(f) if f else None
        rec = tp / pos if pos else 0.0
        f05 = (1.25 * prec * rec / (0.25 * prec + rec)) if prec and rec else 0.0
        rows.append({"kinds": list(kinds), "contradiction": con, "fires": len(f), "tp": tp, "precision": prec,
                     "recall": rec, "f05": f05})
    ok = [r for r in rows if r["precision"] is not None and r["precision"] >= MIN_PRECISION and r["fires"] >= MIN_FIRES]
    best = max(ok, key=lambda r: (r["f05"], r["contradiction"])) if ok else None
    by_kind = {}
    for c, y in S:
        k = "/".join(c["kinds"])
        by_kind.setdefault(k, [0, 0])
        by_kind[k][0] += 1
        by_kind[k][1] += int(y)
    out = {"stores": len(paths), "candidates": len(cands), "labeled": len(S), "positives": pos, "by_kinds": by_kind,
           "variants": rows, "best": best,
           "chosen": ({"enabled": True, "kinds": best["kinds"], "contradiction": best["contradiction"]} if best
                      else {"enabled": False}),
           "rule": f"max F0.5 with precision >= {MIN_PRECISION} and >= {MIN_FIRES} firing labeled pairs; else off",
           "examples": [{"old": c["old"]["statement"], "new": c["new"]["statement"], "slot": c["old"]["slot_key"],
                         "kinds": c["kinds"], "con": round(c["con"], 3), "yes": y} for c, y in S][:400],
           "spend_usd_after": get_llm().spent()}
    write_json(rpath("calibration_w2c.json"), out)
    for r in rows:
        print(f"[calibrate_w2c] {r}", flush=True)
    print(f"[calibrate_w2c] by kinds {by_kind}\n[calibrate_w2c] chosen {out['chosen']}", flush=True)


if __name__ == "__main__":
    main()
