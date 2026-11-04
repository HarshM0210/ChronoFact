"""Is the judge-role LLM of the current profile good enough for W2 rule (c)? Its verdicts on the same-slot pairs of a
Spark run's stores are compared with Spark's labels from that run (calibrate_w2c.py output). Rule (c) should be on only
if the precision against Spark is >= 0.80 with >= 5 yes verdicts.

  CF_PROFILE=local python scripts/calibration/check_rule_c.py <spark results dir>"""
import asyncio
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from chronofact import pipeline as P  # noqa: E402
from chronofact.config import ROOT, rpath  # noqa: E402
from chronofact.reconcile import same_slot_pairs  # noqa: E402

A = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results_spark")
lab = {(e["old"], e["new"]): e["yes"] for e in json.load(open(os.path.join(A, "calibration_w2c.json")))["examples"]}
pairs = []
for p in sorted(glob.glob(os.path.join(A, "snapshots", "v5", "*", "ck4.sqlite"))):
    if os.path.exists(os.path.join(os.path.dirname(p), "meta.json")):
        pairs += [(o, n) for o, n in same_slot_pairs(p) if (o["statement"], n["statement"]) in lab]
upd, fails = asyncio.run(P.judge_updates(pairs))
y_sp = [lab[(o["statement"], n["statement"])] for o, n in pairs]
y_lo = [(o["id"], n["id"]) in upd for o, n in pairs]
tp = sum(a and b for a, b in zip(y_lo, y_sp))
nyes = sum(y_lo)
prec = tp / nyes if nyes else None
rec = tp / max(1, sum(y_sp))
ok = prec is not None and prec >= 0.80 and nyes >= 5
out = {"pairs": len(pairs), "spark_yes": sum(y_sp), "local_yes": nyes, "agree_yes": tp, "precision_vs_spark": prec,
       "recall_vs_spark": rec, "failed_verdicts": fails, "rule_c_on": ok}
print(json.dumps(out), flush=True)
P.write_json(rpath("check_rule_c.json"), out)
