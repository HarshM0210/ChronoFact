"""A flat copy of a finished run (tables, plots, report, rows, audit, steps, config) -> <results>/bundle/."""
import glob, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from chronofact import pipeline as P
from chronofact.config import ROOT, rpath
F, B = rpath("final"), rpath("bundle")
shutil.rmtree(B, ignore_errors=True)
os.makedirs(os.path.join(B, "steps"))
for p in glob.glob(os.path.join(F, "tables", "*.csv")) + glob.glob(os.path.join(F, "plots", "*.png")):
    shutil.copy(p, B)
for f in ("report.md", "runs.parquet", "integrity_audit.json", "summary.json"):
    shutil.copy(os.path.join(F, f), B)
shutil.copy(os.path.join(ROOT, "data", "splits.json"), B)
for p in glob.glob(rpath("steps", "*.json")):
    shutil.copy(p, os.path.join(B, "steps"))
for f in ("calibration.json", "calibration_w2.json", "calibration_w2c.json", "check_rule_c.json"):
    if os.path.exists(rpath(f)):
        shutil.copy(rpath(f), B)
shutil.copy(os.path.join(ROOT, "config.yaml"), os.path.join(B, "config_used.yaml"))
# accuracy by question type (as in results_model4)
ty = pd.read_csv(os.path.join(F, "tables", "accuracy_by_type.csv"), index_col=0)
sys_ = P.SYSTEMS
x = np.arange(len(ty))
fig, ax = plt.subplots(figsize=(9, 4.8))
for k, s in enumerate(sys_):
    ax.bar(x + (k - (len(sys_) - 1) / 2) * 0.27, 100 * ty[s], 0.25, label=s)
ax.set_xticks(x, [f"{t.replace('single-session-', 'single-session-\n')}\n(n={int(n)})" for t, n in
                  zip(ty.index, ty.n_questions)], fontsize=8)
ax.set_ylabel("accuracy, % (all budgets and checkpoints)")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(sys_), frameon=False)
ax.grid(axis="y", alpha=.3)
fig.tight_layout()
fig.savefig(os.path.join(B, "accuracy_by_type.png"), dpi=150)
print(sorted(os.listdir(B)))
