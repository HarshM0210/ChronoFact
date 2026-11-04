"""ChronoFact report (README section 10) -> results/final/{runs.parquet, report.md, tables/*.csv, plots/*.png}.
CIs: 1,000 bootstrap resamples of questions (seed 0); a question's value is its mean over the grid cells.
Paired differences use the same questions and the same resamples."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from chronofact import pipeline as P  # noqa: E402
from chronofact.config import CFG, load_step, rpath  # noqa: E402
from chronofact.data import halves  # noqa: E402

OUT = rpath("final")
SYS = P.SYSTEMS
NB = CFG["eval"]["bootstrap"]
REF14 = {"accuracy": {"Model4": [53.1, 70.4, 73.0, 75.7], "RAG_harness": [56.1, 65.1, 69.8, 74.4],
                      "RAG_plain": [41.3, 45.6, 57.2, 61.0]},
         "write": {"episodes": 493, "ADD": 266, "MERGE": 15.0, "NOOP": 6.5, "REJECT": 97.8, "SKIP": 59.3,
                   "SUPERSEDE": 1.8, "facts_active": 263, "facts_superseded": 3.0, "llm_calls": 460,
                   "tokens_in": 281000, "tokens_out": 51000, "rich_share": 0.69}}


def load_rows():
    a, b = halves()
    rows = []
    for s in SYS:
        for q in a + b:
            rows += list(P.read_gz(P.judged_path(s, q)).values())
    return pd.DataFrame(rows)


def qvals(df, system, budget=None):
    x = df[df.system == system]
    if budget is not None:
        x = x[x.budget == budget]
    return x.groupby("question_id").correct.mean()


def main():
    df = load_rows()
    for col in ("final_recovered", "answer_error", "answer_model"):       # fields newer than some row files
        if col not in df:
            df[col] = None
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "plots"), exist_ok=True)
    tmp = os.path.join(OUT, "runs.parquet.part")
    df.drop(columns=["write_role_stats"]).to_parquet(tmp)
    os.replace(tmp, os.path.join(OUT, "runs.parquet"))
    budgets = sorted(df.budget.unique())
    ckpts = sorted(df.checkpoint.unique())
    qids = sorted(df.question_id.unique())
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(qids), (NB, len(qids)))           # the same resamples for every number

    def boot(series):
        v = series.reindex(qids).to_numpy(dtype=float)
        bs = v[idx].mean(1)
        return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    rel = load_step("relevance")
    L = ["# ChronoFact (Model5) results", "",
         f"Rows: {len(df)}; questions: {len(qids)}; systems {SYS}; budgets {budgets}; checkpoints {ckpts}. "
         f"MODEL5 gate: tau {rel['A']['chosen_tau']} (chosen on half A, scores half B = fold A), tau "
         f"{rel['B']['chosen_tau']} (chosen on half B, scores half A = fold B). 95% CIs: {NB} bootstrap resamples "
         "of questions.", ""]

    # 1 accuracy per budget
    acc = []
    for s in SYS:
        for b in budgets:
            m, lo, hi = boot(qvals(df, s, b))
            acc.append({"system": s, "budget": b, "acc": m, "lo": lo, "hi": hi})
        m, lo, hi = boot(qvals(df, s))
        acc.append({"system": s, "budget": "all", "acc": m, "lo": lo, "hi": hi})
    acc = pd.DataFrame(acc)
    acc.to_csv(os.path.join(OUT, "tables", "accuracy.csv"), index=False)
    cols = budgets + ["all"]
    L += ["## 1. Accuracy by budget (mean over 5 checkpoints), %", "",
          "| system | " + " | ".join(map(str, cols)) + " |", "|---|" + "---|" * len(cols)]
    for s in SYS:
        a = acc[acc.system == s].set_index("budget")
        L.append(f"| {s} | " + " | ".join(f"{100 * a.acc[b]:.1f} [{100 * a.lo[b]:.1f}, {100 * a.hi[b]:.1f}]"
                                         for b in cols) + " |")
    L += ["", "Model4 reference (other LLM/embedder/NLI; compare leads, not raw numbers): " +
          "; ".join(f"{k} {v}" for k, v in REF14["accuracy"].items()), ""]

    # 2 paired differences
    pdiff = []
    for base in SYS[1:]:
        for b in cols:
            x = qvals(df, "MODEL5", None if b == "all" else b)
            y = qvals(df, base, None if b == "all" else b)
            m, lo, hi = boot(x - y)
            pdiff.append({"vs": base, "budget": b, "diff": m, "lo": lo, "hi": hi, "significant": lo > 0})
    pdiff = pd.DataFrame(pdiff)
    pdiff.to_csv(os.path.join(OUT, "tables", "paired_model5_minus.csv"), index=False)
    L += ["## 2. MODEL5 minus baseline (paired, points)", "",
          "| baseline | " + " | ".join(map(str, cols)) + " |", "|---|" + "---|" * len(cols)]
    for base in SYS[1:]:
        a = pdiff[pdiff.vs == base].set_index("budget")
        L.append(f"| {base} | " + " | ".join(
            f"{100 * a['diff'][b]:+.1f} [{100 * a.lo[b]:+.1f}, {100 * a.hi[b]:+.1f}]{' *' if a.significant[b] else ''}"
            for b in cols) + " |")
    lead_h = float(np.mean([pdiff[(pdiff.vs == "RAG_harness") & (pdiff.budget == b)]["diff"].iloc[0] for b in budgets]))
    plain_all = pdiff[(pdiff.vs == "RAG_plain") & (pdiff.budget == "all")].iloc[0]
    bar = bool(plain_all.lo > 0) and lead_h >= 0.017
    L += ["", "`*` = 95% CI above 0.", "",
          f"**Bar (README 14):** lead over RAG_plain significant: {'yes' if plain_all.lo > 0 else 'no'} "
          f"({100 * plain_all['diff']:+.1f} [{100 * plain_all.lo:+.1f}, {100 * plain_all.hi:+.1f}]); mean paired lead over "
          f"RAG_harness across the four budgets {100 * lead_h:+.1f} points (needs >= +1.7). "
          f"**Model5 {'meets' if bar else 'does not meet'} the bar.**", ""]

    # 3 by type, checkpoint, fold
    df["qtype"] = np.where(df.is_abstention, "abstention", df.question_type)
    ty = df.groupby(["qtype", "system"]).correct.mean().unstack("system")[SYS]
    ty.insert(0, "n_questions", df.groupby("qtype").question_id.nunique())
    ty.to_csv(os.path.join(OUT, "tables", "accuracy_by_type.csv"))
    L += ["## 3. Accuracy by question type (all budgets and checkpoints)", "",
          "| type | n | " + " | ".join(SYS) + " |", "|---|---|" + "---|" * len(SYS)]
    for qt, r in ty.iterrows():
        L.append(f"| {qt} | {int(r.n_questions)} | " + " | ".join(f"{100 * r[s]:.1f}" for s in SYS) + " |")
    ck = df.groupby(["system", "checkpoint"]).correct.mean().unstack("checkpoint")
    ck.to_csv(os.path.join(OUT, "tables", "accuracy_by_checkpoint.csv"))
    L += ["", "## 4. Accuracy by checkpoint (all budgets)", "",
          "| system | " + " | ".join(f"{c:g}" for c in ckpts) + " |", "|---|" + "---|" * len(ckpts)]
    for s in SYS:
        L.append(f"| {s} | " + " | ".join(f"{100 * ck.loc[s, c]:.1f}" for c in ckpts) + " |")
    fo = df.groupby(["fold", "system", "budget"]).correct.mean().unstack("budget")
    fo.to_csv(os.path.join(OUT, "tables", "accuracy_by_fold.csv"))
    L += ["", "## 5. Per fold (fold A = tau from half A, scored on half B; fold B the reverse)", "",
          "| fold | system | " + " | ".join(map(str, budgets)) + " |", "|---|---|" + "---|" * len(budgets)]
    for (f, s), r in fo.iterrows():
        L.append(f"| {f} | {s} | " + " | ".join(f"{100 * r[b]:.1f}" for b in budgets) + " |")

    # 4 behaviour
    ku = df[(df.question_type == "knowledge-update") & ~df.is_abstention]
    ans = df[~df.is_abstention]
    beh = pd.DataFrame({
        "stale_rate_knowledge_update": ku.groupby("system").stale.apply(lambda s: s.astype(float).mean()),
        "false_abstention_rate": ans.groupby("system").abstained.mean(),
        "abstention_question_accuracy": df[df.is_abstention].groupby("system").correct.mean(),
        "evidence_recall_any": df[df.ev_any.notna()].groupby("system").ev_any.apply(lambda s: s.astype(float).mean()),
        "evidence_recall_all": df[df.ev_all.notna()].groupby("system").ev_all.apply(lambda s: s.astype(float).mean()),
        "final_line_found": df[df["mode"] == "reason"].groupby("system").final_found.apply(lambda s: s.astype(float).mean()),
        "final_line_by_recovery_call": df[df["mode"] == "reason"].groupby("system").final_recovered.apply(
            lambda s: s.astype(float).mean()),
        "mean_fact_lines": df.groupby("system").fact_lines.mean(),
    }).reindex(SYS)
    beh.to_csv(os.path.join(OUT, "tables", "behaviour.csv"))
    tier = df.groupby(["answer_model", "system"]).correct.mean().unstack("system")[SYS]
    tier.insert(0, "n_questions", df.groupby("answer_model").question_id.nunique())
    tier.to_csv(os.path.join(OUT, "tables", "accuracy_by_answer_tier.csv"))
    ev = df[df.ev_any.notna()].groupby(["system", "budget"]).ev_any.apply(lambda s: s.astype(float).mean()).unstack("budget")
    fl = df[df.system == "MODEL5"].groupby("budget").fact_lines.mean()
    ct = df.groupby(["system", "budget"]).context_tokens.mean().unstack("budget")
    L += ["", "## 6. Behaviour", "", beh.to_markdown(floatfmt=".3f"), "",
          "Accuracy by answering model (one model unless config answer.split sends a fixed share of questions to "
          "another; all systems of a question use the same model, so paired differences stay within one model):", "", tier.to_markdown(floatfmt=".3f"), "",
          "Evidence recall (any evidence turn in the context) by budget:", "", ev.to_markdown(floatfmt=".3f"), "",
          "MODEL5 fact lines shown by budget: " + ", ".join(f"{b}: {fl[b]:.2f}" for b in budgets) +
          " (Model4: 1.8 / 3.5 / 5.1 / 5.7)", "", "Context tokens by budget:", "", ct.to_markdown(floatfmt=".0f"), ""]

    # 5 cost
    a, b = halves()
    metas = [P.read_json(P.meta_path("v5", q)) for q in a + b]
    last = [m["checkpoints"][-1] for m in metas]
    roles = {}
    for c in last:
        for r, v in c["role_stats"].items():
            for k, x in v.items():
                roles.setdefault(r, {}).setdefault(k, 0)
                roles[r][k] += x / len(last)
    ops = pd.DataFrame([c["ops"] for c in last]).fillna(0).mean()
    sto = pd.DataFrame([c["storage"] for c in last]).mean()
    w2 = load_step("w2")["final_checkpoint_totals"]
    wcalls = np.mean([c["write_cum"]["llm_calls"] for c in last])
    win = np.mean([c["write_cum"]["tokens_in"] for c in last])
    wout = np.mean([c["write_cum"]["tokens_out"] for c in last])
    p50 = np.median([m["write_turn_latency_p50"] for m in metas])
    p95 = np.median([m["write_turn_latency_p95"] for m in metas])
    from chronofact.llm import get_llm
    spend = get_llm().spent()
    kinds = sto[["kind_state", "kind_event", "kind_attribute"]]
    L += ["## 7. Cost", "",
          f"Write path per history (final checkpoint): {wcalls:.0f} LLM calls, {win / 1e3:.0f}k input and "
          f"{wout / 1e3:.0f}k output tokens (Model4: 460 calls, 281k / 51k). Per role: "
          + "; ".join(f"{r}: {v['calls']:.0f} calls, {v['tokens_in'] / 1e3:.0f}k in, {v['tokens_out'] / 1e3:.1f}k out"
                      for r, v in roles.items()) + ".",
          f"Write latency per turn: median of per-history p50 {p50:.0f} ms, of per-history p95 {p95:.0f} ms. "
          f"Mean wall time per history {np.mean([m['wall_seconds'] for m in metas]) / 60:.1f} min (with "
          f"{CFG['resources']['ingest_workers']}x{CFG['resources']['ingest_per_worker']} histories in flight).",
          "Ops per history (final checkpoint): " + ", ".join(f"{k} {v:.1f}" for k, v in ops.items()) +
          ". Model4: " + ", ".join(f"{k} {v}" for k, v in REF14["write"].items()) + ".",
          f"Store per history: {sto.episodes:.0f} episodes, {sto.facts_active:.0f} active, {sto.facts_superseded:.1f} "
          f"superseded, {sto.facts_rejected:.0f} rejected facts; rich statements on "
          f"{sto.facts_rich / (sto.facts_active + sto.facts_superseded):.2f} of facts. Kinds: "
          + ", ".join(f"{k[5:]} {v / kinds.sum():.0%}" for k, v in kinds.items()) + ".",
          f"W2 (final checkpoints, all histories): {w2}.", ""]
    cost = df.groupby("system").agg(answer_tokens_in=("answer_tokens_in", "mean"),
                                    answer_tokens_out=("answer_tokens_out", "mean"),
                                    retrieve_ms_p50=("retrieve_latency_ms", "median"),
                                    retrieve_ms_p95=("retrieve_latency_ms", lambda s: s.quantile(.95)),
                                    answer_ms_p50=("answer_latency_ms", "median"),
                                    answer_ms_p95=("answer_latency_ms", lambda s: s.quantile(.95))).reindex(SYS)
    cost.to_csv(os.path.join(OUT, "tables", "read_cost.csv"))
    L += ["Read side per answered row:", "", cost.to_markdown(floatfmt=".1f"), "",
          f"LLM spend, all calls (profile {CFG['profile']}): ${spend:.2f} of the ${CFG['llm']['spend_cap_usd']:.0f} cap.",
          ""]
    jp = load_step("judge_pilot")
    L += [f"Rows without a valid judge verdict (counted WRONG): {int((df.judge_rule == 'judge_error').sum())}; "
          f"knowledge-update rows without a stale verdict: {int(ku.stale.isna().sum())}.", ""]
    L += [f"Judge pilot: max_tokens {jp['judge_max_tokens']}, hand check {jp['handcheck_wrong_of_50']}/50 verdicts wrong, "
          f"projected ${jp['projected_total_usd']:.2f}.", ""]

    # plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s in SYS:
        x = acc[(acc.system == s) & (acc.budget != "all")]
        bb = x.budget.astype(int)
        ax.errorbar(bb, 100 * x.acc, yerr=[100 * (x.acc - x.lo), 100 * (x.hi - x.acc)], marker="o", capsize=3, label=s)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("context budget (tokens)")
    ax.set_ylabel("accuracy, % (500 questions)")
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plots", "accuracy_vs_budget.png"), dpi=150)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for k, base in enumerate(SYS[1:]):
        x = pdiff[(pdiff.vs == base) & (pdiff.budget != "all")]
        ax.errorbar(np.arange(len(budgets)) + (k - .5) * .1, 100 * x["diff"],
                    yerr=[100 * (x["diff"] - x.lo), 100 * (x.hi - x["diff"])], marker="o", ls="", capsize=3, label=base)
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(range(len(budgets)), [str(b) for b in budgets])
    ax.set_ylabel("MODEL5 - baseline (points)")
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "plots", "model5_minus_baselines.png"), dpi=150)
    with open(os.path.join(OUT, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    P.write_json(os.path.join(OUT, "summary.json"), {"bar_met": bar, "lead_vs_rag_harness_mean_4_budgets": lead_h,
                                                     "lead_vs_rag_plain": dict(plain_all), "spend_usd": spend})
    print("\n".join(L))


if __name__ == "__main__":
    main()
