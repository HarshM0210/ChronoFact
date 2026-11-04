"""Write-threshold calibration for THIS model stack (Muse Spark extractions, Contriever cosines, BART-MNLI), from
the pilot stores (results/snapshots/v5/<pilot qid>/ck4.sqlite) -> results/calibration.json.

The README thresholds were set for Model4's LLM and embedder; their cosine scale does not carry over to Contriever.
For each threshold, candidate pairs are binned by score, up to N_PER_BIN per bin are sampled (seed 0) and labeled by
Spark (judge role, standard tier, temperature 0). The threshold is the lower edge of the lowest bin from which every
bin up to the top has precision >= that threshold's bar (bins with fewer than MIN_BIN labels are not counted).
The bar weighs what an error costs against what a miss costs:
  entity_merge_sim  0.80  two entity names of one history: same real-world entity? A false merge only re-points an
                          entity id (fact lines show the fact's own statement); a missed merge splits an entity.
  near_dup_high     0.90  two values of one subject + family: the same value? A false NOOP drops an update.
  min_entail        0.70  a verified-quote fact: is the statement supported by its message? A rejected fact is lost
                          from memory, a kept unsupported one can mislead; Spark's label ("fully supported, no added
                          detail") is strict, so a bin is kept while at least 70% of its facts are supported.

Run with a profile whose judge is Muse Spark, after the pilot of a Spark-judged profile:  CF_PROFILE=hybrid python scripts/calibration/calibrate.py
"""
import asyncio
import json
import os
import random
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np  # noqa: E402

from chronofact import rules as RU  # noqa: E402
from chronofact.config import rpath  # noqa: E402
from chronofact.data import halves  # noqa: E402
from chronofact.llm import RUN_TAG, LLMJSONError, acall, get_llm  # noqa: E402
from chronofact.models import embedder, nli  # noqa: E402
from chronofact.pipeline import snap_dir, write_json  # noqa: E402

N_PER_BIN, MIN_BIN = 30, 8
BAR = {"entity_merge_sim": 0.80, "near_dup_high": 0.90, "min_entail": 0.70}
COS_EDGES = [round(x, 2) for x in np.arange(0.50, 1.0001, 0.05)]
P_EDGES = [round(x, 2) for x in np.arange(0.0, 1.0001, 0.1)]
SCHEMA = {"type": "object", "properties": {"reason": {"type": "string"}, "yes": {"type": "boolean"}},
          "required": ["reason", "yes"]}
Q_ENT = ('Two names from one person\'s conversations with an AI assistant. Do they refer to the SAME real-world '
         'thing (the same person, pet, place, organization, product, ...)? A variant spelling or a longer/shorter '
         'form of the same name counts as the same; two different things of the same kind do not.\nA: {a}\nB: {b}\n'
         'Return JSON {{"reason": "<one short sentence>", "yes": true | false}}')
Q_VAL = ('Two recorded values of the same attribute of a person (category: {fam}). Do they state the SAME value '
         '(a reworded or shorter form counts as the same; a different amount, date, name or item does not)?\n'
         'A: {a}\nB: {b}\nReturn JSON {{"reason": "<one short sentence>", "yes": true | false}}')
Q_ENT_FACT = ('A user wrote this message to an AI assistant:\n"""{m}"""\n\nIs this statement about the user fully '
              'supported by the message (stated or directly implied, with no added detail)?\nStatement: {s}\n'
              'Return JSON {{"reason": "<one short sentence>", "yes": true | false}}')


def stores():
    a, b = halves()
    for q in a[:15] + b[:15]:
        p = os.path.join(snap_dir("v5", q), "ck4.sqlite")
        if os.path.exists(p):
            yield q, sqlite3.connect(f"file:{p}?mode=ro", uri=True)


def bin_of(x, edges):
    for lo, hi in zip(edges[:-1], edges[1:]):
        if lo <= x < hi or (hi == edges[-1] and x == hi):
            return lo
    return None


def sample(items, edges, rng):
    """items: [(score, payload)] -> {bin lower edge: [items]}"""
    by = {}
    for it in items:
        b = bin_of(it[0], edges)
        if b is not None:
            by.setdefault(b, []).append(it)
    return {b: rng.sample(v, min(N_PER_BIN, len(v))) for b, v in sorted(by.items())}


async def label(prompts):
    async def one(p):
        try:
            out, _ = await acall("judge", [{"role": "user", "content": p}], SCHEMA)
            return bool(out.get("yes")) if isinstance(out, dict) and isinstance(out.get("yes"), bool) else None
        except LLMJSONError:
            return None
    return await asyncio.gather(*[one(p) for p in prompts])


def choose(labels, bar, default):
    """lowest lower edge from which every counted bin up to the top has precision >= bar (no counted bin: default)"""
    table, chosen, going = [], None, True
    for b in sorted(labels, reverse=True):
        ys = [y for y in labels[b] if y is not None]
        prec = float(np.mean(ys)) if ys else None
        counted = len(ys) >= MIN_BIN
        table.append({"bin": b, "n": len(ys), "precision": prec, "counted": counted})
        if going and counted:
            if prec >= bar:
                chosen = b
            else:
                going = False
    return (chosen if chosen is not None else default), sorted(table, key=lambda r: r["bin"])


def main():
    rng = random.Random(0)
    emb = embedder()
    ents, vals, facts = [], [], []
    for q, d in stores():
        names = []
        for name, aliases in d.execute("SELECT canonical_name, aliases FROM entities WHERE canonical_name != 'user'"):
            names.append(name)
            for al in json.loads(aliases or "[]"):            # pairs the write path merged at the current threshold
                ents.append((name, al))
        if len(names) > 1:
            v = emb.encode(names)
            s = v @ v.T
            iu = np.triu_indices(len(names), 1)
            for i, j in zip(*iu):
                if s[i, j] >= COS_EDGES[0]:
                    ents.append((names[i], names[j]))
        en = {r[0]: r[1] for r in d.execute("SELECT id, canonical_name FROM entities")}
        rows = d.execute("SELECT subj_id, family, obj_id, obj_literal FROM facts WHERE status IN ('active','superseded')").fetchall()
        grp = {}
        for subj, fam, oid, lit in rows:
            val = en.get(oid, "") if oid is not None else (lit or "")
            if val:
                grp.setdefault((subj, fam), set()).add(val)
        for (_, fam), vs in grp.items():
            vs = sorted(vs)
            for i in range(len(vs)):
                for j in range(i + 1, len(vs)):
                    a, b = vs[i], vs[j]
                    if a.strip().lower() != b.strip().lower() and sorted(RU._digits(a)) == sorted(RU._digits(b)):
                        vals.append((fam, a, b))           # the pairs _same_value decides by cosine
        ep = {r[0]: r[1] for r in d.execute("SELECT id, text FROM episodes")}
        for quote, st, src in d.execute("SELECT quote, statement, source_episode_ids FROM facts"):
            t = ep.get(json.loads(src or "[]")[0]) if src else None
            if t and st and quote and RU.quote_ok(quote, t):
                facts.append((t, quote, st))
        d.close()
    ents = sorted(set(ents))
    ev = emb.encode([a for a, _ in ents] + [b for _, b in ents])
    n = len(ents)
    ent_items = [(float(ev[k] @ ev[n + k]), p) for k, p in enumerate(ents)]
    vv = emb.encode([a for _, a, _ in vals] + [b for _, _, b in vals])
    m = len(vals)
    val_items = [(float(vv[k] @ vv[m + k]), p) for k, p in enumerate(vals)]
    fp = nli().entail([(RU.third_person(RU.premise_window(qt, t)), st) for t, qt, st in facts])
    fact_items = [(float(p), x) for p, x in zip(fp, facts)]

    S_ent, S_val, S_fact = sample(ent_items, COS_EDGES, rng), sample(val_items, COS_EDGES, rng), sample(fact_items, P_EDGES, rng)
    jobs = [("ent", b, Q_ENT.format(a=p[0], b=p[1])) for b, its in S_ent.items() for _, p in its]
    jobs += [("val", b, Q_VAL.format(fam=p[0].replace("_", " "), a=p[1], b=p[2])) for b, its in S_val.items() for _, p in its]
    jobs += [("fact", b, Q_ENT_FACT.format(m=RU._clip(p[0], 400), s=p[2])) for b, its in S_fact.items() for _, p in its]
    print(f"[calibrate] candidates: entities {len(ent_items)}, values {len(val_items)}, facts {len(fact_items)}; "
          f"labeling {len(jobs)}", flush=True)
    RUN_TAG.set("calibrate")
    ys = asyncio.run(label([j[2] for j in jobs]))
    get_llm().flush()
    lab = {"ent": {}, "val": {}, "fact": {}}
    for (kind, b, _), y in zip(jobs, ys):
        lab[kind].setdefault(b, []).append(y)
    out = {}
    for kind, name, edges, default in (("ent", "entity_merge_sim", COS_EDGES, 0.85), ("val", "near_dup_high", COS_EDGES, 0.90),
                                       ("fact", "min_entail", P_EDGES, 0.6)):
        th, table = choose(lab[kind], BAR[name], default)
        out[name] = {"chosen": th, "bar": BAR[name], "bins": table}
        print(f"[calibrate] {name}: {th}  " + "  ".join(f"{r['bin']}:{r['precision'] if r['precision'] is None else round(r['precision'], 2)}"
                                                         f"/{r['n']}" for r in table), flush=True)
    examples = {k: [(round(s, 3), p, y) for (s, p), y in zip([it for b in sorted(S) for it in S[b]],
                                                               [y for (kk, _, _), y in zip(jobs, ys) if kk == k])][:400]
                for k, S in (("ent", S_ent), ("val", S_val))}
    write_json(rpath("calibration.json"), {"rule": f"per-bin precision >= bar {BAR} from the chosen bin up; "
                                                   f"{N_PER_BIN} labels per bin; bins with < {MIN_BIN} labels not counted",
                                           "thresholds": out, "examples": examples,
                                           "spend_usd_after": get_llm().spent()})


if __name__ == "__main__":
    main()
