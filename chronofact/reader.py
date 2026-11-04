"""ChronoFact read path (README section 7) and the RAG baseline (README section 9). No LLM call.

retrieve(question, as_of, budget, system, tau, visible_until):
  MODEL5       facts (gate tau) + turns, skip packing
  RAG_plain    turns only, stop packing
One Retriever is kept per opened snapshot, so the 4 budgets x 2 systems of a checkpoint share the store
connection, the indexes and the per-question work (README 7, last paragraph).
"""
from collections import defaultdict

import numpy as np

from chronofact import relations as R
from chronofact.config import CFG
from chronofact.models import tokens
from chronofact.rules import _PLAN
from chronofact.store import norm_name
from chronofact.timeparse import question_mode

FACT_HEADER = "Known facts about the user (with validity dates):"
TURN_HEADER = "Conversation excerpts (oldest first):"

# README Appendix C.8
_QSTOP = set("what which who whom when where why how many much did do does is are was were the a an "
             "my i me to of in on at for with and or have has had you your it its this that "
             "can could would should will been being about from by as than then there".split())


def rrf(lists, k):
    score = defaultdict(float)
    for lst in lists:
        for rank, (eid, _) in enumerate(lst, start=1):
            score[eid] += 1.0 / (k + rank)
    return sorted(score.items(), key=lambda x: -x[1])


def time_filter(facts, mode, t_start=None, t_end=None, as_of=None):
    """README Appendix C.6 (as_of = visible_until here)."""
    out = []
    for f in facts:
        if f["status"] in ("rejected", "forgotten"):
            continue
        if mode in ("current", "none"):
            if f["status"] == "active":
                out.append(f)
        elif mode == "as_of":
            vf = f["valid_from"] or ""
            vt = f["valid_to"] or "9999"
            if vf < t_end.isoformat() and vt > t_start.isoformat():
                out.append(f)
        else:
            out.append(f)
    if as_of is not None:
        cut = as_of.isoformat()
        out = [f for f in out if (f["recorded_at"] or "") <= cut]
    return out


def turn_line(e):
    return f"[{e['ts'][:10]}, {e['speaker']}] {e['text']}"


class Retriever:
    def __init__(self, store, uid):
        self.store, self.uid = store, uid
        self.c = CFG["read"]
        self._rich = None
        self._q = {}            # per-question work shared by budgets and systems

    def rich(self):
        if self._rich is None:
            self._rich = self.store.rich_map()
        return self._rich

    # ------------------------------------------------------------ question entities (README C.8 _seeds)
    def seeds(self, question):
        names = self.store.entity_names(self.uid)
        user_eid = names.get("user")
        toks = norm_name(question).split()
        found = {}
        grams = []
        for n in (4, 3, 2, 1):
            for i in range(len(toks) - n + 1):
                g = " ".join(toks[i:i + n])
                if n == 1 and (g in _QSTOP or len(g) < 3):
                    continue
                grams.append(g)
                if g in names and names[g] != user_eid:
                    found[names[g]] = 1.0
        content = [g for g in grams if not all(w in _QSTOP for w in g.split())][: self.c["entity_ngrams"]]
        if content:
            for v in self.store.emb.encode(content):
                for eid, _, sim in self.store.similar_entities(self.uid, v, 3):
                    if sim >= self.c["entity_seed_sim"] and eid != user_eid:
                        found.setdefault(eid, sim)
        return set(found)

    # ------------------------------------------------------------ shared per question
    def _prepare(self, question, as_of, vis):
        key = (question, as_of, vis)
        if key in self._q:
            return self._q[key]
        st, uid = self.store, self.uid
        mode, ts, te = question_mode(question, as_of)                                   # 1
        qvec = st.emb.encode([question], is_query=True)[0]                              # 3
        idx = st.episode_index(uid)                                                     # 4
        fused = rrf([idx.bm25(question, self.c["k"]), idx.dense(qvec, self.c["k"])], self.c["rrf_k"])
        eps = st.episodes(uid, [e for e, _ in fused])
        cut = vis.isoformat()
        cands = [eps[e] for e, _ in fused if e in eps and eps[e]["text"] and eps[e]["ts"] <= cut]
        cands = cands[: self.c["candidate_turns"]]
        self._q[key] = {"mode": mode, "ts": ts, "te": te, "qvec": qvec, "cands": cands}
        return self._q[key]

    def _fact_part(self, question, vis, P, tau):
        """README 7 steps 2, 5, 6 -> (ranked facts [(fact, text)], gate-pass fact ids, by_slot, ents, slots)"""
        key = ("facts", question, vis, tau)
        if key in self._q:
            return self._q[key]
        st, uid = self.store, self.uid
        all_facts = st.facts(uid, statuses=("active", "superseded"))
        facts = time_filter(all_facts, P["mode"], P["ts"], P["te"], vis)               # 2
        ents = st.entity_rows(uid)
        slots = R.question_slots(question)
        by_slot = defaultdict(list)
        cut = vis.isoformat()
        for f in all_facts:
            if f["status"] == "superseded" and (f["recorded_at"] or "") <= cut:
                by_slot[R.slot_of(f)].append(f)
        pool = facts
        if not _PLAN.search(question):                                                  # 6: intentions
            pool = [f for f in pool if not R.is_intention(f)]
        out = {"ranked": [], "gate": [], "by_slot": by_slot, "ents": ents, "slots": slots, "n_eligible": len(facts)}
        if pool:
            rich = self.rich()
            texts = [rich.get(f["id"]) or f["statement"] or "" for f in pool]
            vecs = st.emb.encode(texts)
            cos = vecs @ P["qvec"]
            seeds = self.seeds(question) if pool else set()                             # 5
            topic = [any(R.matches_slot(f["relation"] or "", s) for s in slots) for f in pool]
            keep = [i for i in range(len(pool))
                    if cos[i] >= tau or pool[i]["subj_id"] in seeds or pool[i]["obj_id"] in seeds or topic[i]]
            out["gate"] = [pool[i]["id"] for i in keep]
            score = np.array([cos[i] + (self.c["w_topic"] if topic[i] else 0.0) for i in keep], dtype=np.float64)
            ranked = [keep[j] for j in np.argsort(-score, kind="stable")]
            uniq = []
            for i in ranked:                                                            # near-duplicates
                if not any(float(vecs[i] @ vecs[j]) >= self.c["fact_dup"] for j in uniq):
                    uniq.append(i)
            out["ranked"] = [(pool[i], texts[i]) for i in uniq[: self.c["max_facts"]]]
        self._q[key] = out
        return out

    # ------------------------------------------------------------ lines
    @staticmethod
    def fact_line(f, text, ents, mode, slots, by_slot):
        """README Appendix C.9 _fact_line with dates=False, conflict_display=False, history_lines='v1'."""
        sent = text.rstrip(".")
        vf = (f["valid_from"] or "")[:10] or "?"
        if f["status"] == "active":
            line = f"- {sent} (since {vf}"
        else:
            line = f"- {sent} ({vf} to {(f['valid_to'] or '?')[:10]}, no longer current"
        show_hist = mode in ("history", "as_of") or any(R.matches_slot(f["relation"] or "", s) for s in slots)
        prev = []
        if f["status"] == "active" and f["slot_type"] == "single":
            prev = sorted(by_slot.get(R.slot_of(f), []), key=lambda x: x["valid_from"] or "", reverse=True)[:3]
        if show_hist:
            for p in prev:
                obj = ents.get(p["obj_id"], {}).get("name") if p["obj_id"] is not None else p["obj_literal"]
                line += f"; previously {obj}, {(p['valid_from'] or '?')[:10]} to {(p['valid_to'] or '?')[:10]}"
        return line + ")"

    def _pack_turns(self, cands, budget, used, packer):
        """README 7.8 (skip) and 9 (stop: cut the first misfit to the tokens left if >= 32, append ' ...', stop)."""
        tc = tokens()
        total = used + tc.count(TURN_HEADER) + 1
        chosen = []
        for e in cands:
            line = turn_line(e)
            n = tc.count(line) + 1
            if total + n <= budget:
                chosen.append((e, line))
                total += n
                continue
            if packer == "stop":
                left = budget - total - 1
                if left >= self.c["stop_min_tokens"]:
                    line = tc.truncate(line, left) + " ..."
                    chosen.append((e, line))
                    total += tc.count(line) + 1
                break
        if not chosen:
            return [], [], used
        chosen.sort(key=lambda x: (x[0]["ts"], x[0]["id"]))
        items = [{"id": e["id"], "session_id": e["session_id"], "turn_idx": e["turn_idx"], "ts": e["ts"]}
                 for e, _ in chosen]
        return [TURN_HEADER] + [l for _, l in chosen], items, total

    # ------------------------------------------------------------ retrieve
    def retrieve(self, question, as_of, budget, system, tau=None, visible_until=None):
        vis = visible_until or as_of
        P = self._prepare(question, as_of, vis)
        tc = tokens()
        flines, fshown, f_used, gate_ids = [], [], 0, []
        if system == "MODEL5":
            if tau is None:
                raise ValueError("MODEL5 needs the fold's gate tau")
            F = self._fact_part(question, vis, P, tau)
            gate_ids = F["gate"]
            cap = int(self.c["facts_budget_frac"] * budget)                            # 7
            f_used = tc.count(FACT_HEADER) + 1
            for f, text in F["ranked"]:
                line = self.fact_line(f, text, F["ents"], P["mode"], F["slots"], F["by_slot"])
                n = tc.count(line) + 1
                if f_used + n > cap:
                    continue
                flines.append(line)
                fshown.append(f)
                f_used += n
            if not flines:
                f_used = 0
        if system not in ("MODEL5", "RAG_plain"):
            raise ValueError(f"unknown system {system!r}")
        packer = "stop" if system == "RAG_plain" else "skip"
        elines, eitems, used = self._pack_turns(P["cands"], budget, f_used, packer)       # 8
        lines = ([FACT_HEADER] + flines + [""]) if flines else []                         # 9
        lines += elines
        return {"text": "\n".join(lines).strip(), "tokens": used, "mode": P["mode"], "fact_lines": len(flines),
                "facts": [{"id": f["id"], "status": f["status"], "kind": f["kind"],
                           "src": f["source_episode_ids"]} for f in fshown],
                "turns": eitems, "gate": gate_ids, "tau": tau}
