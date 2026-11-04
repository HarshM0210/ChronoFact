"""W2: strict reconciliation of a finished store (README section 6; helpers from Appendix C.10). No LLM.
src is copied to dst first; src is never changed. Only ACTIVE, NON-EVENT facts are looked at. The newer fact
is the one with the later (recorded_at, valid_from, id); the older becomes superseded with
valid_to = newer.recorded_at and superseded_by = newer.id, logged W2_SUPERSEDE.

Rule (a): same subject and slot key; the newest of the group vs each older one: both values contain a digit and
          differ (case-insensitive), number-free statement skeletons have cosine >= 0.80, NLI contradiction
          >= 0.50 in either direction.
Rule (b): two STATE facts of one subject: skeletons (value, digits, number words removed; letters only) with at
          least 2 content words each, skeleton cosine >= 0.90, different values, slot-key cosine >= 0.92,
          NLI contradiction >= 0.50 in either direction.
Rule (c) (Model5): same subject and the same, non-generic slot key, different values; the newest of the
          group vs each older one; superseded when the judge-role LLM judged that the newer fact replaces the older one
          (`updates`, asked by the caller for the pairs of same_slot_pairs()). Rules a and b go first.
"""
import os
import re
import sqlite3

from chronofact.config import CFG
from chronofact.relations import kind_of
from chronofact.rules import GENERIC_SLOTS

# ---------------------------------------------------------------- README Appendix C.10
_NUMW = r"(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|hundred|dozen|half|once|twice)"

_HASNUM = re.compile(r"\d|\b" + _NUMW + r"\b", re.I)


def _value(f, ents):
    if f["obj_id"] is not None:
        return ents.get(f["obj_id"], "")
    return f["obj_literal"] or ""


def _skeleton(statement, value):
    s = (statement or "").lower()
    if value:
        s = s.replace(value.lower(), " ")
    s = re.sub(r"\d+([.,:/-]\d+)*", " ", s)
    s = re.sub(r"\b" + _NUMW + r"\b", " ", s, flags=re.I)
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", s)).strip()


_WEAK = set("the a an user users their his her its is are was were be been being has have had do does did "
            "of in on at to for from with by and or as it this that they them very really recently "
            "currently new now".split())


def _content(sk):
    return [w for w in sk.split() if w not in _WEAK]


def _key(f):
    return (f["recorded_at"] or "", f["valid_from"] or "", f["id"])


def _active_nonevent(db):
    cols = [r[1] for r in db.execute("PRAGMA table_info(facts)")]
    facts = [dict(zip(cols, r)) for r in db.execute("SELECT * FROM facts WHERE status='active' ORDER BY id")]
    return [f for f in facts if kind_of(f) != "event"]


def _same_slot(facts, ents):
    """rule (c) candidates: [(older, newest)] per (subject, non-generic slot key) with different values"""
    by = {}
    for f in facts:
        if f.get("slot_key") and f["slot_key"] not in GENERIC_SLOTS:
            by.setdefault((f["subj_id"], f["slot_key"]), []).append(f)
    out = []
    for grp in by.values():
        if len(grp) < 2:
            continue
        grp = sorted(grp, key=_key)
        new = grp[-1]
        out += [(o, new) for o in grp[:-1]
                if _value(o, ents).strip().lower() != _value(new, ents).strip().lower()]
    return out


def same_slot_pairs(src):
    """the rule (c) pairs of a store, for the caller to judge: [(older fact, newer fact)] (dicts of the facts table)"""
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    ents = {r[0]: r[1] for r in s.execute("SELECT id, canonical_name FROM entities")}
    out = _same_slot(_active_nonevent(s), ents)
    s.close()
    return out


# ---------------------------------------------------------------- W2
def reconcile(src, dst, encode, contradict, updates=frozenset()):
    """encode(list[str]) -> L2-normalised vectors (context side); contradict(list[(a, b)]) -> probabilities;
    updates: {(older id, newer id)} judged updates among same_slot_pairs(src) (rule c).
    Returns {"superseded", "rule_a", "rule_b", "rule_c", "active_before"}."""
    c = CFG["w2"]
    tmp = dst + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    s = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    d = sqlite3.connect(tmp)
    s.backup(d)
    s.close()
    d.execute("PRAGMA journal_mode=DELETE")
    facts = _active_nonevent(d)
    ents = {r[0]: r[1] for r in d.execute("SELECT id, canonical_name FROM entities")}
    sup = {}                                     # older fact id -> (newer fact, rule)

    # rule (a)
    by_slot = {}
    for f in facts:
        if f.get("slot_key"):
            by_slot.setdefault((f["subj_id"], f["slot_key"]), []).append(f)
    pairs_a = []
    for grp in by_slot.values():
        if len(grp) < 2:
            continue
        grp = sorted(grp, key=_key)
        newest = grp[-1]
        for f in grp[:-1]:
            v_old, v_new = _value(f, ents), _value(newest, ents)
            if v_old.strip().lower() == v_new.strip().lower():
                continue
            if re.search(r"\d", v_old) and re.search(r"\d", v_new):
                pairs_a.append((f, newest))
    if pairs_a:
        va = encode([_skeleton(o["statement"], "") or "none" for o, _ in pairs_a])
        vb = encode([_skeleton(n["statement"], "") or "none" for _, n in pairs_a])
        pairs_a = [p for p, sim in zip(pairs_a, (va * vb).sum(1)) if sim >= c["num_sim"]]
    if pairs_a:
        c1 = contradict([(o["statement"] or "", n["statement"] or "") for o, n in pairs_a])
        c2 = contradict([(n["statement"] or "", o["statement"] or "") for o, n in pairs_a])
        for (o, n), x, y in zip(pairs_a, c1, c2):
            if max(x, y) >= c["contradiction"]:
                sup.setdefault(o["id"], (n, "a"))

    # rule (b)
    states = [f for f in facts if kind_of(f) == "state" and f["id"] not in sup]
    by_subj = {}
    for f in states:
        by_subj.setdefault(f["subj_id"], []).append(f)
    for grp in by_subj.values():
        if len(grp) < 2:
            continue
        sk = [_skeleton(f.get("statement"), _value(f, ents)) for f in grp]
        ok = [i for i, x in enumerate(sk) if len(_content(x)) >= 2]
        if len(ok) < 2:
            continue
        v = encode([sk[i] for i in ok])
        sims = v @ v.T
        pairs = []
        for a in range(len(ok)):
            for b in range(a + 1, len(ok)):
                if sims[a, b] < c["skel_sim"]:
                    continue
                fa, fb = grp[ok[a]], grp[ok[b]]
                if _value(fa, ents).strip().lower() == _value(fb, ents).strip().lower():
                    continue
                pairs.append((fa, fb) if _key(fa) < _key(fb) else (fb, fa))
        if not pairs:
            continue
        a_v = encode([(p[0].get("slot_key") or "").replace("_", " ") or "none" for p in pairs])
        b_v = encode([(p[1].get("slot_key") or "").replace("_", " ") or "none" for p in pairs])
        near = [p for p, sim in zip(pairs, (a_v * b_v).sum(1)) if sim >= c["slot_sim"]]
        if not near:
            continue
        c1 = contradict([(o["statement"] or "", n["statement"] or "") for o, n in near])
        c2 = contradict([(n["statement"] or "", o["statement"] or "") for o, n in near])
        for (old, new), x, y in zip(near, c1, c2):
            if max(x, y) >= c["contradiction"] and old["id"] not in sup:
                sup[old["id"]] = (new, "b")

    # rule (c)
    for old, new in _same_slot(facts, ents):
        if (old["id"], new["id"]) in updates and old["id"] not in sup:
            sup[old["id"]] = (new, "c")

    for fid, (new, rule) in sorted(sup.items()):
        d.execute("UPDATE facts SET status='superseded', valid_to=?, superseded_by=? WHERE id=?",
                  (new["recorded_at"], new["id"], fid))
        d.execute("INSERT INTO ops_log (user_id, ts, op, fact_id, reason, model, tokens_in, tokens_out, latency_ms) "
                  "VALUES (?,?,?,?,?,?,?,?,?)", (new["user_id"], new["recorded_at"], "W2_SUPERSEDE", fid,
                                                 f"rule {rule}: replaced by fact {new['id']}", "", 0, 0, 0.0))
    d.commit()
    d.close()
    os.replace(tmp, dst)
    return {"superseded": len(sup), "rule_a": sum(1 for _, r in sup.values() if r == "a"),
            "rule_b": sum(1 for _, r in sup.values() if r == "b"), "rule_c": sum(1 for _, r in sup.values() if r == "c"),
            "active_before": len(facts)}
