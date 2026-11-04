"""One SQLite file per history (WAL), README section 5, plus the in-memory per-user indexes:
turn vectors (FAISS IndexFlatIP), BM25 over turn tokens, and an entity index over canonical names (aliases are
only used for exact matching). The indexes are never saved: reopening a snapshot rebuilds them from the
tables (episodes with non-empty text in id order; entities in id order), using the embedding cache.
Every query filters on user_id. Fact lists are returned in id order (Model4 relied on SQLite's implicit order)."""
import hashlib
import json
import os
import re
import sqlite3
from datetime import datetime

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from chronofact.models import embedder

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL,
    session_id TEXT, turn_idx INTEGER, speaker TEXT, text TEXT, ts TEXT, content_hash TEXT UNIQUE);
CREATE INDEX IF NOT EXISTS ep_user ON episodes(user_id);
CREATE TABLE IF NOT EXISTS entities (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL,
    canonical_name TEXT, type TEXT, aliases TEXT);
CREATE INDEX IF NOT EXISTS ent_user ON entities(user_id);
CREATE TABLE IF NOT EXISTS facts (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL,
    subj_id INTEGER, relation TEXT, obj_id INTEGER, obj_literal TEXT, slot_type TEXT,
    valid_from TEXT, valid_to TEXT, recorded_at TEXT, status TEXT, confidence REAL, quote TEXT,
    source_episode_ids TEXT, superseded_by INTEGER, kind TEXT, family TEXT, slot_key TEXT,
    assertion_type TEXT, statement TEXT, conflict_with INTEGER, date_precision TEXT);
CREATE INDEX IF NOT EXISTS fact_user ON facts(user_id, subj_id, relation, status);
CREATE TABLE IF NOT EXISTS ops_log (id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, ts TEXT,
    op TEXT, fact_id INTEGER, reason TEXT, model TEXT, tokens_in INTEGER, tokens_out INTEGER,
    latency_ms REAL);
CREATE TABLE IF NOT EXISTS fact_rich (fact_id INTEGER PRIMARY KEY, rich TEXT);
"""

# README Appendix C.7
_STOP = set("a an the and or but if of to in on at for with by from as is are was were be been "
            "being i me my we our you your it its this that these those do does did have has had "
            "what which who whom when where why how can could would should will shall may might "
            "there their they them he she his her so not no yes just about into than then also "
            "am".split())


def bm25_tokens(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


def norm_name(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())).strip()


def iso(dt):
    return dt.isoformat(timespec="seconds") if isinstance(dt, datetime) else dt


FACT_COLS = ["id", "user_id", "subj_id", "relation", "obj_id", "obj_literal", "slot_type", "valid_from", "valid_to",
             "recorded_at", "status", "confidence", "quote", "source_episode_ids", "superseded_by", "kind", "family",
             "slot_key", "assertion_type", "statement", "conflict_with", "date_precision"]


class EpisodeIndex:
    def __init__(self, dim):
        self.ids, self.texts, self.ts, self.speakers, self.toks = [], [], [], [], []
        self.faiss = faiss.IndexFlatIP(dim)
        self._bm25 = None

    def add(self, ids, texts, ts, speakers, vecs):
        self.ids += ids
        self.texts += texts
        self.ts += ts
        self.speakers += speakers
        self.toks += [bm25_tokens(t) for t in texts]
        if len(ids):
            self.faiss.add(np.ascontiguousarray(vecs, dtype=np.float32))
        self._bm25 = None

    def bm25(self, query, k):
        """top k by BM25Okapi (default parameters), positive scores only; empty token list -> ["_"]"""
        if not self.ids:
            return []
        if self._bm25 is None:
            self._bm25 = BM25Okapi([t or ["_"] for t in self.toks])
        s = self._bm25.get_scores(bm25_tokens(query))
        top = np.argsort(-s, kind="stable")[:k]
        return [(self.ids[i], float(s[i])) for i in top if s[i] > 0]

    def dense(self, qvec, k):
        if not self.ids:
            return []
        D, I = self.faiss.search(np.ascontiguousarray(qvec.reshape(1, -1), dtype=np.float32), min(k, len(self.ids)))
        return [(self.ids[i], float(d)) for d, i in zip(D[0], I[0]) if i >= 0]


class EntityIndex:
    def __init__(self, dim):
        self.ids, self.types = [], []
        self.faiss = faiss.IndexFlatIP(dim)

    def add(self, eid, etype, vec):
        self.ids.append(eid)
        self.types.append(etype)
        self.faiss.add(np.ascontiguousarray(vec.reshape(1, -1), dtype=np.float32))

    def search(self, qvec, k):
        if not self.ids:
            return []
        D, I = self.faiss.search(np.ascontiguousarray(qvec.reshape(1, -1), dtype=np.float32), min(k, len(self.ids)))
        return [(self.ids[i], self.types[i], float(d)) for d, i in zip(D[0], I[0]) if i >= 0]


class Store:
    def __init__(self, db_path, readonly=False):
        self.path = db_path
        if readonly:
            self.db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
        else:
            os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
            self.db = sqlite3.connect(db_path, timeout=60, check_same_thread=False)
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.execute("PRAGMA synchronous=NORMAL")
            self.db.executescript(SCHEMA)
            self.db.commit()
        self.emb = embedder()
        self.ep, self.ent, self.names = {}, {}, {}

    def _load_user(self, uid):
        if uid in self.ep:
            return
        self.ep[uid] = EpisodeIndex(self.emb.dim)
        rows = self.db.execute("SELECT id, text, ts, speaker FROM episodes WHERE user_id=? AND text != '' "
                               "ORDER BY id", (uid,)).fetchall()
        if rows:
            self.ep[uid].add([r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows],
                             self.emb.encode([r[1] for r in rows]))
        self.ent[uid] = EntityIndex(self.emb.dim)
        self.names[uid] = {}
        rows = self.db.execute("SELECT id, canonical_name, type, aliases FROM entities WHERE user_id=? ORDER BY id",
                               (uid,)).fetchall()
        if rows:
            for (eid, name, etype, aliases), v in zip(rows, self.emb.encode([r[1] for r in rows])):
                self.ent[uid].add(eid, etype, v)
                for a in [name] + json.loads(aliases or "[]"):
                    self.names[uid][norm_name(a)] = eid

    # ---------------------------------------------------------------- episodes
    @staticmethod
    def content_hash(t):
        return hashlib.sha256(f"{t.user_id}|{t.session_id}|{t.turn_idx}|{t.speaker}|{t.text}".encode()).hexdigest()

    def add_episodes(self, turns):
        """Insert new turns (a repeated content hash is not stored twice), embed them (context side), add them
        to BM25 and the dense index. Returns the episode id of every turn (the existing id for a repeat)."""
        if not turns:
            return []
        uid = turns[0].user_id
        self._load_user(uid)
        ids, new = [], []
        for t in turns:
            h = self.content_hash(t)
            row = self.db.execute("SELECT id FROM episodes WHERE content_hash=?", (h,)).fetchone()
            if row:
                ids.append(row[0])
                continue
            cur = self.db.execute("INSERT INTO episodes (user_id, session_id, turn_idx, speaker, text, ts, "
                                  "content_hash) VALUES (?,?,?,?,?,?,?)",
                                  (t.user_id, t.session_id, t.turn_idx, t.speaker, t.text, iso(t.ts), h))
            ids.append(cur.lastrowid)
            new.append((cur.lastrowid, t))
        self.db.commit()
        new = [(i, t) for i, t in new if t.text != ""]          # same rule as the rebuild: non-empty text
        if new:
            self.ep[uid].add([i for i, _ in new], [t.text for _, t in new], [iso(t.ts) for _, t in new],
                             [t.speaker for _, t in new], self.emb.encode([t.text for _, t in new]))
        return ids

    def episodes(self, uid, ids):
        if not ids:
            return {}
        out = {}
        ids = list(ids)
        for j in range(0, len(ids), 900):
            ch = ids[j:j + 900]
            for r in self.db.execute(f"SELECT id, session_id, speaker, text, ts, turn_idx FROM episodes WHERE "
                                     f"user_id=? AND id IN ({','.join('?' * len(ch))})", (uid, *ch)):
                out[r[0]] = {"id": r[0], "session_id": r[1], "speaker": r[2], "text": r[3], "ts": r[4],
                             "turn_idx": r[5]}
        return out

    def episode_index(self, uid) -> EpisodeIndex:
        self._load_user(uid)
        return self.ep[uid]

    def snapshot(self, path):
        """Consistent copy of the whole store (SQLite backup API), written atomically."""
        self.db.commit()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        if os.path.exists(tmp):
            os.remove(tmp)
        dst = sqlite3.connect(tmp)
        self.db.backup(dst)
        dst.execute("PRAGMA journal_mode=DELETE")       # a snapshot is one self-contained file (no -wal/-shm)
        dst.close()
        os.replace(tmp, path)

    # ---------------------------------------------------------------- entities
    def find_entity_exact(self, uid, name):
        self._load_user(uid)
        return self.names[uid].get(norm_name(name))

    def similar_entities(self, uid, vec, k=5):
        self._load_user(uid)
        return self.ent[uid].search(vec, k)

    def add_entity(self, uid, name, etype, vec):
        self._load_user(uid)
        eid = self.db.execute("INSERT INTO entities (user_id, canonical_name, type, aliases) VALUES (?,?,?,?)",
                              (uid, name, etype, "[]")).lastrowid
        self.ent[uid].add(eid, etype, vec)
        self.names[uid][norm_name(name)] = eid
        return eid

    def add_alias(self, uid, eid, alias):
        row = self.db.execute("SELECT aliases FROM entities WHERE id=? AND user_id=?", (eid, uid)).fetchone()
        al = json.loads(row[0] or "[]")
        if alias not in al:
            al.append(alias)
            self.db.execute("UPDATE entities SET aliases=? WHERE id=? AND user_id=?", (json.dumps(al), eid, uid))
        self.names[uid][norm_name(alias)] = eid

    def entity_names(self, uid):
        self._load_user(uid)
        return self.names[uid]

    def entity_rows(self, uid):
        return {r[0]: {"id": r[0], "name": r[1], "type": r[2]} for r in self.db.execute(
            "SELECT id, canonical_name, type FROM entities WHERE user_id=? ORDER BY id", (uid,))}

    # ---------------------------------------------------------------- facts
    @staticmethod
    def _fact(row):
        f = dict(zip(FACT_COLS, row))
        f["source_episode_ids"] = json.loads(f["source_episode_ids"] or "[]")
        return f

    def add_fact(self, uid, **kw):
        cols = [c for c in FACT_COLS if c not in ("id", "user_id")]
        vals = [kw.get(c) for c in cols]
        vals[cols.index("source_episode_ids")] = json.dumps(kw.get("source_episode_ids", []))
        return self.db.execute(f"INSERT INTO facts (user_id, {','.join(cols)}) VALUES (?{',?' * len(cols)})",
                               (uid, *vals)).lastrowid

    def update_fact(self, uid, fid, **kw):
        if "source_episode_ids" in kw:
            kw["source_episode_ids"] = json.dumps(kw["source_episode_ids"])
        sets = ",".join(f"{k}=?" for k in kw)
        self.db.execute(f"UPDATE facts SET {sets} WHERE id=? AND user_id=?", (*kw.values(), fid, uid))

    def set_rich(self, fid, rich):
        self.db.execute("INSERT OR REPLACE INTO fact_rich VALUES (?,?)", (fid, rich))

    def forget(self, uid, fact_id, ts=None):
        """Right to be forgotten: deletes a fact, every other fact taken from the same messages, the text of those
        messages and of the assistant's reply to each of them (a reply can repeat what the user said). The episode
        rows stay as tombstones with empty text, so ids and provenance links remain valid; the in-memory indexes are
        rebuilt without them. Returns (forgotten fact ids, deleted episode ids)."""
        row = self.db.execute("SELECT source_episode_ids, statement FROM facts WHERE id=? AND user_id=?",
                              (fact_id, uid)).fetchone()
        if not row:
            raise KeyError(f"no fact {fact_id} for user {uid}")
        eps = set(json.loads(row[0] or "[]"))
        gone = [r[0] for r in self.db.execute("SELECT id, source_episode_ids FROM facts WHERE user_id=? AND "
                                              "status != 'forgotten'", (uid,)) if eps & set(json.loads(r[1] or "[]"))]
        for fid in gone:
            self.update_fact(uid, fid, status="forgotten")
        for e in list(eps):
            r = self.db.execute("SELECT session_id, turn_idx FROM episodes WHERE id=? AND user_id=?", (e, uid)).fetchone()
            if r:
                nxt = self.db.execute("SELECT id FROM episodes WHERE user_id=? AND session_id=? AND turn_idx=? AND "
                                      "speaker='assistant'", (uid, r[0], r[1] + 1)).fetchone()
                if nxt:
                    eps.add(nxt[0])
        for e in eps:
            self.db.execute("UPDATE episodes SET text='' WHERE id=? AND user_id=?", (e, uid))
        self.log(uid, "FORGET", fact_id, f"forgot: {row[1]} ({len(gone)} facts, {len(eps)} messages deleted)", ts)
        self.commit()
        for d in (self.ep, self.ent, self.names):
            d.pop(uid, None)
        return gone, sorted(eps)

    def rich_map(self):
        return dict(self.db.execute("SELECT fact_id, rich FROM fact_rich"))

    def facts(self, uid, subj_id=None, family=None, statuses=("active",)):
        q, args = f"SELECT {','.join(FACT_COLS)} FROM facts WHERE user_id=?", [uid]
        if subj_id is not None:
            q += " AND subj_id=?"
            args.append(subj_id)
        if family is not None:
            q += " AND family=?"
            args.append(family)
        if statuses:
            q += f" AND status IN ({','.join('?' * len(statuses))})"
            args += list(statuses)
        return [self._fact(r) for r in self.db.execute(q + " ORDER BY id", args)]

    # ---------------------------------------------------------------- log + stats
    def log(self, uid, op, fact_id=None, reason="", ts=None, model="", stats=None):
        s = stats
        self.db.execute("INSERT INTO ops_log (user_id, ts, op, fact_id, reason, model, tokens_in, tokens_out, "
                        "latency_ms) VALUES (?,?,?,?,?,?,?,?,?)",
                        (uid, iso(ts) if ts else None, op, fact_id, (reason or "")[:500], model,
                         s.tokens_in if s else 0, s.tokens_out if s else 0, s.latency_ms if s else 0.0))

    def commit(self):
        self.db.commit()

    def stats(self, uid):
        one = lambda q, *a: self.db.execute(q, (uid, *a)).fetchone()[0]  # noqa: E731
        st = {"episodes": one("SELECT COUNT(*) FROM episodes WHERE user_id=?"),
              "entities": one("SELECT COUNT(*) FROM entities WHERE user_id=?")}
        for s in ("active", "superseded", "rejected"):
            st[f"facts_{s}"] = one("SELECT COUNT(*) FROM facts WHERE user_id=? AND status=?", s)
        st["facts_rich"] = self.db.execute("SELECT COUNT(*) FROM fact_rich").fetchone()[0]
        for k in ("state", "event", "attribute"):
            st[f"kind_{k}"] = one("SELECT COUNT(*) FROM facts WHERE user_id=? AND status!='rejected' AND kind=?", k)
        return st

    def close(self):
        try:
            self.db.commit()
        except sqlite3.Error:
            pass
        self.db.close()
