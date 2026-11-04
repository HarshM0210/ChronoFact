"""Local Meta models, CPU only: embeddings (Contriever), NLI (BART-large-MNLI) and
the answerer's tokenizer for budgets (config tokenizer). Lazily loaded process-wide singletons.

Embeddings: questions use the QUERY encoder, everything else the CONTEXT encoder (README 17). No prefix.
Max 512 tokens. L2-normalised, so inner product = cosine. Vectors are cached on disk by sha256 of
(model, side, text) - the side and model must be part of the key, otherwise a question embedded on the
query side could be served as a context vector.

Determinism (fix over Model4): Model4 used a freshly computed float32 vector in the process that computed it,
but the float16 copy from the cache everywhere else, so a re-run could see slightly different cosines. Here a
new vector is rounded to float16 before first use, and the cache row that wins (INSERT OR IGNORE) is the one
every process uses. The same is done for NLI probabilities (cached per pair), so a re-run is byte-identical.
"""
import hashlib
import os
import sqlite3
import threading

import numpy as np

from chronofact.config import CFG, cpath, embedder_name, pin


def _db(name):
    p = cpath(name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    db = sqlite3.connect(p, timeout=300, check_same_thread=False)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    return db


def _fetch(db, keys, table, col):
    out = {}
    for j in range(0, len(keys), 900):
        ch = keys[j:j + 900]
        out.update(db.execute(f"SELECT key, {col} FROM {table} WHERE key IN ({','.join('?' * len(ch))})", ch))
    return out


class Embedder:
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer
        spec = CFG["embeddings"]
        self.name = embedder_name()
        self.pooling = spec["pooling"]
        self.maxlen = CFG["embeddings"]["max_tokens"]
        self.batch = CFG["embeddings"]["batch_size"]
        self.tok, self.enc = {}, {}
        for side in ("query", "context"):
            rid = spec[side]
            self.tok[side] = AutoTokenizer.from_pretrained(rid, revision=pin(rid))
            self.enc[side] = AutoModel.from_pretrained(rid, revision=pin(rid)).eval()
        self.torch = torch
        self.dim = self.enc["context"].config.hidden_size
        self.db = _db(CFG["embeddings"]["cache_path"])
        self.db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB)")
        self.db.commit()
        self.lock = threading.Lock()
        self.mem = {}

    def _key(self, side, text):
        return hashlib.sha256(f"{self.name}|{side}|{text}".encode()).hexdigest()

    def _compute(self, side, texts):
        torch = self.torch
        tok, enc = self.tok[side], self.enc[side]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = [None] * len(texts)
        with torch.inference_mode():
            for j in range(0, len(order), self.batch):
                idx = order[j:j + self.batch]
                b = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=self.maxlen,
                        return_tensors="pt")
                h = enc(**b).last_hidden_state
                if self.pooling == "cls":
                    v = h[:, 0]
                else:
                    m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
                    v = (h * m).sum(1) / m.sum(1).clamp(min=1e-9)
                v = torch.nn.functional.normalize(v.float(), dim=-1).numpy()
                for k, i in enumerate(idx):
                    out[i] = v[k]
        return out

    def encode(self, texts, is_query=False):
        """float32 array (n, dim), L2-normalised (up to float16 rounding)."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        side = "query" if is_query else "context"
        keys = [self._key(side, t) for t in texts]
        res = {k: self.mem[k] for k in keys if k in self.mem}
        miss = list(dict.fromkeys(k for k in keys if k not in res))
        if miss:
            with self.lock:
                got = _fetch(self.db, miss, "emb", "vec")
            for k, v in got.items():
                res[k] = np.frombuffer(v, dtype=np.float16).astype(np.float32)
            todo = {}
            for t, k in zip(texts, keys):
                if k not in res:
                    todo[k] = t
            if todo:
                tk = list(todo)
                vecs = self._compute(side, [todo[k] for k in tk])
                with self.lock:
                    self.db.executemany("INSERT OR IGNORE INTO emb VALUES (?,?)",
                                        [(k, v.astype(np.float16).tobytes()) for k, v in zip(tk, vecs)])
                    self.db.commit()
                    got = _fetch(self.db, tk, "emb", "vec")     # the stored row wins (another process may
                for k in tk:                                    # have written it first)
                    res[k] = np.frombuffer(got[k], dtype=np.float16).astype(np.float32)
            if len(self.mem) > 500_000:
                self.mem.clear()
            for k in miss:
                self.mem[k] = res[k]
        return np.stack([res[k] for k in keys]).astype(np.float32)


class NLI:
    """BART-large-MNLI. Pair order (premise, hypothesis). Label order read from the model config; softmax
    over all three logits (README 17). Probabilities cached per pair (float32)."""

    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        rid = CFG["nli"]["model"]
        self.tok = AutoTokenizer.from_pretrained(rid, revision=pin(rid))
        self.model = AutoModelForSequenceClassification.from_pretrained(rid, revision=pin(rid)).eval()
        lab = {i: l.lower() for i, l in self.model.config.id2label.items()}
        self.idx = {name: [i for i, l in lab.items() if l.startswith(name)][0]
                    for name in ("contradiction", "neutral", "entailment")}
        self.torch = torch
        self.batch = CFG["nli"]["batch_size"]
        self.db = _db(CFG["nli"]["cache_path"])
        self.db.execute("CREATE TABLE IF NOT EXISTS nli (key TEXT PRIMARY KEY, probs BLOB)")
        self.db.commit()
        self.lock = threading.Lock()

    @staticmethod
    def _key(p, h):
        return hashlib.sha256(f"{CFG['nli']['model']}\x00{p}\x00{h}".encode()).hexdigest()

    def _compute(self, pairs):
        torch = self.torch
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        out = [None] * len(pairs)
        with torch.inference_mode():
            for j in range(0, len(order), self.batch):
                idx = order[j:j + self.batch]
                b = self.tok([pairs[i][0] for i in idx], [pairs[i][1] for i in idx], padding=True,
                             truncation=True, max_length=1024, return_tensors="pt")
                p = torch.softmax(self.model(**b).logits.float(), dim=-1).numpy()
                for k, i in enumerate(idx):
                    out[i] = p[k]
        return out

    def probs(self, pairs):
        """(n, 3) float32: columns contradiction, neutral, entailment."""
        if not pairs:
            return np.zeros((0, 3), dtype=np.float32)
        keys = [self._key(p, h) for p, h in pairs]
        with self.lock:
            got = _fetch(self.db, list(dict.fromkeys(keys)), "nli", "probs")
        todo = {}
        for (p, h), k in zip(pairs, keys):
            if k not in got:
                todo[k] = (p, h)
        if todo:
            tk = list(todo)
            pr = self._compute([todo[k] for k in tk])
            rows = [(k, np.array([v[self.idx["contradiction"]], v[self.idx["neutral"]],
                                  v[self.idx["entailment"]]], dtype=np.float32).tobytes()) for k, v in zip(tk, pr)]
            with self.lock:
                self.db.executemany("INSERT OR IGNORE INTO nli VALUES (?,?)", rows)
                self.db.commit()
                got.update(_fetch(self.db, tk, "nli", "probs"))
        return np.stack([np.frombuffer(got[k], dtype=np.float32) for k in keys])

    def entail(self, pairs):
        return self.probs(pairs)[:, 2] if pairs else np.zeros(0)

    def contradict(self, pairs):
        return self.probs(pairs)[:, 0] if pairs else np.zeros(0)


class TokenCounter:
    """The tokenizer of config `tokenizer` (files only; no model is loaded), no special tokens."""

    def __init__(self):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(CFG["tokenizer"]["repo"], revision=CFG["tokenizer"]["revision"])

    def ids(self, text):
        return self.tok.encode(text, add_special_tokens=False)

    def count(self, text):
        return len(self.ids(text))

    def truncate(self, text, n):
        ids = self.ids(text)
        return text if len(ids) <= n else self.tok.decode(ids[:n])


_inst, _lock = {}, threading.Lock()


def _get(name, cls):
    with _lock:
        if name not in _inst:
            _inst[name] = cls()
        return _inst[name]


def embedder() -> Embedder:
    return _get("emb", Embedder)


def nli() -> NLI:
    return _get("nli", NLI)


def tokens() -> TokenCounter:
    return _get("tok", TokenCounter)
