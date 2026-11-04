"""ChronoFact write path, README section 5 (5.1 - 5.8).

Per batch of turns (60): store episodes -> choose user turns (salience) -> extract (concurrent) -> prepare
-> verify (quote + batched NLI) -> label (concurrent) + rich statement check (batched NLI) -> update the store
strictly in turn order, one fact at a time.

Where the README prose is more precise than Model4's code, the README is followed (marked "README" below):
- event duplicates: |valid_from difference| <= 3 days measured exactly (Model4 used timedelta.days, which
  floors, so the window was asymmetric: +3 d 23 h counted as within, -3 d 1 h did not);
- supersession guard: "both values contain a digit" uses both VALUES (Model4 only looked at the old fact's
  literal, so an old value stored as an entity, e.g. "iPhone 13", never counted);
- negation: the controller is called only when the slot is empty (Model4 also called it when the slot held
  only events);
- "same value": the numbers of the two values must be the same (compared as sorted lists).
"""
import asyncio
import re
import time
from datetime import datetime

import numpy as np

from chronofact import prompts as PR
from chronofact import relations as R
from chronofact import rules as RU
from chronofact.config import CFG
from chronofact.interface import CallStats
from chronofact.llm import LLMJSONError, acall
from chronofact.models import nli
from chronofact.store import iso
from chronofact.timeparse import resolve


def _s(x):
    return x if isinstance(x, str) else ("" if x is None else str(x))


def chunk_jobs(jobs, n, max_words):
    """Consecutive extraction jobs (i, turn, ctx) -> chunks of at most n jobs and max_words words (turn text
    clipped to 1,200 words, context to 600, as in the prompt). Shared by ingest and step batch_check."""
    chunks, cur, words = [], [], 0
    for j in jobs:
        w = min(len(j[1].text.split()), 1200) + (min(len(j[2].text.split()), 600) if j[2] is not None else 0)
        if cur and (len(cur) >= n or words + w > max_words):
            chunks.append(cur)
            cur, words = [], 0
        cur.append(j)
        words += w
    if cur:
        chunks.append(cur)
    return chunks


def extraction_jobs(turns, prev=None):
    """README 5.2 for one batch of Turn objects: -> [(index, turn, assistant context or None)] of the user turns
    that pass salience, plus [(index, skip reason)]. `prev` = {user: last turn} carried across batches."""
    prev = {} if prev is None else prev
    jobs, skips = [], []
    for i, t in enumerate(turns):
        last = prev.get(t.user_id)
        if last is not None and last.session_id != t.session_id:
            last = None
        prev[t.user_id] = t
        if t.speaker != "user":
            continue
        why = RU.salience(t.text)
        if why:
            skips.append((i, why))
            continue
        ctx = last if (last is not None and last.speaker == "assistant" and
                       (RU._REFERENTIAL.search(t.text) or len(t.text.split()) < 12)) else None
        jobs.append((i, t, ctx))
    return jobs, skips


class Writer:
    def __init__(self, store, cfg=None):
        self.store = store
        self.cfg = dict(cfg or CFG["write"])
        self._prev = {}                 # user -> last turn of the current session
        self.role_stats = {}
        self.on_stage = None            # optional callback(stage, data) for the live demo; None in every run

    def _emit(self, stage, data):
        if self.on_stage is not None:
            self.on_stage(stage, data)

    def _role(self, role, st):
        r = self.role_stats.setdefault(role, {"calls": 0, "tokens_in": 0, "tokens_out": 0})
        r["calls"] += st.llm_calls
        r["tokens_in"] += st.tokens_in
        r["tokens_out"] += st.tokens_out

    # ================================================================== batch
    async def ingest_batch(self, turns):
        """Returns one CallStats per turn (LLM cost of that turn; latency_ms = its share of the batch time)."""
        if not turns:
            return []
        uid = turns[0].user_id
        t0 = time.perf_counter()
        ep_ids = self.store.add_episodes(turns)                                         # 5.1
        stats = [CallStats() for _ in turns]
        work = [0.0] * len(turns)
        ext_lat = [0.0] * len(turns)
        t_store = (time.perf_counter() - t0) * 1000 / len(turns)

        jobs, skips = extraction_jobs(turns, self._prev)                                 # 5.2
        for i, why in skips:
            self.store.log(uid, "SKIP", None, f"episode {ep_ids[i]}: {why}", turns[i].ts)
        self._emit("salience", {"skipped": [{"text": turns[i].text, "why": why} for i, why in skips
                                            if turns[i].speaker == "user"],
                                "extract": [t.text for _, t, _ in jobs]})
        results = await self._extract_jobs(uid, jobs)                                   # 5.3 (batched)
        cands, shown = [], []
        for (i, t, _), (facts, st, err) in zip(jobs, results):
            stats[i].add(st)
            ext_lat[i] = st.latency_ms
            self._role("extractor", st)
            if err:
                self.store.log(uid, "ERROR", None, f"episode {ep_ids[i]}: extraction failed: {err}", t.ts,
                               stats=st)
            for f in facts:
                c = self._prepare(f, t, ep_ids[i])                                       # 5.4
                shown.append({k: _s(f.get(k)) for k in ("subject", "family", "object", "statement", "quote",
                                                        "time_expression", "assertion_type")} | {"ok": bool(c)})
                if c:
                    cands.append((i, c))
                elif f.get("family") in R.FAMILIES and all(k in f for k in ("subject", "object")):
                    # schema-valid but an empty subject/object: a placeholder fact the model made up ("The user has
                    # an age."); dropped (README 5.4), counted apart from schema failures
                    self.store.log(uid, "DROP_EMPTY", None, f"episode {ep_ids[i]}: empty subject/object: "
                                   f"{str(f)[:300]}", t.ts)
                else:          # README 5.4 drop; logged so a schema problem can never hide (pilot/audit check it)
                    self.store.log(uid, "DROP", None, f"episode {ep_ids[i]}: missing subject/object or invalid "
                                   f"family: {str(f)[:300]}", t.ts)

        self._emit("extracted", shown)
        t1 = time.perf_counter()                                                         # 5.5
        keep, pairs, pidx, ver = [], [], [], []
        for k, (i, c) in enumerate(cands):
            if not RU.quote_ok(c["quote"], turns[i].text):
                self.store.log(uid, "REJECT", None, f"quote not in turn: {c['quote'][:120]}", turns[i].ts)
                self._reject(uid, c)
                ver.append({"statement": c["statement"], "quote": c["quote"], "quote_ok": False, "nli": None,
                            "kept": False})
                continue
            pairs.append((RU.third_person(RU.premise_window(c["quote"], turns[i].text)), c["statement"]))
            pidx.append(k)
        if pairs:
            for k, p in zip(pidx, nli().entail(pairs)):
                i, c = cands[k]
                if p < self.cfg["min_entail"]:
                    self.store.log(uid, "REJECT", None, f"nli {p:.2f} < min: {c['statement']}", turns[i].ts)
                    self._reject(uid, c)
                else:
                    keep.append((i, c))
                ver.append({"statement": c["statement"], "quote": c["quote"], "quote_ok": True, "nli": float(p),
                            "kept": bool(p >= self.cfg["min_entail"])})
        self._emit("verified", {"facts": ver, "min_entail": self.cfg["min_entail"]})
        cands = keep
        cands = await self._label_all(uid, cands, turns, stats)                          # 5.6
        self._emit("labeled", [{"statement": c["statement"], "rich": c.get("rich"), "kind": c["kind"],
                                "family": c["family"], "slot_key": c["slot_key"]} for _, c in cands])
        t_mid = (time.perf_counter() - t1) * 1000

        last = self.store.db.execute("SELECT COALESCE(MAX(id), 0) FROM ops_log").fetchone()[0] \
            if self.on_stage is not None else 0
        for i, c in cands:                                                               # 5.7
            t2 = time.perf_counter()
            st = await self._apply(uid, c, turns[i])
            stats[i].add(st)
            work[i] += (time.perf_counter() - t2) * 1000
            if self.on_stage is not None:
                rows = self.store.db.execute("SELECT id, op, fact_id, reason FROM ops_log WHERE id > ? ORDER BY id",
                                             (last,)).fetchall()
                if rows:
                    last = rows[-1][0]
                self._emit("applied", [{"op": o, "fact_id": f, "reason": r} for _, o, f, r in rows])
        for i, _, _ in jobs:
            work[i] += t_mid / max(1, len(jobs))
        for i in range(len(turns)):
            # wall time attributable to the turn: its share of storing, its extraction call, its share of
            # verify + label (they run concurrently for the batch), and its own updates
            stats[i].latency_ms = t_store + ext_lat[i] + work[i]
        self.store.commit()
        return stats

    # ================================================================== 5.3 extract
    async def _extract(self, t, ctx):
        lines = [f"Message date: {t.ts.strftime('%Y/%m/%d (%a)')}"]
        if ctx is not None:
            lines.append("The assistant's previous message (only to understand references; do not extract from it):")
            lines.append(RU._clip(ctx.text, 600))
        lines.append("User message:")
        lines.append(RU._clip(t.text, 1200))
        msgs = [{"role": "system", "content": PR.EXTRACT_SYSTEM}, {"role": "user", "content": "\n".join(lines)}]
        st = CallStats()
        try:
            out, s1 = await acall("extractor", msgs, PR.EXTRACT_SCHEMA)
            st.add(s1)
            return self._facts_of(out), st, None
        except LLMJSONError as e:
            st.add(e.stats)
            if e.rejected:                     # refused by the API's content filter: never re-sent
                return [], st, str(e)
        # one more call: compact instruction, only the date and the user message, max_tokens 512
        msgs = [{"role": "system", "content": PR.EXTRACT_SYSTEM + "\n" + PR.EXTRACT_FALLBACK_LINE},
                {"role": "user", "content": "\n".join([lines[0], "User message:", RU._clip(t.text, 1200)])}]
        try:
            out, s2 = await acall("extractor", msgs, PR.EXTRACT_SCHEMA, max_tokens=512)
            st.add(s2)
            return self._facts_of(out), st, None
        except LLMJSONError as e:
            st.add(e.stats)
            return [], st, str(e)

    async def _extract_jobs(self, uid, jobs):
        """Extraction for the chosen user turns of one batch, write.extract_chunk messages per call (consecutive
        turns, at most write.extract_chunk_words words). A chunk of one uses the per-turn README prompt. A chunk
        whose call fails (invalid JSON after the retries) is redone turn by turn. Returns (facts, stats, error) per
        job; a chunk's LLM cost is booked on its first turn, its latency on every turn of it."""
        chunks = chunk_jobs(jobs, self.cfg["extract_chunk"], self.cfg["extract_chunk_words"])
        res = await asyncio.gather(*[self._extract_chunk(uid, ch) for ch in chunks])
        return [r for rr in res for r in rr]

    async def _extract_chunk(self, uid, chunk):
        if len(chunk) == 1:
            return [await self._extract(chunk[0][1], chunk[0][2])]
        lines = []
        for k, (_, t, ctx) in enumerate(chunk, 1):
            lines += [f"### Message {k}", f"Message date: {t.ts.strftime('%Y/%m/%d (%a)')}"]
            if ctx is not None:
                lines.append("The assistant's previous message (only to understand references; do not extract from it):")
                lines.append(RU._clip(ctx.text, 600))
            lines += ["User message:", RU._clip(t.text, 1200), ""]
        msgs = [{"role": "system", "content": PR.EXTRACT_BATCH_SYSTEM}, {"role": "user", "content": "\n".join(lines).strip()}]
        try:
            out, st = await acall("extractor", msgs, PR.extract_batch_schema(len(chunk)))
        except LLMJSONError as e:              # redo the chunk turn by turn (README per-turn path + its fallback)
            if e.rejected:                     # refused by the content filter: the chunk is not re-sent (logged)
                return [([], e.stats if k == 0 else CallStats(), str(e)) for k in range(len(chunk))]
            per = await asyncio.gather(*[self._extract(t, ctx) for _, t, ctx in chunk])
            first = per[0][1]
            first.add(e.stats)
            return per
        facts = out.get("facts", []) if isinstance(out, dict) else []
        facts = facts if isinstance(facts, list) else []
        by = [[] for _ in chunk]
        for f in facts:
            m = f.get("message") if isinstance(f, dict) else None
            if isinstance(m, int) and 1 <= m <= len(chunk):
                by[m - 1].append(f)
            else:              # a fact that names no valid message cannot be checked against a turn: dropped
                t = chunk[0][1]
                self.store.log(uid, "DROP", None, f"batch extraction: no valid message number: {str(f)[:300]}", t.ts)
        out_rows = []
        for k, (_, t, _) in enumerate(chunk):
            s_k = CallStats(st.llm_calls, st.tokens_in, st.tokens_out, st.latency_ms) if k == 0 else \
                CallStats(0, 0, 0, st.latency_ms)
            out_rows.append((by[k][: self.cfg["max_facts_per_turn"]], s_k, None))
        return out_rows

    def _facts_of(self, out):
        facts = out.get("facts", []) if isinstance(out, dict) else []
        facts = facts if isinstance(facts, list) else []
        return [f for f in facts if isinstance(f, dict)][: self.cfg["max_facts_per_turn"]]

    # ================================================================== 5.4 prepare (README Appendix C.3)
    def _prepare(self, f, t, ep_id):
        subj = _s(f.get("subject")).strip()
        obj = _s(f.get("object")).strip()
        fam = f.get("family")
        if not subj or not obj or fam not in R.FAMILIES:
            return None
        if subj.lower() in ("user", "i", "me", "myself", "the user"):
            subj = "user"
        kind = R.FAMILIES[fam]
        rng = resolve(_s(f.get("time_expression")), t.ts)
        quote = _s(f.get("quote"))
        statement = _s(f.get("statement")).strip() or f"The user {fam.replace('_', ' ')} {obj}."
        assertion = f.get("assertion_type") if f.get("assertion_type") in R.ASSERTION_TYPES else "fact"
        c = {"subject": subj, "family": fam, "kind": kind, "relation": fam,
             "slot_key": R.norm_slot_key(_s(f.get("slot_key")) or obj), "object": obj,
             "object_type": f.get("object_type") if f.get("object_type") in R.OBJECT_TYPES else "other",
             "assertion_type": assertion, "quote": quote,
             "statement": statement, "episode_id": ep_id, "recorded_at": iso(t.ts),
             "slot_type": "single" if kind == "state" else "multi",
             "negated": bool(RU._NEG.search(quote)) and assertion != "intention",
             "literal": R.literal_object(fam, obj), "rich": None}
        c["valid_from"] = iso(rng[0]) if rng else iso(t.ts)
        c["date_precision"] = rng[2] if rng else "turn"
        c["valid_to"] = None
        if kind == "state" and rng and rng[2] in ("year", "month") and rng[1] <= t.ts \
                and (t.ts - rng[1]).days > 0:
            c["valid_to"] = iso(rng[1])                 # a state stated about the past
        return c

    def _reject(self, uid, c):
        self.store.add_fact(uid, subj_id=None, relation=c["family"], obj_id=None,
                            obj_literal=f"{c['subject']} | {c['object']}", slot_type=c["slot_type"],
                            valid_from=c["valid_from"], valid_to=c["valid_to"], recorded_at=c["recorded_at"],
                            status="rejected", confidence=0.0, quote=c["quote"], source_episode_ids=[c["episode_id"]],
                            superseded_by=None, kind=c["kind"], family=c["family"], slot_key=c["slot_key"],
                            assertion_type=c["assertion_type"], statement=c["statement"],
                            date_precision=c["date_precision"])

    # ================================================================== 5.6 label + rich statement
    def _near_keys(self, uid, slot_key):
        keys = sorted({r[0] for r in self.store.db.execute(
            "SELECT DISTINCT slot_key FROM facts WHERE user_id=? AND status IN ('active','superseded') "
            "AND slot_key IS NOT NULL AND slot_key != ''", (uid,))})
        if not keys:
            return []
        v = self.store.emb.encode([slot_key.replace("_", " ")] + [k.replace("_", " ") for k in keys])
        order = np.argsort(-(v[1:] @ v[0]), kind="stable")[:8]
        return [keys[int(i)] for i in order]

    async def _label(self, uid, c, turn):
        msg = "\n".join(["Message: " + RU._clip(turn.text, 300),
                         "Draft statement: " + c["statement"],
                         "Draft family: " + c["family"],
                         "Draft slot key: " + c["slot_key"],
                         "Assertion type: " + c["assertion_type"],
                         "Existing keys: " + (", ".join(self._near_keys(uid, c["slot_key"])) or "(none)")])
        msgs = [{"role": "system", "content": PR.LABEL_SYSTEM}, {"role": "user", "content": msg}]
        try:
            out, st = await acall("labeler", msgs, PR.LABEL_SCHEMA)
            return (out if isinstance(out, dict) else None), st
        except LLMJSONError as e:
            return None, e.stats

    async def _label_group(self, uid, group, turns):
        """One labeling call for up to write.label_chunk facts of a batch (Model4 labeled a batch's facts
        concurrently, none seeing another's label, so a joint call sees the same information). A group of one uses
        the per-fact README prompt; a failed call is redone fact by fact. -> [(label or None, stats)] per fact;
        the call's cost is booked on the group's first fact."""
        if len(group) == 1:
            i, c = group[0]
            return [await self._label(uid, c, turns[i])]
        blocks = []
        for k, (i, c) in enumerate(group, 1):
            blocks.append("\n".join([f"### Fact {k}",
                                     "Message: " + RU._clip(turns[i].text, 300),
                                     "Draft statement: " + c["statement"],
                                     "Draft family: " + c["family"],
                                     "Draft slot key: " + c["slot_key"],
                                     "Assertion type: " + c["assertion_type"],
                                     "Existing keys: " + (", ".join(self._near_keys(uid, c["slot_key"])) or "(none)")]))
        msgs = [{"role": "system", "content": PR.LABEL_BATCH_SYSTEM}, {"role": "user", "content": "\n\n".join(blocks)}]
        try:
            out, st = await acall("labeler", msgs, PR.label_batch_schema(len(group)))
        except LLMJSONError as e:
            if e.rejected:                     # refused by the content filter: not re-sent; drafts kept unlabeled
                return [(None, e.stats if k == 0 else CallStats()) for k in range(len(group))]
            per = await asyncio.gather(*[self._label(uid, c, turns[i]) for i, c in group])
            per[0][1].add(e.stats)
            return per
        got = {}
        for x in (out.get("labels", []) if isinstance(out, dict) else []) or []:
            if isinstance(x, dict) and isinstance(x.get("id"), int) and x["id"] not in got:
                got[x["id"]] = x
        for k, (i, c) in enumerate(group, 1):
            if k not in got:   # treated like a failed label (draft family kept, no rich statement); counted
                self.store.log(uid, "LABEL_MISS", None, f"batch label missing for: {c['statement'][:200]}", turns[i].ts)
        return [(got.get(k), st if k == 1 else CallStats()) for k in range(1, len(group) + 1)]

    async def _label_all(self, uid, cands, turns, stats):
        if not cands:
            return cands
        n = self.cfg["label_chunk"]
        groups = [cands[j:j + n] for j in range(0, len(cands), n)]
        res = [r for g in await asyncio.gather(*[self._label_group(uid, g, turns) for g in groups]) for r in g]
        checks = []
        for (i, c), (out, st) in zip(cands, res):
            stats[i].add(st)
            self._role("labeler", st)
            kind = fam = None
            if out:
                kind, fam = out.get("kind"), out.get("family")
                sk = R.norm_slot_key(_s(out.get("slot_key")))
                if sk:
                    c["slot_key"] = sk
                rich = _s(out.get("statement")).strip()
                if rich and rich != c["statement"]:
                    if RU.rich_ok(rich, turns[i].text, self.cfg["rich_max_words"]):
                        checks.append((i, c, rich))
                    else:          # diagnostics for README 13.2 ("read 30 dropped ones")
                        self.store.log(uid, "RICH_DROP", None, f"words/numbers: {rich}", turns[i].ts)
            if fam in R.FAMILIES and kind in ("event", "state", "attribute"):
                if self.cfg.get("label_kind_from_family") and R.FAMILIES[fam] != kind:
                    kind = R.FAMILIES[fam]     # the labeler's family wins over its contradicting kind (config note)
                fam = RU.consistent_family(fam, kind, c["assertion_type"], c["statement"])
            else:
                fam = RU.consistent_family(c["family"], c["kind"], c["assertion_type"], c["statement"])
            kind = R.FAMILIES[fam]
            if kind != "state":
                c["valid_to"] = None
            c.update(family=fam, relation=fam, kind=kind, slot_type="single" if kind == "state" else "multi",
                     literal=R.literal_object(fam, c["object"]))
        if checks:
            pairs = [(RU.third_person(RU._clip(turns[i].text, 200)), rich) for i, _, rich in checks]
            for (i, c, rich), p in zip(checks, nli().entail(pairs)):
                if p >= self.cfg["min_entail"]:
                    c["rich"] = rich
                else:
                    self.store.log(uid, "RICH_DROP", None, f"nli {p:.2f}: {rich}", turns[i].ts)
        return cands

    # ================================================================== 5.7 update
    def _entity(self, uid, name, etype, ts):
        if name == "user":
            eid = self.store.find_entity_exact(uid, "user")
            if eid is None:
                eid = self.store.add_entity(uid, "user", "person", self.store.emb.encode(["user"])[0])
            return eid
        eid = self.store.find_entity_exact(uid, name)
        if eid is not None:
            return eid
        vec = self.store.emb.encode([name])[0]
        for cand_id, ctype, sim in self.store.similar_entities(uid, vec, 5):
            if sim < self.cfg["entity_merge_sim"]:
                break
            if ctype == etype or "other" in (ctype, etype):
                self.store.add_alias(uid, cand_id, name)
                self.store.log(uid, "MERGE", None, f"'{name}' -> entity {cand_id} (sim {sim:.2f})", ts)
                return cand_id
        return self.store.add_entity(uid, name, etype, vec)

    def _canonical_slot(self, uid, subj_id, family, slot_key):
        keys = {r[0] for r in self.store.db.execute(
            "SELECT DISTINCT slot_key FROM facts WHERE user_id=? AND subj_id=? AND family=? AND "
            "status IN ('active','superseded')", (uid, subj_id, family)) if r[0] is not None}
        if not keys or slot_key in keys:
            return slot_key
        keys = sorted(keys)
        v = self.store.emb.encode([slot_key.replace("_", " ")] + [k.replace("_", " ") for k in keys])
        sims = v[1:] @ v[0]
        j = int(np.argmax(sims))
        return keys[j] if sims[j] >= self.cfg["entity_merge_sim"] else slot_key

    @staticmethod
    def _value(f, ents):
        if f["obj_id"] is not None:
            return ents.get(f["obj_id"], {}).get("name", "") or ""
        return f["obj_literal"] or ""

    def _same_value(self, a, b):
        if a.strip().lower() == b.strip().lower():
            return True
        if sorted(RU._digits(a)) != sorted(RU._digits(b)):       # numbers that differ: different values
            return False
        v = self.store.emb.encode([a, b])
        return float(v[0] @ v[1]) >= self.cfg["near_dup_high"]

    def _add(self, uid, c, status, superseded_by=None, conflict_with=None):
        fid = self.store.add_fact(uid, subj_id=c["subj_id"], relation=c["family"], obj_id=c["obj_id"],
                                  obj_literal=c["obj_literal"], slot_type=c["slot_type"], valid_from=c["valid_from"],
                                  valid_to=c["valid_to"], recorded_at=c["recorded_at"], status=status,
                                  confidence=0.7, quote=c["quote"], source_episode_ids=[c["episode_id"]],
                                  superseded_by=superseded_by, kind=c["kind"], family=c["family"],
                                  slot_key=c["slot_key"], assertion_type=c["assertion_type"],
                                  statement=c["statement"], conflict_with=conflict_with,
                                  date_precision=c["date_precision"])
        if c.get("rich"):
            self.store.set_rich(fid, c["rich"])
        return fid

    def _noop(self, uid, f, c, reason, turn):
        src = f["source_episode_ids"]
        if c["episode_id"] not in src:
            src = src + [c["episode_id"]]
        self.store.update_fact(uid, f["id"], source_episode_ids=src, confidence=min(1.0, (f["confidence"] or 0) + 0.1))
        self.store.log(uid, "NOOP", f["id"], f"{reason}: {c['statement']}", turn.ts)

    def _conflicts(self, slot, c, ents):
        """README 5.7.9 supersession guard."""
        if not slot or c["slot_key"] in RU.GENERIC_SLOTS or c["slot_key"] == c["family"]:
            return []
        new_num = bool(re.search(r"\d", c["object"]))
        cand = [f for f in slot if f["family"] in RU.SINGLE_FAMILIES
                or (new_num and re.search(r"\d", self._value(f, ents)))]
        if not cand:
            return []
        olds = [f["statement"] or "" for f in cand]
        x = nli().contradict([(o, c["statement"]) for o in olds])
        y = nli().contradict([(c["statement"], o) for o in olds])
        return [f for f, a, b in zip(cand, x, y) if max(a, b) >= self.cfg["guard_contradiction"]]

    async def _apply(self, uid, c, turn):
        st = CallStats()
        subj_id = self._entity(uid, c["subject"], "person" if c["subject"] == "user" else "other", turn.ts)
        if c["literal"]:
            obj_id, obj_lit = None, c["object"]
        else:
            obj_id, obj_lit = self._entity(uid, c["object"], c["object_type"], turn.ts), None
        c.update(subj_id=subj_id, obj_id=obj_id, obj_literal=obj_lit)
        c["slot_key"] = self._canonical_slot(uid, subj_id, c["family"], c["slot_key"])
        ents = self.store.entity_rows(uid)
        slot = [f for f in self.store.facts(uid, subj_id, c["family"], ("active",)) if (f["slot_key"] or "") == c["slot_key"]]

        if c["negated"]:                                                     # 5.7.4
            if not slot:
                return await self._controller(uid, c, turn, st)
            closable = [f for f in slot if f["kind"] != "event"]
            for f in closable:
                self.store.update_fact(uid, f["id"], valid_to=c["recorded_at"], status="superseded")
                self.store.log(uid, "SUPERSEDE", f["id"], f"negation closes slot {c['family']}/{c['slot_key']}: "
                               f"{c['statement']}", turn.ts)
            if not closable:
                self.store.log(uid, "NOOP", None, f"negation, slot holds only events: {c['statement']}", turn.ts)
            return st

        same = [f for f in slot if self._same_value(self._value(f, ents), c["object"])]     # 5.7.5
        if c["kind"] == "event":                                             # 5.7.6
            win = self.cfg["event_window_days"] * 86400
            for f in same:
                gap = abs((datetime.fromisoformat(f["valid_from"]) - datetime.fromisoformat(c["valid_from"])).total_seconds())
                if gap <= win:
                    self._noop(uid, f, c, "same event", turn)
                    return st
            fid = self._add(uid, c, "active")
            self.store.log(uid, "ADD", fid, f"event: {c['statement']}", turn.ts)
            return st
        if same:                                                             # 5.7.7
            self._noop(uid, same[0], c, "same slot and value", turn)
            return st
        if c["kind"] == "attribute":                                         # 5.7.8
            fid = self._add(uid, c, "active")
            self.store.log(uid, "ADD", fid, f"attribute: {c['statement']}", turn.ts)
            return st
        conf = self._conflicts(slot, c, ents)                                # 5.7.9 state
        if not conf:
            fid = self._add(uid, c, "superseded" if c["valid_to"] else "active")
            self.store.log(uid, "ADD", fid, f"state: {c['statement']}", turn.ts)
            return st
        old = max(conf, key=lambda f: (f["valid_from"] or "", f["recorded_at"] or ""))
        newer = (c["valid_from"] or "", c["recorded_at"]) >= (old["valid_from"] or "", old["recorded_at"] or "")
        if c["valid_to"] is None and newer:
            gap = abs((datetime.fromisoformat(c["recorded_at"]) - datetime.fromisoformat(old["recorded_at"])).total_seconds())
            cw = old["id"] if gap < self.cfg["conflict_window_h"] * 3600 else None
            fid = self._add(uid, c, "active", conflict_with=cw)
            self.store.update_fact(uid, old["id"], valid_to=c["valid_from"], status="superseded", superseded_by=fid)
            self.store.log(uid, "SUPERSEDE", old["id"], f"state {c['family']}/{c['slot_key']}: '{self._value(old, ents)}'"
                           f" -> '{c['object']}' (fact {fid})" + (" [recorded <24h apart]" if cw else ""), turn.ts)
            self.store.log(uid, "ADD", fid, f"state: {c['statement']} (replaces {old['id']})", turn.ts)
            return st
        vt = c["valid_to"] or old["valid_from"]
        fid = self._add(uid, dict(c, valid_to=vt), "superseded")
        self.store.log(uid, "ADD", fid, f"past state: {c['statement']} (to {(vt or '?')[:10]})", turn.ts)
        return st

    # ================================================================== 5.8 controller
    async def _controller(self, uid, c, turn, st):
        active = [f for f in self.store.facts(uid, statuses=("active",)) if f["kind"] != "event"]
        if not active:
            self.store.log(uid, "NOOP", None, f"negation with nothing to close: {c['statement']}", turn.ts)
            return st
        sents = [f["statement"] or "" for f in active]
        v = self.store.emb.encode(sents + [c["statement"]])
        order = np.argsort(-(v[:-1] @ v[-1]), kind="stable")[: self.cfg["controller_top_k"]]
        shown = [active[int(i)] for i in order]
        listing = "\n".join(f"[{f['id']}] {f['statement']}" for f in shown)
        msg = f"Message date: {turn.ts:%Y-%m-%d}\nUser said: \"{c['quote']}\"\n\nACTIVE facts:\n{listing}"
        try:
            out, cst = await acall("controller", [{"role": "system", "content": PR.CONTROLLER_SYSTEM},
                                                  {"role": "user", "content": msg}], PR.CONTROLLER_SCHEMA)
        except LLMJSONError as e:
            st.add(e.stats)
            self._role("controller", e.stats)
            self.store.log(uid, "ERROR", None, f"controller: {e}", turn.ts, stats=e.stats)
            return st
        st.add(cst)
        self._role("controller", cst)
        out = out if isinstance(out, dict) else {}
        tid = out.get("target_fact_id")
        if isinstance(tid, str) and tid.strip().isdigit():
            tid = int(tid.strip())
        ids = {f["id"] for f in shown}
        if out.get("op") == "CLOSE" and isinstance(tid, int) and tid in ids:
            self.store.update_fact(uid, tid, valid_to=c["recorded_at"], status="superseded")
            self.store.log(uid, "SUPERSEDE", tid, f"controller CLOSE (negation): {_s(out.get('reason'))}", turn.ts,
                           model="controller", stats=cst)
        else:
            self.store.log(uid, "NOOP", tid if isinstance(tid, int) else None,
                           f"controller NOOP (negation): {_s(out.get('reason'))}", turn.ts, model="controller", stats=cst)
        return st
