"""LongMemEval-S (cleaned) loading, checkpoints and the two fixed halves (README section 4).

The halves come from data/splits.json (m3_A: 251, m3_B: 249) and are used as they are. Nothing here ever
writes that file (Model4's download script regenerated splits.json, which would have replaced the halves).
"""
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from chronofact.config import CFG, path

RAW = path("data", "raw", "longmemeval_s_cleaned.json")
SPLITS = path("data", "splits.json")
URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"


def parse_date(s: str) -> datetime:
    """'2023/05/20 (Sat) 02:21' -> datetime"""
    day, rest = s.split(" (")
    return datetime.strptime(day + " " + rest.split(") ")[1], "%Y/%m/%d %H:%M")


@dataclass
class Session:
    session_id: str
    date: datetime
    turns: list
    is_evidence: bool = False


@dataclass
class Question:
    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: datetime
    is_abstention: bool
    answer_session_ids: list
    sessions: list = field(default_factory=list)       # sorted by date (stable)

    @property
    def evidence_end(self) -> int:
        """e = number of sessions up to and including the last evidence session.
        A history with no evidence session in its haystack gets e = n (no such history is expected; the
        smoke test reports the count)."""
        idx = [i for i, s in enumerate(self.sessions) if s.is_evidence]
        return (max(idx) + 1) if idx else len(self.sessions)

    def checkpoints(self, fracs=None):
        """[(fraction, sessions ingested)]: e + round(f * (n - e)), Python round (halves to even)."""
        fracs = fracs if fracs is not None else CFG["eval"]["checkpoint_fractions"]
        e, n = self.evidence_end, len(self.sessions)
        return [(f, e + round(f * (n - e))) for f in fracs]

    def turns(self):
        """All turns in time order; turn timestamp = session date + turn_index seconds."""
        out = []
        for s in self.sessions:
            for i, t in enumerate(s.turns):
                out.append({"session_id": s.session_id, "turn_idx": i, "speaker": t["role"], "text": t["content"],
                            "ts": s.date + timedelta(seconds=i), "has_answer": bool(t.get("has_answer", False))})
        return out

    def visible_until(self):
        """max(question date, latest turn timestamp): the whole history precedes the question by design."""
        return max([self.question_date] + [t["ts"] for t in self.turns()])

    def evidence_turns(self):
        """{(session_id, turn_idx)} of turns with has_answer"""
        return {(s.session_id, i) for s in self.sessions for i, t in enumerate(s.turns) if t.get("has_answer")}


_CACHE = {}


def load_all() -> dict:
    if "all" in _CACHE:
        return _CACHE["all"]
    with open(RAW) as f:
        raw = json.load(f)
    out = {}
    for r in raw:
        ev = set(r["answer_session_ids"])
        sess = [Session(sid, parse_date(d), turns, sid in ev)
                for sid, d, turns in zip(r["haystack_session_ids"], r["haystack_dates"], r["haystack_sessions"])]
        sess.sort(key=lambda s: s.date)                 # Python's sort is stable: ties keep the file order
        out[r["question_id"]] = Question(
            question_id=r["question_id"], question_type=r["question_type"], question=r["question"],
            answer=str(r["answer"]), question_date=parse_date(r["question_date"]),
            is_abstention=r["question_id"].endswith("_abs"), answer_session_ids=list(r["answer_session_ids"]),
            sessions=sess)
    _CACHE["all"] = out
    return out


def halves():
    with open(SPLITS) as f:
        sp = json.load(f)
    return list(sp["m3_A"]), list(sp["m3_B"])


def half_of(qid):
    a, _ = halves()
    return "A" if qid in set(a) else "B"
