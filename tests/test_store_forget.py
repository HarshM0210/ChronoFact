"""Store.forget: the fact, its sibling facts, the source message and the assistant's reply are gone from memory and
from transcript search; unrelated facts stay."""
import os
import sys
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from chronofact.interface import Turn  # noqa: E402
from chronofact.store import Store  # noqa: E402


def test_forget(tmp_path):
    st = Store(str(tmp_path / "s.sqlite"))
    t0 = datetime(2025, 6, 2, 10)
    ids = st.add_episodes([Turn("u", "s1", 0, "user", "We moved to Munich because my partner got a job at BMW.", t0),
                           Turn("u", "s1", 1, "assistant", "How is your partner liking BMW?", t0),
                           Turn("u", "s1", 2, "user", "I run 5 times a week now.", t0)])
    common = dict(subj_id=None, obj_id=None, slot_type="single", valid_from=t0.isoformat(), valid_to=None,
                  recorded_at=t0.isoformat(), status="active", confidence=0.7, quote="", superseded_by=None,
                  kind="state", assertion_type="fact", conflict_with=None, date_precision="turn")
    f_bmw = st.add_fact("u", relation="partner", obj_literal="BMW", source_episode_ids=[ids[0]], family="partner",
                        slot_key="partner_job", statement="The user's partner works at BMW.", **common)
    f_mun = st.add_fact("u", relation="residence", obj_literal="Munich", source_episode_ids=[ids[0]], family="residence",
                        slot_key="current_city", statement="The user lives in Munich.", **common)
    f_run = st.add_fact("u", relation="routine", obj_literal="5", source_episode_ids=[ids[2]], family="routine",
                        slot_key="runs_per_week", statement="The user runs 5 times a week.", **common)
    assert ids[0] in [e for e, _ in st.episode_index("u").bm25("BMW partner", 5)]   # findable before
    gone, eps = st.forget("u", f_bmw, t0)
    assert sorted(gone) == sorted([f_bmw, f_mun]) and eps == sorted([ids[0], ids[1]])
    status = {f["id"]: f["status"] for f in st.facts("u", statuses=("active", "forgotten"))}
    assert status == {f_bmw: "forgotten", f_mun: "forgotten", f_run: "active"}
    assert st.episode_index("u").bm25("BMW partner", 5) == []          # gone from transcript search
    assert st.episode_index("u").ids == [ids[2]]                          # only the unrelated message is indexed
    assert st.db.execute("SELECT COUNT(*) FROM ops_log WHERE op='FORGET'").fetchone()[0] == 1
