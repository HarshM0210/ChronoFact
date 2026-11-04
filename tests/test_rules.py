"""Unit tests for the README rules (run: python -m pytest -q tests). No model or server needed."""
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chronofact import relations as R  # noqa: E402
from chronofact import rules as RU  # noqa: E402
from chronofact.answer import _fill, answer_prompt, is_abstention, parse_final  # noqa: E402
from chronofact.reader import Retriever, rrf, time_filter  # noqa: E402
from chronofact.reconcile import _content, _skeleton  # noqa: E402
from chronofact.timeparse import question_mode, resolve  # noqa: E402

REF = datetime(2023, 5, 20, 2, 21, 5)      # a Saturday


def test_salience():
    assert RU.salience("Thanks!") == "acknowledgement"
    assert RU.salience("sounds good.") == "acknowledgement"
    assert RU.salience("what is the capital") is not None            # short, no signal
    assert RU.salience("Can you list five good books about history please") == "no first-person claim"
    assert RU.salience("I adopted a dog named Max last week") is None


def test_quote_and_third_person():
    assert RU.quote_ok("I’ve got 3 cats", "Well, I've got 3 cats at home.")
    assert not RU.quote_ok("ab", "ab cd")
    assert RU.third_person("I'm sure my dog likes me") == "the user is sure the user's dog likes the user"


def test_premise_window():
    t = "We moved last year. My dog Luna is old. She's been getting a bit lame. Any tips?"
    w = RU.premise_window("She's been getting a bit lame", t)
    # README C.2 code: the window starts one sentence before the first 3-sentence span holding the quote
    assert w == "We moved last year. My dog Luna is old. She's been getting a bit lame." and "tips" not in w
    assert RU.premise_window("not there at all", t) == "not there at all"


def test_negation():
    assert RU._NEG.search("I sold my car") and RU._NEG.search("I don't play tennis anymore")
    assert not RU._NEG.search("I bought a car")


def test_resolve():
    assert resolve("yesterday", REF)[:2] == (datetime(2023, 5, 19), datetime(2023, 5, 20))
    assert resolve("last year", REF)[2] == "year"
    assert resolve("two weeks ago", REF)[0] == datetime(2023, 5, 6)
    assert resolve("in 2019", REF)[:2] == (datetime(2019, 1, 1), datetime(2020, 1, 1))
    assert resolve("4000 years ago", REF) is None                     # fix: no crash
    assert resolve("", REF) is None


def test_question_mode():
    assert question_mode("What did I used to drive?", REF)[0] == "history"
    m = question_mode("Where did I go last week?", REF)
    assert m[0] == "as_of" and m[1] == datetime(2023, 5, 8)
    assert question_mode("Where do I live now?", REF)[0] == "current"
    assert question_mode("What was the name of the band?", REF)[0] == "none"
    question_mode("What happened 4000 years ago?", REF)               # must not crash


def test_relations():
    assert R.norm_slot_key("Times Worn: Black Converses") == "time_worn_black_converse"
    assert R.norm_slot_key("glass") == "glass"
    assert R.literal_object("count", "Max") and R.literal_object("likes", "12 km") and not R.literal_object("likes", "Max")
    assert R.question_slots("How many cats do I have?") == ["number_of_"]
    assert R.matches_slot("favourite_x", "favourite_") and not R.matches_slot("favourite", "favourite_")
    assert len(R.FAMILIES) == 50


def test_consistent_family():
    assert RU.consistent_family("likes", "state", "fact", "x") == "other_state"
    assert RU.consistent_family("visit", "event", "intention", "x") == "plan"
    assert RU.consistent_family("likes", "attribute", "event", "x") == "other_event"
    assert RU.consistent_family("residence", "state", "fact", "The user is in Paris.") == "other_state"
    assert RU.consistent_family("residence", "state", "fact", "The user lives in Paris.") == "residence"
    assert RU.consistent_family("workplace", "state", "fact", "The user works at Google.") == "workplace"


def test_rich_ok():
    assert RU.rich_ok("The user ran 5 km in 25 minutes.", "I ran 5 km in 25 minutes today", 36)
    assert not RU.rich_ok("The user ran 6 km.", "I ran 5 km", 36)
    assert not RU.rich_ok(" ".join(["w"] * 37), "w", 36)


def test_w2_helpers():
    assert _skeleton("The user has 3 cats and two dogs.", "") == "the user has cats and dogs"
    assert _skeleton("The user lives in Paris.", "Paris") == "the user lives in"
    assert _content("the user lives in") == ["lives"]


def test_answer_parse():
    assert parse_final("thinking\nFINAL: Paris") == "Paris"
    assert parse_final("**Final Answer:** \"42\"") == "42"
    assert parse_final("a\nFINAL:\nBerlin") == "Berlin"
    assert parse_final("no final line\nRome") == "Rome"
    assert is_abstention("final: not in memory")
    p = answer_prompt("Q?", REF, "", "reason")
    assert "Memory:\n(empty)\nQuestion: Q?" in p and p.endswith("does not contain the answer>")
    assert p.startswith("You are answering a question about a user, using only the memory below.\n"
                        "Current date: 2023/05/20 (Sat) 02:21")


def test_fill_single_pass():
    assert _fill("Q: {question} A: {answer}", question="{answer}", answer="x") == "Q: {answer} A: x"


def test_rrf_and_time_filter():
    assert [e for e, _ in rrf([[(1, 0), (2, 0)], [(2, 0), (3, 0)]], 60)][0] == 2
    fs = [{"status": "active", "valid_from": "2023-01-01T00:00:00", "valid_to": None, "recorded_at": "2023-01-01T00:00:00"},
          {"status": "superseded", "valid_from": "2022-01-01T00:00:00", "valid_to": "2022-06-01T00:00:00",
           "recorded_at": "2022-01-01T00:00:00"},
          {"status": "rejected", "valid_from": "", "valid_to": None, "recorded_at": ""}]
    assert len(time_filter(fs, "current")) == 1
    assert len(time_filter(fs, "history")) == 2
    assert len(time_filter(fs, "as_of", datetime(2022, 3, 1), datetime(2022, 4, 1))) == 1
    assert len(time_filter(fs, "history", as_of=datetime(2022, 6, 1))) == 1


def test_fact_line():
    f = {"status": "active", "valid_from": "2023-01-02T00:00:00", "valid_to": None, "relation": "residence",
         "slot_type": "single", "subj_id": 1, "family": "residence", "slot_key": "city"}
    old = {"status": "superseded", "valid_from": "2021-01-01T00:00:00", "valid_to": "2023-01-02T00:00:00",
           "obj_id": None, "obj_literal": "Rome", "subj_id": 1, "family": "residence", "slot_key": "city"}
    by = {(1, "residence", "city"): [old]}
    assert Retriever.fact_line(f, "The user lives in Paris.", {}, "current", [], by) == "- The user lives in Paris (since 2023-01-02)"
    assert Retriever.fact_line(f, "The user lives in Paris.", {}, "history", [], by) == \
        "- The user lives in Paris (since 2023-01-02; previously Rome, 2021-01-01 to 2023-01-02)"
    g = dict(old, statement="x")
    assert Retriever.fact_line(g, "The user lived in Rome.", {}, "current", [], by) == \
        "- The user lived in Rome (2021-01-01 to 2023-01-02, no longer current)"


def test_prompts_verbatim():
    from chronofact import prompts as PR
    readme = open(os.path.expanduser(os.environ.get("CF_README", "README_Model5.md"))).read() \
        if os.path.exists(os.environ.get("CF_README", "README_Model5.md")) else None
    if readme:
        for t in (PR.EXTRACT_SYSTEM, PR.LABEL_SYSTEM, PR.CONTROLLER_SYSTEM, PR.JUDGE_TEMPLATE, PR.STALE_TEMPLATE):
            assert t in readme


def test_checkpoints():
    from chronofact.data import Question, Session
    ss = [Session(str(i), datetime(2023, 1, 1 + i), [], is_evidence=(i == 3)) for i in range(10)]
    q = Question("q", "t", "?", "a", datetime(2023, 2, 1), False, ["3"], ss)
    # e = 4, n = 10: 4 + round(f * 6): 4, round(1.5)=2 -> 6, 7, round(4.5)=4 -> 8, 10
    assert q.checkpoints() == [(0.0, 4), (0.25, 6), (0.5, 7), (0.75, 8), (1.0, 10)]
