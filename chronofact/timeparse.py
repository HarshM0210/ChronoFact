"""Time expression resolution and question time mode: README Appendix C.5 code, unchanged, except that
resolve() wraps it: an expression that cannot become a calendar date ("4000 years ago" -> year -1977, or a
dateparser overflow) resolves to None instead of crashing the write/read path (a Model3 run crashed on it)."""
import re
from datetime import datetime, timedelta

import dateparser

MONTHS = ("january|february|march|april|may|june|july|august|september|october|november|december|"
          "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec")
_NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9, "ten": 10, "a couple of": 2, "couple of": 2,
        "a few": 3, "few": 3, "several": 3}
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _month_start(d):
    return datetime(d.year, d.month, 1)


def _add_months(d, n):
    m = d.month - 1 + n
    return datetime(d.year + m // 12, m % 12 + 1, 1)


def _day(d):
    return datetime(d.year, d.month, d.day)


def _resolve(expr: str, ref: datetime):
    """README Appendix C.5, unchanged."""
    if not expr:
        return None
    e = expr.lower().strip()
    e = re.sub(r"^(on|in|at|during|since|from|around|about|back in)\s+", "", e)
    if e in ("now", "currently", "today", "tonight", "this morning", "this evening", "recently",
             "just", "lately", "these days", "right now", "at the moment"):
        return _day(ref), _day(ref) + timedelta(days=1), "day"
    if e in ("yesterday", "last night"):
        d = _day(ref) - timedelta(days=1)
        return d, d + timedelta(days=1), "day"
    if e in ("this week",):
        d = _day(ref) - timedelta(days=ref.weekday())
        return d, d + timedelta(days=7), "week"
    if e in ("last week", "the other week", "a week ago"):
        d = _day(ref) - timedelta(days=ref.weekday() + 7)
        return d, d + timedelta(days=7), "week"
    if e in ("last weekend", "this past weekend", "over the weekend", "the weekend"):
        sat = _day(ref) - timedelta(days=(ref.weekday() - 5) % 7 or 7)
        return sat, sat + timedelta(days=2), "day"
    if e in ("this month",):
        return _month_start(ref), _add_months(ref, 1), "month"
    if e in ("last month", "a month ago"):
        return _add_months(ref, -1), _month_start(ref), "month"
    if e in ("this year",):
        return datetime(ref.year, 1, 1), datetime(ref.year + 1, 1, 1), "year"
    if e in ("last year", "a year ago"):
        return datetime(ref.year - 1, 1, 1), datetime(ref.year, 1, 1), "year"
    m = re.fullmatch(r"(last|this past|past)\s+(" + "|".join(_WEEKDAYS) + r")", e)
    if m:
        wd = _WEEKDAYS.index(m.group(2))
        back = (ref.weekday() - wd) % 7 or 7
        d = _day(ref) - timedelta(days=back)
        return d, d + timedelta(days=1), "day"
    m = re.fullmatch(r"(\d+|" + "|".join(sorted(_NUM, key=len, reverse=True)) +
                     r")\s+(day|week|month|year)s?\s+ago", e)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else _NUM[m.group(1)]
        unit = m.group(2)
        if unit == "day":
            d = _day(ref) - timedelta(days=n)
            return d, d + timedelta(days=1), "day"
        if unit == "week":
            d = _day(ref) - timedelta(weeks=n)
            return d, d + timedelta(days=1), "day"
        if unit == "month":
            d = _add_months(ref, -n)
            return d, _add_months(d, 1), "month"
        return datetime(ref.year - n, 1, 1), datetime(ref.year - n + 1, 1, 1), "year"
    m = re.fullmatch(r"((19|20)\d\d)", e)
    if m:
        y = int(m.group(1))
        return datetime(y, 1, 1), datetime(y + 1, 1, 1), "year"
    m = re.fullmatch(r"(" + MONTHS + r")\.?(\s+((19|20)\d\d))?", e)
    if m:
        d = dateparser.parse(e, settings={"RELATIVE_BASE": ref, "PREFER_DATES_FROM": "past",
                                          "PREFER_DAY_OF_MONTH": "first"})
        if d:
            return _month_start(d), _add_months(d, 1), "month"
    # anything else: dateparser (absolute dates like "March 3rd", "2023-04-12", "next Friday")
    try:
        d = dateparser.parse(e, settings={"RELATIVE_BASE": ref, "PREFER_DATES_FROM": "past",
                                          "RETURN_AS_TIMEZONE_AWARE": False})
    except Exception:
        d = None
    if d:
        d = _day(d)
        return d, d + timedelta(days=1), "day"
    return None


def resolve(expr: str, ref: datetime):
    """Resolve a time expression against the reference (turn) timestamp -> (start, end, precision) or None."""
    try:
        return _resolve(expr, ref)
    except (ValueError, OverflowError):
        return None


# ---------------------------------------------------------------------------- question time mode
_HISTORY = re.compile(r"\b(used to|before|previous(ly)?|formerly|originally|at first|initially|"
                      r"what changed|how many times|change[ds]?|history|over time|first time|"
                      r"earlier|old\b|no longer|anymore|switch(ed)?)\b")
_ASOF = re.compile(r"\b(in (" + MONTHS + r")|last (year|month|week|weekend|" + "|".join(_WEEKDAYS) +
                   r")|on (" + MONTHS + r")|\d+ (days?|weeks?|months?|years?) ago|"
                   r"(a|one|two|three|four|five|six|a couple of|a few) (days?|weeks?|months?|years?) ago|"
                   r"in (19|20)\d\d|yesterday|(" + MONTHS + r") \d{1,2})\b")
_CURRENT = re.compile(r"\b(now|currently|current|these days|at the moment|nowadays|still|"
                      r"today|presently|latest|most recent(ly)?)\b")
_PRESENT = re.compile(r"\b(do i|am i|is my|are my|does my|have i|i have|i am|i'm|my)\b")


def question_mode(question: str, as_of: datetime):
    q = question.lower()
    if _HISTORY.search(q):
        return "history", None, None
    m = _ASOF.search(q)
    if m:
        r = resolve(m.group(0), as_of)
        if r:
            return "as_of", r[0], r[1]
    if _CURRENT.search(q) or _PRESENT.search(q):
        return "current", None, None
    return "none", None, None
