"""Write-path rules that need no model: README Appendix C.1, C.2, C.11 (code unchanged) and B.5, B.9."""
import re

from chronofact import relations as R

# ------------------------------------------------------------------------------- C.1
_ACK = re.compile(r"^(thanks?( you)?|thank you( so much)?|ok(ay)?|lol|sure|great|cool|nice|got it|"
                  r"yes|no|yeah|yep|nope|perfect|awesome|sounds good|will do|hmm+|ah|oh)[\s!.,]*$", re.I)

_FIRST = re.compile(r"\b(i|i'm|i've|i'd|i'll|my|mine|me|myself|we|our|us|we're|we've)\b", re.I)

_NUM = re.compile(r"\d")

_NE = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]+")

_DATEW = re.compile(r"\b(today|yesterday|tomorrow|ago|last|next|monday|tuesday|wednesday|thursday|"
                    r"friday|saturday|sunday|january|february|march|april|may|june|july|august|"
                    r"september|october|november|december|week|month|year)\b", re.I)

_NEG = re.compile(r"\b(no longer|not anymore|n't\b.{0,40}\banymore|stopped|quit|gave up|got rid of|"
                  r"sold (my|our)|moved out|broke up|used to|anymore)\b", re.I)


def salience(text):
    """Returns a skip reason, or None if the turn should go to extraction."""
    t = text.strip()
    words = len(t.split())
    if _ACK.match(t):
        return "acknowledgement"
    has_first = bool(_FIRST.search(t))
    signal = has_first or _NUM.search(t) or _NE.search(t) or _DATEW.search(t)
    if not signal and words < 6:
        return "short, no entity/number/date/first-person"
    if not has_first:
        return "no first-person claim"
    return None


def _norm_text(s):
    s = s.lower().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"[^a-z0-9'$%.,\- ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip(" .,")


def quote_ok(quote, turn_text):
    q = _norm_text(quote)
    return len(q) >= 3 and q in _norm_text(turn_text)


_FP = [(r"\bI'm\b", "the user is"), (r"\bI am\b", "the user is"), (r"\bI've\b", "the user has"),
       (r"\bI'd\b", "the user would"), (r"\bI'll\b", "the user will"), (r"\bI\b", "the user"),
       (r"\bmy\b", "the user's"), (r"\bMy\b", "The user's"), (r"\bme\b", "the user"),
       (r"\bmyself\b", "the user"), (r"\bmine\b", "the user's")]


def third_person(s):
    for a, b in _FP:
        s = re.sub(a, b, s)
    return s


def _clip(text, words):
    w = text.split()
    return text if len(w) <= words else " ".join(w[:words]) + " ..."


def _digits(s):
    return re.findall(r"\d+(?:\.\d+)?", s)


# ------------------------------------------------------------------------------- C.2
_REFERENTIAL = re.compile(r"\b(that one|this one|the (first|second|third|fourth|last|other|latter|former)|"
                          r"option \d|number \d|#\d|you (said|mentioned|suggested|recommended)|"
                          r"your (suggestion|recommendation|idea|list)|sounds (good|great|perfect)|"
                          r"i'?ll (go with|take|try) (that|it|those|the))\b", re.I)

_SENT = re.compile(r"(?<=[.!?])\s+|\n+")


def premise_window(quote, turn_text, max_words=90):
    """NLI premise for a quote: the sentence(s) of the turn that contain it plus the sentence before."""
    sents = [x for x in _SENT.split(turn_text) if x.strip()]
    nq = _norm_text(quote)
    for i, snt in enumerate(sents):
        span = " ".join(sents[i:i + 3])
        if _norm_text(quote)[:40] in _norm_text(span):
            j = i
            while j < len(sents) and nq[:40] not in _norm_text(" ".join(sents[i:j + 1])):
                j += 1
            k = j
            while k < len(sents) and nq not in _norm_text(" ".join(sents[i:k + 1])) and k < i + 3:
                k += 1
            w = " ".join(sents[max(0, i - 1):k + 1]).split()
            return " ".join(w[-max_words:]) if len(w) > max_words else " ".join(w)
    return quote


# ------------------------------------------------------------------------------- B.5, C.11, B.9
_LIVES = re.compile(r"\b(liv(e|es|ing|ed)|moved?|moving|reside|home|apartment|flat|house|rent|based in)\b", re.I)

_WORKS = re.compile(r"\b(work(s|ing|ed)?|job|employ\w*|office|company|boss|manager|colleague|coworker|startup)\b", re.I)

SINGLE_FAMILIES = {"residence", "workplace", "job_title", "school", "relationship_status", "partner", "age", "commute",
                   "record", "count", "measurement", "diet", "vehicle"}

GENERIC_SLOTS = {"current_activity", "activity", "current_task", "current_project", "current_interest", "device", "like",
                 "interest", "preference", "progress", "hobby", "goal", "routine", "plan", "habit", "status",
                 "current_status", "current_state", "current_event", "other", "general", "thing", "item", "task"}


def consistent_family(fam, kind, assertion, statement):
    """Label rules that need no LLM. Returns the corrected family."""
    if R.FAMILIES.get(fam) != kind:
        fam = f"other_{kind}"
    if assertion in ("intention", "confirmed_plan"):
        return "plan"
    if assertion == "event" and R.FAMILIES[fam] != "event":
        return "other_event"
    if fam == "residence" and not _LIVES.search(statement):
        return "other_state"
    if fam == "workplace" and not _WORKS.search(statement):
        return "other_state"
    return fam


def rich_ok(rich, turn_text, max_words):
    """numbers must come from the message; length is capped. NLI is applied by the caller in a batch."""
    if not rich or len(rich.split()) > max_words:
        return False
    return set(_digits(rich)) <= set(_digits(turn_text))


# ------------------------------------------------------------------------------- C.9 (read side)
_PLAN = re.compile(r"\b(plan|plans|planning|going to|will i|next|upcoming|future|intend|want to|"
                   r"thinking (about|of)|consider)\b", re.I)
