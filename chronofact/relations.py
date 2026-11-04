"""Families, kinds, literal objects, question slots (README B.6, B.7, B.13; code from Appendix C.4).
Model5 facts always carry kind/family/assertion_type, so the Model1 fallbacks of kind_of/is_intention are not
needed; they are kept minimal and only fall back to the README's family table."""
import re

LITERAL = {"age", "salary", "rent", "weight", "height", "budget", "zip_code", "commute_duration",
           "workout_frequency", "gym_schedule", "sleep_schedule", "marital_status", "diet",
           "relationship_status", "degree", "job_title", "home_type", "spent_on"}

LITERAL_PREFIXES = ("personal_best_", "number_of_", "progress_", "frequency_of_", "total_",
                    "amount_")

LITERAL_FAMILIES = ("count", "record", "progress", "measurement", "finance", "age", "routine", "commute")


def is_literal(rel: str, obj: str) -> bool:
    if rel in LITERAL or rel.startswith(LITERAL_PREFIXES):
        return True
    o = obj.strip()
    return bool(re.fullmatch(r"[\d\s.,:$%/\-]+|\d.*(minutes?|hours?|days?|weeks?|months?|years?|"
                             r"miles?|km|kilometers?|pounds?|lbs?|kg|dollars?|times?)", o.lower())) \
        or len(o.split()) > 8


def literal_object(family: str, obj: str) -> bool:
    """README B.7: is_literal(family, object) or the family is one of the value families."""
    return is_literal(family, obj) or family in LITERAL_FAMILIES


SLOT_TRIGGERS = {
    "lives_in": r"\b(where do i live|live in|living in|my (city|town|home ?town|address))\b",
    "works_at": r"\b(where do i work|work(ing)? (at|for)|my (employer|company))\b",
    "job_title": r"\b(my (job|role|position|title|occupation))\b",
    "studies_at": r"\b(where do i study|my (school|university|college))\b",
    "has_partner": r"\b(my (partner|wife|husband|boyfriend|girlfriend|spouse))\b",
    "has_pet": r"\b(my (pet|dog|cat|hamster|bird|fish|rabbit|parrot|turtle)|pet'?s name)\b",
    "owns_car": r"\b(my car|what car|which car)\b",
    "phone_model": r"\b(my phone|what phone)\b",
    "degree": r"\b(degree|graduate[d]? (with|in)|major)\b",
    "age": r"\b(how old|my age)\b",
    "has_doctor": r"\b(my (doctor|dermatologist|dentist|therapist|physician))\b",
    "commute_duration": r"\b(commute)\b",
    "favourite_": r"\b(favou?rite)\b",
    "personal_best_": r"\b(personal best|record time|fastest)\b",
    "number_of_": r"\bhow many\b",
}


def question_slots(question: str) -> list:
    q = question.lower()
    return [rel for rel, pat in SLOT_TRIGGERS.items() if re.search(pat, q)]


def matches_slot(rel: str, slot: str) -> bool:
    return rel.startswith(slot) if slot.endswith("_") else rel == slot


FAMILIES = {
    # state
    "residence": "state", "workplace": "state", "job_title": "state", "school": "state",
    "relationship_status": "state", "partner": "state", "vehicle": "state", "device": "state",
    "diet": "state", "age": "state", "degree": "state", "commute": "state", "routine": "state",
    "measurement": "state", "finance": "state", "count": "state", "record": "state",
    "progress": "state", "favourite": "state", "current_activity": "state", "name": "state",
    "other_state": "state",
    # event
    "visit": "event", "purchase": "event", "attended": "event", "activity_done": "event",
    "started": "event", "finished": "event", "media_consumed": "event", "gift": "event",
    "health_event": "event", "met_person": "event", "achievement": "event", "other_event": "event",
    # attribute
    "likes": "attribute", "dislikes": "attribute", "owns": "attribute", "pet": "attribute",
    "family_member": "attribute", "friend": "attribute", "skill": "attribute", "hobby": "attribute",
    "goal": "attribute", "plan": "attribute", "interest": "attribute", "uses": "attribute",
    "member_of": "attribute", "health_condition": "attribute", "preference": "attribute",
    "other_attribute": "attribute",
}
FAMILY_LIST = sorted(FAMILIES)
ASSERTION_TYPES = ["fact", "event", "confirmed_plan", "intention", "preference"]
OBJECT_TYPES = ["person", "place", "organization", "product", "event", "activity", "media", "animal", "food",
                "quantity", "date", "other"]


def kind_of(f):
    return f.get("kind") or FAMILIES.get(f.get("family") or f.get("relation") or "", "attribute")


def is_intention(f):
    return f.get("assertion_type") == "intention"


def slot_of(f):
    """Slot identity (subject, family, slot key)."""
    return (f["subj_id"], f.get("family") or f.get("relation"), f.get("slot_key") or "")


def norm_slot_key(s):
    toks = re.findall(r"[a-z0-9]+", (s or "").lower())
    toks = [t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t for t in toks]
    return "_".join(toks)[:80]
