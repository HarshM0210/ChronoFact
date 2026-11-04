"""Prompts and JSON schemas: the design spec's prompts verbatim (do not edit by hand), plus the batched-extraction /
labeling variants (profile spark) and the W2 update check."""



EXTRACT_SYSTEM = 'Extract durable personal facts about the USER from the user\'s message to an AI assistant.\nReturn JSON {"facts": [...]} with at most 6 facts, most important first, or {"facts": []}.\n\nOnly facts the user asserts about their own life: people, pets, possessions, events, confirmed plans, preferences, habits, numbers and changes. Users often ask for help and mention personal facts in passing (e.g. "Any tips for my garden? By the way, I planted 12 tomato saplings last week"): always extract those facts. Do not turn the request itself into a fact (asking about something does not mean the user has or does it), and skip hypotheticals, general knowledge and anything the assistant said.\n\nFields:\n- subject: "user", or a named person/pet in the user\'s life (e.g. "Max (user\'s dog)").\n- family: one of the allowed values. residence and workplace are only for where the user currently lives or works; trips, visits and planned or temporary stays are visit or plan. state (one current value): age, commute, count, current_activity, degree, device, diet, favourite, finance, job_title, measurement, name, other_state, partner, progress, record, relationship_status, residence, routine, school, vehicle, workplace. event (a dated occurrence): achievement, activity_done, attended, finished, gift, health_event, media_consumed, met_person, other_event, purchase, started, visit. attribute (ongoing, can have many): dislikes, family_member, friend, goal, health_condition, hobby, interest, likes, member_of, other_attribute, owns, pet, plan, preference, skill, uses.\n- slot_key: short snake_case name of the exact attribute, e.g. times_worn_black_converse, current_city, 5k_personal_best, korean_restaurants_tried.\n- object: the value, at most 8 words, with amounts, prices and names.\n- assertion_type: fact | event | confirmed_plan | intention (only thinking about / might / considering) | preference.\n- time_expression: the phrase that dates it, copied verbatim ("yesterday", "two weeks ago", "in 2019"), or "".\n- statement: one short third-person sentence stating the fact, e.g. "The user has worn their black Converse six times."\n- quote: an exact, contiguous substring of the user\'s message (at most 30 words) that states it.'

EXTRACT_SCHEMA = {'type': 'object', 'properties': {'facts': {'type': 'array', 'maxItems': 6, 'items': {'type': 'object', 'properties': {'subject': {'type': 'string'}, 'family': {'type': 'string', 'enum': ['achievement', 'activity_done', 'age', 'attended', 'commute', 'count', 'current_activity', 'degree', 'device', 'diet', 'dislikes', 'family_member', 'favourite', 'finance', 'finished', 'friend', 'gift', 'goal', 'health_condition', 'health_event', 'hobby', 'interest', 'job_title', 'likes', 'measurement', 'media_consumed', 'member_of', 'met_person', 'name', 'other_attribute', 'other_event', 'other_state', 'owns', 'partner', 'pet', 'plan', 'preference', 'progress', 'purchase', 'record', 'relationship_status', 'residence', 'routine', 'school', 'skill', 'started', 'uses', 'vehicle', 'visit', 'workplace']}, 'slot_key': {'type': 'string'}, 'object': {'type': 'string'}, 'object_type': {'type': 'string', 'enum': ['person', 'place', 'organization', 'product', 'event', 'activity', 'media', 'animal', 'food', 'quantity', 'date', 'other']}, 'assertion_type': {'type': 'string', 'enum': ['fact', 'event', 'confirmed_plan', 'intention', 'preference']}, 'time_expression': {'type': 'string'}, 'statement': {'type': 'string'}, 'quote': {'type': 'string'}}, 'required': ['subject', 'family', 'slot_key', 'object', 'object_type', 'assertion_type', 'time_expression', 'statement', 'quote']}}}, 'required': ['facts']}

LABEL_SYSTEM = 'You file one fact about a user. Answer these in order.\n1. kind. Did it happen once and finish? event. Is it one value that can change over time, so a new value replaces the old? state. Can several be true at once? attribute.\n2. family, matching the kind. event: achievement, activity_done, attended, finished, gift, health_event, media_consumed, met_person, other_event, purchase, started, visit. state: age, commute, count, current_activity, degree, device, diet, favourite, finance, job_title, measurement, name, other_state, partner, progress, record, relationship_status, residence, routine, school, vehicle, workplace. attribute: dislikes, family_member, friend, goal, health_condition, hobby, interest, likes, member_of, other_attribute, owns, pet, plan, preference, skill, uses. Use other_event, other_state or other_attribute only if nothing else fits. residence and workplace are only for where the user currently lives or works; plans go under plan.\n3. slot_key: short snake_case name of the exact thing. If one of the existing keys names the same thing, copy it exactly.\n4. statement: one third-person sentence, at most 30 words, with the useful detail from the message: who, what, how much, when, and why it matters. Use only what the message says. Never add a detail, number or date that is not in it.\nReturn JSON {"kind": ..., "family": ..., "slot_key": ..., "statement": ...}.'

LABEL_SCHEMA = {'type': 'object', 'properties': {'kind': {'type': 'string', 'enum': ['event', 'state', 'attribute']}, 'family': {'type': 'string', 'enum': ['achievement', 'activity_done', 'age', 'attended', 'commute', 'count', 'current_activity', 'degree', 'device', 'diet', 'dislikes', 'family_member', 'favourite', 'finance', 'finished', 'friend', 'gift', 'goal', 'health_condition', 'health_event', 'hobby', 'interest', 'job_title', 'likes', 'measurement', 'media_consumed', 'member_of', 'met_person', 'name', 'other_attribute', 'other_event', 'other_state', 'owns', 'partner', 'pet', 'plan', 'preference', 'progress', 'purchase', 'record', 'relationship_status', 'residence', 'routine', 'school', 'skill', 'started', 'uses', 'vehicle', 'visit', 'workplace']}, 'slot_key': {'type': 'string'}, 'statement': {'type': 'string'}}, 'required': ['kind', 'family', 'slot_key', 'statement']}

CONTROLLER_SYSTEM = 'The user said something is no longer true. Pick the existing ACTIVE fact it ends.\n- CLOSE: that fact is no longer true (give its id).\n- NOOP: none of the listed facts is ended by this statement.\nReturn JSON {"op": "CLOSE" | "NOOP", "target_fact_id": <id or null>, "reason": "<one short sentence>"}.'

CONTROLLER_SCHEMA = {'type': 'object', 'properties': {'op': {'type': 'string', 'enum': ['CLOSE', 'NOOP']}, 'target_fact_id': {'type': ['integer', 'null']}, 'reason': {'type': 'string'}}, 'required': ['op', 'target_fact_id', 'reason']}

JUDGE_TEMPLATE = 'Grade the answer as CORRECT or WRONG.\nQuestion: {question}\nGold answer: {gold}\nModel answer: {answer}\nRules:\n- CORRECT only if the model answer states the same fact as the gold answer. Paraphrase and date formats are fine.\n- A vaguer answer, a list that includes the gold among wrong options, or a hedge is WRONG.\n- If the gold says the information is not available, CORRECT only if the model answer says it is not in memory.\nReturn JSON: {"reason": "<one sentence>", "label": "CORRECT" | "WRONG"}'

JUDGE_SCHEMA = {'type': 'object', 'properties': {'reason': {'type': 'string'}, 'label': {'type': 'string', 'enum': ['CORRECT', 'WRONG']}}, 'required': ['reason', 'label']}

STALE_TEMPLATE = 'A user\'s information changed over time. Below are the user\'s own messages that contain the relevant information, with dates.\n{evidence lines}\n\nQuestion: {question}\nCorrect (latest) answer: {gold}\nModel answer: {answer}\n\nIs the model answer an OLDER value of this information, i.e. one that was true earlier according to the messages but was later updated? Answer false if the model answer is correct, unrelated, made up, or says it does not know.\nReturn JSON: {"reason": "<one sentence>", "stale": true | false}'

STALE_SCHEMA = {'type': 'object', 'properties': {'reason': {'type': 'string'}, 'stale': {'type': 'boolean'}}, 'required': ['reason', 'stale']}

EXTRACT_FALLBACK_LINE = "Return compact JSON on one line, at most 3 facts."   # README 5.3

JUDGE_RETRY_LINE = "\nKeep the reason under 12 words."                   # README 10


# ---------------------------------------------------------------------------- batched write path (Model5 on Spark)
# The contributor tier allows 150 requests/minute, so the write path asks for several user messages (extraction)
# or several facts (labeling) per call. The instructions are the README prompts above, word for word; only the
# framing (numbered items, one output list) is added. Each fact names the message it comes from, and its quote is
# still checked against THAT message only.
_EXTRACT_RULES = EXTRACT_SYSTEM.split("\n\n", 1)[1]

EXTRACT_BATCH_SYSTEM = (
    "Extract durable personal facts about the USER from the user's messages to an AI assistant.\n"
    "You get several numbered user messages, each with its date. Treat every message on its own, exactly as if it "
    "were the only one. For each message return at most 6 facts, most important first. Put all facts in one list "
    "and give each fact the number of the message it comes from in \"message\". Return JSON {\"facts\": [...]}, or "
    "{\"facts\": []} if no message states a fact.\n\n"
    + _EXTRACT_RULES.replace("- quote: an exact, contiguous substring of the user's message",
                             "- quote: an exact, contiguous substring of THAT user message")
    + "\n- message: the number of the message the fact comes from.")
assert EXTRACT_BATCH_SYSTEM.count("THAT user message") == 1


def extract_batch_schema(n_messages):
    item = dict(EXTRACT_SCHEMA["properties"]["facts"]["items"])
    item = {"type": "object", "properties": dict(item["properties"], message={"type": "integer"}),
            "required": item["required"] + ["message"]}
    return {"type": "object", "properties": {"facts": {"type": "array", "maxItems": 6 * n_messages, "items": item}},
            "required": ["facts"]}


# The framing asks for the per-fact care explicitly: on 60 smoke facts, a plain "file each one on its own" framing
# enriched 6 statements at 20 per call (per-fact README prompt: 13); this framing at 10 per call enriched 27, with
# every rich statement passing the numbers and NLI checks (2026-11-04).
LABEL_BATCH_SYSTEM = (
    LABEL_SYSTEM.replace("You file one fact about a user. Answer these in order.",
                         "You file facts about a user. The facts are numbered. File every fact on its own, with the "
                         "same care as if it were the only one, using only its own message and its own existing keys: "
                         "for each statement, reread that fact's message and add every useful detail it gives. For "
                         "each fact, answer these in order.")
    .replace('Return JSON {"kind": ..., "family": ..., "slot_key": ..., "statement": ...}.',
             'Return JSON {"labels": [{"id": <fact number>, "kind": ..., "family": ..., "slot_key": ..., '
             '"statement": ...}, ...]} with exactly one entry per fact.'))
assert LABEL_BATCH_SYSTEM.count("<fact number>") == 1 and "You file one fact" not in LABEL_BATCH_SYSTEM


# W2 rule (c): same subject + same slot key + different values -> does the newer fact replace the older one? The
# README's cosine/NLI gates never fire on Spark + Contriever stores, and NLI alone reaches only 50% precision on these
# pairs (scripts/calibrate_w2c.py, results/calibration_w2c.json); this prompt is the one those labels were made with.
W2_UPDATE_TEMPLATE = (
    'Two facts a memory system stored about a user, the second recorded later.\nOlder ({d1}): {a}\nNewer ({d2}): {b}\n'
    'Does the newer fact REPLACE the older one, i.e. both are about the same single attribute of the user (the same '
    'thing, not just a similar topic) and the value changed, so the older fact is no longer true? Two facts that can '
    'both be true at once do not replace each other.\nReturn JSON {{"reason": "<one short sentence>", "yes": true | false}}')

W2_UPDATE_SCHEMA = {"type": "object", "properties": {"reason": {"type": "string"}, "yes": {"type": "boolean"}},
                    "required": ["reason", "yes"]}


def label_batch_schema(n_facts):
    item = {"type": "object", "properties": dict({"id": {"type": "integer"}}, **LABEL_SCHEMA["properties"]),
            "required": ["id"] + LABEL_SCHEMA["required"]}
    return {"type": "object", "properties": {"labels": {"type": "array", "maxItems": n_facts, "items": item}},
            "required": ["labels"]}
