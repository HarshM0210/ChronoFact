"""Answering (README section 8) and the judge + stale check (README section 10)."""
import re

from chronofact import prompts as PR
from chronofact.config import CFG
from chronofact.llm import LLMJSONError, acall

ABSTAIN = "NOT IN MEMORY"
_FINAL = re.compile(r"^\s*\**\s*final\s*(answer)?\s*\**\s*[:\-]\s*(.*)$", re.I)       # README C.12


def answer_prompt(question, question_date, context_text, mode):
    lines = ["You are answering a question about a user, using only the memory below.",
             f"Current date: {question_date.strftime('%Y/%m/%d (%a) %H:%M')}",
             "Memory:", context_text or "(empty)", f"Question: {question}"]
    if mode == "reason":
        lines += ["First think step by step in at most 6 short lines, using only the memory (find the relevant "
                  "entries, compare dates, count carefully, prefer the most recent value when values changed). "
                  "Then write the final answer on its own last line as:",
                  f"FINAL: <one short phrase, or exactly {ABSTAIN} if the memory does not contain the answer>"]
    else:
        lines += [f"Answer in one short phrase. If the memory does not contain the answer, reply exactly: {ABSTAIN}"]
    return "\n".join(lines)


def has_final(text):
    return any(_FINAL.match(l) for l in (text or "").splitlines())


def parse_final(text):
    """README C.12: last 'FINAL: ...' line; empty -> the next line; none -> last non-empty line."""
    lines = [l for l in (text or "").splitlines() if l.strip()]
    for i in range(len(lines) - 1, -1, -1):
        m = _FINAL.match(lines[i])
        if m:
            val = m.group(2).strip().strip("*\"' ")
            if val:
                return val
            if i + 1 < len(lines):
                return lines[i + 1].strip().strip("*\"' ")
    return lines[-1].strip().strip("*\"' ") if lines else ""


def is_abstention(answer_text):
    return ABSTAIN.lower() in (answer_text or "").lower()


FINAL_NUDGE = (f"Stop reasoning. Reply with only the final answer line: FINAL: <one short phrase, or exactly "
               f"{ABSTAIN} if the memory does not contain the answer>")


async def answer(question, question_date, context_text, mode, model=None):
    """-> (final answer, CallStats, raw output, error or None, recovered). A request the API rejects (4xx, e.g. a
    content filter) gets the empty answer: the row is kept, judged WRONG and counted (answer_error), never dropped,
    and one bad row cannot stop the step.
    Reason mode: when the reply has no FINAL line (Llama 8B can loop in its reasoning until max_tokens), one more
    call asks for the final line only (the same for every reason-mode system); raw is then both replies, so the
    answer still re-parses from raw, and recovered=True records it."""
    prompt = answer_prompt(question, question_date, context_text, mode)
    msgs = [{"role": "user", "content": prompt}]
    mt = CFG["answer"]["reason_max_tokens" if mode == "reason" else "short_max_tokens"]
    try:
        raw, st = await acall("answerer", msgs, max_tokens=mt, model=model)
    except LLMJSONError as e:
        return "", e.stats, "", str(e)[:300], False
    if mode != "reason":
        return raw.strip(), st, raw, None, False
    if has_final(raw):
        return parse_final(raw), st, raw, None, False
    try:
        raw2, st2 = await acall("answerer", msgs + [{"role": "assistant", "content": raw},
                                                    {"role": "user", "content": FINAL_NUDGE}], max_tokens=48, model=model)
    except LLMJSONError as e:
        return parse_final(raw), st.add(e.stats), raw, None, False
    st.add(st2)
    both = raw.rstrip() + "\n" + raw2.strip()
    return parse_final(both), st, both, None, True


def _fill(tpl, **kw):
    """single pass, so a placeholder-like string inside an answer is never substituted"""
    pat = re.compile("|".join(re.escape("{" + k + "}") for k in kw))
    return pat.sub(lambda m: str(kw[m.group(0)[1:-1]]), tpl)


async def _judge_call(prompt, schema):
    try:
        return await acall("judge", [{"role": "user", "content": prompt}], schema)
    except LLMJSONError as e:     # the failed attempts are cached: change the prompt, not just retry
        if e.rejected:            # refused by the content filter: never re-sent
            raise
        return await acall("judge", [{"role": "user", "content": prompt + PR.JUDGE_RETRY_LINE}], schema)


async def judge(question, gold, answer_text):
    p = _fill(PR.JUDGE_TEMPLATE, question=question, gold=gold, answer=answer_text)
    out, st = await _judge_call(p, PR.JUDGE_SCHEMA)
    label = out.get("label") if isinstance(out, dict) else None
    if label not in ("CORRECT", "WRONG"):
        raise LLMJSONError(f"judge returned no valid label: {out!r}", st)
    return label == "CORRECT", str(out.get("reason", "")), st


async def stale_check(question, gold, answer_text, evidence_lines):
    ev = "\n".join(evidence_lines)[:12000]
    p = _fill(PR.STALE_TEMPLATE, **{"evidence lines": ev, "question": question, "gold": gold, "answer": answer_text})
    out, st = await _judge_call(p, PR.STALE_SCHEMA)
    if not isinstance(out, dict) or not isinstance(out.get("stale"), bool):
        raise LLMJSONError(f"stale check returned no boolean: {out!r}", st)
    return out["stale"], str(out.get("reason", "")), st
