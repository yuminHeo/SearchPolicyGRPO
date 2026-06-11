from __future__ import annotations

import re
from typing import Any


TRUE_SET = {"true", "1", "yes", "correct", "supports", "supported"}
FALSE_SET = {"false", "0", "no", "incorrect", "refutes", "refuted"}
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL)
BOXED_RE = re.compile(r"\\boxed\{(.*?)\}", flags=re.DOTALL)
DOC_ID_RE = re.compile(r"\[([^\]\n]{1,160})\]")


def normalize_label(value: Any) -> str:
    text = str(value).strip().lower()
    if text in TRUE_SET:
        return "true"
    if text in FALSE_SET:
        return "false"
    return "unknown"


def extract_boxed_answer_label(text: str) -> str:
    answers = ANSWER_RE.findall(text or "")
    if not answers:
        return "unknown"
    boxed = BOXED_RE.findall(answers[-1])
    if not boxed:
        return "unknown"
    return normalize_label(boxed[-1])


def retrieved_doc_ids(text: str) -> list[str]:
    seen: set[str] = set()
    doc_ids: list[str] = []
    for match in DOC_ID_RE.finditer(text or ""):
        doc_id = match.group(1).strip()
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            doc_ids.append(doc_id)
    return doc_ids


def gold_evidence_hit(response: str, extra_info: dict[str, Any]) -> bool:
    metadata = dict((extra_info or {}).get("metadata") or {})
    gold = (extra_info or {}).get("gold_evidence") or metadata.get("gold_evidence") or []
    if isinstance(gold, str):
        gold = [gold]
    gold = [str(item).strip() for item in gold if str(item).strip()]
    if not gold:
        return False
    doc_id_set = set(retrieved_doc_ids(response))
    lower_response = (response or "").lower()
    return any(item in doc_id_set or item.lower() in lower_response for item in gold)


def strip_prompt_leakage(text: str) -> str:
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant")[-1]
    if "<think>" in text:
        text = text[text.find("<think>") :]
    return text.strip()


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Any | None = None,
    search_penalty: float = -0.05,
    evidence_reward: float = 0.5,
) -> dict[str, Any]:
    del data_source
    extra = dict(extra_info or {})
    response = strip_prompt_leakage(solution_str or "")
    gold = normalize_label(extra.get("label", ground_truth))
    pred = extract_boxed_answer_label(response)
    is_correct = pred == gold and gold in {"true", "false"}

    if pred == "unknown":
        r_correct = -0.5
    elif is_correct:
        r_correct = 1.0
    else:
        r_correct = 0.0

    num_searches = len(re.findall(r"<search>.*?</search>", response, flags=re.DOTALL))
    r_search = max(0, num_searches - 1) * float(search_penalty)
    evidence_hit = gold_evidence_hit(response, extra)
    r_evidence = float(evidence_reward) if is_correct and evidence_hit else 0.0
    reward = r_correct + r_evidence + r_search

    return {
        "score": round(float(reward), 4),
        "pred": pred,
        "gold": gold,
        "correct": int(is_correct),
        "unknown": int(pred == "unknown"),
        "num_searches": num_searches,
        "r_correct": round(r_correct, 4),
        "r_evidence": round(r_evidence, 4),
        "r_search": round(r_search, 4),
        "gold_evidence_hit": int(evidence_hit),
    }

