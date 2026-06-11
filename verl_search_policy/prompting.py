from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


TRUE_SET = {"true", "1", "yes", "correct", "supports", "supported"}
FALSE_SET = {"false", "0", "no", "incorrect", "refutes", "refuted"}


def normalize_label(value: Any) -> str:
    text = str(value).strip().lower()
    if text in TRUE_SET:
        return "true"
    if text in FALSE_SET:
        return "false"
    return "unknown"


@dataclass
class TripleExample:
    id: str
    subject: str
    predicate: str
    object: str
    label: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
    gold_evidence: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "TripleExample":
        metadata = dict(row.get("metadata") or {})
        gold_evidence = row.get("gold_evidence") or metadata.get("gold_evidence") or []
        if isinstance(gold_evidence, str):
            gold_evidence = [gold_evidence]
        return cls(
            id=str(row.get("id") or row.get("triple_id") or ""),
            subject=str(row.get("subject") or row.get("s") or ""),
            predicate=str(row.get("predicate") or row.get("p") or ""),
            object=str(row.get("object") or row.get("o") or ""),
            label=normalize_label(row.get("label", row.get("gold_label", "unknown"))),
            metadata=metadata,
            gold_evidence=[str(item) for item in gold_evidence],
        )

    def to_prompt_input(self) -> str:
        return (
            "Verify whether the following knowledge-graph triple is true or false.\n"
            f"Subject: {self.subject}\n"
            f"Predicate: {self.predicate}\n"
            f"Object: {self.object}\n"
            "Use iterative search when needed. Search queries should be based on the subject, "
            "predicate, object, and later refined using retrieved evidence. The final answer must "
            "be exactly <answer>\\boxed{true}</answer> or <answer>\\boxed{false}</answer>."
        )


SEARCH_POLICY_SYSTEM_PROMPT = """You are a triple verification search agent.
Given a knowledge-graph triple (subject, predicate, object), decide whether the triple is true or false.
First decide whether the triple can be answered from stable general knowledge or must be verified by search.
Use the search tool for time-sensitive facts, dated facts, recent/current facts, obscure facts, or any uncertain information.

Use one of these trajectory formats.

If you decide NOT to search because the triple is stable and certain from general knowledge, output immediately:
<answer>\\boxed{true}</answer> or <answer>\\boxed{false}</answer>

If you decide to search, use this format:
<think>why search is needed and what evidence is needed</think>
<search>search query</search>

After the retrieval system provides evidence, use this format:
<think>updated reasoning and evidence sufficiency check</think>
<search>next search query</search>
or
<think>updated reasoning and evidence sufficiency check</think>
<answer>\\boxed{true}</answer> or <answer>\\boxed{false}</answer>

Search-control rules:
1. First decide whether to search or answer directly.
2. If the triple involves a point in time, date, temporal relation, recent/current status, obscure entity, or uncertain fact, you must search.
3. If you decide not to search, output only the final <answer> immediately.
4. If you decide to search, treat the retrieved <result> and <result_summary> evidence as the context information. Given the context information and without prior knowledge, evaluate whether the information in the documents supports the triple.
5. After every <result>, write a <think> step that explicitly checks whether the evidence covers subject identity, object identity, and the predicate relation between them.
6. If any of those three parts are missing, ambiguous, or only indirectly implied, do not answer yet. Issue another <search> with a different query targeting the missing or ambiguous part.
7. Older search results may appear as <result_summary>. Use them as compact memory and use the latest full <result> as the most detailed current evidence.
8. The final answer must be exactly \\boxed{true} or \\boxed{false}.
9. Never repeat the same search query."""


SEARCH_ENV_TEMPLATE = """import requests

def triple_search(query: str, top_n: int = 5):
    url = "<search-url-placeholder>/search"
    if query == "":
        return "invalid query"
    payload = {"query": query, "top_n": top_n}
    response = requests.post(url, json=payload, timeout=20)
    if response.status_code == 422:
        response = requests.post(url, params=payload, timeout=20)
    if response.status_code >= 400:
        return f"search_error status={response.status_code} body={response.text[:500]}"
    payload = response.json()
    if isinstance(payload, dict):
        if isinstance(payload.get("documents"), list):
            docs = payload["documents"]
        elif isinstance(payload.get("results"), list):
            docs = payload["results"]
        else:
            return f"search_error unexpected_response={payload}"
    elif isinstance(payload, list):
        docs = payload[0] if len(payload) == 2 and isinstance(payload[0], list) else payload
    else:
        return f"search_error unexpected_response={payload}"
    text = ""
    for doc in docs:
        if not isinstance(doc, dict):
            text += f"{doc}\\n\\n"
            continue
        doc_id = doc.get("id", "")
        title = doc.get("title", "")
        contents = doc.get("contents", "")
        text += f"[{doc_id}] {title}\\n{contents}\\n\\n"
    return text.strip()
"""


TOOL_SCHEMAS = json.dumps(
    [
        {
            "type": "function",
            "function": {
                "name": "triple_search",
                "description": "Search evidence for verifying a subject-predicate-object triple.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Evidence search query."},
                        "top_n": {"type": "integer", "description": "Number of documents to return.", "default": 5},
                    },
                    "required": ["query"],
                },
            },
        }
    ],
    indent=2,
)


def build_env(search_url: str) -> str:
    return SEARCH_ENV_TEMPLATE.replace("<search-url-placeholder>", search_url.rstrip("/"))

