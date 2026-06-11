#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

try:
    from accelerate import Accelerator, DeepSpeedPlugin
except ImportError as exc:
    raise RuntimeError(
        "stage2_search_grpo requires accelerate. Install accelerate and, for ZeRO training, deepspeed."
    ) from exc


TRUE_SET = {"true", "1", "yes", "correct", "supports", "supported"}
FALSE_SET = {"false", "0", "no", "incorrect", "refutes", "refuted"}
SEARCH_OPEN = "<search>"
SEARCH_CLOSE = "</search>"
ANSWER_OPEN = "<answer>"
ANSWER_CLOSE = "</answer>"
DOC_ID_RE = re.compile(r"\[([^\]\n]{1,160})\]")
SEARCH_RE = re.compile(r"<search>(.*?)</search>", flags=re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", flags=re.DOTALL)
BOXED_RE = re.compile(r"\\boxed\{(.*?)\}", flags=re.DOTALL)


SYSTEM_PROMPT = """You are a triple verification search agent.
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


FINAL_ANSWER_PREFIX = (
    "\n<think>Maximum search turn budget reached. Given the retrieved context information above "
    "and without prior knowledge, evaluate whether the documents support the triple. Do not search "
    "again. Output exactly \\boxed{true} if the triple is correct according to the documents. "
    "Output exactly \\boxed{false} if the triple is incorrect according to the documents, with no "
    "explanation.</think>\n<answer>"
)


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

    def statement(self) -> str:
        return f"({self.subject}, {self.predicate}, {self.object})"

    def user_prompt(self) -> str:
        return (
            "Verify whether the following knowledge-graph triple is true or false.\n"
            f"Subject: {self.subject}\n"
            f"Predicate: {self.predicate}\n"
            f"Object: {self.object}\n"
            "Use iterative search when needed. Search queries should be based on the subject, "
            "predicate, object, and later refined using retrieved evidence. The final answer must "
            "be exactly <answer>\\boxed{true}</answer> or <answer>\\boxed{false}</answer>."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "label": self.label,
            "metadata": self.metadata,
            "gold_evidence": self.gold_evidence,
        }


@dataclass
class ModelStep:
    prompt: str
    output: str
    forced_final: bool = False


@dataclass
class Trajectory:
    example: TripleExample
    group_index: int
    sample_index: int
    prompt: str
    trajectory: str
    model_steps: list[ModelStep] = field(default_factory=list)
    search_count: int = 0
    retrieved_doc_ids: list[str] = field(default_factory=list)
    gold_evidence_hits: set[str] = field(default_factory=set)
    result_history: list[tuple[str, str, str]] = field(default_factory=list)
    finished: bool = False
    forced_final: bool = False
    forced_final_invalid: bool = False
    prediction: str = "unknown"
    reward: float = 0.0
    reward_components: dict[str, Any] = field(default_factory=dict)


class SearchCostController:
    def __init__(
        self,
        initial: float,
        max_value: float,
        step: float,
        target_unknown_rate: float,
        tolerance: float,
        enabled: bool,
    ) -> None:
        self.value = float(initial)
        self.initial = float(initial)
        self.max_value = float(max_value)
        self.step = float(step)
        self.target_unknown_rate = float(target_unknown_rate)
        self.tolerance = float(tolerance)
        self.enabled = enabled

    def update(self, metrics: dict[str, float]) -> float:
        if not self.enabled:
            return self.value
        unknown_rate = float(metrics.get("unknown_rate", 0.0))
        if unknown_rate > self.target_unknown_rate + self.tolerance:
            self.value = max(0.0, self.value - self.step)
        elif unknown_rate < self.target_unknown_rate - self.tolerance:
            self.value = min(self.max_value, self.value + self.step)
        return self.value


class HTTPRetriever:
    def __init__(self, search_url: str, top_n: int, max_result_chars: int, timeout: int = 30) -> None:
        self.search_url = search_url.rstrip("/")
        self.top_n = top_n
        self.max_result_chars = max_result_chars
        self.timeout = timeout

    def search_many(self, queries: list[str]) -> list[str]:
        if not queries:
            return []
        workers = min(len(queries), 32)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(self.search, queries))

    def search(self, query: str) -> str:
        query = query.strip()
        if not query:
            return "search_error: empty query"
        try:
            response = requests.post(
                f"{self.search_url}/search",
                json={"query": query, "top_n": self.top_n},
                timeout=self.timeout,
            )
            if response.status_code == 422:
                response = requests.post(
                    f"{self.search_url}/search",
                    params={"query": query, "top_n": self.top_n},
                    timeout=self.timeout,
                )
            if response.status_code >= 400:
                return f"search_error status={response.status_code} body={response.text[:500]}"
            docs = self._extract_docs(response.json())
            text = self._format_docs(docs)
        except Exception as exc:
            text = f"search_error: {exc}"
        if self.max_result_chars > 0 and len(text) > self.max_result_chars:
            text = text[: self.max_result_chars].rstrip() + "\n...[truncated]"
        return text

    @staticmethod
    def _extract_docs(payload: Any) -> list[Any]:
        if isinstance(payload, dict):
            if isinstance(payload.get("documents"), list):
                return payload["documents"]
            if isinstance(payload.get("results"), list):
                return payload["results"]
            return [payload]
        if (
            isinstance(payload, list)
            and len(payload) == 2
            and isinstance(payload[0], list)
            and isinstance(payload[1], list)
        ):
            return payload[0]
        if isinstance(payload, list):
            return payload
        return [payload]

    @staticmethod
    def _format_docs(docs: list[Any]) -> str:
        chunks = []
        for doc in docs:
            if not isinstance(doc, dict):
                chunks.append(str(doc))
                continue
            doc_id = str(doc.get("id", "")).strip()
            title = str(doc.get("title", "")).strip()
            contents = str(doc.get("contents", doc.get("text", ""))).strip()
            prefix = f"[{doc_id}] {title}".strip()
            chunks.append(f"{prefix}\n{contents}".strip())
        return "\n\n".join(chunk for chunk in chunks if chunk).strip()


def normalize_label(value: Any) -> str:
    text = str(value).strip().lower()
    if text in TRUE_SET:
        return "true"
    if text in FALSE_SET:
        return "false"
    return "unknown"


def read_jsonl(path: str, limit: int = 0) -> list[TripleExample]:
    rows: list[TripleExample] = []
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(TripleExample.from_dict(json.loads(line)))
            if limit and len(rows) >= limit:
                break
    return rows


def qwen_chat_prompt(system_prompt: str, user_prompt: str, assistant_prefix: str = "<think>") -> str:
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_prefix}"
    )


def truncate_at_action_stop(text: str) -> str:
    candidates = []
    for marker in (SEARCH_CLOSE, ANSWER_CLOSE):
        pos = text.find(marker)
        if pos >= 0:
            candidates.append(pos + len(marker))
    if not candidates:
        return text
    return text[: min(candidates)]


def extract_last_search(text: str) -> str:
    matches = SEARCH_RE.findall(text)
    return matches[-1].strip() if matches else ""


def extract_answer_label(text: str) -> str:
    answers = ANSWER_RE.findall(text)
    if not answers:
        return "unknown"
    answer = answers[-1]
    boxed = BOXED_RE.findall(answer)
    if boxed:
        return normalize_label(boxed[-1])
    return normalize_label(answer)


def answer_format_valid(text: str) -> bool:
    if text.count(ANSWER_OPEN) != 1 or text.count(ANSWER_CLOSE) != 1:
        return False
    if text.count("<think>") != text.count("</think>"):
        return False
    answer = ANSWER_RE.findall(text)
    if not answer:
        return False
    boxed = BOXED_RE.findall(answer[-1])
    if not boxed:
        return False
    return normalize_label(boxed[-1]) in {"true", "false"}


def retrieved_doc_ids(result_text: str) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for match in DOC_ID_RE.finditer(result_text):
        doc_id = match.group(1).strip()
        if doc_id and doc_id not in seen:
            ids.append(doc_id)
            seen.add(doc_id)
    return ids


def update_gold_hits(example: TripleExample, result_text: str, hits: set[str]) -> list[str]:
    doc_ids = set(retrieved_doc_ids(result_text))
    lower_result = result_text.lower()
    new_hits: list[str] = []
    for item in example.gold_evidence:
        gold = str(item).strip()
        if not gold:
            continue
        if gold in doc_ids or gold.lower() in lower_result:
            if gold not in hits:
                new_hits.append(gold)
            hits.add(gold)
    return new_hits


def result_summary(query: str, result_text: str, max_chars: int) -> str:
    compact = result_text.strip()
    if len(compact) > max_chars:
        compact = compact[:max_chars].rstrip() + "\n...[truncated]"
    return (
        "<result_summary>\n"
        f"Query: {query}\n"
        "Earlier retrieved evidence, compacted by the retrieval system:\n"
        f"{compact}\n"
        "</result_summary>\n"
    )


def compact_previous_results(state: Trajectory) -> None:
    for turn_output, result_block, summary in state.result_history:
        state.prompt = state.prompt.replace(turn_output + result_block, summary, 1)


def normalize_final_target(text: str) -> str:
    text = truncate_at_action_stop(text.strip())
    if ANSWER_OPEN in text:
        start = text.find(ANSWER_OPEN) + len(ANSWER_OPEN)
        text = text[start:]
    if ANSWER_CLOSE in text:
        text = text.split(ANSWER_CLOSE, 1)[0]
    return text.strip() + ANSWER_CLOSE


def compute_reward(
    state: Trajectory,
    search_cost_coef: float,
    reward_clip_min: float,
    reward_clip_max: float,
) -> tuple[float, dict[str, Any]]:
    label = state.example.label
    pred = extract_answer_label(state.trajectory)
    state.prediction = pred
    valid_format = answer_format_valid(state.trajectory)

    if pred == label and pred in {"true", "false"}:
        r_correct = 1.0
    elif pred == "unknown":
        r_correct = -1.5
    else:
        r_correct = -1.0

    r_format = 0.2 if valid_format else -0.5
    r_search = -float(search_cost_coef) * state.search_count

    if label == "true" and pred == "false":
        r_label = -1.3
    elif label == "false" and pred == "true":
        r_label = -1.0
    else:
        r_label = 0.0

    prediction_correct = pred == label and pred in {"true", "false"}
    r_evidence = 0.2 if state.gold_evidence_hits and prediction_correct else 0.0

    raw = r_correct + r_format + r_search + r_label + r_evidence
    clipped = max(reward_clip_min, min(reward_clip_max, raw))
    components = {
        "reward_raw": raw,
        "reward": clipped,
        "R_correct": r_correct,
        "R_format": r_format,
        "R_search": r_search,
        "R_label": r_label,
        "R_evidence": r_evidence,
        "prediction": pred,
        "label": label,
        "answer_format_valid": valid_format,
        "search_count": state.search_count,
        "gold_evidence_retrieved": bool(state.gold_evidence_hits),
        "forced_final": state.forced_final,
        "forced_final_invalid": state.forced_final and not valid_format,
        "search_cost_coef": search_cost_coef,
    }
    state.forced_final_invalid = bool(components["forced_final_invalid"])
    state.reward = clipped
    state.reward_components = components
    return clipped, components


def aggregate_metrics(states: list[Trajectory]) -> dict[str, float]:
    total = max(1, len(states))
    true_rows = [s for s in states if s.example.label == "true"]
    false_rows = [s for s in states if s.example.label == "false"]
    correct = sum(1 for s in states if s.prediction == s.example.label and s.prediction in {"true", "false"})
    unknown = sum(1 for s in states if s.prediction == "unknown")
    true_to_false = sum(1 for s in true_rows if s.prediction == "false")
    false_to_true = sum(1 for s in false_rows if s.prediction == "true")
    false_unknown = sum(1 for s in false_rows if s.prediction == "unknown")
    true_correct = sum(1 for s in true_rows if s.prediction == "true")
    false_correct = sum(1 for s in false_rows if s.prediction == "false")
    return {
        "overall_accuracy": correct / total,
        "unknown_rate": unknown / total,
        "false_unknown_rate": false_unknown / max(1, len(false_rows)),
        "true_to_false_rate": true_to_false / max(1, len(true_rows)),
        "false_to_true_rate": false_to_true / max(1, len(false_rows)),
        "avg_search_count": sum(s.search_count for s in states) / total,
        "forced_final_invalid_count": float(sum(1 for s in states if s.forced_final_invalid)),
        "false_recall": false_correct / max(1, len(false_rows)),
        "true_recall": true_correct / max(1, len(true_rows)),
    }


def collate_examples(batch: list[TripleExample]) -> list[TripleExample]:
    return batch


def csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def configure_deepspeed_batch_sizes(args: argparse.Namespace, plugin: Any | None) -> None:
    if plugin is None:
        return

    world_size = int(os.environ.get("WORLD_SIZE") or os.environ.get("ACCELERATE_NUM_PROCESSES") or "1")
    micro_batch = max(1, int(args.logprob_micro_batch_size))
    grad_accum = max(1, int(args.gradient_accumulation_steps))
    train_batch = micro_batch * grad_accum * max(1, world_size)

    updates = {
        "train_micro_batch_size_per_gpu": micro_batch,
        "gradient_accumulation_steps": grad_accum,
        "train_batch_size": train_batch,
    }
    if args.max_grad_norm > 0:
        updates["gradient_clipping"] = float(args.max_grad_norm)

    configs = []
    deepspeed_config = getattr(plugin, "deepspeed_config", None)
    if isinstance(deepspeed_config, dict):
        configs.append(deepspeed_config)
    hf_ds_config = getattr(plugin, "hf_ds_config", None)
    hf_config = getattr(hf_ds_config, "config", None)
    if isinstance(hf_config, dict) and hf_config not in configs:
        configs.append(hf_config)

    for config in configs:
        for key, value in updates.items():
            if config.get(key, "auto") == "auto":
                config[key] = value


def apply_policy_lora(model: torch.nn.Module, args: argparse.Namespace) -> torch.nn.Module:
    if not args.use_lora and not args.lora_adapter_path:
        return model
    try:
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError("LoRA training requires peft. Install it with `pip install peft`.") from exc

    if args.lora_adapter_path:
        return PeftModel.from_pretrained(model, args.lora_adapter_path, is_trainable=True)

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=csv_list(args.lora_target_modules),
        bias=args.lora_bias,
    )
    return get_peft_model(model, config)


def apply_reference_lora(model: torch.nn.Module, adapter_path: str) -> torch.nn.Module:
    if not adapter_path:
        return model
    try:
        from peft import PeftModel
    except ImportError as exc:
        raise RuntimeError("Reference LoRA adapters require peft. Install it with `pip install peft`.") from exc
    return PeftModel.from_pretrained(model, adapter_path, is_trainable=False)


def make_dataloader(
    examples: list[TripleExample],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_examples,
        generator=generator,
    )


def generate_text_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    do_sample: bool,
    device: torch.device,
    synced_gpus: bool,
) -> list[str]:
    if not prompts:
        return []
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    input_width = inputs["input_ids"].shape[1]
    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "synced_gpus": synced_gpus,
    }
    if do_sample:
        generation_kwargs.update(
            {
                "temperature": max(temperature, 1e-6),
                "top_p": top_p,
            }
        )
        if top_k > 0:
            generation_kwargs["top_k"] = top_k
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            **generation_kwargs,
        )
    texts = []
    for row in output_ids:
        gen_ids = row[input_width:]
        texts.append(truncate_at_action_stop(tokenizer.decode(gen_ids, skip_special_tokens=False)))
    return texts


def rollout_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    examples: list[TripleExample],
    retriever: HTTPRetriever,
    args: argparse.Namespace,
    device: torch.device,
    group_size: int,
    do_sample: bool = True,
) -> list[Trajectory]:
    states: list[Trajectory] = []
    for group_index, example in enumerate(examples):
        for sample_index in range(group_size):
            prompt = qwen_chat_prompt(args.system_prompt, example.user_prompt(), "<think>")
            states.append(
                Trajectory(
                    example=example,
                    group_index=group_index,
                    sample_index=sample_index,
                    prompt=prompt,
                    trajectory="<think>",
                )
            )

    model.eval()
    for _turn in range(1, args.max_turns + 1):
        active = [state for state in states if not state.finished]
        if not active:
            break
        prompts = [state.prompt for state in active]
        outputs = generate_text_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=args.max_turn_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=do_sample,
            device=device,
            synced_gpus=args.synced_gpus,
        )
        search_items: list[tuple[Trajectory, str, str]] = []
        for state, output in zip(active, outputs):
            output = output.strip()
            state.model_steps.append(ModelStep(prompt=state.prompt, output=output))
            state.prompt += output
            state.trajectory += output
            if ANSWER_CLOSE in output or ANSWER_OPEN in output:
                state.finished = True
                continue
            query = extract_last_search(output)
            if not query:
                if not output:
                    state.finished = True
                continue
            search_items.append((state, output, query))

        if search_items:
            results = retriever.search_many([query for _, _, query in search_items])
            for (state, output, query), result in zip(search_items, results):
                state.search_count += 1
                ids = retrieved_doc_ids(result)
                state.retrieved_doc_ids.extend(ids)
                update_gold_hits(state.example, result, state.gold_evidence_hits)
                compact_previous_results(state)
                result_block = f"<result>{result}\n</result>\n"
                state.prompt += result_block
                state.trajectory += result_block
                summary = result_summary(query, result_block, args.result_summary_chars)
                state.result_history.append((output, result_block, summary))

    unfinished = [state for state in states if not state.finished]
    if args.force_final and unfinished:
        prompts = [state.prompt + FINAL_ANSWER_PREFIX for state in unfinished]
        outputs = generate_text_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=args.force_final_new_tokens,
            temperature=args.eval_temperature if not do_sample else args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=do_sample,
            device=device,
            synced_gpus=args.synced_gpus,
        )
        for state, output in zip(unfinished, outputs):
            target = normalize_final_target(output)
            forced_prompt = state.prompt + FINAL_ANSWER_PREFIX
            state.model_steps.append(ModelStep(prompt=forced_prompt, output=target, forced_final=True))
            state.prompt = forced_prompt + target
            state.trajectory += FINAL_ANSWER_PREFIX + target
            state.finished = True
            state.forced_final = True

    for state in states:
        state.prediction = extract_answer_label(state.trajectory)
    model.train()
    return states


def tokenize_prompt_target(tokenizer: Any, prompt: str, target: str, max_length: int) -> tuple[list[int], list[int]]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    if not target_ids:
        target_ids = [tokenizer.eos_token_id]
    total = len(prompt_ids) + len(target_ids)
    if total > max_length:
        overflow = total - max_length
        if overflow < len(prompt_ids):
            prompt_ids = prompt_ids[overflow:]
        else:
            target_ids = target_ids[overflow - len(prompt_ids) :]
            prompt_ids = []
    return prompt_ids, target_ids


def sequence_logprobs(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt_target_pairs: list[tuple[str, str]],
    max_length: int,
    micro_batch_size: int,
    device: torch.device,
    requires_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    sums: list[torch.Tensor] = []
    counts: list[torch.Tensor] = []
    context = torch.enable_grad() if requires_grad else torch.no_grad()
    with context:
        for start in range(0, len(prompt_target_pairs), micro_batch_size):
            chunk = prompt_target_pairs[start : start + micro_batch_size]
            encoded = [tokenize_prompt_target(tokenizer, p, t, max_length) for p, t in chunk]
            max_len = max(len(prompt_ids) + len(target_ids) for prompt_ids, target_ids in encoded)
            input_rows = []
            label_rows = []
            attention_rows = []
            for prompt_ids, target_ids in encoded:
                input_ids = prompt_ids + target_ids
                labels = [-100] * len(prompt_ids) + target_ids
                pad_len = max_len - len(input_ids)
                input_rows.append(input_ids + [tokenizer.pad_token_id] * pad_len)
                label_rows.append(labels + [-100] * pad_len)
                attention_rows.append([1] * len(input_ids) + [0] * pad_len)
            input_ids_t = torch.tensor(input_rows, dtype=torch.long, device=device)
            labels_t = torch.tensor(label_rows, dtype=torch.long, device=device)
            attention_t = torch.tensor(attention_rows, dtype=torch.long, device=device)
            outputs = model(input_ids=input_ids_t, attention_mask=attention_t)
            logits = outputs.logits[:, :-1, :]
            shifted_labels = labels_t[:, 1:]
            mask = shifted_labels.ne(-100)
            safe_labels = shifted_labels.masked_fill(~mask, 0)
            log_probs = F.log_softmax(logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
            log_probs = log_probs * mask
            sums.append(log_probs.sum(dim=1))
            counts.append(mask.sum(dim=1).to(log_probs.dtype))
    return torch.cat(sums, dim=0), torch.cat(counts, dim=0)


def trajectory_logprobs(
    policy_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    tokenizer: Any,
    trajectories: list[Trajectory],
    max_length: int,
    micro_batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pairs: list[tuple[str, str]] = []
    step_to_traj: list[int] = []
    for traj_idx, trajectory in enumerate(trajectories):
        for step in trajectory.model_steps:
            if step.output.strip():
                pairs.append((step.prompt, step.output))
                step_to_traj.append(traj_idx)
    if not pairs:
        zero = torch.zeros(len(trajectories), dtype=torch.float32, device=device)
        return zero, zero, zero

    policy_step_logp, token_counts = sequence_logprobs(
        policy_model,
        tokenizer,
        pairs,
        max_length=max_length,
        micro_batch_size=micro_batch_size,
        device=device,
        requires_grad=True,
    )
    ref_step_logp, _ = sequence_logprobs(
        ref_model,
        tokenizer,
        pairs,
        max_length=max_length,
        micro_batch_size=micro_batch_size,
        device=device,
        requires_grad=False,
    )
    policy_sums = torch.zeros(len(trajectories), dtype=policy_step_logp.dtype, device=device)
    ref_sums = torch.zeros(len(trajectories), dtype=policy_step_logp.dtype, device=device)
    counts = torch.zeros(len(trajectories), dtype=policy_step_logp.dtype, device=device)
    traj_index = torch.tensor(step_to_traj, dtype=torch.long, device=device)
    policy_sums.scatter_add_(0, traj_index, policy_step_logp)
    ref_sums.scatter_add_(0, traj_index, ref_step_logp.to(policy_step_logp.dtype))
    counts.scatter_add_(0, traj_index, token_counts.to(policy_step_logp.dtype))
    return policy_sums, ref_sums, counts.clamp_min(1.0)


def select_grpo_samples(
    states: list[Trajectory],
    group_size: int,
    epsilon: float,
) -> tuple[list[Trajectory], list[float], int]:
    selected: list[Trajectory] = []
    advantages: list[float] = []
    skipped = 0
    groups: dict[int, list[Trajectory]] = {}
    for state in states:
        groups.setdefault(state.group_index, []).append(state)
    for group in groups.values():
        if len(group) != group_size:
            skipped += 1
            continue
        rewards = [state.reward for state in group]
        mean = sum(rewards) / len(rewards)
        variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
        std = math.sqrt(variance)
        if std <= epsilon or max(rewards) == min(rewards):
            skipped += 1
            continue
        for state, reward in zip(group, rewards):
            selected.append(state)
            advantages.append((reward - mean) / (std + epsilon))
    return selected, advantages, skipped


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_checkpoint(accelerator: Accelerator, model: torch.nn.Module, tokenizer: Any, output_dir: Path, step: int) -> None:
    save_dir = output_dir / f"checkpoint-step-{step}"
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(
        save_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        safe_serialization=True,
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(save_dir)
    accelerator.wait_for_everyone()


def evaluate(
    model: torch.nn.Module,
    tokenizer: Any,
    eval_examples: list[TripleExample],
    retriever: HTTPRetriever,
    args: argparse.Namespace,
    accelerator: Accelerator,
    search_cost: float,
    step: int,
) -> dict[str, float]:
    if not eval_examples:
        return {}
    local_examples = eval_examples[accelerator.process_index :: accelerator.num_processes]
    if args.max_eval_examples:
        local_examples = local_examples[: args.max_eval_examples]
    unwrapped = accelerator.unwrap_model(model)
    all_states: list[Trajectory] = []
    for start in range(0, len(local_examples), args.eval_batch_size):
        batch = local_examples[start : start + args.eval_batch_size]
        states = rollout_batch(
            model=unwrapped,
            tokenizer=tokenizer,
            examples=batch,
            retriever=retriever,
            args=args,
            device=accelerator.device,
            group_size=1,
            do_sample=False,
        )
        for state in states:
            compute_reward(state, search_cost, args.reward_clip_min, args.reward_clip_max)
        all_states.extend(states)
    local_metrics = aggregate_metrics(all_states)
    local_count = torch.tensor([len(all_states)], dtype=torch.float32, device=accelerator.device)
    counts = accelerator.gather(local_count).sum().item()
    reduced: dict[str, float] = {}
    for key, value in local_metrics.items():
        weighted = torch.tensor([value * len(all_states)], dtype=torch.float32, device=accelerator.device)
        reduced[key] = accelerator.gather(weighted).sum().item() / max(1.0, counts)
    if accelerator.is_main_process:
        row = {"step": step, "split": "eval", **reduced, "search_cost_coef": search_cost}
        write_jsonl(Path(args.output_dir) / "metrics.jsonl", row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return reduced


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 trajectory-level GRPO for iterative search policy.")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--ref_model_name_or_path", default="")
    parser.add_argument("--train_file", default="../TrajRL/dataset/trex_renlg/train.jsonl")
    parser.add_argument("--eval_file", default="../TrajRL/dataset/trex_renlg/dev.jsonl")
    parser.add_argument("--output_dir", default="outputs/stage2_search_grpo")
    parser.add_argument("--search_url", default="http://localhost:8090")
    parser.add_argument("--system_prompt_file", default="")
    parser.add_argument("--train_limit", type=int, default=0)
    parser.add_argument("--eval_limit", type=int, default=0)
    parser.add_argument("--max_eval_examples", type=int, default=128)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--train_batch_size", type=int, default=1, help="Number of triples per optimizer rollout batch.")
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--deepspeed_config", default="")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--no_lora", dest="use_lora", action="store_false")
    parser.add_argument("--lora_adapter_path", default="", help="Optional trainable policy LoRA adapter to continue from.")
    parser.add_argument("--ref_lora_adapter_path", default="", help="Optional frozen reference LoRA adapter for KL.")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--lora_bias", choices=["none", "all", "lora_only"], default="none")
    parser.add_argument("--max_turns", type=int, default=4)
    parser.add_argument("--max_turn_new_tokens", type=int, default=384)
    parser.add_argument("--force_final", action="store_true", default=True)
    parser.add_argument("--no_force_final", dest="force_final", action="store_false")
    parser.add_argument("--force_final_new_tokens", type=int, default=96)
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument("--max_result_chars", type=int, default=2000)
    parser.add_argument("--result_summary_chars", type=int, default=480)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--eval_temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--synced_gpus", action="store_true")
    parser.add_argument("--beta_kl", type=float, default=0.001)
    parser.add_argument("--advantage_epsilon", type=float, default=1e-6)
    parser.add_argument("--reward_clip_min", type=float, default=-3.0)
    parser.add_argument("--reward_clip_max", type=float, default=1.5)
    parser.add_argument("--search_cost_coef", type=float, default=0.02)
    parser.add_argument("--search_cost_coef_max", type=float, default=0.04)
    parser.add_argument("--search_cost_adjust_step", type=float, default=0.002)
    parser.add_argument("--target_unknown_rate", type=float, default=0.05)
    parser.add_argument("--unknown_rate_tolerance", type=float, default=0.02)
    parser.add_argument("--auto_adjust_search_cost", action="store_true")
    parser.add_argument("--logprob_micro_batch_size", type=int, default=1)
    parser.add_argument("--normalize_logprob_by_tokens", action="store_true")
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.system_prompt_file:
        args.system_prompt = Path(args.system_prompt_file).expanduser().read_text(encoding="utf-8").strip()
    else:
        args.system_prompt = SYSTEM_PROMPT
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    ds_plugin = None
    if args.deepspeed_config:
        ds_plugin = DeepSpeedPlugin(hf_ds_config=args.deepspeed_config)
        configure_deepspeed_batch_sizes(args, ds_plugin)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        deepspeed_plugin=ds_plugin,
    )
    configure_deepspeed_batch_sizes(args, getattr(accelerator.state, "deepspeed_plugin", None))
    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16 if args.mixed_precision == "fp16" else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model = apply_policy_lora(model, args)
    if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if accelerator.is_main_process and (args.use_lora or args.lora_adapter_path):
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(
            json.dumps(
                {
                    "event": "lora_enabled",
                    "trainable_parameters": trainable,
                    "total_parameters": total,
                    "trainable_ratio": trainable / max(1, total),
                    "lora_adapter_path": args.lora_adapter_path,
                    "lora_target_modules": csv_list(args.lora_target_modules),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    ref_path = args.ref_model_name_or_path or args.model_name_or_path
    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_path,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    ref_model = apply_reference_lora(ref_model, args.ref_lora_adapter_path)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)

    train_examples = read_jsonl(args.train_file, args.train_limit)
    eval_examples = read_jsonl(args.eval_file, args.eval_limit) if args.eval_file else []
    local_train_examples = train_examples[accelerator.process_index :: accelerator.num_processes]
    dataloader = make_dataloader(local_train_examples, args.train_batch_size, shuffle=True, seed=args.seed)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    updates_per_epoch = max(1, math.ceil(len(dataloader) / args.gradient_accumulation_steps))
    total_steps = args.max_steps or (updates_per_epoch * args.num_train_epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    ref_model.to(accelerator.device)
    retriever = HTTPRetriever(args.search_url, args.top_n, args.max_result_chars)
    search_cost = SearchCostController(
        initial=args.search_cost_coef,
        max_value=args.search_cost_coef_max,
        step=args.search_cost_adjust_step,
        target_unknown_rate=args.target_unknown_rate,
        tolerance=args.unknown_rate_tolerance,
        enabled=args.auto_adjust_search_cost,
    )

    global_step = 0
    last_metrics: dict[str, float] = {}
    model.train()
    for epoch in range(args.num_train_epochs):
        for batch in dataloader:
            if args.max_steps and global_step >= args.max_steps:
                break
            unwrapped = accelerator.unwrap_model(model)
            with torch.no_grad():
                states = rollout_batch(
                    model=unwrapped,
                    tokenizer=tokenizer,
                    examples=batch,
                    retriever=retriever,
                    args=args,
                    device=accelerator.device,
                    group_size=args.group_size,
                    do_sample=True,
                )
            for state in states:
                compute_reward(state, search_cost.value, args.reward_clip_min, args.reward_clip_max)
            selected, advantages, skipped_groups = select_grpo_samples(states, args.group_size, args.advantage_epsilon)
            train_metrics = aggregate_metrics(states)
            if not selected:
                if accelerator.is_main_process:
                    row = {
                        "step": global_step,
                        "epoch": epoch,
                        "split": "train",
                        "skipped_groups": skipped_groups,
                        "selected_trajectories": 0,
                        "search_cost_coef": search_cost.value,
                        **train_metrics,
                    }
                    write_jsonl(output_dir / "metrics.jsonl", row)
                    print(json.dumps(row, ensure_ascii=False), flush=True)
                search_cost.update(train_metrics)
                continue

            with accelerator.accumulate(model):
                policy_logp, ref_logp, token_counts = trajectory_logprobs(
                    policy_model=model,
                    ref_model=ref_model,
                    tokenizer=tokenizer,
                    trajectories=selected,
                    max_length=args.max_seq_length,
                    micro_batch_size=args.logprob_micro_batch_size,
                    device=accelerator.device,
                )
                if args.normalize_logprob_by_tokens:
                    policy_objective_logp = policy_logp / token_counts
                    ref_objective_logp = ref_logp / token_counts
                else:
                    policy_objective_logp = policy_logp
                    ref_objective_logp = ref_logp
                advantage_t = torch.tensor(advantages, dtype=policy_objective_logp.dtype, device=accelerator.device)
                sampled_kl = policy_objective_logp - ref_objective_logp
                objective = advantage_t * policy_objective_logp - args.beta_kl * sampled_kl
                loss = -objective.mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            kl_value = accelerator.gather(sampled_kl.detach().float().mean()).mean().item()
            loss_value = accelerator.gather(loss.detach().float()).mean().item()
            train_metrics["kl"] = kl_value
            train_metrics["loss"] = loss_value
            train_metrics["mean_reward"] = sum(s.reward for s in states) / max(1, len(states))
            train_metrics["selected_trajectories"] = float(len(selected))
            train_metrics["skipped_groups"] = float(skipped_groups)
            train_metrics["search_cost_coef"] = search_cost.value
            last_metrics = train_metrics

            if accelerator.is_main_process and (global_step % args.logging_steps == 0):
                row = {"step": global_step, "epoch": epoch, "split": "train", **train_metrics}
                write_jsonl(output_dir / "metrics.jsonl", row)
                write_jsonl(
                    output_dir / "rollouts.jsonl",
                    {
                        "step": global_step,
                        "examples": [
                            {
                                **state.example.to_dict(),
                                "trajectory": state.trajectory,
                                "reward_components": state.reward_components,
                                "retrieved_doc_ids": state.retrieved_doc_ids,
                                "gold_evidence_hits": sorted(state.gold_evidence_hits),
                            }
                            for state in states
                        ],
                    },
                )
                print(json.dumps(row, ensure_ascii=False), flush=True)

            if args.eval_steps > 0 and global_step % args.eval_steps == 0:
                eval_metrics = evaluate(
                    model,
                    tokenizer,
                    eval_examples,
                    retriever,
                    args,
                    accelerator,
                    search_cost.value,
                    global_step,
                )
                if eval_metrics:
                    search_cost.update(eval_metrics)
            else:
                search_cost.update(train_metrics)

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                save_checkpoint(accelerator, model, tokenizer, output_dir, global_step)

        if args.max_steps and global_step >= args.max_steps:
            break

    save_checkpoint(accelerator, model, tokenizer, output_dir, global_step)
    if accelerator.is_main_process:
        summary = {"step": global_step, "split": "final", **last_metrics}
        write_jsonl(output_dir / "metrics.jsonl", summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
