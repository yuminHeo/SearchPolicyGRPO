from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from verl_search_policy.prompting import TOOL_SCHEMAS, TripleExample, build_env


def read_records(path: str) -> list[dict[str, Any]]:
    input_path = Path(path)
    if input_path.suffix.lower() == ".jsonl":
        rows = []
        with input_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("data", "records", "examples"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"Unsupported data format: {path}")


def build_row(record: dict[str, Any], index: int, data_source: str, search_url: str) -> dict[str, Any]:
    example = TripleExample.from_dict(record)
    metadata = dict(example.metadata or {})
    extra_info = {
        "index": index,
        "id": record.get("id", example.id),
        "subject": example.subject,
        "predicate": example.predicate,
        "object": example.object,
        "label": example.label,
        "metadata": metadata,
        "gold_evidence": example.gold_evidence,
        "env": build_env(search_url),
        "func_schemas": TOOL_SCHEMAS,
    }
    return {
        "data_source": data_source,
        "prompt": example.to_prompt_input(),
        "ability": "search_policy_grpo",
        "reward_model": {"style": "rule", "ground_truth": example.label},
        "extra_info": extra_info,
    }


def load_rows(
    input_file: str,
    data_source: str,
    search_url: str,
    max_records: int | None,
    seed: int,
    shuffle: bool,
) -> list[dict[str, Any]]:
    records = read_records(input_file)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(records)
    if max_records is not None and max_records >= 0:
        records = records[:max_records]
    return [
        build_row(record, index, data_source, search_url)
        for index, record in enumerate(tqdm(records, desc=f"convert {Path(input_file).name}", unit="record"))
    ]


def split_train_val(rows: list[dict[str, Any]], val_size: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if val_size <= 0:
        return rows, []
    if val_size >= len(rows):
        raise ValueError(f"--val-size must be smaller than the number of rows ({len(rows)}).")
    return rows[:-val_size], rows[-val_size:]


def write_parquet(rows: list[dict[str, Any]], output_file: str) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("prepare_verl_data.py requires pandas and pyarrow/fastparquet.") from exc
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert SearchPolicyGRPO JSON/JSONL data to VERL parquet.")
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--val-file", default="")
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--val-output", required=True)
    parser.add_argument("--search-url", default="http://localhost:8090")
    parser.add_argument("--data-source", default="search_policy_grpo")
    parser.add_argument("--max-train-records", type=int, default=-1)
    parser.add_argument("--max-val-records", type=int, default=-1)
    parser.add_argument("--val-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()

    max_train = args.max_train_records if args.max_train_records >= 0 else None
    max_val = args.max_val_records if args.max_val_records >= 0 else None
    train_rows = load_rows(args.train_file, args.data_source, args.search_url, max_train, args.seed, args.shuffle)
    if args.val_file:
        val_rows = load_rows(args.val_file, args.data_source, args.search_url, max_val, args.seed, False)
    else:
        train_rows, val_rows = split_train_val(train_rows, args.val_size)

    write_parquet(train_rows, args.train_output)
    write_parquet(val_rows, args.val_output)
    print(f"wrote train={len(train_rows)} rows to {args.train_output}")
    print(f"wrote val={len(val_rows)} rows to {args.val_output}")


if __name__ == "__main__":
    main()

