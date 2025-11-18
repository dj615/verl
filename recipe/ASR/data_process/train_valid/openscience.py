import argparse
import json
import os
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=8192)
    parser.add_argument("--max_train_data", type=int, default=5000)
    parser.add_argument("--max_valid_data", type=int, default=1000)
    parser.add_argument("--max_test_data", type=int, default=0)
    args = parser.parse_args()

    save_dirs = {
        "train": "/root/workspace/ASR_data/train",
        "valid": "/root/workspace/ASR_data/valid",
        "test": "/root/workspace/ASR_data/test",
    }
    for d in save_dirs.values():
        os.makedirs(d, exist_ok=True)

    jsonl_paths = {k: os.path.join(v, "openscience.jsonl") for k, v in save_dirs.items()}
    parquet_paths = {k: os.path.join(v, "openscience.parquet") for k, v in save_dirs.items()}

    jsonl_files = {k: open(p, "w", encoding="utf-8") for k, p in jsonl_paths.items()}
    selected_rows = {"train": [], "valid": [], "test": []}

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    ds = load_dataset("nvidia/OpenScienceReasoning-2", split="train", streaming=True)

    counts = {"train": 0, "valid": 0, "test": 0}

    for sample in ds:
        if counts["train"] < args.max_train_data:
            split = "train"
        elif counts["valid"] < args.max_valid_data:
            split = "valid"
        elif counts["test"] < args.max_test_data:
            split = "test"
        else:
            break

        question = sample.get("input", "")
        answer = sample.get("output", "")
        groundtruth = sample.get("expected_answer", "")

        ability = sample.get("ability", "")
        data_source = sample.get("source", "openscience")
        extra_info = sample.get("metadata", {})

        text = question + "\n" + answer
        if len(tokenizer(text).input_ids) > args.max_len:
            continue

        record = {
            "question": question,
            "answer": answer,
            "groundtruth": groundtruth,
            "data_source": data_source,
            "ability": ability,
            "reward_model": {"style": "rule", "ground_truth": groundtruth},
            "extra_info": extra_info,
        }

        jsonl_files[split].write(json.dumps(record, ensure_ascii=False) + "\n")
        selected_rows[split].append(record)
        counts[split] += 1

    for f in jsonl_files.values():
        f.close()

    for split in ["train", "valid", "test"]:
        table = pa.Table.from_pylist(selected_rows[split])
        pq.write_table(table, parquet_paths[split])


if __name__ == "__main__":
    main()