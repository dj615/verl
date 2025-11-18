import argparse
import json
import os

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer


def flush_chunk(chunk, writer, path):
    """
    将一个 Python list[dict] 写入 Parquet。
    如果 writer 还没创建，就在这里创建。
    写完后清空 chunk 并返回 writer。
    """
    if not chunk:
        return writer

    table = pa.Table.from_pylist(chunk)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema)
    writer.write_table(table)
    chunk.clear()
    return writer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--max_len", type=int, default=8192)
    parser.add_argument("--max_train_data", type=int, default=5000)
    parser.add_argument("--max_valid_data", type=int, default=1000)
    parser.add_argument("--max_test_data", type=int, default=0)
    parser.add_argument(
        "--parquet_chunk_size",
        type=int,
        default=10000,
        help="How many rows per parquet write chunk to keep memory small.",
    )
    args = parser.parse_args()

    save_dirs = {
        "train": "/root/workspace/ASR_data/train",
        "valid": "/root/workspace/ASR_data/valid",
        "test": "/root/workspace/ASR_data/test",
    }
    for d in save_dirs.values():
        os.makedirs(d, exist_ok=True)

    jsonl_paths = {
        split: os.path.join(dir_path, "openscience.jsonl")
        for split, dir_path in save_dirs.items()
    }
    parquet_paths = {
        split: os.path.join(dir_path, "openscience.parquet")
        for split, dir_path in save_dirs.items()
    }

    # JSONL 流式写
    jsonl_files = {
        split: open(path, "w", encoding="utf-8")
        for split, path in jsonl_paths.items()
    }

    # Parquet 分块写
    parquet_writers = {"train": None, "valid": None, "test": None}
    parquet_chunks = {"train": [], "valid": [], "test": []}
    CHUNK_SIZE = args.parquet_chunk_size

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    print(f"Loaded tokenizer: {args.tokenizer}")

    # 使用 streaming 模式，不在本地缓存完整数据集
    ds = load_dataset("nvidia/OpenScienceReasoning-2", split="train", streaming=True)
    print("Loaded nvidia/OpenScienceReasoning-2 with streaming=True.")

    counts = {"train": 0, "valid": 0, "test": 0}

    try:
        for sample in ds:
            # 按顺序填满 train -> valid -> test
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

            # JSONL（流式写）
            jsonl_files[split].write(json.dumps(record, ensure_ascii=False) + "\n")

            # Parquet 分块写
            parquet_chunks[split].append(record)
            counts[split] += 1

            if len(parquet_chunks[split]) >= CHUNK_SIZE:
                parquet_writers[split] = flush_chunk(
                    parquet_chunks[split],
                    parquet_writers[split],
                    parquet_paths[split],
                )

    finally:
        # 关闭 JSONL
        for f in jsonl_files.values():
            f.close()

        # 刷新剩余 chunk，并关闭 Parquet writer
        for split in ["train", "valid", "test"]:
            writer = parquet_writers[split]
            chunk = parquet_chunks[split]
            writer = flush_chunk(chunk, writer, parquet_paths[split])
            if writer is not None:
                writer.close()

    print("Done.")
    print(f"Final counts: train={counts['train']}, valid={counts['valid']}, test={counts['test']}")
    print(f"Train jsonl:   {jsonl_paths['train']}")
    print(f"Valid jsonl:   {jsonl_paths['valid']}")
    print(f"Test jsonl:    {jsonl_paths['test']}")
    print(f"Train parquet: {parquet_paths['train']}")
    print(f"Valid parquet: {parquet_paths['valid']}")
    print(f"Test parquet:  {parquet_paths['test']}")


if __name__ == "__main__":
    main()
