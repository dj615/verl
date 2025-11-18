import os
import json
import argparse
import random

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

"""
python3 recipe/ASR/data_process/train_valid/HARP_download.py

python3 recipe/ASR/data_process/train_valid/HARP.py
"""

# ===== 路径设置 =====
SAVE_DIR = "/root/workspace/ASR_data"

TRAIN_JSONL = os.path.join(SAVE_DIR, "train", "HARP.jsonl")
VALID_JSONL = os.path.join(SAVE_DIR, "valid", "HARP.jsonl")
TRAIN_PARQUET = os.path.join(SAVE_DIR, "train", "HARP.parquet")
VALID_PARQUET = os.path.join(SAVE_DIR, "valid", "HARP.parquet")

os.makedirs(os.path.join(SAVE_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "valid"), exist_ok=True)


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
    parser.add_argument(
        "--max_len",
        type=int,
        default=None,
        help="Filter out samples whose question+answer token count exceeds max_len.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer name when using --max_len.",
    )
    parser.add_argument(
        "--max_train_data",
        type=int,
        default=None,
        help="Maximum number of filtered train samples to keep.",
    )
    parser.add_argument(
        "--max_valid_data",
        type=int,
        default=None,
        help="Maximum number of filtered valid samples to keep.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help=(
            "Path to unzipped HARP.jsonl. "
            "If not set, default is SAVE_DIR/raw/HARP.jsonl"
        ),
    )
    parser.add_argument(
        "--parquet_chunk_size",
        type=int,
        default=10000,
        help="How many rows per parquet write chunk to keep memory small.",
    )
    args = parser.parse_args()

    # ========= tokenizer =========
    tokenizer = None
    if args.max_len is not None:
        if args.tokenizer is None:
            raise ValueError("--tokenizer must be specified when --max_len is set.")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
        print(f"Loaded tokenizer: {args.tokenizer}")

    # ========= 数据路径 =========
    if args.data_path is None:
        # 你可以把 HARP.jsonl 解压到这个默认位置
        args.data_path = os.path.join(SAVE_DIR, "raw", "HARP.jsonl")
    print(f"Loading HARP from {args.data_path} ...")

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"HARP jsonl not found at {args.data_path}")

    # ========= 第一次遍历：只统计总行数，用于精确 9:1 划分 =========
    total = 0
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1

    raw_train_size = int(total * 0.9)

    print(
        f"Total samples: {total}, raw train={raw_train_size}, "
        f"raw valid={total - raw_train_size}"
    )
    print(
        f"max_train_data={args.max_train_data}, "
        f"max_valid_data={args.max_valid_data}"
    )

    # ========= 构造一个长度为 total 的 bool 列表表示 train / valid，并打乱 =========
    # True -> train, False -> valid
    is_train_list = [True] * raw_train_size + [False] * (total - raw_train_size)
    random.seed(42)
    random.shuffle(is_train_list)

    # ========= 打开输出文件（流式写 JSONL） =========
    train_jsonl_f = open(TRAIN_JSONL, "w", encoding="utf-8")
    valid_jsonl_f = open(VALID_JSONL, "w", encoding="utf-8")

    # ========= 分块写 Parquet =========
    train_writer = None
    valid_writer = None
    train_chunk = []
    valid_chunk = []
    CHUNK_SIZE = args.parquet_chunk_size

    train_count = 0
    valid_count = 0

    try:
        # ========= 第二次遍历：逐行读取、过滤、划分、写出 =========
        with open(args.data_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if not line.strip():
                    continue
                # 如果两个 split 都达到上限，可以直接提前结束
                if (
                    args.max_train_data is not None
                    and args.max_valid_data is not None
                    and train_count >= args.max_train_data
                    and valid_count >= args.max_valid_data
                ):
                    break

                sample = json.loads(line)

                # ====== question / answer ======
                # HARP: problem -> question
                question = str(sample.get("problem", "")).strip()

                # HARP: answer -> groundtruth
                groundtruth = str(sample.get("answer", "")).strip()
                answer = groundtruth  # 和 DAPO 脚本一样，只保留最终答案

                # ====== token 过滤 ======
                if args.max_len is not None:
                    total_tokens = len(tokenizer(question + "\n" + answer).input_ids)
                    if total_tokens > args.max_len:
                        continue

                # ====== extra_info：把除 problem/answer 外的所有字段都丢进去 ======
                extra_info = {
                    k: v
                    for k, v in sample.items()
                    if k not in ("problem", "answer")
                }

                # ====== 统一成和 DAPO_MATH 一样的结构 ======
                record = {
                    "question": question,
                    "answer": answer,
                    "groundtruth": groundtruth,
                    "data_source": "harp",  # 标记数据来源
                    # 这里用 level 当作 ability，你也可以改成 subject
                    "ability": str(sample.get("level", "")),
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": groundtruth,
                    },
                    "extra_info": extra_info,
                }

                # 当前样本理论上属于 train 还是 valid（在过滤之后可能数量会变少）
                is_train = is_train_list[idx]

                # ====== 9:1 划分并截断数量 ======
                if is_train:
                    if args.max_train_data is not None and train_count >= args.max_train_data:
                        continue

                    # 写 JSONL（单行流式）
                    train_jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    # 加入 Parquet chunk
                    train_chunk.append(record)
                    train_count += 1

                    if len(train_chunk) >= CHUNK_SIZE:
                        train_writer = flush_chunk(train_chunk, train_writer, TRAIN_PARQUET)
                else:
                    if args.max_valid_data is not None and valid_count >= args.max_valid_data:
                        continue

                    valid_jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    valid_chunk.append(record)
                    valid_count += 1

                    if len(valid_chunk) >= CHUNK_SIZE:
                        valid_writer = flush_chunk(valid_chunk, valid_writer, VALID_PARQUET)

    finally:
        # 把剩余 chunk 写完
        train_writer = flush_chunk(train_chunk, train_writer, TRAIN_PARQUET)
        valid_writer = flush_chunk(valid_chunk, valid_writer, VALID_PARQUET)

        if train_writer is not None:
            train_writer.close()
        if valid_writer is not None:
            valid_writer.close()

        train_jsonl_f.close()
        valid_jsonl_f.close()

    print(f"Final train kept: {train_count}")
    print(f"Final valid kept: {valid_count}")

    print("Done!")
    print(f"Train jsonl:   {TRAIN_JSONL}")
    print(f"Valid jsonl:   {VALID_JSONL}")
    print(f"Train parquet: {TRAIN_PARQUET}")
    print(f"Valid parquet: {VALID_PARQUET}")

    # ========= 删掉原始 HARP.jsonl（和你原脚本行为一致） =========
    try:
        os.remove(args.data_path)
        print(f"Removed raw file: {args.data_path}")
    except OSError as e:
        print(f"Warning: failed to remove {args.data_path}: {e}")


if __name__ == "__main__":
    main()
