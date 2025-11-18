import os
import json
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
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

    # HARP 主文件是一个 JSONL，我们用 datasets 的 json loader 来读
    ds = load_dataset("json", data_files={"train": args.data_path})["train"]

    # ========= 划分 9:1 =========
    ds = ds.shuffle(seed=42)
    total = len(ds)
    raw_train_size = int(total * 0.9)

    train_records = []
    valid_records = []
    train_count = 0
    valid_count = 0

    print(
        f"Total samples: {total}, raw train={raw_train_size}, "
        f"raw valid={total - raw_train_size}"
    )
    print(
        f"max_train_data={args.max_train_data}, "
        f"max_valid_data={args.max_valid_data}"
    )

    for idx, sample in enumerate(ds):
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

        # ====== 9:1 划分并截断数量 ======
        if idx < raw_train_size:
            if args.max_train_data is not None and train_count >= args.max_train_data:
                continue
            train_records.append(record)
            train_count += 1
        else:
            if args.max_valid_data is not None and valid_count >= args.max_valid_data:
                continue
            valid_records.append(record)
            valid_count += 1

    print(f"Final train kept: {len(train_records)}")
    print(f"Final valid kept: {len(valid_records)}")

    # ========= 写 JSONL =========
    print("Writing JSONL files...")
    with open(TRAIN_JSONL, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(VALID_JSONL, "w", encoding="utf-8") as f:
        for r in valid_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ========= 写 Parquet =========
    print("Writing Parquet files...")
    pq.write_table(pa.Table.from_pylist(train_records), TRAIN_PARQUET)
    pq.write_table(pa.Table.from_pylist(valid_records), VALID_PARQUET)

    print("Done!")
    print(f"Train jsonl:   {TRAIN_JSONL}")
    print(f"Valid jsonl:   {VALID_JSONL}")
    print(f"Train parquet: {TRAIN_PARQUET}")
    print(f"Valid parquet: {VALID_PARQUET}")

    os.remove(args.data_path)
    
if __name__ == "__main__":
    main()
