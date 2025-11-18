import os
import json
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer


SAVE_DIR = "/root/workspace/ASR_data"

# 这里用 NuminaMath_1_5 作文件名前缀，避免和别的冲突
PREFIX = "NuminaMath_1.5"
TRAIN_JSONL = os.path.join(SAVE_DIR, "train", f"{PREFIX}.jsonl")
VALID_JSONL = os.path.join(SAVE_DIR, "valid", f"{PREFIX}.jsonl")
TRAIN_PARQUET = os.path.join(SAVE_DIR, "train", f"{PREFIX}.parquet")
VALID_PARQUET = os.path.join(SAVE_DIR, "valid", f"{PREFIX}.parquet")

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
        default=10000,
        help="Maximum number of filtered train samples to keep.",
    )
    parser.add_argument(
        "--max_valid_data",
        type=int,
        default=1000,
        help="Maximum number of filtered valid samples to keep.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="AI-MO/NuminaMath-1.5",
        help="HF dataset name, default: AI-MO/NuminaMath-1.5",
    )
    args = parser.parse_args()

    # ----- tokenizer -----
    tokenizer = None
    if args.max_len is not None:
        if args.tokenizer is None:
            raise ValueError("--tokenizer must be specified when --max_len is set.")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
        print(f"Loaded tokenizer: {args.tokenizer}")

    # ----- load dataset -----
    print(f"Loading dataset {args.dataset} (split=train)...")
    # NuminaMath-1.5 只有 train 一个 split
    ds = load_dataset(args.dataset, split="train")

    # Shuffle for 9:1 split
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
        # ----- 基本字段 -----
        # Numina: problem, solution, answer, problem_type, question_type, ...
        question = str(sample.get("problem", "")).strip()

        # 完整 CoT 推理
        solution = str(sample.get("solution", "")).strip()
        # 简短最终答案（通常是一个表达式或数字）
        final_answer = str(sample.get("answer", "")).strip()

        # 如果 solution 为空，就退回只用 final_answer
        answer_text = solution if solution else final_answer
        groundtruth = final_answer

        # ----- token 过滤 -----
        if args.max_len is not None:
            total_tokens = len(tokenizer(question + "\n" + answer_text).input_ids)
            if total_tokens > args.max_len:
                continue

        # ----- extra_info：除了 problem / solution / answer 之外都塞进去 -----
        extra_info = {
            k: v
            for k, v in sample.items()
            if k not in ("problem", "solution", "answer")
        }

        record = {
            "question": question,
            "answer": answer_text,          # 用 CoT 作为 answer
            "groundtruth": groundtruth,     # 单独保留最终答案
            "data_source": sample.get("source", "numinamath_1.5"),
            "ability": str(sample.get("problem_type", "")),
            "reward_model": {
                "style": "cot",
                "ground_truth": groundtruth,
            },
            "extra_info": extra_info,
        }

        # ----- 9:1 划分 + 数量截断 -----
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

    # ----- 写 JSONL -----
    print("Writing JSONL files...")
    with open(TRAIN_JSONL, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(VALID_JSONL, "w", encoding="utf-8") as f:
        for r in valid_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ----- 写 Parquet -----
    print("Writing Parquet files...")
    pq.write_table(pa.Table.from_pylist(train_records), TRAIN_PARQUET)
    pq.write_table(pa.Table.from_pylist(valid_records), VALID_PARQUET)

    print("Done!")
    print(f"Train jsonl:   {TRAIN_JSONL}")
    print(f"Valid jsonl:   {VALID_JSONL}")
    print(f"Train parquet: {TRAIN_PARQUET}")
    print(f"Valid parquet: {VALID_PARQUET}")


if __name__ == "__main__":
    main()
