import os
import json
import re
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer


SAVE_DIR = "/root/workspace/ASR_data"
TRAIN_JSONL = os.path.join(SAVE_DIR, "train", "MATH.jsonl")
VALID_JSONL = os.path.join(SAVE_DIR, "valid", "MATH.jsonl")
TRAIN_PARQUET = os.path.join(SAVE_DIR, "train", "MATH.parquet")
VALID_PARQUET = os.path.join(SAVE_DIR, "valid", "MATH.parquet")

os.makedirs(os.path.join(SAVE_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "valid"), exist_ok=True)


def extract_boxed(text: str):
    """从 solution 中提取 boxed{...} """
    if not isinstance(text, str):
        return ""
    match = re.search(r"boxed\{(.*?)\}", text.replace("\\", ""), flags=re.IGNORECASE)
    if not match:
        return ""
    ans = match.group(1).strip().replace("\n", " ").strip()
    return ans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=None, help="Filter out samples whose question+answer token count exceeds max_len.")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer name when using max_len.")
    parser.add_argument("--max_train_data", type=int, default=None, help="Maximum number of filtered training samples to keep.")
    parser.add_argument("--max_valid_data", type=int, default=None, help="Maximum number of filtered validation samples to keep.")
    args = parser.parse_args()

    # setup tokenizer if needed
    tokenizer = None
    if args.max_len is not None:
        if args.tokenizer is None:
            raise ValueError("--tokenizer must be provided when --max_len is used.")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
        print(f"Loaded tokenizer: {args.tokenizer}")

    print("Loading dataset EleutherAI/hendrycks_math...")
    ds = load_dataset("EleutherAI/hendrycks_math", split="train")

    # shuffle for 9:1 split
    ds = ds.shuffle(seed=42)
    total = len(ds)
    raw_train_size = int(total * 0.9)

    train_records = []
    valid_records = []

    print(f"Total samples: {total}, raw train: {raw_train_size}, raw valid: {total - raw_train_size}")
    print(f"max_train_data={args.max_train_data}, max_valid_data={args.max_valid_data}")

    train_count = 0
    valid_count = 0

    for idx, sample in enumerate(ds):
        question = sample.get("problem", "").strip()
        answer = sample.get("solution", "").strip()

        # token filtering
        if args.max_len is not None:
            total_tokens = len(tokenizer(question + "\n" + answer).input_ids)
            if total_tokens > args.max_len:
                continue

        # extract reward ground truth
        reward_gt = extract_boxed(answer)

        record = {
            "question": question,
            "answer": answer,
            "groundtruth": reward_gt,
            "data_source": sample.get("source", "hendrycks_math"),
            "ability": sample.get("type", ""),
            "reward_model": {"style": "rule", "ground_truth": reward_gt},
            "extra_info": sample.get("metadata", {}),
        }

        # 9:1 split index-based
        if idx < raw_train_size:
            # train part
            if args.max_train_data is not None and train_count >= args.max_train_data:
                continue
            train_records.append(record)
            train_count += 1
        else:
            # valid part
            if args.max_valid_data is not None and valid_count >= args.max_valid_data:
                continue
            valid_records.append(record)
            valid_count += 1

    print(f"Final kept train samples: {len(train_records)}")
    print(f"Final kept valid samples: {len(valid_records)}")

    # write jsonl
    print("Writing JSONL files...")
    with open(TRAIN_JSONL, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(VALID_JSONL, "w", encoding="utf-8") as f:
        for r in valid_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # write parquet
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
