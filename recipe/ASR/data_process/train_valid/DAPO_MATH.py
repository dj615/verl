import os
import json
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer


SAVE_DIR = "/root/workspace/ASR_data"
TRAIN_JSONL = os.path.join(SAVE_DIR, "train", "DAPO_MATH.jsonl")
VALID_JSONL = os.path.join(SAVE_DIR, "valid", "DAPO_MATH.jsonl")
TRAIN_PARQUET = os.path.join(SAVE_DIR, "train", "DAPO_MATH.parquet")
VALID_PARQUET = os.path.join(SAVE_DIR, "valid", "DAPO_MATH.parquet")

os.makedirs(os.path.join(SAVE_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "valid"), exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=None,
                        help="Filter out samples whose question+answer token count exceeds max_len.")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Tokenizer name when using --max_len.")
    parser.add_argument("--max_train_data", type=int, default=None,
                        help="Maximum number of filtered train samples to keep.")
    parser.add_argument("--max_valid_data", type=int, default=None,
                        help="Maximum number of filtered valid samples to keep.")
    args = parser.parse_args()

    # Load tokenizer if needed
    tokenizer = None
    if args.max_len is not None:
        if args.tokenizer is None:
            raise ValueError("--tokenizer must be specified when --max_len is set.")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
        print(f"Loaded tokenizer: {args.tokenizer}")

    print("Loading dataset BytedTsinghua-SIA/DAPO-Math-17k...")
    ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")

    # Shuffle for 9:1 split
    ds = ds.shuffle(seed=42)
    total = len(ds)
    raw_train_size = int(total * 0.9)

    train_records = []
    valid_records = []

    train_count = 0
    valid_count = 0

    print(f"Total samples: {total}, raw train={raw_train_size}, raw valid={total - raw_train_size}")
    print(f"max_train_data={args.max_train_data}, max_valid_data={args.max_valid_data}")

    for idx, sample in enumerate(ds):
        # prompt: list of {"content", "role"}
        prompt_list = sample.get("prompt", [])
        if isinstance(prompt_list, list) and len(prompt_list) > 0:
            question = prompt_list[0].get("content", "").strip()
        else:
            question = ""

        # answer = reward ground truth
        reward = sample.get("reward_model", {})
        groundtruth = str(reward.get("ground_truth", "")).strip()
        answer = groundtruth  # DAPO only provides final answer

        # token filtering
        if args.max_len is not None:
            total_tokens = len(tokenizer(question + "\n" + answer).input_ids)
            if total_tokens > args.max_len:
                continue

        record = {
            "question": question,
            "answer": answer,
            "groundtruth": groundtruth,
            "data_source": sample.get("data_source", "dapo_math"),
            "ability": sample.get("ability", ""),
            "reward_model": {
                "style": "rule",
                "ground_truth": groundtruth
            },
            "extra_info": sample.get("extra_info", {}),
        }

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

    # Write JSONL
    print("Writing JSONL files...")
    with open(TRAIN_JSONL, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(VALID_JSONL, "w", encoding="utf-8") as f:
        for r in valid_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write Parquet
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
