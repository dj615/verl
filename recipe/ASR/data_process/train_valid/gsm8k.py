import os
import json
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer


SAVE_DIR = "/root/workspace/ASR_data"
os.makedirs(os.path.join(SAVE_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "valid"), exist_ok=True)


def extract_final_answer(full_answer: str) -> str:
    """
    GSM8K 的答案格式一般是：
    "... 多步推理 ... #### 72"
    这里提取最后的 "72" 作为 groundtruth。
    """
    if full_answer is None:
        return ""
    ans = str(full_answer).strip()
    marker = "####"
    if marker in ans:
        return ans.split(marker)[-1].strip()
    return ans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=None, help="Filter out samples whose question+answer token count exceeds max_len.")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer name when using --max_len.")
    parser.add_argument("--max_train_data", type=int, default=None, help="Maximum number of filtered train samples to keep.")
    parser.add_argument("--max_valid_data", type=int, default=None, help="Maximum number of filtered valid samples to keep.")
    parser.add_argument("--subset", type=str, default="main", choices=["main", "socratic"], help="GSM8K subset to use: 'main' or 'socratic'. Default: main")
    args = parser.parse_args()

    # ===== tokenizer =====
    tokenizer = None
    if args.max_len is not None:
        if args.tokenizer is None:
            raise ValueError("--tokenizer must be specified when --max_len is set.")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
        print(f"Loaded tokenizer: {args.tokenizer}")

    # ===== load dataset =====
    print(f"Loading dataset openai/gsm8k, subset={args.subset}, split=train ...")
    ds = load_dataset("openai/gsm8k", args.subset, split="train")

    # 只对官方 train 做 9:1 划分，官方 test 保留给评测用
    ds = ds.shuffle(seed=42)
    total = len(ds)
    raw_train_size = int(total * 0.9)

    print(f"Total samples in train split: {total}")
    print(f"raw train={raw_train_size}, raw valid={total - raw_train_size}")
    print(
        f"max_train_data={args.max_train_data}, "
        f"max_valid_data={args.max_valid_data}"
    )

    # 根据 subset 决定输出文件前缀
    prefix = "GSM8K"
    if args.subset != "main":
        prefix += f"_{args.subset}"

    train_jsonl = os.path.join(SAVE_DIR, "train", f"{prefix}.jsonl")
    valid_jsonl = os.path.join(SAVE_DIR, "valid", f"{prefix}.jsonl")
    train_parquet = os.path.join(SAVE_DIR, "train", f"{prefix}.parquet")
    valid_parquet = os.path.join(SAVE_DIR, "valid", f"{prefix}.parquet")

    train_records = []
    valid_records = []
    train_count = 0
    valid_count = 0

    for idx, sample in enumerate(ds):
        # ===== question / answer =====
        question = str(sample.get("question", "")).strip()
        full_answer = str(sample.get("answer", "")).strip()

        groundtruth = extract_final_answer(full_answer)

        # 这里和 DAPO 有一点不同：
        # - answer: 保留完整 CoT（GSM8K 提供的 solution）
        # - groundtruth: 只保留最后的数字答案
        answer = full_answer

        # ===== token 过滤 =====
        if args.max_len is not None:
            total_tokens = len(tokenizer(question + "\n" + answer).input_ids)
            if total_tokens > args.max_len:
                continue

        record = {
            "question": question,
            "answer": answer,
            "groundtruth": groundtruth,
            "data_source": f"gsm8k_{args.subset}",
            # GSM8K 没有 ability 相关字段，这里留空即可
            "ability": "",
            "reward_model": {
                # 这里用 "cot" 表示来自 chain-of-thought 解答
                "style": "cot",
                "ground_truth": groundtruth,
            },
            # GSM8K 没有额外字段，保持结构一致留一个空 dict
            "extra_info": {},
        }

        # ===== 9:1 划分 + 数量截断 =====
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

    # ===== 写 JSONL =====
    print("Writing JSONL files...")
    with open(train_jsonl, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(valid_jsonl, "w", encoding="utf-8") as f:
        for r in valid_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ===== 写 Parquet =====
    print("Writing Parquet files...")
    pq.write_table(pa.Table.from_pylist(train_records), train_parquet)
    pq.write_table(pa.Table.from_pylist(valid_records), valid_parquet)

    print("Done!")
    print(f"Train jsonl:   {train_jsonl}")
    print(f"Valid jsonl:   {valid_jsonl}")
    print(f"Train parquet: {train_parquet}")
    print(f"Valid parquet: {valid_parquet}")


if __name__ == "__main__":
    main()
