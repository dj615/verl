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
        "--subset",
        type=str,
        default="default",
        choices=["default", "extended", "all"],
        help="Subset of OpenR1-Math-220k to use: default / extended / all. Default: default",
    )
    args = parser.parse_args()

    # ----- tokenizer -----
    tokenizer = None
    if args.max_len is not None:
        if args.tokenizer is None:
            raise ValueError("--tokenizer must be specified when --max_len is set.")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
        print(f"Loaded tokenizer: {args.tokenizer}")

    # ----- load dataset (train split of given subset) -----
    dataset_name = "open-r1/OpenR1-Math-220k"
    print(f"Loading dataset {dataset_name}, subset={args.subset}, split=train ...")
    ds = load_dataset(dataset_name, args.subset, split="train")

    # ----- prefix & 输出路径 -----
    # 例如 subset=default -> OpenR1_Math_220k_default
    prefix = f"OpenR1_Math_220k_{args.subset}"

    train_jsonl = os.path.join(SAVE_DIR, "train", f"{prefix}.jsonl")
    valid_jsonl = os.path.join(SAVE_DIR, "valid", f"{prefix}.jsonl")
    train_parquet = os.path.join(SAVE_DIR, "train", f"{prefix}.parquet")
    valid_parquet = os.path.join(SAVE_DIR, "valid", f"{prefix}.parquet")

    # ----- Shuffle for 9:1 split -----
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
        # HF: problem, solution, answer, problem_type, question_type, source, ...
        question = str(sample.get("problem", "")).strip()
        solution = str(sample.get("solution", "")).strip()
        final_answer = str(sample.get("answer", "")).strip()

        # 优先用完整 CoT 作为 answer；如果 solution 为空就退回 answer
        answer_text = solution if solution else final_answer
        groundtruth = final_answer  # 直接用官方 answer 字段

        # ----- token 过滤 -----
        if args.max_len is not None:
            total_tokens = len(tokenizer(question + "\n" + answer_text).input_ids)
            if total_tokens > args.max_len:
                continue

        # ----- extra_info：除 problem/solution/answer 外其余字段全塞进去 -----
        extra_info = {
            k: v
            for k, v in sample.items()
            if k not in ("problem", "solution", "answer")
        }

        record = {
            "question": question,
            "answer": answer_text,          # DeepSeek R1 的 CoT 推理
            "groundtruth": groundtruth,     # 最终答案（字符串）
            "data_source": sample.get("source", f"openr1_math_220k_{args.subset}"),
            "ability": str(sample.get("problem_type", "")),  # Algebra / Geometry ...
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
    with open(train_jsonl, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(valid_jsonl, "w", encoding="utf-8") as f:
        for r in valid_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ----- 写 Parquet -----
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
