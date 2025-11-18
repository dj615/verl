import os
import re
import json
import argparse
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer


SAVE_DIR = "/root/workspace/ASR_data"
os.makedirs(os.path.join(SAVE_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(SAVE_DIR, "valid"), exist_ok=True)


def extract_groundtruth_from_solution(solution: str) -> str:
    """
    从 NuminaMath-CoT 的 solution 文本里尽量抽出“最终答案”：
    1）优先抓最后一个 \\boxed{...}
    2）如果没有 boxed，就退而求其次抓最后一个 $...$ 内的内容
    都没有就返回空字符串
    """
    if solution is None:
        return ""
    text = str(solution)

    # 1. \boxed{...}
    boxed = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxed:
        return boxed[-1].strip()

    # 2. 最后一个 $...$
    dollars = re.findall(r"\$([^$]+)\$", text)
    if dollars:
        return dollars[-1].strip()

    return ""


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
        default="AI-MO/NuminaMath-CoT",
        help="HF dataset name. Default: AI-MO/NuminaMath-CoT",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Prefix for output filenames. "
             "If not set, will use the dataset name suffix, e.g. NuminaMath_CoT.",
    )
    args = parser.parse_args()

    # ----- tokenizer -----
    tokenizer = None
    if args.max_len is not None:
        if args.tokenizer is None:
            raise ValueError("--tokenizer must be specified when --max_len is set.")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
        print(f"Loaded tokenizer: {args.tokenizer}")

    # ----- load dataset (train split) -----
    print(f"Loading dataset {args.dataset} (split=train)...")
    ds = load_dataset(args.dataset, split="train")

    # ----- prefix & 输出路径 -----
    if args.prefix is not None:
        prefix = args.prefix
    else:
        # 例如 "AI-MO/NuminaMath-CoT" -> "NuminaMath_CoT"
        base = args.dataset.split("/")[-1]
        prefix = base.replace(".", "_").replace("-", "_")

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
        # NuminaMath-CoT: source, problem, solution, messages
        question = str(sample.get("problem", "")).strip()
        solution = str(sample.get("solution", "")).strip()

        # 从 CoT 文本里抽 groundtruth（如果抽不到，就为空字符串）
        groundtruth = extract_groundtruth_from_solution(solution)

        # 这里把完整 CoT 作为 answer
        answer_text = solution

        # ----- token 过滤 -----
        if args.max_len is not None:
            total_tokens = len(tokenizer(question + "\n" + answer_text).input_ids)
            if total_tokens > args.max_len:
                continue

        # ----- extra_info：除了已经用到的字段之外，其余都塞进去 -----
        extra_info = {}
        # 保留 messages 方便之后做对话式训练
        if "messages" in sample:
            extra_info["messages"] = sample["messages"]

        record = {
            "question": question,
            "answer": answer_text,          # CoT 解答
            "groundtruth": groundtruth,     # 从 CoT 里抽出来的最终答案（可能为空）
            "data_source": sample.get("source", "numinamath_cot"),
            # 这里能力字段就先用 source 塞一下（synthetic_math / cn_k12 / olympiads 等）
            "ability": str(sample.get("source", "")),
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
