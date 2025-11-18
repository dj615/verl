import os
import re
import json
import argparse
import random

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
        help=(
            "Prefix for output filenames. "
            "If not set, will use the dataset name suffix, e.g. NuminaMath_CoT."
        ),
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use HF streaming mode (IterableDataset)，不在本地缓存完整数据集。",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="(optional) HF datasets cache dir. 非 streaming 模式下生效，方便你之后手动删除。",
    )
    parser.add_argument(
        "--parquet_chunk_size",
        type=int,
        default=10000,
        help="How many rows per parquet write chunk to keep memory small.",
    )
    parser.add_argument(
        "--shuffle_buffer_size",
        type=int,
        default=10000,
        help="(仅 streaming 模式有效) 近似 shuffle 的 buffer size。",
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

    total = None
    raw_train_size = None

    if args.streaming:
        # streaming 模式：不在本地落完整缓存
        ds = load_dataset(
            args.dataset,
            split="train",
            streaming=True,
        )
        if args.shuffle_buffer_size and args.shuffle_buffer_size > 0:
            ds = ds.shuffle(seed=42, buffer_size=args.shuffle_buffer_size)
        print("Using streaming=True, total samples unknown; using approximate 9:1 random split.")
        random.seed(42)
    else:
        # 普通模式：会下载 + 缓存到本地（可通过 cache_dir 指定目录）
        if args.cache_dir is not None:
            ds = load_dataset(
                args.dataset,
                split="train",
                cache_dir=args.cache_dir,
            )
        else:
            ds = load_dataset(
                args.dataset,
                split="train",
            )

        # Shuffle for 9:1 split
        ds = ds.shuffle(seed=42)
        total = len(ds)
        raw_train_size = int(total * 0.9)

        print(
            f"Total samples: {total}, raw train={raw_train_size}, "
            f"raw valid={total - raw_train_size}"
        )

    print(
        f"max_train_data={args.max_train_data}, "
        f"max_valid_data={args.max_valid_data}"
    )

    # ----- prefix & 输出路径 -----
    prefix = "Numina_CoT"

    train_jsonl = os.path.join(SAVE_DIR, "train", f"{prefix}.jsonl")
    valid_jsonl = os.path.join(SAVE_DIR, "valid", f"{prefix}.jsonl")
    train_parquet = os.path.join(SAVE_DIR, "train", f"{prefix}.parquet")
    valid_parquet = os.path.join(SAVE_DIR, "valid", f"{prefix}.parquet")

    # ----- 打开 JSONL（流式写） -----
    train_jsonl_f = open(train_jsonl, "w", encoding="utf-8")
    valid_jsonl_f = open(valid_jsonl, "w", encoding="utf-8")

    # ----- 分块写 Parquet -----
    train_writer = None
    valid_writer = None
    train_chunk = []
    valid_chunk = []
    CHUNK_SIZE = args.parquet_chunk_size

    train_count = 0
    valid_count = 0

    try:
        for idx, sample in enumerate(ds):
            # 如果两个 split 都达到上限，可以直接提前结束
            if (
                args.max_train_data is not None
                and args.max_valid_data is not None
                and train_count >= args.max_train_data
                and valid_count >= args.max_valid_data
            ):
                break

            # ----- 基本字段 -----
            # NuminaMath-CoT: source, problem, solution, messages
            question = str(sample.get("problem", "")).strip()
            solution = str(sample.get("solution", "")).strip()

            # 从 CoT 文本里抽 groundtruth（如果抽不到，就为空字符串）
            groundtruth = extract_groundtruth_from_solution(solution)

            # 把完整 CoT 作为 answer
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

            # ----- 9:1 划分策略 -----
            if not args.streaming:
                # 非 streaming：严格按打乱后的前 90% 做 train
                is_train = idx < raw_train_size
            else:
                # streaming：近似随机 9:1 划分
                if (
                    args.max_train_data is not None
                    and train_count >= args.max_train_data
                ):
                    is_train = False
                elif (
                    args.max_valid_data is not None
                    and valid_count >= args.max_valid_data
                ):
                    is_train = True
                else:
                    is_train = random.random() < 0.9

            # ----- 写入对应 split，并控制数量上限 -----
            if is_train:
                if args.max_train_data is not None and train_count >= args.max_train_data:
                    continue

                # 写 JSONL（单行流式）
                train_jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                # Parquet chunk
                train_chunk.append(record)
                train_count += 1

                if len(train_chunk) >= CHUNK_SIZE:
                    train_writer = flush_chunk(train_chunk, train_writer, train_parquet)
            else:
                if args.max_valid_data is not None and valid_count >= args.max_valid_data:
                    continue

                valid_jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                valid_chunk.append(record)
                valid_count += 1

                if len(valid_chunk) >= CHUNK_SIZE:
                    valid_writer = flush_chunk(valid_chunk, valid_writer, valid_parquet)

    finally:
        # 把剩余 chunk 写完
        train_writer = flush_chunk(train_chunk, train_writer, train_parquet)
        valid_writer = flush_chunk(valid_chunk, valid_writer, valid_parquet)

        if train_writer is not None:
            train_writer.close()
        if valid_writer is not None:
            valid_writer.close()

        train_jsonl_f.close()
        valid_jsonl_f.close()

    print(f"Final train kept: {train_count}")
    print(f"Final valid kept: {valid_count}")
    print("Done!")
    print(f"Train jsonl:   {train_jsonl}")
    print(f"Valid jsonl:   {valid_jsonl}")
    print(f"Train parquet: {train_parquet}")
    print(f"Valid parquet: {valid_parquet}")


if __name__ == "__main__":
    main()
