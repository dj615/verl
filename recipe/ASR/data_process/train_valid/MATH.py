import os
import json
import re
import argparse
import random

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
    """从 solution 中提取 boxed{...} 作为最终答案"""
    if not isinstance(text, str):
        return ""
    match = re.search(r"boxed\{(.*?)\}", text.replace("\\", ""), flags=re.IGNORECASE)
    if not match:
        return ""
    ans = match.group(1).strip().replace("\n", " ").strip()
    return ans


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
        help="Tokenizer name when using max_len.",
    )
    parser.add_argument(
        "--max_train_data",
        type=int,
        default=None,
        help="Maximum number of filtered training samples to keep.",
    )
    parser.add_argument(
        "--max_valid_data",
        type=int,
        default=None,
        help="Maximum number of filtered validation samples to keep.",
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

    # ===== tokenizer =====
    tokenizer = None
    if args.max_len is not None:
        if args.tokenizer is None:
            raise ValueError("--tokenizer must be provided when --max_len is used.")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
        print(f"Loaded tokenizer: {args.tokenizer}")

    print("Loading dataset EleutherAI/hendrycks_math...")

    total = None
    raw_train_size = None

    if args.streaming:
        # streaming 模式：不在本地落完整缓存
        ds = load_dataset(
            "EleutherAI/hendrycks_math",
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
                "EleutherAI/hendrycks_math",
                split="train",
                cache_dir=args.cache_dir,
            )
        else:
            ds = load_dataset(
                "EleutherAI/hendrycks_math",
                split="train",
            )

        # shuffle for 9:1 split
        ds = ds.shuffle(seed=42)
        total = len(ds)
        raw_train_size = int(total * 0.9)

        print(
            f"Total samples: {total}, raw train: {raw_train_size}, "
            f"raw valid: {total - raw_train_size}"
        )

    print(
        f"max_train_data={args.max_train_data}, "
        f"max_valid_data={args.max_valid_data}"
    )

    # ===== 打开 JSONL（流式写） =====
    train_jsonl_f = open(TRAIN_JSONL, "w", encoding="utf-8")
    valid_jsonl_f = open(VALID_JSONL, "w", encoding="utf-8")

    # ===== 分块写 Parquet =====
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

            question = str(sample.get("problem", "")).strip()
            answer = str(sample.get("solution", "")).strip()

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

            # ===== 9:1 划分策略 =====
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

            # ===== 写入对应 split，并控制数量上限 =====
            if is_train:
                if args.max_train_data is not None and train_count >= args.max_train_data:
                    continue

                # 写 JSONL（单行流式）
                train_jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                # Parquet chunk
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

    print(f"Final kept train samples: {train_count}")
    print(f"Final kept valid samples: {valid_count}")

    print("Writing finished.")
    print(f"Train jsonl:   {TRAIN_JSONL}")
    print(f"Valid jsonl:   {VALID_JSONL}")
    print(f"Train parquet: {TRAIN_PARQUET}")
    print(f"Valid parquet: {VALID_PARQUET}")


if __name__ == "__main__":
    main()
