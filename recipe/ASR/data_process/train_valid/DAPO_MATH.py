import os
import json
import argparse
import random

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


def flush_chunk(chunk, writer, path):
    """
    将一个 Python list[dict] 写入 Parquet。如果 writer 还没创建，就在这里创建。
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
        "--streaming",
        action="store_true",
        help="Use HF streaming mode (IterableDataset), 不在本地缓存完整数据集。",
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
        help="Streaming 模式下 shuffle 的 buffer size；非 streaming 模式忽略该参数。",
    )

    args = parser.parse_args()

    # Load tokenizer if needed
    tokenizer = None
    if args.max_len is not None:
        if args.tokenizer is None:
            raise ValueError("--tokenizer must be specified when --max_len is set.")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
        print(f"Loaded tokenizer: {args.tokenizer}")

    print("Loading dataset BytedTsinghua-SIA/DAPO-Math-17k...")

    if args.streaming:
        # Streaming 模式：不在本地缓存完整数据集
        ds = load_dataset(
            "BytedTsinghua-SIA/DAPO-Math-17k",
            split="train",
            streaming=True,
        )
        # IterableDataset 的 shuffle 是基于 buffer 的近似随机
        if args.shuffle_buffer_size and args.shuffle_buffer_size > 0:
            ds = ds.shuffle(seed=42, buffer_size=args.shuffle_buffer_size)
        total = None
        raw_train_size = None
        print("Using streaming=True, approximate 9:1 train/valid split (no exact length).")
    else:
        # 普通模式：会下载 + 缓存到 cache_dir（如果提供）或默认缓存目录
        if args.cache_dir is not None:
            ds = load_dataset(
                "BytedTsinghua-SIA/DAPO-Math-17k",
                split="train",
                cache_dir=args.cache_dir,
            )
        else:
            ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")

        # Shuffle for 9:1 split
        ds = ds.shuffle(seed=42)
        total = len(ds)
        raw_train_size = int(total * 0.9)
        print(f"Total samples: {total}, raw train={raw_train_size}, raw valid={total - raw_train_size}")

    print(f"max_train_data={args.max_train_data}, max_valid_data={args.max_valid_data}")

    # 打开 JSONL 文件（流式写入）
    train_jsonl_f = open(TRAIN_JSONL, "w", encoding="utf-8")
    valid_jsonl_f = open(VALID_JSONL, "w", encoding="utf-8")

    # Parquet 分块写入
    train_writer = None
    valid_writer = None
    train_chunk = []
    valid_chunk = []
    CHUNK_SIZE = args.parquet_chunk_size

    train_count = 0
    valid_count = 0

    # 为 streaming 模式的近似 9:1 划分设置随机种子，方便复现
    random.seed(42)

    try:
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
            if tokenizer is not None and args.max_len is not None:
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
                    "ground_truth": groundtruth,
                },
                "extra_info": sample.get("extra_info", {}),
            }

            # ----------- 决定当前样本分到 train 还是 valid -----------
            if not args.streaming:
                # 非 streaming：严格按打乱后的前 90% 做 train
                is_train = idx < raw_train_size
            else:
                # streaming：近似 9:1，根据随机数划分
                # 同时兼顾 max_train_data / max_valid_data 的上限
                if args.max_train_data is not None and train_count >= args.max_train_data:
                    is_train = False
                elif args.max_valid_data is not None and valid_count >= args.max_valid_data:
                    is_train = True
                else:
                    is_train = random.random() < 0.9
            # ---------------------------------------------------

            if is_train:
                if args.max_train_data is not None and train_count >= args.max_train_data:
                    continue

                # 写 JSONL（单行）
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
        # 把剩余的 chunk 写完
        train_writer = flush_chunk(train_chunk, train_writer, TRAIN_PARQUET)
        valid_writer = flush_chunk(valid_chunk, valid_writer, VALID_PARQUET)

        # 关闭 writer
        if train_writer is not None:
            train_writer.close()
        if valid_writer is not None:
            valid_writer.close()

        # 关闭 JSONL 文件
        train_jsonl_f.close()
        valid_jsonl_f.close()

    print(f"Final train kept: {train_count}")
    print(f"Final valid kept: {valid_count}")
    print("Done!")
    print(f"Train jsonl:   {TRAIN_JSONL}")
    print(f"Valid jsonl:   {VALID_JSONL}")
    print(f"Train parquet: {TRAIN_PARQUET}")
    print(f"Valid parquet: {VALID_PARQUET}")


if __name__ == "__main__":
    main()
