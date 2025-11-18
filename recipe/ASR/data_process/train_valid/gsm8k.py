import os
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
        "--subset",
        type=str,
        default="main",
        choices=["main", "socratic"],
        help="GSM8K subset to use: 'main' or 'socratic'. Default: main",
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

    total = None
    raw_train_size = None

    if args.streaming:
        # streaming 模式，不在本地落完整缓存
        ds = load_dataset(
            "openai/gsm8k",
            args.subset,
            split="train",
            streaming=True,
        )
        print("Using streaming=True, total samples unknown; using approximate 9:1 random split.")
        random.seed(42)
    else:
        # 普通模式：会下载 + 缓存到本地（可通过 cache_dir 指定目录）
        if args.cache_dir is not None:
            ds = load_dataset(
                "openai/gsm8k",
                args.subset,
                split="train",
                cache_dir=args.cache_dir,
            )
        else:
            ds = load_dataset(
                "openai/gsm8k",
                args.subset,
                split="train",
            )

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
    prefix = "gsm8K"
    if args.subset != "main":
        prefix += f"_{args.subset}"

    train_jsonl = os.path.join(SAVE_DIR, "train", f"{prefix}.jsonl")
    valid_jsonl = os.path.join(SAVE_DIR, "valid", f"{prefix}.jsonl")
    train_parquet = os.path.join(SAVE_DIR, "train", f"{prefix}.parquet")
    valid_parquet = os.path.join(SAVE_DIR, "valid", f"{prefix}.parquet")

    # 流式写 JSONL
    train_jsonl_f = open(train_jsonl, "w", encoding="utf-8")
    valid_jsonl_f = open(valid_jsonl, "w", encoding="utf-8")

    # 分块写 Parquet
    train_writer = None
    valid_writer = None
    train_chunk = []
    valid_chunk = []
    CHUNK_SIZE = args.parquet_chunk_size

    train_count = 0
    valid_count = 0

    try:
        for idx, sample in enumerate(ds):
            # 如果两个 split 都已经达到上限，可以提前结束循环（特别是 streaming 时节省网络 IO）
            if (
                args.max_train_data is not None
                and args.max_valid_data is not None
                and train_count >= args.max_train_data
                and valid_count >= args.max_valid_data
            ):
                break

            # ===== question / answer =====
            question = str(sample.get("question", "")).strip()
            full_answer = str(sample.get("answer", "")).strip()

            groundtruth = extract_final_answer(full_answer)

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

                # JSONL（单行流式写）
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
        # 把剩余的 chunk 写完
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
