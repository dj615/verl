import os
import json
import argparse
import random
import itertools

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer


SAVE_DIR = "/root/workspace/ASR_data"
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
    tokenizer = None    # type: ignore
    if args.max_len is not None:
        if args.tokenizer is None:
            raise ValueError("--tokenizer must be specified when --max_len is set.")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
        print(f"Loaded tokenizer: {args.tokenizer}")

    # ----- load dataset (train split of given subset) -----
    dataset_name = "open-r1/OpenR1-Math-220k"
    print(f"Loading dataset {dataset_name}, subset={args.subset} ...")

    total = None
    raw_train_size = None

    if args.streaming:
        print("Using streaming=True (IterableDataset)，不会在本地落完整缓存。")
        random.seed(42)

        if args.subset == "all":
            # 默认 + extended 串起来
            ds_default = load_dataset(dataset_name, split="default", streaming=True)
            ds_extended = load_dataset(dataset_name, split="extended", streaming=True)
            ds_iter = itertools.chain(ds_default, ds_extended)
        else:
            # subset=default / extended -> 对应 split
            ds_iter = load_dataset(
                dataset_name,
                split=args.subset,
                streaming=True,
            )

        # streaming 模式用「逐样本随机 9:1」做近似划分，不再对 dataset 本身 shuffle
        ds = ds_iter
        print("Total samples: unknown in streaming mode; using approximate 9:1 random split.")
    else:
        # 非 streaming：会下载 + 缓存到本地（可以通过 cache_dir 控制路径）
        if args.subset == "all":
            # all = default + extended 合并
            if args.cache_dir is not None:
                ds_default = load_dataset(
                    dataset_name,
                    split="default",
                    cache_dir=args.cache_dir,
                )
                ds_extended = load_dataset(
                    dataset_name,
                    split="extended",
                    cache_dir=args.cache_dir,
                )
            else:
                ds_default = load_dataset(dataset_name, split="default")
                ds_extended = load_dataset(dataset_name, split="extended")

            ds = concatenate_datasets([ds_default, ds_extended])
        else:
            # subset=default / extended -> 对应 split
            if args.cache_dir is not None:
                ds = load_dataset(
                    dataset_name,
                    split=args.subset,
                    cache_dir=args.cache_dir,
                )
            else:
                ds = load_dataset(dataset_name, split=args.subset)

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
    prefix = "OpenR1_Math_220k"

    train_jsonl = os.path.join(SAVE_DIR, "train", f"{prefix}.jsonl")
    valid_jsonl = os.path.join(SAVE_DIR, "valid", f"{prefix}.jsonl")
    train_parquet = os.path.join(SAVE_DIR, "train", f"{prefix}.parquet")
    valid_parquet = os.path.join(SAVE_DIR, "valid", f"{prefix}.parquet")

    # ----- 打开 JSONL 文件（流式写入） -----
    train_jsonl_f = open(train_jsonl, "w", encoding="utf-8")
    valid_jsonl_f = open(valid_jsonl, "w", encoding="utf-8")

    # ----- Parquet 分块写入 -----
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

            # ----- 基本字段 -----
            # HF: problem, solution, answer, problem_type, question_type, source, ...
            question = str(sample.get("problem", "")).strip()
            solution = str(sample.get("solution", "")).strip()
            final_answer = str(sample.get("answer", "")).strip()

            # 优先用完整 CoT 作为 answer；如果 solution 为空就退回 answer
            answer_text = solution if solution else final_answer
            groundtruth = final_answer  # 直接用官方 answer 字段

            # ----- token 过滤 -----
            if tokenizer is not None and args.max_len is not None:
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
                "data_source": sample.get(
                    "source",
                    f"openr1_math_220k_{args.subset}",
                ),
                "ability": str(sample.get("problem_type", "")),  # Algebra / Geometry ...
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
                # streaming：用近似随机 9:1 划分
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

                # 写 JSONL（单行）
                train_jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                # 加入 Parquet chunk
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
    print(f"Train jsonl:   {train_jsonl}")
    print(f"Valid jsonl:   {valid_jsonl}")
    print(f"Train parquet: {train_parquet}")
    print(f"Valid parquet: {valid_parquet}")


if __name__ == "__main__":
    main()
