import os
import json

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

SAVE_DIR = "/root/workspace/ASR_data/test"
DATA_NAME = "hle"
OUTPUT_JSONL = os.path.join(SAVE_DIR, f"{DATA_NAME}.jsonl")
OUTPUT_PARQUET = os.path.join(SAVE_DIR, f"{DATA_NAME}.parquet")

os.makedirs(SAVE_DIR, exist_ok=True)


def main():
    print("Loading cais/hle (split='test')...")
    ds = load_dataset("cais/hle", split="test")

    records = []
    for ex in ds:
        # 只保留 image 为 None 的样本
        if ex.get("image") is not None:
            continue

        question = ex.get("question", "")
        answer = ex.get("answer", "")
        answer_type = ex.get("answer_type", "")

        records.append(
            {
                "prompt": question,
                "groundtruth": answer,
                "answer_type": answer_type,
            }
        )

    print(f"Kept {len(records)} text-only samples (image is None).")

    # 写 JSONL
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 写 Parquet
    table = pa.Table.from_pylist(records)
    pq.write_table(table, OUTPUT_PARQUET)

    print(f"Saved to:\n  {OUTPUT_JSONL}\n  {OUTPUT_PARQUET}")


if __name__ == "__main__":
    main()
