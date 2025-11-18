import os
import json

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

SAVE_DIR = "/root/workspace/ASR_data/test"
DATA_NAME = "mmlu_pro"
OUTPUT_JSONL = os.path.join(SAVE_DIR, f"{DATA_NAME}.jsonl")
OUTPUT_PARQUET = os.path.join(SAVE_DIR, f"{DATA_NAME}.parquet")

INSTRUCTION = (
    # " Based on the question and the given choices, reason step by step, then select the single correct answer from A/B/C/D and output only one of \boxed{A}, \boxed{B}, \boxed{C}, or \boxed{D}, with no additional text."
    "",
)

os.makedirs(SAVE_DIR, exist_ok=True)


def to_letter(ans):
    """
    mmlu 的 answer 可能是：
    - int: 0/1/2/3
    - str: 'A'/'B'/'C'/'D' 或 '0'/'1'/'2'/'3'
    统一转成 'A'~'D'
    """
    if isinstance(ans, int):
        mapping = ["A", "B", "C", "D"]
        return mapping[ans] if 0 <= ans < 4 else str(ans)

    s = str(ans).strip()
    if s in ["A", "B", "C", "D"]:
        return s
    if s in ["0", "1", "2", "3"]:
        mapping = ["A", "B", "C", "D"]
        return mapping[int(s)]
    return s


def main():
    # 通常用于评测的是 config="all", split="test"
    print("Loading cais/mmlu (config='all', split='test')...")
    ds = load_dataset("cais/mmlu", "all", split="test")

    records = []
    for ex in ds:
        q = ex.get("question", "").strip()
        ans = ex.get("answer", "")
        gt = to_letter(ans)

        prompt = q + INSTRUCTION

        records.append(
            {
                "prompt": prompt,
                "groundtruth": gt,
            }
        )

    print(f"Total samples: {len(records)}")

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
