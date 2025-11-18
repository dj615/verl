import os
import json
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

SAVE_DIR = "/root/workspace/ASR_data/test"

# 只处理 text-only 的 subset（名字里带 TO）
TEXT_ONLY_CONFIGS = [
    "OE_TO_maths_en_COMP",
    "OE_TO_maths_zh_CEE",
    "OE_TO_maths_zh_COMP",
    "OE_TO_physics_en_COMP",
    "OE_TO_physics_zh_CEE",
    "TP_TO_maths_en_COMP",
    "TP_TO_maths_zh_CEE",
    "TP_TO_maths_zh_COMP",
    "TP_TO_physics_en_COMP",
]

# 映射成你想要的 8 个组合维度：english/chinese, math/physics, comp/cee
LANG_MAP = {
    "English": "en",
    "Chinese": "zh",
}
SUBJECT_MAP = {
    "Math": "math",
    "Physics": "physics",
}
DIFFICULTY_MAP = {
    "Competition": "comp",
    "CEE": "cee",
}


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # group_name -> List[{"prompt", "response", "groundtruth"}]
    groups = defaultdict(list)

    for cfg in TEXT_ONLY_CONFIGS:
        print(f"Loading subset: {cfg}")
        ds = load_dataset("Hothan/OlympiadBench", cfg, split="train")

        for ex in ds:
            subject = ex.get("subject", "")
            language = ex.get("language", "")
            difficulty = ex.get("difficulty", "")

            s_key = SUBJECT_MAP.get(subject)
            l_key = LANG_MAP.get(language)
            d_key = DIFFICULTY_MAP.get(difficulty)

            # 如果有意料之外的取值就跳过
            if s_key is None or l_key is None or d_key is None:
                continue

            group_name = f"{s_key}_{l_key}_{d_key}"  # e.g. math_en_comp

            question = ex.get("question", "") or ""
            # 不再把 List 转成字符串，直接保留原始结构
            solution = ex.get("solution", [])
            final_answer = ex.get("final_answer", [])

            record = {
                "prompt": question,
                "response": solution,
                "groundtruth": final_answer,
            }
            groups[group_name].append(record)

    # 写出每个组合的 jsonl + parquet
    for group_name, records in groups.items():
        if not records:
            continue

        jsonl_path = os.path.join(
            SAVE_DIR, f"OlympiadBench_{group_name}.jsonl"
        )
        parquet_path = os.path.join(
            SAVE_DIR, f"OlympiadBench_{group_name}.parquet"
        )

        print(f"Saving {len(records)} samples to {jsonl_path} and {parquet_path}")

        # JSONL（list 会自动序列化成 JSON array）
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Parquet（pyarrow 会推断 list type）
        table = pa.Table.from_pylist(records)
        pq.write_table(table, parquet_path)

    print("Done.")


if __name__ == "__main__":
    main()
