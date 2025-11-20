import os
import json
from typing import Dict, List, Union

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import (
    load_dataset,
    Dataset,
    DatasetDict,
    concatenate_datasets,
)

SAVE_DIR = "/root/workspace/ASR_data/test"


def load_hf_dataset(
    hf_name: str,
    split: Union[str, None],
    subset: Union[str, None],
):
    """
    通用加载 HuggingFace 数据集：
    - 支持有 / 没有 subset(name)
    - 支持有 / 没有 split（无 split 时自动合并所有 split）
    """
    load_kwargs = {}

    if subset not in (None, "", "none", "None"):
        load_kwargs["name"] = subset

    if split not in (None, "", "none", "None"):
        load_kwargs["split"] = split

    ds_raw = load_dataset(hf_name, **load_kwargs)

    if isinstance(ds_raw, DatasetDict):
        # 没指定 split 时，合并所有 split
        ds = concatenate_datasets(list(ds_raw.values()))
    else:
        ds = ds_raw

    return ds


from typing import List, Optional, Tuple

# (hf_name, data_name, split, subset, prompt_key, groundtruth_key)
DatasetConfig = Tuple[str, str, Optional[str], Optional[str], str, str]


def process_datasets(configs: List[DatasetConfig]):
    """
    每个 config:
      (hf_name, data_name, split, subset, prompt_key, groundtruth_key)

    例如：
      ("Idavidrein/gpqa", "gpqa_diamond", "train", "gpqa_diamond", "Question", "Correct Answer")
    """
    os.makedirs(SAVE_DIR, exist_ok=True)

    for hf_name, data_name, split, subset, prompt_key, groundtruth_key in configs:
        print(f"Loading dataset: {hf_name}, split={split}, subset={subset}")
        ds = load_hf_dataset(hf_name, split, subset)

        records = []
        for ex in ds:
            # ====== gpqa 特殊处理：改 prompt + groundtruth ======
            if hf_name == "Idavidrein/gpqa":
                # 问题
                q_val = ex.get(prompt_key, "")
                if isinstance(q_val, (dict, list)):
                    question = json.dumps(q_val, ensure_ascii=False)
                else:
                    question = str(q_val)

                # 选项：来自原始字段
                opt_A = str(ex.get("Correct Answer", ""))
                opt_B = str(ex.get("Incorrect Answer 1", ""))
                opt_C = str(ex.get("Incorrect Answer 2", ""))
                opt_D = str(ex.get("Incorrect Answer 3", ""))

                # 拼接新的 prompt
                prompt_str = (
                    question
                    + ' Please choose one of the following options as the answer: '
                    + f'(A) {opt_A}, (B) {opt_B}, (C) {opt_C}, (D) {opt_D}. '
                    + 'Reason step by step and output your final answer as only one letter '
                      '(A, B, C, or D) inside \\boxed{}.'
                )

                # groundtruth 改成选项字母（正确答案放在 A）
                gt_str = "A"

                # 如果以后想随机打乱选项顺序，可以在这里：
                #   1. 先构建 options = [("A", opt_A), ("B", opt_B), ...]
                #   2. 随机打乱
                #   3. 找到文本等于原 ex["Correct Answer"] 的那一项，取对应字母作为 gt_str

            else:
                # ====== 其它数据集：保持原逻辑 ======
                p_val = ex.get(prompt_key, "")
                if isinstance(p_val, (dict, list)):
                    prompt_str = json.dumps(p_val, ensure_ascii=False)
                else:
                    prompt_str = str(p_val)

                g_val = ex.get(groundtruth_key, "")
                if isinstance(g_val, (dict, list)):
                    gt_str = json.dumps(g_val, ensure_ascii=False)
                else:
                    gt_str = str(g_val)

            records.append(
                {
                    "prompt": prompt_str,
                    "groundtruth": gt_str,
                }
            )

        jsonl_path = os.path.join(SAVE_DIR, f"{data_name}.jsonl")
        parquet_path = os.path.join(SAVE_DIR, f"{data_name}.parquet")

        # 写 JSONL
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # 写 Parquet
        table = pa.Table.from_pylist(records)
        pq.write_table(table, parquet_path)

        print(
            f"Saved {len(records)} rows for {hf_name} "
            f"to {jsonl_path} and {parquet_path}"
        )


if __name__ == "__main__":
    # "huggingface_name": ["save_name", "split", "subset", "prompt_key", "groundtruth_key"]
    example_map = {
        # for openscience
        "Idavidrein/gpqa": ["gpqa_diamond", "train", "gpqa_diamond", "Question", "Correct Answer"],
        "Idavidrein/gpqa": ["gpqa_main", "train", "gpqa_main", "Question", "Correct Answer"],
        "Idavidrein/gpqa": ["gpqa_extended", "train", "gpqa_extended", "Question", "Correct Answer"],
        "basicv8vc/SimpleQA": ["simpleqa", "test", None, "problem", "answer"],
        "google/frames-benchmark": ["frames", "test", None, "prompt", "answer"],
        # for math & dapo_math
        "HuggingFaceH4/aime_2024": ["aime24", "train", None, "problem", "answer"],
        "math-ai/aime25": ["aime25", "test", None, "problem", "answer"],
        "math-ai/amc23": ["amc23", "test", None, "question", "answer"],
        "HuggingFaceH4/MATH-500": ["math500", "test", None, "problem", "answer"],
        "KbsdJames/Omni-MATH": ["omni_math", "test", None, "problem", "answer"],
        "math-ai/minervamath": ["minerva", "test", None, "problem", "answer"],
    }

    process_datasets(example_map)


if __name__ == "__main__":
    # (hf_name, data_name, split, subset, prompt_key, groundtruth_key)
    example_configs: List[DatasetConfig] = [
        # for openscience
        ("Idavidrein/gpqa", "gpqa_diamond", "train", "gpqa_diamond", "Question", "Correct Answer"),
        ("Idavidrein/gpqa", "gpqa_main", "train", "gpqa_main", "Question", "Correct Answer"),
        ("Idavidrein/gpqa", "gpqa_extended", "train", "gpqa_extended", "Question", "Correct Answer"),
        ("basicv8vc/SimpleQA", "simpleqa", "test", None, "problem", "answer"),
        ("google/frames-benchmark", "frames", "test", None, "prompt", "answer"),
        # for math & dapo_math
        ("HuggingFaceH4/aime_2024", "aime24", "train", None, "problem", "answer"),
        ("math-ai/aime25", "aime25", "test", None, "problem", "answer"),
        ("math-ai/amc23", "amc23", "test", None, "question", "answer"),
        ("HuggingFaceH4/MATH-500", "math500", "test", None, "problem", "answer"),
        ("KbsdJames/Omni-MATH", "omni_math", "test", None, "problem", "answer"),
        ("math-ai/minervamath", "minerva", "test", None, "problem", "answer"),
    ]

    process_datasets(example_configs)
