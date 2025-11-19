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


def process_datasets(datasets_map: Dict[str, List[str]]):
    """
    datasets_map 形如：
    {
        "EleutherAI/hendrycks_math": ["MATH_eval", "test", None, "problem", "solution"],
        "BytedTsinghua-SIA/DAPO-Math-17k": ["DAPO_MATH_eval", "train", None, "prompt", "answer"],
        "some_dataset/without_split": ["MY_DATA", None, "text", "label"],
    }

    每个 value:
      长度为 5: [data_name, split, subset, prompt_key, groundtruth_key]
      长度为 4: [data_name, split, prompt_key, groundtruth_key]（无 subset 时）
    """
    os.makedirs(SAVE_DIR, exist_ok=True)

    for hf_name, cfg in datasets_map.items():
        if len(cfg) == 5:
            data_name, split, subset, prompt_key, groundtruth_key = cfg
        elif len(cfg) == 4:
            data_name, split, prompt_key, groundtruth_key = cfg
            subset = None
        else:
            raise ValueError(
                f"Config for {hf_name} must have 4 or 5 elements: "
                "[data_name, split, subset, prompt_key, groundtruth_key] "
                "or [data_name, split, prompt_key, groundtruth_key]"
            )

        print(f"Loading dataset: {hf_name}, split={split}, subset={subset}")
        ds = load_hf_dataset(hf_name, split, subset)

        records = []
        for ex in ds:
            # prompt
            p_val = ex.get(prompt_key, "")
            if isinstance(p_val, (dict, list)):
                prompt_str = json.dumps(p_val, ensure_ascii=False)
            else:
                prompt_str = str(p_val)

            # groundtruth
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
