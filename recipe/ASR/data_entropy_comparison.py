import argparse
import json
import math
from collections import Counter
from transformers import AutoTokenizer

"""
python3 recipe/ASR/data_entropy_comparison.py --dataset_path /root/workspace/ASR_data/train/DAPO_MATH.jsonl --tokenizer_name Qwen/Qwen3-8B

python3 recipe/ASR/data_entropy_comparison.py --dataset_path /root/workspace/ASR_data/train/gsm8k.jsonl --tokenizer_name Qwen/Qwen3-8B

python3 recipe/ASR/data_entropy_comparison.py --dataset_path /root/workspace/ASR_data/train/HARP.jsonl --tokenizer_name Qwen/Qwen3-8B

python3 recipe/ASR/data_entropy_comparison.py --dataset_path /root/workspace/ASR_data/train/MATH.jsonl --tokenizer_name Qwen/Qwen3-8B

python3 recipe/ASR/data_entropy_comparison.py --dataset_path /root/workspace/ASR_data/train/NuminaMath_1.5.jsonl --tokenizer_name Qwen/Qwen3-8B

python3 recipe/ASR/data_entropy_comparison.py --dataset_path /root/workspace/ASR_data/train/NuminaMath_CoT.jsonl --tokenizer_name Qwen/Qwen3-8B

python3 recipe/ASR/data_entropy_comparison.py --dataset_path /root/workspace/ASR_data/train/OpenR1_Math_220k.jsonl --tokenizer_name Qwen/Qwen3-8B
"""

def load_dataset(path):
    data = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    return data

def compute_dataset_entropy(dataset, tokenizer):
    counter = Counter()
    total_tokens = 0

    for item in dataset:
        answer = item["answer"]
        tokens = tokenizer(answer, add_special_tokens=False)["input_ids"]
        counter.update(tokens)
        total_tokens += len(tokens)

    H = 0.0
    for token, count in counter.items():
        p = count / total_tokens
        H -= p * math.log(p + 1e-12)

    return H, total_tokens, len(counter)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    dataset = load_dataset(args.dataset_path)

    H, total, vocab = compute_dataset_entropy(dataset, tokenizer)

    print(f"Token-level entropy H* = {H}")

if __name__ == "__main__":
    main()
