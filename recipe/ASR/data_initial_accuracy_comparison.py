import os
import json
import re
import argparse
from typing import Dict, Any, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


DATA_FILES = [
    "/root/workspace/ASR_data/valid/DAPO_MATH.jsonl",
    "/root/workspace/ASR_data/valid/gsm8k.jsonl",
    "/root/workspace/ASR_data/valid/HARP.jsonl",
    "/root/workspace/ASR_data/valid/MATH.jsonl",
    "/root/workspace/ASR_data/valid/NuminaMath_1.5.jsonl",
    "/root/workspace/ASR_data/valid/NuminaMath_CoT.jsonl",
    "/root/workspace/ASR_data/valid/OpenR1_Math_220k.jsonl",
]


def load_jsonl(path: str, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load up to max_samples samples from a jsonl file."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if max_samples is not None and len(samples) >= max_samples:
                break
    return samples


def build_prompt(question: str) -> str:
    """Add instruction before question."""
    prefix = "Please output your final answer in \\boxed{}.\n\n"
    return prefix + question


def extract_boxed(text: str) -> str:
    """
    从模型输出中提取 \\boxed{...} 的内容：
    1）优先取最后一个 \\boxed{...}
    2）如果没有，就取最后一行非空文本
    """
    if text is None:
        return ""
    text = str(text)

    # 1. 匹配 \boxed{ ... }
    matches = re.findall(r"\\boxed\{([^}]*)\}", text)
    if matches:
        return matches[-1].strip()

    # 2. 没有 boxed, 就用最后一行非空
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        return lines[-1]
    return ""


def normalize_ans(ans: str) -> str:
    """
    轻量归一化：
    - 去掉首尾空格
    - 去掉首尾的 $, 以及句末的点号
    - 把多个空格缩成一个
    """
    if ans is None:
        return ""
    x = str(ans).strip()

    # 去掉外层 $ ... $
    if x.startswith("$") and x.endswith("$") and len(x) >= 2:
        x = x[1:-1].strip()

    # 去掉末尾句号
    x = x.rstrip(" .")

    # 去掉首尾括号（有些答案是 (42)）
    if x.startswith("(") and x.endswith(")"):
        x = x[1:-1].strip()

    # 多空格 -> 单空格
    x = re.sub(r"\s+", " ", x)

    return x


def get_groundtruth(sample: Dict[str, Any]) -> str:
    """优先用 sample['groundtruth']，否则从 reward_model['ground_truth'] 取。"""
    if "groundtruth" in sample and sample["groundtruth"] is not None:
        return str(sample["groundtruth"])
    rm = sample.get("reward_model", {})
    return str(rm.get("ground_truth", ""))


def run_model(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device: str = "cuda",
) -> str:
    """
    调用模型生成：
    - 如果 tokenizer 有 apply_chat_template，则走 chat 格式
    - 否则直接把 prompt 丢进去
    """
    model_inputs = None

    # chat 模式
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer(text, return_tensors="pt")
    else:
        model_inputs = tokenizer(prompt, return_tensors="pt")

    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # 只取新生成部分
    gen_ids = output_ids[0][model_inputs["input_ids"].shape[1]:]
    out_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return out_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen3-8B",
        help="HF model name, e.g. Qwen/Qwen3-8B",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference, e.g. cuda / cpu",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=8192,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max total samples across all datasets (for quick test).",
    )
    args = parser.parse_args()

    print(f"Loading tokenizer & model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
        device_map=args.device if args.device != "cpu" else None,
    )
    model.eval()

    total_all = 0
    correct_all = 0

    for path in DATA_FILES:
        if not os.path.exists(path):
            print(f"[WARN] File not found, skip: {path}")
            continue

        # 如果设置了 max_samples，总量到上限就停
        remaining = None
        if args.max_samples is not None:
            remaining = max(args.max_samples - total_all, 0)
            if remaining <= 0:
                break

        print(f"\n=== Evaluating on {path} ===")
        samples = load_jsonl(path, max_samples=remaining)
        if not samples:
            print("No samples loaded, skip.")
            continue

        correct = 0
        total = 0

        for sample in samples:
            question = str(sample.get("question", "")).strip()
            if not question:
                continue

            gt_raw = get_groundtruth(sample)

            prompt = build_prompt(question)
            model_output = run_model(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
            )

            pred_raw = extract_boxed(model_output)

            gt_norm = normalize_ans(gt_raw)
            pred_norm = normalize_ans(pred_raw)

            is_correct = (gt_norm != "" and pred_norm == gt_norm)

            total += 1
            total_all += 1
            if is_correct:
                correct += 1
                correct_all += 1

            if total % 50 == 0:
                acc_tmp = correct / total if total > 0 else 0.0
                print(f"[{os.path.basename(path)}] progress={total}, acc={acc_tmp:.4f}")

            if args.max_samples is not None and total_all >= args.max_samples:
                break

        acc = correct / total if total > 0 else 0.0
        print(f"==> {os.path.basename(path)}: correct={correct}/{total}, acc={acc:.4f}")

        if args.max_samples is not None and total_all >= args.max_samples:
            break

    global_acc = correct_all / total_all if total_all > 0 else 0.0
    print("\n==============================")
    print(f"GLOBAL: correct={correct_all}/{total_all}, acc={global_acc:.4f}")


if __name__ == "__main__":
    main()
