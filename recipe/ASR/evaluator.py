#!/usr/bin/env python
import argparse
import os
import re
import ast
import math
import asyncio
from typing import Literal, List, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel
from datasets import load_dataset
from openai import AsyncOpenAI


exact_match = {"mmlu_pro", "amc23", "aime24", "aime25", "gpqa_diamond", "gpqa_main", "gpqa_extended"}
SUPPORTED_TASKS = exact_match | {"hle"}


# =========================
# 通用小工具
# =========================

def ensure_list(x):
    """把各种形式的数据统一成 list[str]."""
    if isinstance(x, (list, tuple)):
        return list(x)
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    # 字符串形式的 list，比如 '["a", "b"]'
    if isinstance(x, str) and x.startswith("[") and x.endswith("]"):
        try:
            v = ast.literal_eval(x)
            if isinstance(v, (list, tuple)):
                return list(v)
        except Exception:
            pass
    return [x]


def extract_boxed(text: str):
    """
    从字符串中提取第一个 \boxed{...} 的内容，没有就返回 None.
    注意：原始文本里是单反斜杠 \boxed，
    所以正则里写 r"\\boxed" 来匹配一个反斜杠。
    """
    if text is None:
        return None
    text = str(text)
    # 支持换行
    m = re.search(r"\\boxed\s*\{([^}]*)\}", text, flags=re.DOTALL)
    if m:
        return m.group(1)
    return None


def normalize_ans(s: str) -> str:
    """轻量级归一化：去空白、外围 $ 和括号，去 \\left/\\right 等。"""
    if s is None:
        return ""
    s = str(s).strip()

    # 去掉外围的 $$ / $
    if s.startswith("$$") and s.endswith("$$") and len(s) >= 4:
        s = s[2:-2].strip()
    elif s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()

    # 去掉外围的 \(\) 或 \[\]
    if s.startswith(r"\(") and s.endswith(r"\)"):
        s = s[2:-2].strip()
    if s.startswith(r"\[") and s.endswith(r"\]"):
        s = s[2:-2].strip()

    # 去掉 \left \right
    s = s.replace(r"\left", "").replace(r"\right", "")

    # 去掉数字里的逗号（如 1,000 -> 1000）
    s = s.replace(",", "")

    # 去掉所有空白
    s = "".join(s.split())

    # 小写
    s = s.lower()

    # 去掉末尾的句号
    s = s.rstrip(".")

    # 去掉最外层一对括号 () 或 []
    while len(s) >= 2 and (
        (s[0] == "(" and s[-1] == ")")
        or (s[0] == "[" and s[-1] == "]")
    ):
        s = s[1:-1]

    return s


# =========================
# 1) Exact Match 任务：\boxed{} exact match（pass@n）
# =========================

def evaluate_math_like_file(pred_file: str) -> Tuple[float, int, int]:
    """
    对单个 parquet 文件做 pass@n:
    - 每一行视为一个样本
    - responses: list[str]（n 次采样）
    - groundtruth: string
    - 如果该样本的任意一个 response 的 \boxed{} 里答案 == groundtruth（归一化后），
      则该样本记为“通过”。
    返回 (acc, correct, total)，其中 acc = correct / total.
    """
    df = pd.read_parquet(pred_file)

    if "responses" not in df.columns or "groundtruth" not in df.columns:
        raise ValueError(
            f"{pred_file} 中必须包含 'responses' 和 'groundtruth' 这两列。"
        )

    total = 0
    correct = 0

    for _, row in df.iterrows():
        gt_raw = row["groundtruth"]
        gt_norm = normalize_ans(gt_raw)

        responses = ensure_list(row["responses"])
        passed = False

        for resp in responses:
            boxed = extract_boxed(resp)
            if boxed is None:
                continue
            pred_norm = normalize_ans(boxed)
            if pred_norm == gt_norm:
                passed = True
                break

        total += 1
        if passed:
            correct += 1

    acc = correct / total if total > 0 else 0.0
    return acc, correct, total


# =========================
# 2) HLE 官方 evaluator 逻辑（用 judge 模型）
# =========================

JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\%| and 100|\%| from [response]. Put 100 if there is no confidence score available."""


class ExtractedAnswer(BaseModel):
    extracted_final_answer: str
    reasoning: str
    correct: Literal["yes", "no"]
    confidence: int
    strict: Literal[True] = True  # 100% reliability


async def _hle_extract_answer(
    client: AsyncOpenAI,
    judge_model: str,
    question: str,
    correct_answer: str,
    response: str,
):
    """调用 judge 模型，对单个 (question, correct_answer, response) 打分。"""
    prompt = JUDGE_PROMPT.format(
        question=question, correct_answer=correct_answer, response=response
    )
    try:
        resp = await client.beta.chat.completions.parse(
            model=judge_model,
            max_completion_tokens=4096,  # 官方脚本中也是 "overkill for judge"
            messages=[{"role": "user", "content": prompt}],
            response_format=ExtractedAnswer,
        )
        content = resp.choices[0].message.parsed
        return {
            "model_answer": content.extracted_final_answer,
            "reasoning": content.reasoning,
            "correct": content.correct,  # "yes" / "no"
            "confidence": content.confidence,  # 0-100
        }
    except Exception as e:
        print("Judge Error:", e)
        return None


def calib_err(confidence, correct, p="2", beta=100):
    """
    直接照搬官方 calib_err 实现：
    - confidence: np.array, [0,1] 之间
    - correct: bool / 0/1
    """
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
    if not bins:
        return 0.0
    bins[-1] = [bins[-1][0], len(confidence)]

    cerr = 0.0
    total_examples = len(confidence)
    for i in range(len(bins) - 1):
        bin_confidence = confidence[bins[i][0] : bins[i][1]]
        bin_correct = correct[bins[i][0] : bins[i][1]]
        num_examples_in_bin = len(bin_confidence)

        if num_examples_in_bin > 0:
            difference = np.abs(
                np.nanmean(bin_confidence) - np.nanmean(bin_correct)
            )

            if p == "2":
                cerr += num_examples_in_bin / total_examples * np.square(difference)
            elif p == "1":
                cerr += num_examples_in_bin / total_examples * difference
            elif p in ("infty", "infinity", "max"):
                cerr = np.maximum(cerr, difference)
            else:
                raise ValueError("p must be '1', '2', or 'infty'")

    if p == "2":
        cerr = np.sqrt(cerr)

    return float(cerr)


async def _hle_judge_all_async(
    dataset_name: str,
    id2response: dict,
    judge_model: str,
    num_workers: int = 10,
):
    """
    真正的异步评估逻辑：
    - 从 HF 加载 cais/hle（或别的）test split
    - 用 judge 模型打分
    - 返回 correct_list, confidence_list, total_questions
    """
    client = AsyncOpenAI(timeout=300.0, max_retries=1)

    dataset = load_dataset(dataset_name, split="test").to_dict()
    questions = [dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())]
    total_questions = len(questions)

    semaphore = asyncio.Semaphore(num_workers)

    async def _process(q):
        qid = q["id"]
        if qid not in id2response:
            # 该问题没有预测
            return None
        question_text = q["question"]
        correct_answer = q["answer"]
        model_resp = id2response[qid]

        async with semaphore:
            judged = await _hle_extract_answer(
                client, judge_model, question_text, correct_answer, model_resp
            )
        if judged is None:
            return None
        # correct: "yes"/"no"
        is_correct = "yes" in judged["correct"]
        conf = judged["confidence"]
        return is_correct, conf

    tasks = [asyncio.create_task(_process(q)) for q in questions]
    task_results = await asyncio.gather(*tasks)

    correct_list = []
    conf_list = []
    for r in task_results:
        if r is None:
            continue
        c, conf = r
        correct_list.append(bool(c))
        conf_list.append(int(conf))

    return correct_list, conf_list, total_questions


def evaluate_hle_file(
    pred_file: str,
    dataset_name: str = "cais/hle",
    judge_model: str = "o3-mini-2025-01-31",
    num_workers: int = 10,
) -> Tuple[float, int, int]:
    """
    用官方 judge 逻辑评估一个 HLE 预测 parquet。
    pred_file: verl 输出的 parquet，至少包含 'responses' 列，建议包含 'id' 列。
    逻辑：
      1. 从 HF 加载 dataset_name 的 test split（默认 cais/hle）。
      2. 把 parquet 里的 responses 映射到每个 question id：
         - 若 parquet 有 'id' 列，则按 id 对齐；
         - 否则按行顺序与 HF dataset 的顺序一一对应。
      3. 调用 judge 模型，对每个有预测的样本输出 correct / confidence。
      4. 按官方 run_judge_results.py 计算 Accuracy 和 Calibration Error。
    返回：
      (acc, correct_count, total_questions)
      其中 acc 是 Accuracy / 100（0~1），
      correct_count 是 judge 判定的正确题目数（只算有预测的题目），
      total_questions 是 HF test split 的总题数（官方分母）。
    """
    df = pd.read_parquet(pred_file)
    if "responses" not in df.columns:
        raise ValueError(f"{pred_file} 中必须包含 'responses' 列（list[str]）。")

    # 构建 id -> response 映射，只取第一个 sample
    id2response = {}

    if "id" in df.columns:
        for _, row in df.iterrows():
            qid = row["id"]
            responses = ensure_list(row["responses"])
            if not responses:
                continue
            id2response[qid] = responses[0]
    else:
        # 没有 id，就按顺序对齐 HF dataset
        dataset = load_dataset(dataset_name, split="test").to_dict()
        questions = [
            dict(zip(dataset.keys(), values)) for values in zip(*dataset.values())
        ]
        n = min(len(df), len(questions))
        for i in range(n):
            qid = questions[i]["id"]
            responses = ensure_list(df.iloc[i]["responses"])
            if not responses:
                continue
            id2response[qid] = responses[0]

    correct_list, conf_list, total_questions = asyncio.run(
        _hle_judge_all_async(dataset_name, id2response, judge_model, num_workers)
    )

    correct = np.array(correct_list)
    confidence = np.array(conf_list) / 100.0

    if len(correct) != total_questions:
        print(
            f"Available predictions judged: {len(correct)} | Total questions in dataset: {total_questions}"
        )

    # Accuracy，用官方的分母 n = total_questions
    if total_questions == 0:
        acc = 0.0
        confidence_half_width = 0.0
    else:
        accuracy_pct = 100 * float(correct.sum()) / float(total_questions)
        acc = accuracy_pct / 100.0
        # Wald 95% CI
        confidence_half_width = 1.96 * math.sqrt(
            accuracy_pct * (100 - accuracy_pct) / total_questions
        )

    calibration_error = 100 * round(
        calib_err(confidence, correct, p="2", beta=100), 2
    )

    print("*** HLE Metrics ***")
    print(
        f"Accuracy: {acc*100:.2f}% +/- {confidence_half_width:.2f}% | n = {total_questions}"
    )
    print(f"Calibration Error: {calibration_error}")

    # 返回 (acc 比例, “判对题数”, 总题数)
    # 判对题数用 correct.sum()（只算真正 judge 掉的题），便于做 micro 统计。
    return acc, int(correct.sum()), int(total_questions)


# =========================
# 3) 主逻辑：支持 task + 多个文件 + macro avg
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="从 data.output_path 生成的 parquet 中解析 \\boxed{} 或用 HLE 官方 evaluator 计算准确率"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="任务名称，如 mmlu_pro / amc23 / aime24 / aime25 / hle",
    )
    parser.add_argument(
        "--pred_files",
        nargs="+",
        required=True,
        help="一个或多个预测 parquet 文件路径（即 data.output_path）",
    )

    # HLE 专用参数
    parser.add_argument(
        "--hle_dataset",
        type=str,
        default="cais/hle",
        help="HLE 所使用的 HF 数据集名称，默认 cais/hle",
    )
    parser.add_argument(
        "--hle_judge_model",
        type=str,
        default="o3-mini-2025-01-31",
        help="用来当 judge 的 OpenAI 模型名（官方默认 o3-mini-2025-01-31）",
    )
    parser.add_argument(
        "--hle_num_workers",
        type=int,
        default=10,
        help="HLE judge 的异步并发数（受你的 rate limit 限制）",
    )

    args = parser.parse_args()
    task = args.task.lower()

    if task not in SUPPORTED_TASKS:
        print(
            f"[WARN] 未知 task '{task}'，暂时按 math-like exact match 逻辑进行评估。"
        )

    per_file_stats = []
    total_correct = 0
    total_examples = 0

    for path in args.pred_files:
        if task == "hle":
            acc, c, t = evaluate_hle_file(
                path,
                dataset_name=args.hle_dataset,
                judge_model=args.hle_judge_model,
                num_workers=args.hle_num_workers,
            )
        else:
            acc, c, t = evaluate_math_like_file(path)

        per_file_stats.append((path, acc, c, t))
        total_correct += c
        total_examples += t

    print(f"===== Task: {task} | Per-file accuracy =====")
    for path, acc, c, t in per_file_stats:
        print(f"{os.path.basename(path)}: acc={acc*100:.2f}%, correct={c}, total={t}")

    if len(per_file_stats) > 1:
        # 你之前说：data1=0.5, data2=0.6 -> 0.55，所以这里用 macro avg
        macro = sum(acc for _, acc, _, _ in per_file_stats) / len(per_file_stats)
        micro = total_correct / total_examples if total_examples > 0 else 0.0
        print("===== Summary =====")
        print(
            f"Macro avg accuracy over {len(per_file_stats)} files: {macro * 100:.2f}%"
        )
        print(
            f"Micro avg accuracy over {total_examples} questions (用各自任务的官方分母): {micro * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
