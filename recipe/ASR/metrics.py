# -*- coding: utf-8 -*-
"""
为 D1 验证集计算监督 CE 损失 G_n
为 D2 验证集计算策略熵 H_n（平均 token 熵）
- 采用 teacher-forcing：将 prompt+response 拼接，熵/损失仅在 response 段统计
- 兼容 parquet/jsonl；需 datasets 支持
"""
import re
from typing import Optional
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

import os
from typing import Tuple

def _resolve_model_and_tokenizer_paths(model_or_ckpt: str, tokenizer_id: Optional[str] = None):
    # 如果用户显式指定 tokenizer，就用用户的
    if tokenizer_id:
        tok_path = tokenizer_id
    else:
        # 优先看当前目录本身是否是 HF 模型
        if os.path.exists(os.path.join(model_or_ckpt, "config.json")):
            tok_path = model_or_ckpt
        else:
            tok_path = model_or_ckpt  # hub name 等

    model_path = model_or_ckpt
    return model_path, tok_path


def _to_dtype(dtype: str):
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError(dtype)


def _prepare_ds(
    file_path: str,
    prompt_key: str,
    response_key: str,
    tokenizer: AutoTokenizer,
    max_samples: int = 2048,
    max_length: int = 4096,
    truncate_mode: str = "truncate",  # 可选 "truncate" 或 "skip"
):
    ext = file_path.split(".")[-1].lower()
    if ext in ["parquet"]:
        ds = load_dataset("parquet", data_files=file_path, split="train")
    elif ext in ["jsonl", "json"]:
        ds = load_dataset("json", data_files=file_path, split="train")
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    def _fmt(ex):
        prompt = ex[prompt_key]
        resp = ex[response_key]
        text = f"{prompt}\n{resp}"
        return {"text": text, "prompt_len": len(tokenizer(prompt)["input_ids"])}

    ds = ds.map(_fmt, remove_columns=ds.column_names)

    def _tok(ex):
        # 手动分步 tokenize 以便判断长度
        prompt_ids = tokenizer(ex["text"].split("\n")[0], truncation=False)["input_ids"]
        resp_ids = tokenizer(
            ex["text"].split("\n")[1] if "\n" in ex["text"] else "",
            truncation=False,
        )["input_ids"]
        full_ids = prompt_ids + resp_ids

        if len(full_ids) > max_length:
            if truncate_mode == "skip":
                return {
                    "input_ids": [],
                    "attention_mask": [],
                    "labels": [],
                    "valid": 0,
                }
            else:  # 默认 truncate
                full_ids = full_ids[:max_length]

        attn = [1] * len(full_ids)
        labels = [-100] * len(full_ids)
        start = len(prompt_ids)
        for i in range(start, len(full_ids)):
            labels[i] = full_ids[i]

        return {
            "input_ids": full_ids,
            "attention_mask": attn,
            "labels": labels,
            "valid": 1,
        }

    ds = ds.map(_tok, remove_columns=ds.column_names)
    ds = ds.filter(lambda ex: ex["valid"] == 1)
    ds = ds.remove_columns("valid")

    return ds


def _dataloader(ds, batch_size: int, tokenizer):
    def _collate(batch):
        maxlen = max(len(x["input_ids"]) for x in batch)
        pad_id = tokenizer.pad_token_id or 0

        def _pad(seq, pad):
            return seq + [pad] * (maxlen - len(seq))

        input_ids = torch.tensor(
            [_pad(x["input_ids"], pad_id) for x in batch],
            dtype=torch.long,
        )
        attention = torch.tensor(
            [_pad(x["attention_mask"], 0) for x in batch],
            dtype=torch.long,
        )
        labels = torch.tensor(
            [_pad(x["labels"], -100) for x in batch],
            dtype=torch.long,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "labels": labels,
        }

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=_collate)


# ====== RL 用的奖励函数工具 ======


def _extract_boxed_answer(text: str) -> Optional[str]:
    """
    从模型输出中提取最后一个 \\boxed{...} 的内容。
    支持内部嵌套大括号，例如 \\boxed{\\frac{1}{2}}。
    找不到就返回 None。
    """
    if not isinstance(text, str):
        return None

    s = text
    boxed_contents = []
    i = 0

    # 逐个查找 \boxed，并做手动括号匹配
    while True:
        idx = s.find(r"\boxed", i)
        if idx == -1:
            break

        j = idx + len(r"\boxed")
        # 跳过空白，找到紧随其后的 '{'
        while j < len(s) and s[j].isspace():
            j += 1
        if j >= len(s) or s[j] != "{":
            i = idx + len(r"\boxed")
            continue

        depth = 0
        k = j
        while k < len(s):
            ch = s[k]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    # 取出最外层大括号内部的内容
                    boxed_contents.append(s[j + 1 : k])
                    i = k + 1
                    break
            k += 1
        else:
            # 括号不平衡，跳过这一处
            i = idx + len(r"\boxed")
            continue

    if not boxed_contents:
        return None

    ans = boxed_contents[-1].strip()  # 通常取最后一个 boxed
    # 去掉简单尾缀（如句号）
    ans = ans.strip().rstrip(".").strip()
    return ans or None


def _normalize_answer(s: str) -> str:
    """
    轻度归一化：
    - 转小写
    - 去掉首尾空白
    - 折叠中间多余空格

    SciQ 用这个做 exact match。
    """
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().lower()
    return " ".join(s.split())


# ====== MATH 官方 evaluator 逻辑（内联版） ======


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except Exception:
        return string


def _remove_right_units(string):
    # "\\text{ " 只在右侧单位里出现
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{."
    # Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc.
    # Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset,
    # but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def is_equiv(str1, str2, verbose=False):
    """
    MATH 官方等价判断函数：
    先用 _strip_string 规整，再做精确 string 比较。
    """
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


@torch.no_grad()
def compute_policy_entropy_on_parquet_or_jsonl(
    model_or_ckpt: str,
    tokenizer_id: Optional[str],
    file_path: str,
    prompt_key: str,
    response_key: str,
    device: str = "cuda",
    dtype: str = "bfloat16",
    max_samples: int = 2048,
    batch_size: int = 4,
    max_length: int = 4096,
    truncate_mode: str = "truncate",
) -> float:
    model_path, tok_path = _resolve_model_and_tokenizer_paths(model_or_ckpt, tokenizer_id)

    tok = AutoTokenizer.from_pretrained(
        tok_path,
        use_fast=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=_to_dtype(dtype),
    )
    model.to(device)
    model.eval()

    ds = _prepare_ds(
        file_path,
        prompt_key,
        response_key,
        tok,
        max_samples=max_samples,
        max_length=max_length,
        truncate_mode=truncate_mode,
    )
    loader = _dataloader(ds, batch_size, tok)

    import torch.nn.functional as F  # noqa: F401  # 保持原样，虽然这里未直接使用
    total_ent = 0.0
    total_tok = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attn,
            use_cache=False,
        )
        # 对齐到下一 token：labels 是下一 token 的“真值”
        logits = outputs.logits  # [B, T, V]

        # 仅在 labels != -100 的位置计算熵
        valid_mask = (labels != -100).float()  # [B, T]
        # 分布 p = softmax(logits)
        probs = torch.softmax(logits, dim=-1)
        # token 熵：-sum p log p
        ent = -(probs * torch.clamp(probs.log(), min=-1e9)).sum(dim=-1)  # [B, T]
        ent = ent * valid_mask
        total_ent += ent.sum().item()
        total_tok += valid_mask.sum().item()

    return total_ent / max(total_tok, 1)


@torch.no_grad()
def compute_ce_loss_on_parquet_or_jsonl(
    model_or_ckpt: str,
    tokenizer_id: Optional[str],
    file_path: str,
    prompt_key: str,
    response_key: str,
    device: str = "cuda",
    dtype: str = "bfloat16",
    max_samples: int = 2048,
    batch_size: int = 4,
    max_length: int = 4096,
    truncate_mode: str = "truncate",
) -> float:
    model_path, tok_path = _resolve_model_and_tokenizer_paths(model_or_ckpt, tokenizer_id)

    tok = AutoTokenizer.from_pretrained(
        tok_path,
        use_fast=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=_to_dtype(dtype),
    )
    model.to(device)
    model.eval()

    ds = _prepare_ds(
        file_path,
        prompt_key,
        response_key,
        tok,
        max_samples=max_samples,
        max_length=max_length,
        truncate_mode=truncate_mode,
    )
    loader = _dataloader(ds, batch_size, tok)

    import torch.nn.functional as F  # noqa: F401
    total_nll = 0.0
    total_tok = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attn,
            use_cache=False,
        )
        logits = outputs.logits  # [B, T, V]
        # 交叉熵只在 labels!=-100 的 token 上计算
        flat_labels = labels.view(-1)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            flat_labels,
            ignore_index=-100,
            reduction="none",
        ).view_as(labels)
        mask = (labels != -100).float()
        total_nll += (loss * mask).sum().item()
        total_tok += mask.sum().item()

    return total_nll / max(total_tok, 1)


# ========== RL 自定义奖励函数：1 正确 / 0 错误 ==========


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth,
    extra_info=None,
    **kwargs,
) -> float:
    """
    自定义奖励函数（给 Verl 的 custom_reward_function 用）：

    - 从 solution_str 中抽取最后一个 \\boxed{...} 的内容
    - 数据集字段：
        * question
        * answer：full CoT solution（中间推理）
        * groundtruth：最终答案（只用这个来对比）
    - dataset 逻辑：
        * 当 dataset == "MATH" 时，用官方 evaluator (is_equiv) 判断是否等价
        * 当 dataset == "SciQ"（或其它非 MATH）时，用 _normalize_answer 做轻度归一化后 exact match
    - 完全相等/等价 -> 奖励 1.0
    - 否则 -> 奖励 0.0
    - 抽不出 boxed 直接 0.0
    """

    # dataset 可以从 data_source 或 kwargs["dataset"] 传入
    dataset = None
    if data_source is not None:
        dataset = str(data_source)
    if "dataset" in kwargs and kwargs["dataset"] is not None:
        dataset = str(kwargs["dataset"])
    dataset_norm = dataset.lower() if isinstance(dataset, str) else ""

    # ground_truth 可能是 dict（来自数据集整行）：
    # question, answer (full CoT), groundtruth (final answer)
    if isinstance(ground_truth, dict):
        if "groundtruth" in ground_truth:
            ground_truth = ground_truth["groundtruth"]
        elif "ground_truth" in ground_truth:
            ground_truth = ground_truth["ground_truth"]
        elif "answer" in ground_truth:
            # 不推荐，但如果没有 groundtruth 也兼容一下
            ground_truth = ground_truth["answer"]
        elif "label" in ground_truth:
            ground_truth = ground_truth["label"]

    # 没有正确答案信息，那就保守给 0
    if ground_truth is None:
        return 0.0

    boxed = _extract_boxed_answer(solution_str)
    if boxed is None:
        # 没有 boxed，按你要求直接算错
        return 0.0

    # ============ MATH：用官方等价判断 ============ #
    if dataset_norm == "math":
        pred_raw = boxed
        gt_raw = str(ground_truth)

        try:
            correct = is_equiv(pred_raw, gt_raw, verbose=False)
        except Exception:
            # 极端情况下 fallback 一下，尽量保持稳健
            try:
                correct = _strip_string(pred_raw) == _strip_string(gt_raw)
            except Exception:
                correct = (pred_raw == gt_raw)

        return 1.0 if correct else 0.0

    # ============ SciQ（以及其它默认） ：轻量 normalization + exact match ============ #
    pred = _normalize_answer(boxed)
    gt = _normalize_answer(str(ground_truth))

    return 1.0 if pred == gt else 0.0


@torch.no_grad()
def compute_accuracy_on_parquet_or_jsonl(
    model_or_ckpt: str,
    tokenizer_id: Optional[str],
    file_path: str,
    prompt_key: str,
    device: str = "cuda",
    dtype: str = "bfloat16",
    max_samples: int = 2048,
    batch_size: int = 4,
    max_prompt_length: int = 512,
    max_new_tokens: int = 512,
    data_source_key: Optional[str] = None,
) -> float:
    """
    在 D2 验证集上做“生成 + compute_score”来估计 accuracy：
      - prompt 来自 `prompt_key`（默认 question）
      - ground_truth 直接把整行样本 dict 丢给 compute_score，里面有 groundtruth/answer 等字段
      - data_source_key / 'dataset' 字段（如果有）会传给 compute_score 用来区分 MATH / SciQ 等

    返回值：平均 accuracy \in [0,1]
    """
    from datasets import load_dataset

    model_path, tok_path = _resolve_model_and_tokenizer_paths(model_or_ckpt, tokenizer_id)

    tok = AutoTokenizer.from_pretrained(
        tok_path,
        use_fast=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=_to_dtype(dtype),
    )
    model.to(device)
    model.eval()

    ext = file_path.split(".")[-1].lower()
    if ext in ["parquet"]:
        ds = load_dataset("parquet", data_files=file_path, split="train")
    elif ext in ["jsonl", "json"]:
        ds = load_dataset("json", data_files=file_path, split="train")
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    if max_samples and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    total = len(ds)
    if total == 0:
        return 0.0

    total_correct = 0

    # 简单按 batch 做 greedy decoding
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = [ds[i] for i in range(start, end)]
        prompts = [ex[prompt_key] for ex in batch]

        inputs = tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy decoding
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )

        # 去掉 prompt 部分，只保留新生成的 tokens
        gen_ids = outputs[:, inputs["input_ids"].shape[1]:]
        texts = tok.batch_decode(gen_ids, skip_special_tokens=True)

        from math import isfinite  # 只是保证 score 是正常数值时才计数

        for ex, pred in zip(batch, texts):
            if data_source_key and data_source_key in ex:
                dataset_name = ex[data_source_key]
            elif "dataset" in ex:
                dataset_name = ex["dataset"]
            else:
                dataset_name = None

            score = compute_score(
                data_source=dataset_name,
                solution_str=pred,
                ground_truth=ex,
                dataset=dataset_name,
            )
            # compute_score 本来就是 1 正确 / 0 错误，这里只做一个稳健判断
            if isinstance(score, (int, float)) and isfinite(score) and score >= 0.5:
                total_correct += 1

    return float(total_correct) / float(total)
