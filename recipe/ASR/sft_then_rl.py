import argparse 
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Optional, Dict, List

import wandb
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import socket

from .metrics import (
    compute_ce_loss_on_parquet_or_jsonl,
    compute_accuracy_on_parquet_or_jsonl,
)

# ========== 基础工具函数 ==========

def run_multinode_torchrun(cmd_template: str, hosts: list, nnodes: int, env=None):
    assert "{NODE_RANK}" in cmd_template
    local_hostnames = {
        "localhost", "127.0.0.1", socket.gethostname(), socket.getfqdn()
    }

    def _launch(rank, host):
        cmd = cmd_template.format(NODE_RANK=rank)
        if host in local_hostnames:
            return subprocess.run(shlex.split(cmd), env=env).returncode
        remote_cmd = f"cd {os.getcwd()} && {cmd}"
        ssh_cmd = ["ssh", host, "bash", "-lc", remote_cmd]  # 关键：加载环境
        return subprocess.run(ssh_cmd, env=env).returncode

    rets = []
    with ThreadPoolExecutor(max_workers=nnodes) as ex:
        futs = [ex.submit(_launch, r, hosts[r]) for r in range(nnodes)]
        for f in as_completed(futs):
            rets.append(f.result())

    if any(r != 0 for r in rets):
        raise RuntimeError(f"multinode torchrun failed: {rets}")

def run_distributed_ce(model_or_ckpt: str, tokenizer_id: str, args, work: Path, tag: str, hosts):
    out_file = work / f"ce_{tag}.json"
    parts = [
        "torchrun",
        f"--nnodes={args.sft_nnodes}",
        f"--nproc_per_node={args.sft_nproc_per_node}",
        "--node_rank={NODE_RANK}",
        f"--master_addr={args.sft_master_addr}",
        f"--master_port={args.sft_master_port}",
        "-m", "recipe.ASR.metrics",
        "--mode=ce",
        f"--model_or_ckpt={model_or_ckpt}",
        f"--tokenizer_id={tokenizer_id}",
        f"--file_path={args.d2_val}",
        f"--prompt_key={args.prompt_key_d2}",
        f"--response_key={args.response_key_d2}",
        f"--max_samples={args.max_eval_samples}",
        f"--batch_size={args.eval_batch_size}",
        f"--max_length={args.max_length}",
        f"--truncate_mode={args.truncate_mode}",
        f"--dtype={args.dtype}",
        f"--output={out_file}",
    ]
    cmd = " ".join(parts)
    run_multinode_torchrun(cmd, hosts, args.sft_nnodes)
    return json.load(open(out_file))["CE"]

def run_distributed_acc(model_or_ckpt: str, tokenizer_id: str, args, work: Path, tag: str, hosts):
    out_file = work / f"acc_{tag}.json"
    parts = [
        "torchrun",
        f"--nnodes={args.sft_nnodes}",
        f"--nproc_per_node={args.sft_nproc_per_node}",
        "--node_rank={NODE_RANK}",
        f"--master_addr={args.sft_master_addr}",
        f"--master_port={args.sft_master_port}",
        "-m", "recipe.ASR.metrics",
        "--mode=acc",
        f"--model_or_ckpt={model_or_ckpt}",
        f"--tokenizer_id={tokenizer_id}",
        f"--file_path={args.d2_val}",
        f"--prompt_key={args.prompt_key_d2}",
        f"--max_samples={args.max_eval_samples}",
        f"--batch_size={args.eval_batch_size}",
        f"--max_prompt_length={args.rl_max_prompt_length}",
        f"--max_new_tokens={args.rl_max_response_length}",
        f"--dtype={args.dtype}",
        f"--output={out_file}",
    ]
    cmd = " ".join(parts)
    run_multinode_torchrun(cmd, hosts, args.sft_nnodes)
    return json.load(open(out_file))["ACC"]

def run_cmd(cmd: str, env: Optional[Dict[str, str]] = None):
    print("[CMD]", cmd, flush=True)
    proc = subprocess.run(shlex.split(cmd), env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}")


def merge_fsdp_to_hf(fsdp_actor_dir: Path, target_dir: Path):
    """
    使用 verl 自带的 model_merger，将 FSDP actor ckpt 转成 HuggingFace 标准格式。

    官方文档示例：
        python -m verl.model_merger merge \\
            --backend fsdp \\
            --local_dir checkpoints/${proj}/${exp}/global_step_1/actor \\
            --target_dir /path/to/merged_hf_model

    这里 fsdp_actor_dir 应该形如: .../global_step_xxx/actor
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    cmd = (
        "python -m verl.model_merger merge "
        f"--backend fsdp "
        f"--local_dir {fsdp_actor_dir} "
        f"--target_dir {target_dir}"
    )
    run_cmd(cmd)


def _split_paths(paths_str: str) -> List[str]:
    if not paths_str:
        return []
    parts: List[str] = []
    for chunk in str(paths_str).replace(" ", ",").split(","):
        p = chunk.strip()
        if p:
            parts.append(p)
    return parts


def count_samples_in_paths(paths_str: str) -> Optional[int]:
    paths = _split_paths(paths_str)
    if not paths:
        return None

    total = 0
    for p in paths:
        if not os.path.exists(p):
            return None

        if p.endswith(".parquet"):
            try:
                import pyarrow.parquet as pq
                meta = pq.read_metadata(p)
                total += meta.num_rows
                continue
            except Exception:
                pass

        if p.endswith(".jsonl"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for _ in f:
                        total += 1
                continue
            except Exception:
                pass

        if p.endswith(".json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, list):
                    total += len(obj)
                    continue
            except Exception:
                pass

        try:
            with open(p, "r", encoding="utf-8") as f:
                for _ in f:
                    total += 1
        except Exception:
            return None

    return total


def calc_steps(num_samples: Optional[int], batch_size: int, epochs: int = 1) -> Optional[int]:
    if num_samples is None or batch_size <= 0 or epochs <= 0:
        return None
    return (num_samples * epochs + batch_size - 1) // batch_size


def list_fsdp_actor_ckpts(default_local_dir: Path) -> List[Path]:
    """
    根据官方 checkpoint 文档，FSDP backend 默认目录结构类似：
        checkpoints/${project}/${experiment}
        ├── global_steps_${i}
        │   ├── actor
        │   │   ├── huggingface
        │   │   └── dist_ckpt
        │   └── critic (PPO 才有)

    这里只需要所有 global_step(s)_*/actor 目录，并按 step 号排序。
    """
    if not default_local_dir.exists():
        return []

    cands: List[Path] = []
    for child in default_local_dir.iterdir():
        if not child.is_dir():
            continue
        name = child.name.lower()
        if "global_step" in name and (child / "actor").is_dir():
            cands.append(child / "actor")

    if not cands:
        return []

    def _extract_step(p: Path) -> int:
        # 从父目录名里抽 step 号，诸如 global_step_10 / global_steps_10
        name = p.parent.name
        digits = "".join(ch for ch in name if ch.isdigit())
        try:
            return int(digits)
        except Exception:
            return 0

    cands.sort(key=_extract_step)
    return cands


def get_tokenizer_id_for_ckpt(model_or_ckpt: str, explicit_tokenizer: Optional[str] = None) -> str:
    """
    根据 ckpt 路径自动选择合适的 tokenizer 路径：
    1. 如果用户显式传了 --tokenizer，就永远用那个。
    2. 如果存在 {model_or_ckpt}/huggingface/tokenizer_config.json，就用 huggingface 子目录。
    3. 若 ckpt 本身是 HF 目录且有 tokenizer_config.json，就直接用该目录。
    4. 否则退回用 model_or_ckpt 本身（HF hub 名称或本地 HF 目录）。
    """
    if explicit_tokenizer:
        return explicit_tokenizer

    p = Path(model_or_ckpt)
    hf_dir = p / "huggingface"
    if hf_dir.is_dir() and (hf_dir / "tokenizer_config.json").exists():
        return str(hf_dir)

    if p.is_dir() and (p / "tokenizer_config.json").exists():
        return str(p)

    return model_or_ckpt


def resolve_model_arg(model_or_ckpt: str) -> str:
    return model_or_ckpt


# ========== 根据 verl 配置构造子训练命令 ==========

def build_sft_cmd(args, ckpt_in: str, ckpt_out: Path) -> str:
    """
    使用 fsdp_sft_trainer 进行 SFT 训练。

    - LoRA 通过 model.lora_rank / model.lora_alpha / model.target_modules 控制。
    - trainer.save_freq 会根据 D1 数据集大小自动设置为“每个 epoch 保存 1 次”。
    """
    lora_rank = args.sft_lora_rank if args.sft_lora_enable == 1 else 0

    parts = ["torchrun"]
    if args.sft_nnodes == 1:
        parts += ["--standalone", f"--nproc_per_node={args.sft_nproc_per_node}"]
    else:
        parts += [
            f"--nnodes={args.sft_nnodes}",
            "--node_rank={NODE_RANK}",
            f"--nproc_per_node={args.sft_nproc_per_node}",
            f"--master_addr={args.sft_master_addr}",
            f"--master_port={args.sft_master_port}",
        ]

    # 尽量和官方 gsm8k 示例保持一致
    parts.extend([
        "-m", "verl.trainer.fsdp_sft_trainer",
        f"data.train_files={args.d1_train}",
        f"data.val_files={args.d1_val}",
        f"data.prompt_key={args.prompt_key_d1}",
        f"data.response_key={args.response_key_d1}",
        f"data.max_length={args.sft_max_length}",
        f"data.truncation={args.sft_truncation}",
        f"data.train_batch_size={args.sft_batch_size}",
        f"data.micro_batch_size_per_gpu={args.sft_micro_batch_size_per_gpu}",
        f"model.partial_pretrain={ckpt_in}",
        f"model.lora_rank={lora_rank}",
        f"model.lora_alpha={args.sft_lora_alpha}",
        "model.target_modules=all-linear" if lora_rank > 0 else "",
        f"optim.lr={args.sft_learning_rate}",
        f"optim.lr_scheduler={args.sft_lr_schedule}",
        "trainer.logger=[console,wandb]",
        "trainer.project_name=Two_SFT",
        f"trainer.experiment_name={args.sft_experiment_name}",
        f"trainer.default_local_dir={str(ckpt_out)}",
        f"trainer.total_epochs={args.sft_epochs}",
    ])

    # epoch-wise 保存 ckpt：trainer.save_freq 设为“每个 epoch 的 step 数”
    if args.sft_steps_per_epoch is not None:
        parts.append(f"trainer.save_freq={args.sft_steps_per_epoch}")

    return " ".join(str(p) for p in parts if p)


def build_rl_cmd(args, ckpt_in: str, ckpt_out: Path) -> str:
    """
    使用 main_ppo 进行 RL 训练，起点为 ckpt_in（HF 格式，已 merge 了 SFT LoRA）。
    """
    lora_rank = args.rl_lora_rank if args.rl_lora_enable == 1 else 0

    parts = [
        "python", "-m", "verl.trainer.main_ppo",
        "+strategy=fsdp2",
        f"custom_reward_function.path={args.rl_reward_fn_path}",
        f"custom_reward_function.name={args.rl_reward_fn_name}",
        f"data.train_files={args.d2_train}",
        f"data.val_files={args.d2_val}",
        f"data.prompt_key={args.prompt_key_d2}",
        f"data.max_prompt_length={args.rl_max_prompt_length}",
        f"data.max_response_length={args.rl_max_response_length}",
        f"data.train_batch_size={args.rl_batch_size}",
        f"data.train_max_samples={args.rl_train_max_samples}",
        f"data.val_max_samples={args.rl_val_max_samples}",
        f"data.truncation={args.rl_truncation}",
        f"data.image_key={args.rl_image_key}",
        f"data.filter_overlong_prompts={str(args.rl_filter_overlong_prompts)}",
        f"data.filter_overlong_prompts_workers={args.rl_filter_overlong_prompts_workers}",
        f"actor_rollout_ref.model.path={ckpt_in}",
        f"actor_rollout_ref.model.lora_rank={lora_rank}",
        f"actor_rollout_ref.model.lora_alpha={args.rl_lora_alpha}",
        "actor_rollout_ref.model.target_modules=all-linear" if lora_rank > 0 else "",
        "actor_rollout_ref.model.use_shm=True",
        "actor_rollout_ref.actor.use_kl_loss=True" if args.rl_use_kl_loss == 1 else "",
        f"actor_rollout_ref.actor.kl_loss_coef={args.rl_kl_coef}" if args.rl_use_kl_loss == 1 else "",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={args.rl_micro_batch_size_per_gpu}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={args.ppo_mini_batch_size}",
        f"actor_rollout_ref.actor.optim.lr={args.rl_learning_rate}",
        "actor_rollout_ref.actor.optim.lr_scheduler_type=constant" if not args.rl_lr_schedule else f"actor_rollout_ref.actor.optim.lr_scheduler_type={args.rl_lr_schedule}",
        "actor_rollout_ref.actor.optim.lr_warmup_steps=0",
        "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0",
        "actor_rollout_ref.rollout.load_format=safetensors",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={args.rl_rollout_gpu_memory_utilization}",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={args.rl_rollout_tensor_model_parallel_size}",
        f"actor_rollout_ref.rollout.temperature={args.rl_rollout_temperature}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={args.rollout_log_prob_micro_batch_size_per_gpu}",
        f"actor_rollout_ref.rollout.top_k={args.rl_rollout_topk}",
        f"actor_rollout_ref.rollout.top_p={args.rl_rollout_topp}",
        f"actor_rollout_ref.rollout.n={args.rl_rollout_n}",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={args.ref_log_prob_micro_batch_size_per_gpu}",
        f"algorithm.adv_estimator={args.rl_adv_estimator}",
        f"algorithm.kl_ctrl.kl_coef={args.rl_kl_coef}",
        f"trainer.total_epochs={args.rl_epochs}",
        "trainer.resume_mode=disable",
        "trainer.logger=[console,wandb]",
        f"trainer.project_name=Two_RL",
        f"trainer.experiment_name={args.rl_experiment_name}",
        f"trainer.default_local_dir={str(ckpt_out)}",
        f"trainer.n_gpus_per_node={args.rl_trainer_n_gpus_per_node}",
        f"trainer.nnodes={args.rl_trainer_nnodes}",
    ]
    # epoch-wise 保存 ckpt：trainer.save_freq 设为“每个 epoch 的 step 数”
    rl_steps_per_epoch = getattr(args, "rl_steps_per_epoch", None)
    if rl_steps_per_epoch is not None:
        parts.append(f"trainer.save_freq={rl_steps_per_epoch}")

    return " ".join(str(p) for p in parts if p)


# ========== 主逻辑：先 SFT 再 RL ==========

def main():
    parser = argparse.ArgumentParser()

    # ===== 基础参数 =====
    parser.add_argument("--base_model_or_ckpt", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--work_dir", type=str, default="/root/workspace/checkpoints")
    parser.add_argument("--sft_ckpt_dir", type=str, default=None)
    parser.add_argument("--rl_ckpt_dir", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--best_rl_hf_subdir", type=str, default=None, help="相对于 work_dir 的子目录名，用于保存最佳 RL HF 合并后的 HF 模型")
    parser.add_argument("--sft_hosts", type=str, default=None, help="逗号分隔 host 列表，长度= sft_nnodes")

    # ===== 数据集 =====
    parser.add_argument("--sft_task", type=str, choices=["DAPO_MATH", "gsm8k", "HARP", "MATH", "NuminaMath_1.5", "NuminaMath_CoT", "OpenR1_Math_220k", "openscience"], required=True)
    parser.add_argument("--rl_task", type=str, choices=["DAPO_MATH", "gsm8k", "HARP", "MATH", "NuminaMath_1.5", "NuminaMath_CoT", "OpenR1_Math_220k", "openscience"], required=True)
    parser.add_argument("--d1_train", type=str, default=None)
    parser.add_argument("--d1_val", type=str, default=None)
    parser.add_argument("--d2_train", type=str, default=None)
    parser.add_argument("--d2_val", type=str, default=None)
    parser.add_argument("--prompt_key_d1", type=str, default="question")
    parser.add_argument("--response_key_d1", type=str, default="answer")
    parser.add_argument("--prompt_key_d2", type=str, default="question")
    parser.add_argument("--response_key_d2", type=str, default="answer")

    # ===== SFT 配置 =====
    parser.add_argument("--sft_max_length", type=int, default=30000)
    parser.add_argument("--sft_truncation", type=str, default="right", choices=["error", "left", "right", "middle"])
    parser.add_argument("--sft_lora_enable", type=int, default=1)
    parser.add_argument("--sft_lora_rank", type=int, default=8)
    parser.add_argument("--sft_lora_alpha", type=int, default=16)
    parser.add_argument("--sft_batch_size", type=int, default=32)
    parser.add_argument("--sft_micro_batch_size_per_gpu", type=int, default=1)
    parser.add_argument("--sft_learning_rate", type=float, default=5e-5)
    parser.add_argument("--sft_lr_schedule", type=str, default="constant")
    parser.add_argument("--sft_epochs", type=int, default=5)
    parser.add_argument("--sft_experiment_name", type=str, default="sft_then_rl_sft")

    # 多机多卡
    parser.add_argument("--sft_nproc_per_node", type=int, default=8)
    parser.add_argument("--sft_nnodes", type=int, default=1)
    parser.add_argument("--sft_node_rank", type=int, default=0)
    parser.add_argument("--sft_master_addr", type=str, default=None)
    parser.add_argument("--sft_master_port", type=str, default="29500")

    # ===== RL 配置（main_ppo）=====
    parser.add_argument("--rl_ray_address", type=str, default=None, help="Ray dashboard or gcs address, e.g. http://head:8265")
    parser.add_argument("--rl_rollout_gpu_memory_utilization", type=float, default=0.5)
    parser.add_argument("--rl_train_max_samples", type=int, default=-1)
    parser.add_argument("--rl_val_max_samples", type=int, default=-1)
    parser.add_argument("--rl_filter_overlong_prompts", action="store_true", default=True)
    parser.add_argument("--rl_filter_overlong_prompts_workers", type=int, default=2)
    parser.add_argument("--rl_truncation", type=str, default="right", choices=["error", "left", "right", "middle"])
    parser.add_argument("--rl_image_key", type=str, default="images")

    parser.add_argument("--rl_lora_enable", type=int, default=1)
    parser.add_argument("--rl_lora_rank", type=int, default=8)
    parser.add_argument("--rl_lora_alpha", type=int, default=16)

    parser.add_argument("--rl_batch_size", type=int, default=128)
    parser.add_argument("--ppo_mini_batch_size", type=int, default=256)
    parser.add_argument("--rl_micro_batch_size_per_gpu", type=int, default=2)
    parser.add_argument("--ref_log_prob_micro_batch_size_per_gpu", type=int, default=2)
    parser.add_argument("--rollout_log_prob_micro_batch_size_per_gpu", type=int, default=2)
    parser.add_argument("--rl_learning_rate", type=float, default=5e-5)
    parser.add_argument("--rl_lr_schedule", type=str, default="constant")
    parser.add_argument("--rl_epochs", type=int, default=5)
    parser.add_argument("--rl_max_prompt_length", type=int, default=30000)
    parser.add_argument("--rl_max_response_length", type=int, default=8192)

    parser.add_argument("--rl_adv_estimator", type=str, default="grpo")
    parser.add_argument("--rl_use_kl_loss", type=int, default=1)
    parser.add_argument("--rl_kl_coef", type=float, default=0.001)

    parser.add_argument("--rl_rollout_n", type=int, default=8)
    parser.add_argument("--rl_rollout_temperature", type=float, default=1.0)
    parser.add_argument("--rl_rollout_topk", type=int, default=-1)
    parser.add_argument("--rl_rollout_topp", type=float, default=0.95)

    parser.add_argument("--rl_reward_fn_path", type=str, default="recipe/ASR/metrics.py")
    parser.add_argument("--rl_reward_fn_name", type=str, default="compute_score")

    parser.add_argument("--rl_rollout_tensor_model_parallel_size", type=int, default=1)
    parser.add_argument("--rl_trainer_nnodes", type=int, default=4)
    parser.add_argument("--rl_trainer_n_gpus_per_node", type=int, default=8)
    parser.add_argument("--rl_experiment_name", type=str, default="sft_then_rl_rl")

    # ===== 指标计算 =====
    parser.add_argument("--max_eval_samples", type=int, default=2048)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=30000)
    parser.add_argument("--truncate_mode", type=str, default="truncate", choices=["truncate", "skip"])

    # ===== wandb =====
    parser.add_argument("--wandb_project", type=str, default="Two")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])

    args = parser.parse_args()

    if args.sft_nnodes > 1:
        hosts = _split_paths(args.sft_hosts)
        if len(hosts) != args.sft_nnodes:
            raise ValueError("sft_hosts length must == sft_nnodes")
    else:
        hosts = ["localhost"]

    if args.best_rl_hf_subdir is None:
        args.best_rl_hf_subdir = f"Two_{args.sft_task}_{args.rl_task}_lora_{args.sft_lora_enable}_{args.rl_lora_enable}_best_ckpt_on_D2"

    if args.sft_ckpt_dir is None or args.rl_ckpt_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.sft_ckpt_dir = f"Two_sft_{args.sft_task}_{timestamp}"
        args.rl_ckpt_dir = f"Two_rl_{args.rl_task}_{timestamp}"
    
    args.d1_train = f"/root/workspace/ASR_data/train/{args.sft_task}.jsonl"
    args.d1_val   = f"/root/workspace/ASR_data/valid/{args.sft_task}.jsonl"
    args.d2_train = f"/root/workspace/ASR_data/train/{args.rl_task}.jsonl"
    args.d2_val   = f"/root/workspace/ASR_data/valid/{args.rl_task}.jsonl"

    # ===== 准备工作目录 =====
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    sft_ckpts_root = work / args.sft_ckpt_dir
    sft_ckpts_root.mkdir(parents=True, exist_ok=True)

    rl_ckpts_root = work / args.rl_ckpt_dir
    rl_ckpts_root.mkdir(parents=True, exist_ok=True)

    # ===== 统计数据规模，估计 epoch 步数，用于设置 save_freq =====
    d1_train_size = count_samples_in_paths(args.d1_train)
    d2_val_size = count_samples_in_paths(args.d2_val)

    sft_global_batch = args.sft_batch_size
    sft_steps_per_epoch = calc_steps(d1_train_size, sft_global_batch, epochs=1)
    args.sft_steps_per_epoch = sft_steps_per_epoch  # 挂到 args 上，供 build_sft_cmd 使用

    d2_train_size = count_samples_in_paths(args.d2_train)
    rl_global_batch = args.rl_batch_size
    rl_steps_per_epoch = calc_steps(d2_train_size, rl_global_batch, epochs=1)
    args.rl_steps_per_epoch = rl_steps_per_epoch  # 挂到 args 上，供 build_rl_cmd 使用


    def _fmt(v):
        return v if v is not None else "Unknown"

    print("=" * 60)
    print("[SFT-then-RL Controller] Training configuration summary")
    print(f"  D1 train examples: {_fmt(d1_train_size)}")
    print(f"  D2 train examples: {_fmt(d2_train_size)}")
    print(f"  D2 val   examples: {_fmt(d2_val_size)}")
    print()
    print("  SFT:")
    print(f"    epochs: {args.sft_epochs}")
    print(f"    global batch size: {sft_global_batch}")
    print(f"    estimated steps per epoch: {_fmt(sft_steps_per_epoch)}")
    print()
    print("  RL:")
    print(f"    epochs: {args.rl_epochs}")
    print(f"    global batch size: {rl_global_batch}")
    print(f"    estimated steps per epoch: {_fmt(rl_steps_per_epoch)}")
    print("=" * 60, flush=True)


    # ===== wandb init =====
    if args.wandb_project and args.wandb_mode != "disabled":
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"SFT_then_RL_{int(time.time())}",
            mode=args.wandb_mode,
            config={
                "schedule": "sft_then_rl",
                "sft_epochs": args.sft_epochs,
                "rl_epochs": args.rl_epochs,
            },
        )
    else:
        wandb.init(mode="disabled")

    start_time = time.time()

    base_model = resolve_model_arg(args.base_model_or_ckpt)

    # ===== 阶段 1：SFT 训练，保存 epoch-wise ckpt =====
    print("[Stage 1] SFT training starts...", flush=True)
    sft_cmd = build_sft_cmd(...)
    if args.sft_nnodes > 1:
        run_multinode_torchrun(sft_cmd, hosts, args.sft_nnodes)
    else:
        run_cmd(sft_cmd)
    print("[Stage 1] SFT training finished.", flush=True)

    # ===== 阶段 2：遍历所有 SFT ckpt，在 D2 val 上计算 CE，一次性选出 generalization loss 最低的 =====
    print("[Stage 2] Selecting best SFT checkpoint on D2 (generalization CE loss)...", flush=True)

    actor_ckpts = list_fsdp_actor_ckpts(sft_ckpts_root)
    if not actor_ckpts:
        raise RuntimeError(f"No FSDP actor checkpoints found under {sft_ckpts_root}")

    tmp_hf_dir = work / "tmp_sft_hf"
    best_sft_hf_dir = work / "best_sft_on_D2"
    best_ce = None
    best_step = None

    for actor_dir in actor_ckpts:
        step_name = actor_dir.parent.name
        print(f"[Stage 2] Evaluating SFT checkpoint: {actor_dir} (parent={step_name})", flush=True)

        if tmp_hf_dir.exists():
            shutil.rmtree(tmp_hf_dir)

        # 将当前 FSDP actor ckpt merge 成 HF 格式（LoRA 会 merge 进 base）
        merge_fsdp_to_hf(actor_dir, tmp_hf_dir)

        tok_id = get_tokenizer_id_for_ckpt(str(tmp_hf_dir), args.tokenizer)
        ce = run_distributed_ce(
            model_or_ckpt=str(tmp_hf_dir),
            tokenizer_id=tok_id,
            args=args,
            work=work,
            tag=step_name,
            hosts=hosts,
        )
        print(f"[Stage 2]  {step_name}  CE (D2 val) = {ce:.6f}", flush=True)
        wandb.log({
            "sft_ckpt_step_name": step_name,
            "sft_ce_on_D2": ce,
        })

        if (best_ce is None) or (ce < best_ce):
            best_ce = ce
            best_step = step_name
            if best_sft_hf_dir.exists():
                shutil.rmtree(best_sft_hf_dir)
            shutil.copytree(tmp_hf_dir, best_sft_hf_dir)
            print(f"[Stage 2]  New best SFT ckpt: {best_step}  (CE={best_ce:.6f})", flush=True)

    print(f"[Stage 2] Best SFT checkpoint on D2: {best_step}, CE={best_ce:.6f}", flush=True)
    wandb.log({
        "sft_best_step": best_step,
        "sft_best_ce_on_D2": best_ce,
    })

    # best_sft_hf_dir 即后续 RL 的起点
    rl_init_ckpt = str(best_sft_hf_dir)

    # ===== 阶段 3：基于最佳 SFT ckpt 进行 RL 训练 =====
    print("[Stage 3] RL training starts from best SFT checkpoint...", flush=True)
    rl_cmd = build_rl_cmd(args, rl_init_ckpt, rl_ckpts_root)
    env = os.environ.copy()
    if args.rl_ray_address:
        env["RAY_ADDRESS"] = args.rl_ray_address
    run_cmd(rl_cmd, env=env)
    print("[Stage 3] RL training finished.", flush=True)

    # ===== 阶段 4：遍历所有 RL ckpt，在 D2 val 上计算 accuracy，选出最优并只保留它 =====
    print("[Stage 4] Selecting best RL checkpoint on D2 (accuracy)...", flush=True)

    rl_actor_ckpts = list_fsdp_actor_ckpts(rl_ckpts_root)
    if not rl_actor_ckpts:
        raise RuntimeError(f"No FSDP actor checkpoints found under {rl_ckpts_root}")

    tmp_rl_hf_dir = work / "tmp_rl_hf"
    best_rl_hf_dir = work / "best_rl_on_D2"
    best_acc = None
    best_rl_step = None
    best_rl_fsdp_dir = None

    for actor_dir in rl_actor_ckpts:
        step_name = actor_dir.parent.name
        print(f"[Stage 4] Evaluating RL checkpoint: {actor_dir} (parent={step_name})", flush=True)

        # 清空临时 HF 目录
        if tmp_rl_hf_dir.exists():
            shutil.rmtree(tmp_rl_hf_dir)

        # 将当前 FSDP actor ckpt merge 成 HF 格式
        merge_fsdp_to_hf(actor_dir, tmp_rl_hf_dir)

        tok_id = get_tokenizer_id_for_ckpt(str(tmp_rl_hf_dir), args.tokenizer)
        acc = run_distributed_acc(
            model_or_ckpt=str(tmp_rl_hf_dir),
            tokenizer_id=tok_id,
            args=args,
            work=work,
            tag=step_name,
            hosts=hosts,
        )

        print(f"[Stage 4]  {step_name}  ACC (D2 val) = {acc:.6f}", flush=True)
        wandb.log({
            "rl_ckpt_step_name": step_name,
            "rl_acc_on_D2": acc,
        })

        # 维护当前最优
        if (best_acc is None) or (acc > best_acc):
            best_acc = acc
            best_rl_step = step_name
            best_rl_fsdp_dir = actor_dir.parent

            if best_rl_hf_dir.exists():
                shutil.rmtree(best_rl_hf_dir)
            shutil.copytree(tmp_rl_hf_dir, best_rl_hf_dir)

            print(f"[Stage 4]  New best RL ckpt: {best_rl_step}  (ACC={best_acc:.6f})", flush=True)

    # 清理临时 HF 目录
    if tmp_rl_hf_dir.exists():
        shutil.rmtree(tmp_rl_hf_dir)

    # 只保留最优的 RL FSDP ckpt，删掉其它 global_step_xxx 目录
    for actor_dir in rl_actor_ckpts:
        fsdp_dir = actor_dir.parent
        if best_rl_fsdp_dir is None or fsdp_dir == best_rl_fsdp_dir:
            continue
        print(f"[Stage 4] Removing non-best RL ckpt: {fsdp_dir}", flush=True)
        shutil.rmtree(fsdp_dir)

    print(f"[Stage 4] Best RL checkpoint on D2: {best_rl_step}, ACC={best_acc:.6f}", flush=True)
    wandb.log({
        "rl_best_step": best_rl_step,
        "rl_best_acc_on_D2": best_acc,
    })

    # ===== 总结 =====
    total_time = time.time() - start_time
    wandb.log({
        "total_wallclock_time_s": total_time,
        "total_wallclock_time_h": total_time / 3600.0,
    })
    wandb.finish()

    print(f"[Done] Total wallclock: {total_time / 3600.0:.4f} h")
    print(f"[Done] Best SFT checkpoint on D2 (HF merged copy): {best_sft_hf_dir}")
    print(f"[Done] Best RL checkpoint on D2 (HF merged copy): {best_rl_hf_dir}")


if __name__ == "__main__":
    main()
