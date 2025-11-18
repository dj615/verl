import argparse 
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Optional, Dict

import wandb
import shutil
from transformers import AutoConfig

from .metrics import compute_policy_entropy_on_parquet_or_jsonl, compute_ce_loss_on_parquet_or_jsonl, compute_accuracy_on_parquet_or_jsonl

# ========== 基础工具函数 ==========

def merge_fsdp_to_hf(fsdp_ckpt_dir: Path, target_dir: Path, hf_model_config_path: Optional[str] = None):
    """
    调用 verl 自带的 model_merger，将 FSDP ckpt 转成 HuggingFace 标准格式。

    - 对 SFT ckpt：local_dir 里一般自带 huggingface/config.json，可以不传 hf_model_config_path。
    - 对 RL ckpt：通常没有 huggingface/config.json，此时必须显式指定 hf_model_config_path，
      比如初始的 base_model_or_ckpt（如 'Qwen/Qwen3-0.6B' 或某个本地 HF 目录）。
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    cmd = (
        "python -m verl.model_merger merge "
        f"--backend fsdp "
        f"--local_dir {fsdp_ckpt_dir}/actor "
        f"--target_dir {target_dir}"
    )
    run_cmd(cmd)

def _split_paths(paths_str: str):
    # 兼容逗号分隔或空格分隔（如果你未来想传多个文件）
    if not paths_str:
        return []
    parts = []
    for chunk in str(paths_str).replace(" ", ",").split(","):
        p = chunk.strip()
        if p:
            parts.append(p)
    return parts


def count_samples_in_paths(paths_str: str) -> Optional[int]:
    """
    粗略统计数据条数：
      - .parquet: 用 pyarrow.parquet 读 metadata
      - .jsonl: 按行数统计
      - .json: 若为 list，取 len
      - 其它：按行数兜底
    任何一个文件不存在 / 失败则返回 None（不影响主训练）
    """
    paths = _split_paths(paths_str)
    if not paths:
        return None

    total = 0
    for p in paths:
        if not os.path.exists(p):
            return None

        # parquet
        if p.endswith(".parquet"):
            try:
                import pyarrow.parquet as pq
                meta = pq.read_metadata(p)
                total += meta.num_rows
                continue
            except Exception:
                # 失败就走后面的兜底逻辑
                pass

        # jsonl
        if p.endswith(".jsonl"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    for _ in f:
                        total += 1
                continue
            except Exception:
                pass

        # json
        if p.endswith(".json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, list):
                    total += len(obj)
                    continue
            except Exception:
                pass

        # 其它：按行数估计
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


def run_cmd(cmd: str, env: Optional[Dict[str, str]] = None):
    print("[CMD]", cmd, flush=True)
    proc = subprocess.run(shlex.split(cmd), env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}")


def latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """
    在给定 ckpt_dir 下，尽量通用地找“最新的 checkpoint 目录”。

    策略：
      - 递归遍历子目录，按 mtime 逆序排序
      - 优先选择名字包含 ["global_step","epoch","step","actor","checkpoint"] 的目录
      - 否则返回最近修改的目录
      - 没找到则返回 None
    """
    if not ckpt_dir.exists():
        return None

    cands = sorted(
        [p for p in ckpt_dir.glob("**/*") if p.is_dir() ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        return None

    for p in cands:
        name = p.name.lower()
        if any(k in name for k in ["global_step", "epoch", "step", "actor", "checkpoint"]):
            return p

    return cands[0]


def resolve_model_arg(model_or_ckpt: str) -> str:
    return model_or_ckpt


# ========== 根据 verl 配置构造子训练命令 ==========

def build_sft_cmd(args, ckpt_in: str, ckpt_out: Path, phase: int) -> str:
    lora_rank = args.sft_lora_rank if args.sft_lora_enable == 1 else 0

    parts = ["torchrun"]
    if args.sft_nnodes == 1:
        parts += ["--standalone", f"--nproc_per_node={args.sft_nproc_per_node}"]
    else:
        parts += [
            f"--nnodes={args.sft_nnodes}",
            f"--node_rank={args.sft_node_rank}",
            f"--nproc_per_node={args.sft_nproc_per_node}",
            f"--master_addr={args.sft_master_addr}",
            f"--master_port={args.sft_master_port}",
        ]

    if getattr(args, "validation_strategy", None) == "steps":
        trainer_step_cfg = f"trainer.total_training_steps={args.validation_steps}"
    elif getattr(args, "validation_strategy", None) == "epochs":
        trainer_step_cfg = f"trainer.total_epochs={args.sft_epochs}"
    else:
        raise ValueError("Only supports args.validation_strategy as epochs or steps.")

    parts.extend([
        "-m", "verl.trainer.fsdp_sft_trainer",
        f"data.train_batch_size={args.sft_batch_size}",
        f"data.micro_batch_size_per_gpu={args.sft_micro_batch_size_per_gpu}",
        f"data.train_files={args.d1_train}",
        f"data.val_files={args.d1_val}",
        f"data.prompt_key={args.prompt_key_d1}",
        f"data.response_key={args.response_key_d1}",
        f"data.max_length={args.sft_max_length}",
        f"data.truncation={args.sft_truncation}",
        f"model.partial_pretrain={ckpt_in}",
        f"model.lora_rank={lora_rank}",
        f"model.lora_alpha={args.sft_lora_alpha}",
        f"optim.lr={args.sft_learning_rate}",
        "optim.lr_scheduler=constant" if not args.sft_lr_schedule else f"optim.lr_scheduler={args.sft_lr_schedule}",
        "optim.lr_warmup_steps_ratio=0.0",
        trainer_step_cfg,
        "trainer.resume_mode=disable",
        "trainer.logger=[console,wandb]",
        "trainer.project_name=ASR_SFT",
        f"trainer.experiment_name=phase_{phase}",
        f"trainer.default_local_dir={str(ckpt_out)}",
        f"trainer.log_freq={args.log_every_n_steps}",
    ])
    return " ".join(str(p) for p in parts if p)

def run_distributed_eval(model_or_ckpt: str, tokenizer_id: str, args, work: Path, phase: int):
    """
    用 torchrun 多机多卡跑一次 eval（policy entropy + accuracy），
    结果写到 work/eval_phase_{phase}.json 再读回来。
    """
    out_file = work / f"eval_phase_{phase}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    parts = [
        "torchrun",
        f"--nnodes={args.rl_trainer_nnodes}",
        f"--nproc_per_node={args.rl_trainer_n_gpus_per_node}",
        f"--node_rank={args.sft_node_rank}",
        f"--master_addr={args.sft_master_addr}",
        f"--master_port={args.sft_master_port}",
        "-m", "recipe.ASR.metrics",
        f"--model_or_ckpt={model_or_ckpt}",
        f"--tokenizer_id={tokenizer_id}",
        f"--file_path={args.d2_val}",
        f"--prompt_key={args.prompt_key_d2}",
        f"--response_key={args.response_key_d2}",
        f"--max_samples={args.max_eval_samples}",
        f"--batch_size={args.eval_batch_size}",
        f"--max_length={args.max_length}",
        f"--truncate_mode={args.truncate_mode}",
        f"--max_prompt_length={args.rl_max_prompt_length}",
        f"--max_new_tokens={args.rl_max_response_length}",
        f"--dtype={args.dtype}",
        f"--output={out_file}",
    ]
    cmd = " ".join(str(p) for p in parts if p)
    run_cmd(cmd)

    with open(out_file, "r", encoding="utf-8") as f:
        j = json.load(f)
    return j["H"], j["P"]

def build_rl_cmd(args, ckpt_in: str, ckpt_out: Path, phase: int) -> str:
    lora_rank = args.rl_lora_rank if args.rl_lora_enable == 1 else 0

    if getattr(args, "validation_strategy", None) == "steps":
        trainer_step_cfg = f"trainer.total_training_steps={args.validation_steps}"
    elif getattr(args, "validation_strategy", None) == "epochs":
        trainer_step_cfg = f"trainer.total_epochs={args.rl_epochs}"
    else:
        raise ValueError("Only supports args.validation_strategy as epochs or steps.")

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
        "actor_rollout_ref.model.target_modules=all-linear",
        "actor_rollout_ref.model.use_shm=True",
        "actor_rollout_ref.actor.use_kl_loss=True" if args.rl_use_kl_loss == 1 else "",
        f"actor_rollout_ref.actor.kl_loss_coef={args.rl_kl_coef}" if args.rl_use_kl_loss == 1 else "",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={args.rl_micro_batch_size_per_gpu}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={args.ppo_mini_batch_size}",
        "actor_rollout_ref.actor.ppo_epochs=1"
        f"actor_rollout_ref.actor.optim.lr={args.rl_learning_rate}",
        "actor_rollout_ref.actor.optim.lr_scheduler_type=constant" if not args.rl_lr_schedule else f"actor_rollout_ref.actor.optim.lr_scheduler_type={args.rl_lr_schedule}",
        "actor_rollout_ref.actor.optim.lr_warmup_steps=0",
        "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0",
        "actor_rollout_ref.rollout.load_format=safetensors",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={args.rl_rollout_gpu_memory_utilization}",
        f"actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={args.rl_rollout_tensor_model_parallel_size}",
        f"actor_rollout_ref.rollout.temperature={args.rl_rollout_temperature}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={args.rollout_log_prob_micro_batch_size_per_gpu}",
        f"actor_rollout_ref.rollout.top_k={args.rl_rollout_topk}",
        f"actor_rollout_ref.rollout.top_p={args.rl_rollout_topp}",
        f"actor_rollout_ref.rollout.n={args.rl_rollout_n}",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={args.ref_log_prob_micro_batch_size_per_gpu}",
        f"algorithm.adv_estimator={args.rl_adv_estimator}",
        f"algorithm.kl_ctrl.kl_coef={args.rl_kl_coef}",
        trainer_step_cfg,
        "trainer.resume_mode=disable",
        "trainer.logger=[console,wandb]",
        f"trainer.project_name=ASR_RL",
        f"trainer.experiment_name=phase_{phase}",
        f"trainer.default_local_dir={str(ckpt_out)}",
        f"trainer.n_gpus_per_node={args.rl_trainer_n_gpus_per_node}",
        f"trainer.nnodes={args.rl_trainer_nnodes}",
        f"trainer.save_freq={args.validation_steps}",
        f"trainer.log_freq={args.log_every_n_steps}",
    ]
    return " ".join(str(p) for p in parts if p)


# ========== 主逻辑 ==========

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
    parser.add_argument("--schedule_mode", type=str, default="ASR", choices=["ASR"])
    parser.add_argument("--log_every_n_steps", type=int, default=5, help="每 N 次梯度更新后实时 log 一次到 wandb")
    parser.add_argument("--early_patience", type=int, default=5, help="连续 early_patience 次 Pn 下降则提前停止训练（基于 D2 validation 的 Pn）。")
    parser.add_argument("--validation_strategy", type=str, default="steps", choices=["epochs", "steps"])
    parser.add_argument("--max_training_steps", type=int, default=2000)
    parser.add_argument("--validation_steps", type=int, default=100, help="step-wise 验证间隔（单位：梯度更新步数）")
    parser.add_argument("--max_phases", type=int, default=10)
    parser.add_argument("--sft_epochs", type=int, default=1)
    parser.add_argument("--rl_epochs", type=int, default=1)

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

    # ===== SFT 配置（映射到 sft_trainer.yaml）=====
    # Data Process
    parser.add_argument("--sft_max_length", type=int, default=40960)
    parser.add_argument("--sft_truncation", type=str, default="right", choices=["error", "left", "right", "middle"], help="SFT 超过最大长度的样本处理方式（默认右截断）")
    # LoRA
    parser.add_argument("--sft_lora_enable", type=int, default=1)
    parser.add_argument("--sft_lora_rank", type=int, default=8)
    parser.add_argument("--sft_lora_alpha", type=int, default=16)
    # Batch / LR / Epoch
    parser.add_argument("--sft_batch_size", type=int, default=32)
    parser.add_argument("--sft_micro_batch_size_per_gpu", type=int, default=1)
    parser.add_argument("--sft_learning_rate", type=float, default=5e-6)
    parser.add_argument("--sft_lr_schedule", type=str, default="constant")
    # 多机多卡
    parser.add_argument("--sft_nproc_per_node", type=int, default=8)
    parser.add_argument("--sft_nnodes", type=int, default=4, help="一共几台机器")
    parser.add_argument("--sft_node_rank", type=int, default=0, help="当前机器的序号（从 0 开始）")
    parser.add_argument("--sft_master_addr", type=str, default=None, help="告诉所有节点主节点的 IP 地址")
    parser.add_argument("--sft_master_port", type=str, default="29500", help="各节点通过此端口通信，未被占用即可")

    # ===== RL 配置（映射到 ppo_trainer.yaml + main_ppo）=====
    parser.add_argument("--rl_rollout_gpu_memory_utilization", type=float, default=0.5, help="映射到 actor_rollout_ref.rollout.gpu_memory_utilization。对于 vLLM：表示 vLLM 实例使用的 GPU 显存占比。如果启动时提示显存不足，就把这个值再调小一些。")
    # Data Process
    parser.add_argument("--rl_train_max_samples", type=int, default=-1, help="RL 阶段最大训练样本数，-1 表示使用全部数据")
    parser.add_argument("--rl_val_max_samples", type=int, default=-1, help="RL 阶段最大验证样本数，-1 表示使用全部数据")
    parser.add_argument("--rl_max_prompt_length", type=int, default=40960)
    parser.add_argument("--rl_max_response_length", type=int, default=8192)
    parser.add_argument("--rl_filter_overlong_prompts", action="store_true", default=True, help="RL 是否过滤过长 prompt 样本（默认开启）")
    parser.add_argument("--rl_filter_overlong_prompts_workers", type=int, default=2, help="RL 过滤过长样本时的并行 worker 数量")
    parser.add_argument("--rl_truncation", type=str, default="right", choices=["error", "left", "right", "middle"], help="RL 超过最大长度的样本处理方式（默认右截断）")
    parser.add_argument("--rl_image_key", type=str, default="images", help="RL 多模态输入中图像字段名（若有）")
    # LoRA
    parser.add_argument("--rl_lora_enable", type=int, default=1)
    parser.add_argument("--rl_lora_rank", type[int], default=8)
    parser.add_argument("--rl_lora_alpha", type=int, default=16)
    # Batch / LR / Epoch
    parser.add_argument("--rl_batch_size", type=int, default=128)
    parser.add_argument("--ppo_mini_batch_size", type=int, default=32)
    parser.add_argument("--rl_micro_batch_size_per_gpu", type=int, default=2)
    parser.add_argument("--ref_log_prob_micro_batch_size_per_gpu", type=int, default=2)
    parser.add_argument("--rollout_log_prob_micro_batch_size_per_gpu", type=int, default=2)
    parser.add_argument("--rl_learning_rate", type=float, default=5e-6)
    parser.add_argument("--rl_lr_schedule", type=str, default="constant")
    # Algo & KL (GRPO 风格)
    parser.add_argument("--rl_adv_estimator", type=str, default="grpo")
    parser.add_argument("--rl_use_kl_loss", type=int, default=1)
    parser.add_argument("--rl_kl_coef", type=float, default=0.001)
    # Rollout 采样参数
    parser.add_argument("--rl_rollout_n", type=int, default=8)
    parser.add_argument("--rl_rollout_temperature", type=float, default=1.0)
    parser.add_argument("--rl_rollout_topk", type=int, default=-1)
    parser.add_argument("--rl_rollout_topp", type=float, default=0.95)
    # 自定义 reward function，通过 Verl 的 custom_reward_function 接口
    parser.add_argument("--rl_reward_fn_path", type=str, default="recipe/ASR/metrics.py")
    parser.add_argument("--rl_reward_fn_name", type=str, default="compute_score")
    # 多机多卡
    parser.add_argument("--rl_ray_address", type=str, default=None, help="告诉 Ray 客户端如何连接 Ray 集群。Ray 会根据这个字符串调用 ray.init(address=...)")
    parser.add_argument("--rl_rollout_tensor_model_parallel_size", type=int, default=1, help="映射到 actor_rollout_ref.rollout.tensor_model_parallel_size；必须整除 trainer.n_gpus_per_node * trainer.nnodes")
    parser.add_argument("--rl_trainer_nnodes", type=int, default=4)
    parser.add_argument("--rl_trainer_n_gpus_per_node", type=int, default=8)
    parser.add_argument("--rl_ray_num_cpus", type=int, default=None, help="限制 Ray 在当前节点上最多使用多少 CPU 核；None 表示自动检测可用 CPU 数")

    # ===== 指标计算 =====
    parser.add_argument("--max_eval_samples", type[int], default=2048)
    parser.add_argument("--eval_batch_size", type[int], default=4)
    parser.add_argument("--max_length", type[int], default=40960, help="只计算 len(prompt + response) < max_length 的数据")
    parser.add_argument("--truncate_mode", type=str, default="skip", choices=["truncate", "skip"])

    # ===== wandb =====
    parser.add_argument("--wandb_project", type=str, default="ASR")
    parser.add_argument("--wandb_run_name", type[str], default=None)
    parser.add_argument("--wandb_mode", type[str], default="online", choices=["online", "offline", "disabled"])

    # ===== 保存策略 =====
    parser.add_argument("--save_strategy", type=str, default="best_on_D2", choices=["best_on_D2"], help="best_on_D2: 额外保留一份在 D2 val 上 accuracy 最好的 HF ckpt（保存在 work_dir/best_ckpt_D2）。")

    args = parser.parse_args()

    # ---- 一些 sanity check ----
    if args.sft_ckpt_dir is None or args.rl_ckpt_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.sft_ckpt_dir = f"ASR_sft_{args.task}_{timestamp}"
        args.rl_ckpt_dir = f"ASR_rl_{args.task}_{timestamp}"
    
    args.d1_train = f"/root/workspace/ASR_data/train/{args.sft_task}.jsonl"
    args.d1_valid = f"/root/workspace/ASR_data/valid/{args.sft_task}.jsonl"
    args.d2_train = f"/root/workspace/ASR_data/train/{args.rl_task}.jsonl"
    args.d2_valid = f"/root/workspace/ASR_data/valid/{args.rl_task}.jsonl"

    if args.wandb_run_name is None:
        args.wandb_run_name = f"sft_{args.sft_task}_rl_{args.rl_task}"

    if args.validation_strategy == "steps":
        if args.validation_steps <= 0:
            raise ValueError("validation_strategy == 'steps' 时，validation_steps 必须 > 0")
        if args.max_training_steps <= 0:
            raise ValueError("validation_strategy == 'steps' 时，max_training_steps 必须 > 0")

    def get_tokenizer_id_for_ckpt(model_or_ckpt: str, explicit_tokenizer: Optional[str] = None) -> str:
        """
        根据 ckpt 路径自动选择合适的 tokenizer 路径：
        1. 如果用户显式传了 --tokenizer，就永远用那个。
        2. 如果存在 {model_or_ckpt}/huggingface/tokenizer_config.json，就用 huggingface 子目录。
        3. 否则退回用 model_or_ckpt 本身（HF 模型名或本地 HF 目录）。
        """
        if explicit_tokenizer:
            return explicit_tokenizer

        p = Path(model_or_ckpt)
        hf_dir = p / "huggingface"
        if hf_dir.is_dir() and (hf_dir / "tokenizer_config.json").exists():
            return str(hf_dir)

        # 若 ckpt 本身就是 HF 格式目录
        if p.is_dir() and (p / "tokenizer_config.json").exists():
            return str(p)

        # 否则当作 HF hub 名称
        return model_or_ckpt

    # 全局追踪 D2 val 最佳 accuracy 的信息
    best_D2_acc = None          # float
    best_D2_ckpt = None         # 字符串形式（HF 名字或本地路径）
    best_D2_hf_dir = None       # Path: work_dir / "best_ckpt_D2"

    # ===== 开始前打印关键超参 / 数据规模信息 =====
    d1_train_size = count_samples_in_paths(args.d1_train)
    d1_val_size = count_samples_in_paths(args.d1_val)
    d2_train_size = count_samples_in_paths(args.d2_train)
    d2_val_size = count_samples_in_paths(args.d2_val)

    sft_global_batch = args.sft_batch_size
    sft_steps_per_epoch = calc_steps(d1_train_size, sft_global_batch, args.sft_epochs)

    if d2_train_size is not None:
        if args.rl_train_max_samples is not None and args.rl_train_max_samples > 0:
            rl_effective_train = min(d2_train_size, args.rl_train_max_samples)
        else:
            rl_effective_train = d2_train_size
    else:
        rl_effective_train = None

    rl_global_batch = args.rl_batch_size
    rl_steps_per_epoch = calc_steps(rl_effective_train, rl_global_batch, args.rl_epochs)

    def _fmt(v):
        return v if v is not None else "Unknown"

    print("=" * 60)
    print("[ASR Controller] Training configuration summary")
    print(f"  D1 train examples: {_fmt(d1_train_size)}")
    print(f"  D1 val   examples: {_fmt(d1_val_size)}")
    print(f"  D2 train examples: {_fmt(d2_train_size)}")
    print(f"  D2 val   examples: {_fmt(d2_val_size)}")
    print()
    print("  SFT:")
    print(f"    epochs: {args.sft_epochs}")
    print(f"    global batch size: {sft_global_batch}")
    print(f"    estimated grad update steps (per all epochs): {_fmt(sft_steps_per_epoch)}")
    print()
    print("  RL:")
    print(f"    epochs: {args.rl_epochs}")
    print(f"    max train samples (rl_train_max_samples): {args.rl_train_max_samples}")
    print(f"    effective train samples: {_fmt(rl_effective_train)}")
    print(f"    global batch size: {rl_global_batch}")
    print(f"    estimated grad update steps (per all epochs): {_fmt(rl_steps_per_epoch)}")
    print()
    print(f"  validation_strategy: {args.validation_strategy}")
    if args.validation_strategy == 'steps':
        print(f"    validation_steps: {args.validation_steps}")
        print(f"    max_training_steps: {args.max_training_steps}")
    else:
        print(f"    max_phases (epoch-wise): {args.max_phases}")
    print(f"  early_patience: {args.early_patience}")
    print("=" * 60, flush=True)

    # ===== wandb init =====
    if args.wandb_project and args.wandb_mode != "disabled":
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"ASR_run_{int(time.time())}",
            mode=args.wandb_mode,
            config={
                "schedule_mode": "ASR",
                "Hf_ratio": 0.27,
                "max_phases": args.max_phases,
                "validation_strategy": args.validation_strategy,
                "validation_steps": args.validation_steps,
                "max_training_steps": args.max_training_steps,
                "early_patience": args.early_patience,
            },
        )
    else:
        wandb.init(mode="disabled")

    # ===== 路径准备 =====
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    sft_ckpts_root = work / args.sft_ckpt_dir
    sft_ckpts_root.mkdir(parents=True, exist_ok=True)

    rl_ckpts_root = work / args.rl_ckpt_dir
    rl_ckpts_root.mkdir(parents=True, exist_ok=True)

    current_hf_dir = work / "current_ckpt_hf"

    # ===== Phase 0：用初始模型在 D2 val 上计算 H0, P0，并设置阈值 =====
    start_time = time.time()
    base_model = resolve_model_arg(args.base_model_or_ckpt)

    tok0 = get_tokenizer_id_for_ckpt(base_model, args.tokenizer)
    H0, P0 = run_distributed_eval(
        model_or_ckpt=base_model,
        tokenizer_id=tok0,
        args=args,
        work=work,
        phase=0,
    )

    # 固定阈值：Hf = 0.27 * H0, Pf = P0
    Hf = 0.27 * H0
    Pf = P0

    print(f"[Init] H0={H0:.6f}, P0={P0:.6f}", flush=True)
    print(f"[Init] Thresholds: Hf=0.27*H0={Hf:.6f}, Pf=P0={Pf:.6f}", flush=True)
    wandb.log({"phase": 0, "H0": H0, "P0": P0, "Hf": Hf, "Pf": Pf})

    # 初始化 best_on_D2 信息（如果启用）
    if args.save_strategy == "best_on_D2":
        best_D2_acc = P0
        best_D2_ckpt = base_model
        best_D2_hf_dir = None  # 还没复制 HF 目录
        wandb.log({
            "phase": 0,
            "D2_val_accuracy": P0,
            "best_D2_val_accuracy": best_D2_acc,
        })

    # 当前模型与当前度量（用于之后每个 phase 决策）
    current_ckpt = base_model
    current_H = H0
    current_P = P0
    num_sft_done = 0
    num_rl_done = 0

    # 早停相关
    patience_counter = 0  # 连续 Pn 上升的次数
    total_steps_done = 0  # 仅在 validation_strategy == 'steps' 时使用

    # 记录 phase 0 到 alt_log
    with open(work / "alt_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "phase": 0,
            "stage": "INIT",
            "H": H0,
            "P": P0,
            "Hf": Hf,
            "Pf": Pf,
            "time": int(time.time()),
        }, ensure_ascii=False) + "\n")

    # ===== 交替训练循环 =====
    for n in range(1, args.max_phases + 1):
        # --- 当使用 step-wise 验证时，检查总步数预算是否耗尽 ---
        if args.validation_strategy == "steps" and total_steps_done >= args.max_training_steps:
            print(
                f"[Early Stop] Reached max_training_steps={args.max_training_steps} "
                f"(total_steps_done={total_steps_done}), stop alternation.",
                flush=True,
            )
            break

        # 记录上一轮的 P，用于 early_patience（注意：phase 0 的 P0 也参与比较）
        prev_P = current_P

        # --- 根据上一轮的度量决定本阶段跑 SFT 还是 RL ---
        if current_P > Pf and current_H > Hf:
            stage = "RL"   # Pn > Pf 且 Hn > Hf -> 下一阶段 RL（在 D2 train 上做 RL）
        else:
            stage = "SFT"  # 否则 -> 在 D1 train 上做 SFT

        print(
            f"[Phase {n}] "
            f"current_H={current_H:.6f} (threshold Hf={Hf:.6f}) | "
            f"current_P={current_P:.6f} (threshold Pf={Pf:.6f}) | "
            f"next_stage={stage}",
            flush=True,
        )

        wandb.log({
            "phase": n,
            "current_H": current_H,
            "current_P": current_P,
            "Hf": Hf,
            "Pf": Pf,
            "next_stage_is_rl": 1 if stage == "RL" else 0,
            "num_sft_done": num_sft_done,
            "num_rl_done": num_rl_done,
            "total_training_steps_so_far": total_steps_done,
        })

        # --- 运行一个 SFT 或 RL 子阶段 ---
        if stage == "SFT":
            ckpt_out = sft_ckpts_root / f"phase_{n}"
            ckpt_out.mkdir(parents=True, exist_ok=True)
            cmd = build_sft_cmd(args, current_ckpt, ckpt_out, phase=n)
        else:
            ckpt_out = rl_ckpts_root / f"phase_{n}"
            ckpt_out.mkdir(parents=True, exist_ok=True)
            cmd = build_rl_cmd(args, current_ckpt, ckpt_out, phase=n)

        run_cmd(cmd)

        # step-wise 情况下，累计已训练步数
        if args.validation_strategy == "steps":
            total_steps_done += args.validation_steps
            wandb.log({
                "phase": n,
                "total_training_steps_so_far": total_steps_done,
            })

        # --- 读 summary.json，log 子训练曲线 ---
        summary_file = ckpt_out / "summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, "r", encoding="utf-8") as sf:
                    summary = json.load(sf)
                steps = summary.get("steps", [])
                losses = summary.get("train_loss_curve", [])
                if steps and losses and len(steps) == len(losses):
                    for s, l in zip(steps, losses):
                        wandb.log({
                            "train/loss": l,
                            "train/step": s,
                            "phase": n,
                            "stage_is_rl": 1 if stage == "RL" else 0,
                        })
                if "avg_train_loss" in summary:
                    wandb.log({
                        f"{stage.lower()}_avg_train_loss": summary["avg_train_loss"],
                        "phase": n,
                    })
                if "train_wallclock_s" in summary:
                    wandb.log({
                        f"{stage.lower()}_wallclock_s": summary["train_wallclock_s"],
                        "phase": n,
                    })
            except Exception as err:
                print(f"[Warning] Failed to read summary.json in {ckpt_out}: {err}", flush=True)

        # --- 更新 current_ckpt（FSDP -> HF 合并），并清理 FSDP 目录节省空间 ---
        fsdp_ckpt = latest_checkpoint(ckpt_out) or ckpt_out  # e.g. .../global_step_10

        # 每个 phase 都把 HF 权重写到同一个目录：work/current_ckpt_hf
        if current_hf_dir.exists():
            shutil.rmtree(current_hf_dir)

        print(f"[Phase {n}] Merging FSDP ckpt {fsdp_ckpt} -> HF model {current_hf_dir}", flush=True)

        if stage == "SFT":
            # SFT：优先用 ckpt 自带 huggingface config；否则退回 base_model_or_ckpt
            hf_cfg_dir = fsdp_ckpt / "huggingface"
            if (hf_cfg_dir / "config.json").exists():
                merge_fsdp_to_hf(fsdp_ckpt, current_hf_dir)
            else:
                print(
                    f"[Phase {n}] Warning: {hf_cfg_dir}/config.json not found, "
                    f"fallback to base_model_or_ckpt={args.base_model_or_ckpt} as HF config source.",
                    flush=True,
                )
                merge_fsdp_to_hf(fsdp_ckpt, current_hf_dir, hf_model_config_path=args.base_model_or_ckpt)
        else:
            # RL：始终用 base_model_or_ckpt 提供 HF config
            merge_fsdp_to_hf(fsdp_ckpt, current_hf_dir, hf_model_config_path=args.base_model_or_ckpt)

        # 合并成功后，删除本阶段的 FSDP ckpt 目录（包括其中的所有 step/epoch ckpt）
        try:
            shutil.rmtree(ckpt_out)
            print(f"[Phase {n}] Removed FSDP checkpoint directory {ckpt_out} to save space.", flush=True)
        except Exception as err:
            print(f"[Phase {n}] Warning: failed to remove {ckpt_out}: {err}", flush=True)

        # 下一阶段从当前 HF 模型继续
        current_ckpt = str(current_hf_dir)

        # --- 用合并后的 HF 模型在 D2 val 上重新评估 H_n, P_n，用于下一阶段决策 ---
        tok_n = get_tokenizer_id_for_ckpt(current_ckpt, args.tokenizer)
        Hn, Pn = run_distributed_eval(
            model_or_ckpt=current_ckpt,
            tokenizer_id=tok_n,
            args=args,
            work=work,
            phase=n,
        )

        print(
            f"[Phase {n}] After {stage}: Hn={Hn:.6f}, Pn={Pn:.6f}",
            flush=True,
        )

        # --- 更新 best_on_D2 ckpt（如启用） ---
        if args.save_strategy == "best_on_D2":
            if (best_D2_acc is None) or (Pn > best_D2_acc):
                best_D2_acc = Pn
                best_D2_ckpt = current_ckpt
                best_D2_hf_dir = work / "best_ckpt_D2"

                if best_D2_hf_dir.exists():
                    shutil.rmtree(best_D2_hf_dir)
                shutil.copytree(current_ckpt, best_D2_hf_dir)

                print(f"[Phase {n}] New best D2 checkpoint saved to {best_D2_hf_dir}", flush=True)

            wandb.log({
                "phase": n,
                "D2_val_accuracy": Pn,
                "best_D2_val_accuracy": best_D2_acc,
            })

        # --- 更新计数、记录 alt_log，并为下一轮决策准备 current_H / current_P ---
        if stage == "SFT":
            num_sft_done += 1
        else:
            num_rl_done += 1

        with open(work / "alt_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "phase": n,
                "stage": stage,
                "Hn": Hn,
                "Pn": Pn,
                "Hf": Hf,
                "Pf": Pf,
                "num_sft_done": num_sft_done,
                "num_rl_done": num_rl_done,
                "time": int(time.time()),
            }, ensure_ascii=False) + "\n")

        # ---- early_patience: 连续 Pn 下降就累加，否则清零 ----
        if Pn < prev_P:
            patience_counter += 1
        else:
            patience_counter = 0

        wandb.log({
            "phase": n,
            "Pn": Pn,
            "prev_P": prev_P,
            "early_patience_counter": patience_counter,
        })

        current_H = Hn
        current_P = Pn

        print(f"[Phase {n}] Updated current_ckpt -> {current_ckpt}", flush=True)

        # 若 patience_counter 达到阈值，则提前停止交替训练
        if args.early_patience > 0 and patience_counter >= args.early_patience:
            print(
                f"[Early Stop] Pn has increased for {patience_counter} consecutive phases "
                f"(>= early_patience={args.early_patience}), stop alternation.",
                flush=True,
            )
            break

    # ===== 总结 =====
    total_time = time.time() - start_time
    wandb.log({
        "total_wallclock_time_s": total_time,
        "total_wallclock_time_h": total_time / 3600.0,
    })
    wandb.finish()

    print(f"[Done] Total wallclock: {total_time / 3600.0:.4f} h")

    # 输出 D2 最佳 ckpt 信息
    if args.save_strategy == "best_on_D2" and best_D2_acc is not None:
        print(f"[Done] Best D2 val accuracy: {best_D2_acc:.6f}")
        if best_D2_hf_dir is not None:
            print(f"[Done] Best D2 checkpoint directory (HF merged copy): {best_D2_hf_dir}")
        else:
            # 没有任何 phase 超过初始模型，则 best 仍然是 base_model_or_ckpt
            print(f"[Done] Best D2 checkpoint is the initial model_or_ckpt: {best_D2_ckpt}")
    
    print(f"[Done] Finished alternation. Final checkpoint: {current_ckpt}")


if __name__ == "__main__":
    main()
