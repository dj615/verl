#!/usr/bin/env bash

# export HF_HOME="/autodl-pub/data/hf_cache"
# export TRANSFORMERS_CACHE="/autodl-pub/data/hf_cache"
# export WANDB_API_KEY="你的wandb key"
# export CUDA_VISIBLE_DEVICES="0"

python -m recipe.ASR.ASR \
  --base_model_or_ckpt Qwen/Qwen3-0.6B \
  --schedule_mode ASR \
  --work_dir ASR_MATH \
  --d1_train /root/workspace/ASR_data/train/MATH.parquet \
  --d1_val /root/workspace/ASR_data/train/MATH.parquet \
  --d2_train /root/workspace/ASR_data/train/MATH.parquet \
  --d2_val /root/workspace/ASR_data/train/MATH.parquet \
  --sft_lora_rank 2 \
  --sft_lora_alpha 4 \
  --sft_batch_size 2 \
  --sft_micro_batch_size_per_gpu 1 \
  --sft_learning_rate 5e-5 \
  --sft_lr_schedule "cosine" \
  --sft_max_length 512 \
  --sft_nproc_per_node 1 \
  --sft_nnodes 1 \
  --sft_node_rank 0 \
  --sft_master_addr 10.0.0.3 \
  --sft_master_port 29500 \
  --rl_lora_rank 2 \
  --rl_lora_alpha 4 \
  --rl_batch_size 4 \
  --rl_learning_rate 5e-5 \
  --rl_lr_schedule "cosine" \
  --rl_max_prompt_length 512 \
  --rl_max_response_length 2048 \
  --rl_rollout_n 2 \
  --rl_rollout_temperature 0.6 \
  --rl_trainer_nnodes 1 \
  --rl_trainer_n_gpus_per_node 1 \
  --eval_batch_size 2 \
  --wandb_project ASR \
  --wandb_run_name MATH_Train
  