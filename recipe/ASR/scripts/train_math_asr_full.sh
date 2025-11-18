#!/usr/bin/env bash

# export HF_HOME="/autodl-pub/data/hf_cache"
# export TRANSFORMERS_CACHE="/autodl-pub/data/hf_cache"
# export WANDB_API_KEY="你的wandb key"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m recipe.ASR.ASR \
  --base_model_or_ckpt Qwen/Qwen3-8B \
  --sft_master_addr <address> \
  --rl_ray_address <address:port> \
  --sft_lora_enable 0 \
  --rl_lora_enable 0 \
  --sft_task MATH \
  --rl_task OpenR1_Math_220k \
  --sft_master_port 29500 \
  --rl_rollout_gpu_memory_utilization 0.5 \
  --rl_micro_batch_size_per_gpu 2 \
  --ref_log_prob_micro_batch_size_per_gpu 2 \
  --rollout_log_prob_micro_batch_size_per_gpu 2
