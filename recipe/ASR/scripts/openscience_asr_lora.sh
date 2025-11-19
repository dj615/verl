#!/usr/bin/env bash

# export HF_HOME="/autodl-pub/data/hf_cache"
# export TRANSFORMERS_CACHE="/autodl-pub/data/hf_cache"
# export WANDB_API_KEY="你的wandb key"

python -m recipe.ASR.ASR \
  --base_model_or_ckpt meta-llama/Llama-3.1-8B-Instruct \
  --sft_master_addr <address> \
  --rl_ray_address <address:port> \
  --sft_ckpt_dir openscience_ASR_lora_sft \
  --rl_ckpt_dir openscience_ASR_lora_rl \
  --sft_task openscience \
  --rl_task openscience \
  --sft_master_port 29500 \
  --rl_rollout_gpu_memory_utilization 0.5 \
  --rl_micro_batch_size_per_gpu 2 \
  --ref_log_prob_micro_batch_size_per_gpu 2 \
  --rollout_log_prob_micro_batch_size_per_gpu 2



MODEL=/root/workspace/checkpoints/best_openscience_openscience_lora_1_1_ckpt_on_D2
DATA_DIR=/root/workspace/ASR_data/test
OUT_DIR=/root/workspace/ASR_data/predictions
RAY_ADDR=<address>

DATASETS=(
  gpqa_diamond
  gpqa_main
  gpq_extended
  simpleqa
  frames
  hle
  mmlu_pro
)

for name in "${DATASETS[@]}"; do
  echo "===== Running dataset: $name ====="

  python -m verl.trainer.main_generation \
    --model.path=$MODEL \
    --data.path=$DATA_DIR/${name}.parquet \
    --data.output_path=$OUT_DIR/best_openscience_openscience_lora_1_1_ckpt_on_D2_${name}.parquet \
    --trainer.nnodes=4 \
    --trainer.n_gpus_per_node=8 \
    --ray_address=$RAY_ADDR \
    --data.prompt_key=prompt \
    --data.n_samples=1 \
    --data.batch_size=64 \
    --rollout.temperature=0 \
    --rollout.top_p=1.0 \
    --rollout.top_k=-1 \
    --rollout.prompt_length=30000 \
    --rollout.response_length=8192 \
    --rollout.tensor_model_parallel_size=1 \
    --rollout.gpu_memory_utilization=0.8
done
