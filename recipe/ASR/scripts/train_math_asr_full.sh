#!/usr/bin/env bash

# export HF_HOME="/autodl-pub/data/hf_cache"
# export TRANSFORMERS_CACHE="/autodl-pub/data/hf_cache"
# export WANDB_API_KEY="你的wandb key"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m recipe.ASR.ASR \
  --base_model_or_ckpt Qwen/Qwen3-8B \
  --sft_master_addr <address> \
  --rl_ray_address <address:port> \
  --sft_ckpt_dir math_ASR_full_sft \
  --rl_ckpt_dir math_ASR_full_rl \
  --sft_lora_enable 0 \
  --rl_lora_enable 0 \
  --sft_task MATH \
  --rl_task OpenR1_Math_220k \
  --sft_master_port 29500 \
  --rl_rollout_gpu_memory_utilization 0.5 \
  --rl_micro_batch_size_per_gpu 2 \
  --ref_log_prob_micro_batch_size_per_gpu 2 \
  --rollout_log_prob_micro_batch_size_per_gpu 2

python -m verl.trainer.main_generation \
  model.path=/root/workspace/checkpoints/best_MATH_OpenR1_Math_220k_lora_1_1_ckpt_on_D2 \
  data.path=/root/workspace/ASR_data/test/aime24.parquet \
  data.output_path=/root/workspace/ASR_data/predictions/best_MATH_OpenR1_Math_220k_lora_1_1_ckpt_on_D2_aime24.parquet \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.batch_size=1 \
  rollout.temperature=0 \
  rollout.top_p=1.0 \
  rollout.top_k=-1 \
  rollout.prompt_length=40960 \
  rollout.response_length=8192 \
  rollout.tensor_model_parallel_size=1 \
  rollout.gpu_memory_utilization=0.8

python -m verl.trainer.main_generation \
  model.path=/root/workspace/checkpoints/best_MATH_OpenR1_Math_220k_lora_1_1_ckpt_on_D2 \
  data.path=/root/workspace/ASR_data/test/aime25.parquet \
  data.output_path=/root/workspace/ASR_data/predictions/best_MATH_OpenR1_Math_220k_lora_1_1_ckpt_on_D2_aime25.parquet \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.batch_size=1 \
  rollout.temperature=0 \
  rollout.top_p=1.0 \
  rollout.top_k=-1 \
  rollout.prompt_length=40960 \
  rollout.response_length=8192 \
  rollout.tensor_model_parallel_size=1 \
  rollout.gpu_memory_utilization=0.8

python -m verl.trainer.main_generation \
  model.path=/root/workspace/checkpoints/best_MATH_OpenR1_Math_220k_lora_1_1_ckpt_on_D2 \
  data.path=/root/workspace/ASR_data/test/amc23.parquet \
  data.output_path=/root/workspace/ASR_data/predictions/best_MATH_OpenR1_Math_220k_lora_1_1_ckpt_on_D2_amc23.parquet \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.batch_size=1 \
  rollout.temperature=0 \
  rollout.top_p=1.0 \
  rollout.top_k=-1 \
  rollout.prompt_length=40960 \
  rollout.response_length=8192 \
  rollout.tensor_model_parallel_size=1 \
  rollout.gpu_memory_utilization=0.8

python -m verl.trainer.main_generation \
  model.path=/root/workspace/checkpoints/best_MATH_OpenR1_Math_220k_lora_1_1_ckpt_on_D2 \
  data.path=/root/workspace/ASR_data/test/math500.parquet \
  data.output_path=/root/workspace/ASR_data/predictions/best_MATH_OpenR1_Math_220k_lora_1_1_ckpt_on_D2_math500.parquet \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.batch_size=1 \
  rollout.temperature=0 \
  rollout.top_p=1.0 \
  rollout.top_k=-1 \
  rollout.prompt_length=40960 \
  rollout.response_length=8192 \
  rollout.tensor_model_parallel_size=1 \
  rollout.gpu_memory_utilization=0.8

python -m verl.trainer.main_generation \
  model.path=/root/workspace/checkpoints/best_MATH_OpenR1_Math_220k_lora_1_1_ckpt_on_D2 \
  data.path=/root/workspace/ASR_data/test/omni_math.parquet \
  data.output_path=/root/workspace/ASR_data/predictions/best_MATH_OpenR1_Math_220k_lora_1_1_ckpt_on_D2_omni_math.parquet \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.batch_size=1 \
  rollout.temperature=0 \
  rollout.top_p=1.0 \
  rollout.top_k=-1 \
  rollout.prompt_length=40960 \
  rollout.response_length=8192 \
  rollout.tensor_model_parallel_size=1 \
  rollout.gpu_memory_utilization=0.8

python -m verl.trainer.main_generation \
  model.path=/root/workspace/checkpoints/best_MATH_OpenR1_Math_220k_lora_1_1_ckpt_on_D2 \
  data.path=/root/workspace/ASR_data/test/minerva.parquet \
  data.output_path=/root/workspace/ASR_data/predictions/best_MATH_OpenR1_Math_220k_lora_1_1_ckpt_on_D2_minerva.parquet \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.batch_size=1 \
  rollout.temperature=0 \
  rollout.top_p=1.0 \
  rollout.top_k=-1 \
  rollout.prompt_length=40960 \
  rollout.response_length=8192 \
  rollout.tensor_model_parallel_size=1 \
  rollout.gpu_memory_utilization=0.8

python -m verl.trainer.main_generation \
  model.path=/root/workspace/checkpoints/best_MATH_OpenR1_Math_220k_lora_1_1_ckpt_on_D2 \
  data.path=/root/workspace/ASR_data/test/OlympiadBench_math_en_comp.parquet \
  data.output_path=/root/workspace/ASR_data/predictions/best_MATH_OpenR1_Math_220k_lora_1_1_ckpt_on_D2_OlympiadBench_math_en_comp.parquet \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.batch_size=1 \
  rollout.temperature=0 \
  rollout.top_p=1.0 \
  rollout.top_k=-1 \
  rollout.prompt_length=40960 \
  rollout.response_length=8192 \
  rollout.tensor_model_parallel_size=1 \
  rollout.gpu_memory_utilization=0.8

python -m verl.trainer.main_generation \
  model.path=/root/workspace/checkpoints/best_MATH_OpenR1_Math_220k_lora_1_1_ckpt_on_D2 \
  data.path=/root/workspace/ASR_data/test/OlympiadBench_math_zh_comp.parquet \
  data.output_path=/root/workspace/ASR_data/predictions/best_MATH_OpenR1_Math_220k_lora_1_1_ckpt_on_D2_OlympiadBench_math_zh_comp.parquet \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.batch_size=1 \
  rollout.temperature=0 \
  rollout.top_p=1.0 \
  rollout.top_k=-1 \
  rollout.prompt_length=40960 \
  rollout.response_length=8192 \
  rollout.tensor_model_parallel_size=1 \
  rollout.gpu_memory_utilization=0.8

python -m verl.trainer.main_generation \
  model.path=/root/workspace/checkpoints/best_MATH_OpenR1_Math_220k_lora_1_1_ckpt_on_D2 \
  data.path=/root/workspace/ASR_data/test/OlympiadBench_math_en_cee.parquet \
  data.output_path=/root/workspace/ASR_data/predictions/best_MATH_OpenR1_Math_220k_lora_1_1_ckpt_on_D2_OlympiadBench_math_en_cee.parquet \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.batch_size=1 \
  rollout.temperature=0 \
  rollout.top_p=1.0 \
  rollout.top_k=-1 \
  rollout.prompt_length=40960 \
  rollout.response_length=8192 \
  rollout.tensor_model_parallel_size=1 \
  rollout.gpu_memory_utilization=0.8

python -m verl.trainer.main_generation \
  model.path=/root/workspace/checkpoints/best_MATH_OpenR1_Math_220k_lora_1_1_ckpt_on_D2 \
  data.path=/root/workspace/ASR_data/test/OlympiadBench_math_zh_cee.parquet \
  data.output_path=/root/workspace/ASR_data/predictions/best_MATH_OpenR1_Math_220k_lora_1_1_ckpt_on_D2_OlympiadBench_math_zh_cee.parquet \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.batch_size=1 \
  rollout.temperature=0 \
  rollout.top_p=1.0 \
  rollout.top_k=-1 \
  rollout.prompt_length=40960 \
  rollout.response_length=8192 \
  rollout.tensor_model_parallel_size=1 \
  rollout.gpu_memory_utilization=0.8

python -m verl.trainer.main_generation \
  model.path=/root/workspace/checkpoints/best_MATH_OpenR1_Math_220k_lora_1_1_ckpt_on_D2 \
  data.path=/root/workspace/ASR_data/test/OlympiadBench_physics_en_comp.parquet \
  data.output_path=/root/workspace/ASR_data/predictions/best_MATH_OpenR1_Math_220k_lora_1_1_ckpt_on_D2_OlympiadBench_physics_en_comp.parquet \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.batch_size=1 \
  rollout.temperature=0 \
  rollout.top_p=1.0 \
  rollout.top_k=-1 \
  rollout.prompt_length=40960 \
  rollout.response_length=8192 \
  rollout.tensor_model_parallel_size=1 \
  rollout.gpu_memory_utilization=0.8

python -m verl.trainer.main_generation \
  model.path=/root/workspace/checkpoints/best_MATH_OpenR1_Math_220k_lora_1_1_ckpt_on_D2 \
  data.path=/root/workspace/ASR_data/test/OlympiadBench_physics_zh_comp.parquet \
  data.output_path=/root/workspace/ASR_data/predictions/best_MATH_OpenR1_Math_220k_lora_1_1_ckpt_on_D2_OlympiadBench_physics_zh_comp.parquet \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.batch_size=1 \
  rollout.temperature=0 \
  rollout.top_p=1.0 \
  rollout.top_k=-1 \
  rollout.prompt_length=40960 \
  rollout.response_length=8192 \
  rollout.tensor_model_parallel_size=1 \
  rollout.gpu_memory_utilization=0.8

python -m verl.trainer.main_generation \
  model.path=/root/workspace/checkpoints/best_MATH_OpenR1_Math_220k_lora_1_1_ckpt_on_D2 \
  data.path=/root/workspace/ASR_data/test/OlympiadBench_physics_en_cee.parquet \
  data.output_path=/root/workspace/ASR_data/predictions/best_MATH_OpenR1_Math_220k_lora_1_1_ckpt_on_D2_OlympiadBench_physics_en_cee.parquet \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.batch_size=1 \
  rollout.temperature=0 \
  rollout.top_p=1.0 \
  rollout.top_k=-1 \
  rollout.prompt_length=40960 \
  rollout.response_length=8192 \
  rollout.tensor_model_parallel_size=1 \
  rollout.gpu_memory_utilization=0.8

python -m verl.trainer.main_generation \
  model.path=/root/workspace/checkpoints/best_MATH_OpenR1_Math_220k_lora_1_1_ckpt_on_D2 \
  data.path=/root/workspace/ASR_data/test/OlympiadBench_physics_zh_cee.parquet \
  data.output_path=/root/workspace/ASR_data/predictions/best_MATH_OpenR1_Math_220k_lora_1_1_ckpt_on_D2_OlympiadBench_physics_zh_cee.parquet \
  trainer.nnodes=4 \
  trainer.n_gpus_per_node=8 \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.batch_size=1 \
  rollout.temperature=0 \
  rollout.top_p=1.0 \
  rollout.top_k=-1 \
  rollout.prompt_length=40960 \
  rollout.response_length=8192 \
  rollout.tensor_model_parallel_size=1 \
  rollout.gpu_memory_utilization=0.8
