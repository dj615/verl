python -m verl.trainer.main_generation \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=1 \
  data.path=/path/to/prompts.parquet \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.batch_size=1 \
  data.output_path=/path/to/gen_outputs.parquet \
  model.path=/path/to/merged_hf_model \
  rollout.temperature=0 \
  rollout.top_p=1.0 \
  rollout.top_k=-1 \
  rollout.prompt_length=8192 \
  rollout.response_length=8192 \
  rollout.tensor_model_parallel_size=1 \
  rollout.gpu_memory_utilization=0.8
