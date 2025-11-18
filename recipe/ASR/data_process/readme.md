# 数据处理要求

## 训练 / 验证

 - 数据格式：数据应以 **Parquet** 格式保存；为了可读性，也可额外输出对应的 **JSONL** 文件。
 - 数据字段：如下所示。
 - 数据划分：train / valid / test
 - 数据数量：5000 / 1000 / 1000
 - 数据长度：超出 4096 tokens 即舍弃
 - 存储位置：/root/workspace/ASR_data/...

### SFT 数据字段

{
  "question": "string",
  "answer": "string"
}

### RL 数据字段

{
  "question": "原始 input",
  "answer": "原始 output",
  "groundtruth": "expected_answer",
  "data_source": "openscience",
  "ability": "ability 标签",
  "reward_model": {
    "style": "rule",
    "ground_truth": "expected_answer"
  },
  "extra_info": { ... metadata ... }
}

## 推理

### 数据字段

{
  "prompt": "原始 prompt", 
  "groundtruth": "原始 groundtruth",
}

### 调用命令

python -m verl.trainer.main_generation \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=1 \
  data.path=/path/to/prompts.parquet \
  data.prompt_key=prompt \
  data.n_samples=1 \
  data.batch_size=1 \
  data.output_path=/path/to/gen_outputs.parquet \
  model.path=/path/to/merged_hf_model \
  rollout.temperature=0.7 \
  rollout.top_p=0.9 \
  rollout.top_k=-1 \
  rollout.prompt_length=4096 \
  rollout.response_length=1024 \
  rollout.tensor_model_parallel_size=1 \
  rollout.gpu_memory_utilization=0.8
