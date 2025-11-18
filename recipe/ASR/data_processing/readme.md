# 数据处理要求

 - 数据格式：数据应以 **Parquet** 格式保存；为了可读性，也可额外输出对应的 **JSONL** 文件。
 - 数据字段：如下所示。
 - 数据划分：train / valid / test
 - 数据数量：5000 / 1000 / 1000
 - 数据长度：超出 4096 tokens 即舍弃
 - 存储位置：/root/workspace/ASR_data/...

---

## SFT 数据字段

{
  "question": "string",
  "answer": "string"
}

## RL 数据字段

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

注：RL 阶段将使用以下字段来进行训练
 - question
 - answer
 - reward model