#!/bin/bash
set -e

echo "========== 开始运行全部数据处理脚本 =========="

# ------ test 目录 ------
echo "[1/13] general.py"
python3 recipe/ASR/data_process/test/general.py

echo "[2/13] hle.py"
python3 recipe/ASR/data_process/test/hle.py

echo "[3/13] mmlu_pro.py"
python3 recipe/ASR/data_process/test/mmlu_pro.py

echo "[4/13] olympiadbench.py"
python3 recipe/ASR/data_process/test/olympiadbench.py


# ------ train_valid 目录 ------
echo "[5/13] DAPO_MATH.py"
python3 recipe/ASR/data_process/test/train_valid/DAPO_MATH.py

echo "[6/13] gsm8k.py"
python3 recipe/ASR/data_process/test/train_valid/gsm8k.py

echo "[7/13] HARP_download.py"
python3 recipe/ASR/data_process/test/train_valid/HARP_download.py

echo "[8/13] HARP.py"
python3 recipe/ASR/data_process/test/train_valid/HARP.py

echo "[9/13] MATH.py"
python3 recipe/ASR/data_process/test/train_valid/MATH.py

echo "[10/13] NuminaMath_1.5.py"
python3 recipe/ASR/data_process/test/train_valid/NuminaMath_1.5.py

echo "[11/13] NuminaMath_CoT.py"
python3 recipe/ASR/data_process/test/train_valid/NuminaMath_CoT.py

echo "[12/13] OpenR1_Math_220k.py"
python3 recipe/ASR/data_process/test/train_valid/OpenR1_Math_220k.py

echo "[13/13] OpenScienceReasoning_2.py"
python3 recipe/ASR/data_process/test/train_valid/OpenScienceReasoning_2.py


echo "========== 全部脚本运行完成 =========="
