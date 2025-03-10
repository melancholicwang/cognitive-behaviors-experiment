#!/bin/bash
# 数据生成脚本

# 设置环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cognitive-test

# 创建目录
mkdir -p data/raw
mkdir -p data/processed

# 生成倒计时游戏数据集
echo "生成倒计时游戏数据集..."
python src/countdown.py --num_samples 1000 --output_file data/raw/countdown.jsonl

# 生成不同认知行为的思维链数据
echo "生成不同认知行为的思维链数据..."

# 负面对照组
python src/generate_cot_data.py generate \
  --input_file data/raw/countdown.jsonl \
  --output_file data/raw/negative_control.jsonl \
  --dataset_type negative_control \
  --num_samples 1000

# 仅回溯
python src/generate_cot_data.py generate \
  --input_file data/raw/countdown.jsonl \
  --output_file data/raw/only_backtracking.jsonl \
  --dataset_type only_backtracking \
  --num_samples 1000

# 回溯+验证
python src/generate_cot_data.py generate \
  --input_file data/raw/countdown.jsonl \
  --output_file data/raw/backtracking_verification.jsonl \
  --dataset_type backtracking_verification \
  --num_samples 1000

# 回溯+逆向推理
python src/generate_cot_data.py generate \
  --input_file data/raw/countdown.jsonl \
  --output_file data/raw/backtracking_backward.jsonl \
  --dataset_type backtracking_backward \
  --num_samples 1000

# 回溯+子目标
python src/generate_cot_data.py generate \
  --input_file data/raw/countdown.jsonl \
  --output_file data/raw/backtracking_subgoal.jsonl \
  --dataset_type backtracking_subgoal \
  --num_samples 1000

# 所有策略
python src/generate_cot_data.py generate \
  --input_file data/raw/countdown.jsonl \
  --output_file data/raw/all_strategies.jsonl \
  --dataset_type all_strategies \
  --num_samples 1000

# 空思维链
python src/generate_cot_data.py empty \
  --input_file data/raw/countdown.jsonl \
  --output_file data/raw/empty_cot.jsonl \
  --num_samples 1000

# 无正确答案
python src/generate_cot_data.py no_positive \
  --input_file data/raw/all_strategies.jsonl \
  --output_file data/raw/no_positive_cot.jsonl \
  --num_samples 1000

# 处理为Parquet格式
echo "处理为Parquet格式..."

# 负面对照组
python src/generate_cot_data.py to_parquet \
  --input_file data/raw/negative_control.jsonl \
  --output_dir data/processed/negative_control

# 仅回溯
python src/generate_cot_data.py to_parquet \
  --input_file data/raw/only_backtracking.jsonl \
  --output_dir data/processed/only_backtracking

# 回溯+验证
python src/generate_cot_data.py to_parquet \
  --input_file data/raw/backtracking_verification.jsonl \
  --output_dir data/processed/backtracking_verification

# 回溯+逆向推理
python src/generate_cot_data.py to_parquet \
  --input_file data/raw/backtracking_backward.jsonl \
  --output_dir data/processed/backtracking_backward

# 回溯+子目标
python src/generate_cot_data.py to_parquet \
  --input_file data/raw/backtracking_subgoal.jsonl \
  --output_dir data/processed/backtracking_subgoal

# 所有策略
python src/generate_cot_data.py to_parquet \
  --input_file data/raw/all_strategies.jsonl \
  --output_dir data/processed/all_strategies

# 空思维链
python src/generate_cot_data.py to_parquet \
  --input_file data/raw/empty_cot.jsonl \
  --output_dir data/processed/empty_cot

# 无正确答案
python src/generate_cot_data.py to_parquet \
  --input_file data/raw/no_positive_cot.jsonl \
  --output_dir data/processed/no_positive_cot

echo "数据生成完成！"