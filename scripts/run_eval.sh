#!/bin/bash
# 行为评估脚本

# 设置环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cognitive-test

# 创建目录
mkdir -p outputs

# 数据集条件列表
conditions=(
  "all_strategies"
  "backtracking_backward"
  "backtracking_subgoal"
  "backtracking_verification"
  "only_backtracking"
)

# 测试数据集
test_dataset="data/processed/all_strategies/test.parquet"

# 检查测试数据集是否存在
if [ ! -f "$test_dataset" ]; then
  echo "错误: 测试数据集不存在: $test_dataset"
  exit 1
fi

# 评估基础模型
echo "评估基础模型..."
python src/behavior_eval.py \
  --model_path "Qwen/Qwen2.5-7B" \
  --dataset_path "$test_dataset" \
  --output_dir "outputs/base_model" \
  --num_samples 50 \
  --mode "single"

# 遍历每个条件并评估
for condition in "${conditions[@]}"; do
  echo "评估条件: ${condition}..."
  
  # SFT模型路径
  sft_model_path="checkpoints/${condition}_sft"
  
  # PPO模型路径
  ppo_model_path="checkpoints/${condition}_ppo"
  
  # 检查模型目录是否存在
  if [ ! -d "$sft_model_path" ]; then
    echo "警告: SFT模型目录不存在: $sft_model_path，跳过评估"
  else
    # 评估SFT模型
    echo "评估SFT模型: ${condition}..."
    python src/behavior_eval.py \
      --model_path "$sft_model_path" \
      --dataset_path "$test_dataset" \
      --output_dir "outputs/${condition}_sft" \
      --num_samples 50 \
      --mode "single"
  fi
  
  if [ ! -d "$ppo_model_path" ]; then
    echo "警告: PPO模型目录不存在: $ppo_model_path，跳过评估"
  else
    # 评估PPO模型检查点
    echo "评估PPO模型检查点: ${condition}..."
    python src/behavior_eval.py \
      --model_path "$ppo_model_path" \
      --dataset_path "$test_dataset" \
      --output_dir "outputs/${condition}_ppo" \
      --num_samples 50 \
      --mode "checkpoints" \
      --checkpoints_dir "$ppo_model_path"
  fi
  
  echo "条件 ${condition} 的评估完成！"
  echo "----------------------------------------"
done

echo "所有评估完成！"