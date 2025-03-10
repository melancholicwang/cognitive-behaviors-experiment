# 认知行为实验：提升qw2.5-7B的自我改进能力

本项目基于论文《Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs》的思路，对qw2.5-7B模型进行认知行为增强实验。

## 实验背景

论文研究表明，某些语言模型（如Qwen-2.5-3B）在自我改进方面表现优于其他模型（如Llama-3.2-3B），这种差异主要源于模型在推理过程中展现的四种关键认知行为：

1. **验证（Verification）**：系统性地检查中间结果和最终结果
2. **回溯（Backtracking）**：当发现错误或死胡同时，放弃当前路径并尝试新方法
3. **子目标设定（Subgoal Setting）**：将复杂问题分解为可管理的步骤
4. **逆向推理（Backward Chaining）**：从目标状态向初始状态推理

论文发现，通过对模型进行这些认知行为的引导（priming），可以显著提高模型的自我改进能力。

## 实验流程

本实验将复现论文中的方法，并应用于qw2.5-7B模型：

1. **数据准备**：生成Countdown游戏任务数据集
2. **认知行为数据生成**：创建包含不同认知行为模式的示例数据
3. **模型微调**：使用认知行为数据对模型进行SFT（监督微调）
4. **强化学习**：对微调后的模型进行PPO训练
5. **行为评估**：使用GPT-4o-mini评估模型在不同训练阶段展现的认知行为

## 实验优势

通过这种方法，我们期望：

- 提高qw2.5-7B模型的推理能力
- 增强模型利用额外计算资源的效率
- 使模型能够在复杂问题上展现更人类化的解决方案

## 目录结构

```
cognitive_test/
├── data/                # 数据存储目录
├── scripts/             # 实验脚本
├── src/                 # 源代码
└── README.md            # 项目说明
```

## 使用方法

请按照以下步骤运行实验：

1. 准备环境：`bash scripts/setup_env.sh`
2. 生成数据集：`bash scripts/generate_data.sh`
3. 模型微调：`bash scripts/run_sft.sh`
4. 强化学习：`bash scripts/run_ppo.sh`
5. 行为评估：`bash scripts/run_eval.sh`

## 注意事项

- 本实验需要较大的计算资源，建议在多GPU环境下运行
- 由于本地没有大模型，脚本中使用了mock数据进行流程演示
- 实际部署时，请替换为真实的模型和数据路径