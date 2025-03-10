#!/usr/bin/env python3
# 行为评估脚本

import os
import json
import argparse
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import logging
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Tuple
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BehaviorEvaluator:
    """行为评估器"""
    
    def __init__(self, model_path: str, api_key: str = None):
        """
        初始化行为评估器
        
        参数:
            model_path: 模型路径
            api_key: API密钥（用于GPT-4o-mini评估）
        """
        self.model_path = model_path
        self.api_key = api_key
        
        # 加载模型和分词器
        logger.info(f"加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 如果提供了API密钥，则初始化OpenAI客户端
        if api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
                self.use_api = True
            except ImportError:
                logger.warning("未安装openai库，将使用规则匹配进行评估")
                self.use_api = False
        else:
            self.use_api = False
    
    def generate_response(self, prompt: str) -> str:
        """
        生成回答
        
        参数:
            prompt: 提示
            
        返回:
            生成的回答
        """
        inputs = self.tokenizer(f"用户: {prompt}", return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response
    
    def evaluate_behaviors_with_api(self, prompt: str, response: str, numbers: List[int], target: int) -> Dict[str, int]:
        """
        使用API评估行为
        
        参数:
            prompt: 提示
            response: 回答
            numbers: 数字列表
            target: 目标数
            
        返回:
            行为计数字典
        """
        # 创建评估提示
        prompts = [
            # 1. 验证行为
            f"""Here is a chain-of-reasoning that a Language Model generated while trying to play the game of CountDown with the numbers {numbers}. The goal is to reach the target number {target}. The chain-of-reasoning the model used is: {response}. 
Evaluate whether the chain-of-reasoning contains any answer-verification steps. An example of an answer-verification step is: 'This sequence results in 1, which is not equal to 22' and 'Since 25 is not equal to 22' for explicit verification and 'Too high!' or 'This works!' for implicit verification. We want to mark instances where the chain-of-reasoning explicitly checks the current result against the target number. 
If you find any answer-verification steps, please count them and provide the count as between the tags <count> </count>. If the chain-of-reasoning does not contain any answer-verification steps, please provide a count of 0 as <count>0</count>.""",

            # 2. 回溯行为
            f"""Here is a chain-of-reasoning that a Language Model generated while trying to play the game of CountDown with the numbers {numbers}. The goal is to reach the target number {target}. The chain-of-reasoning the model used is: {response}.
Evaluate whether the chain-of-reasoning contains any backtracking behavior, where the model realizes a path won't work and explicitly goes back to try a different approach. Due to the nature of the problem, any attempt at a new combination of numbers that does not directly use the result from the previous computation is considered backtracking. 
Count the number of distinct backtracking instances and provide the count between the tags <count> </count>. If the chain-of-reasoning does not contain any backtracking behavior, please provide a count of 0 as <count>0</count>.""",

            # 3. 子目标设定
            f"""Here is a chain-of-reasoning that a Language Model generated while trying to play the game of CountDown with the numbers {numbers}. The goal is to reach the target number {target}. The chain-of-reasoning the model used is: {response}.
Evaluate whether the chain-of-reasoning contains any explicit subgoal setting, where the model breaks down the problem into smaller, intermediate goals. An example of subgoal setting is: "First, I'll try to get close to {target//2}, then...".
Count the number of distinct subgoals set and provide the count between the tags <count> </count>. If the chain-of-reasoning does not contain any subgoal setting, please provide a count of 0 as <count>0</count>.""",

            # 4. 逆向推理
            f"""Here is a chain-of-reasoning that a Language Model generated while trying to play the game of CountDown with the numbers {numbers}. The goal is to reach the target number {target}. The chain-of-reasoning the model used is: {response}.
Evaluate whether the chain-of-reasoning contains any backward-chaining behavior, where the model starts from the target number and works backwards to the initial numbers. An example of backward-chaining when the target is 24 and the numbers are 12 and 2 is: "Let's work backwards from the target. 24/2 = 12. So, 12*2=24." and if the target is 22 and the numbers are 25 and 3 is: "Since the target is 22, and 22 + 3 = 25, ...".
Count the number of distinct backward-chaining instances and provide the count between the tags <count> </count>. If the chain-of-reasoning does not contain any backward-chaining behavior, please provide a count of 0 as <count>0</count>."""
        ]
        
        # 评估行为
        behavior_counts = {
            "verification": 0,
            "backtracking": 0,
            "subgoal": 0,
            "backward": 0
        }
        
        for i, prompt_text in enumerate(prompts):
            try:
                response_obj = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that analyzes mathematical reasoning."},
                        {"role": "user", "content": prompt_text}
                    ],
                    max_tokens=512,
                )
                
                response_text = response_obj.choices[0].message.content
                count_match = re.search(r'<count>(\d+)</count>', response_text)
                count = int(count_match.group(1)) if count_match else 0
                
                if i == 0:
                    behavior_counts["verification"] = count
                elif i == 1:
                    behavior_counts["backtracking"] = count
                elif i == 2:
                    behavior_counts["subgoal"] = count
                elif i == 3:
                    behavior_counts["backward"] = count
            
            except Exception as e:
                logger.error(f"API调用失败: {str(e)}")
        
        return behavior_counts
    
    def evaluate_behaviors_with_rules(self, response: str) -> Dict[str, int]:
        """
        使用规则匹配评估行为
        
        参数:
            response: 回答
            
        返回:
            行为计数字典
        """
        # 提取思维链部分
        think_pattern = r"<think>(.*?)</think>"
        think_match = re.search(think_pattern, response, re.DOTALL)
        
        if not think_match:
            return {
                "verification": 0,
                "backtracking": 0,
                "subgoal": 0,
                "backward": 0
            }
        
        think_content = think_match.group(1)
        
        # 计数各种行为
        # 1. 验证行为 - 检查结果是否等于目标
        verification_count = len(re.findall(r'不等于|等于|验证|检查|确认', think_content))
        
        # 2. 回溯行为 - 放弃当前路径，尝试新方法
        backtracking_count = len(re.findall(r'尝试其他|尝试另一种|再试一次|这个方法不行|不行，让我|换一种', think_content))
        
        # 3. 子目标设定 - 分解问题
        subgoal_count = len(re.findall(r'子目标|首先.*然后|先.*再|分解|中间步骤', think_content))
        
        # 4. 逆向推理 - 从目标向初始状态推理
        backward_count = len(re.findall(r'从目标反向|反向思考|如果我想得到.*可以|从.*推导', think_content))
        
        return {
            "verification": verification_count,
            "backtracking": backtracking_count,
            "subgoal": subgoal_count,
            "backward": backward_count
        }
    
    def evaluate_dataset(self, dataset_path: str, output_dir: str, num_samples: int = 100) -> Dict[str, Any]:
        """
        评估数据集
        
        参数:
            dataset_path: 数据集路径
            output_dir: 输出目录
            num_samples: 样本数量
            
        返回:
            评估结果
        """
        # 加载数据集
        logger.info(f"加载数据集: {dataset_path}")
        dataset = load_dataset("parquet", data_files=dataset_path)["train"]
        
        # 如果数据集太大，随机采样
        if len(dataset) > num_samples:
            dataset = dataset.shuffle(seed=42).select(range(num_samples))
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 评估结果
        results = []
        
        # 评估每个样本
        for i, example in enumerate(dataset):
            logger.info(f"评估样本 {i+1}/{len(dataset)}")
            
            prompt = example["query"]
            
            # 提取数字和目标
            numbers_str = prompt.split("我有以下数字：")[1].split("。")[0]
            numbers = eval(numbers_str)
            target_str = prompt.split("目标数：")[1].split("。")[0]
            target = int(target_str)
            
            # 生成回答
            response = self.generate_response(prompt)
            
            # 评估行为
            if self.use_api:
                behavior_counts = self.evaluate_behaviors_with_api(prompt, response, numbers, target)
            else:
                behavior_counts = self.evaluate_behaviors_with_rules(response)
            
            # 保存结果
            result = {
                "prompt": prompt,
                "response": response,
                "numbers": numbers,
                "target": target,
                "verification_count": behavior_counts["verification"],
                "backtracking_count": behavior_counts["backtracking"],
                "subgoal_count": behavior_counts["subgoal"],
                "backward_count": behavior_counts["backward"]
            }
            
            results.append(result)
        
        # 计算平均值
        avg_verification = sum(r["verification_count"] for r in results) / len(results)
        avg_backtracking = sum(r["backtracking_count"] for r in results) / len(results)
        avg_subgoal = sum(r["subgoal_count"] for r in results) / len(results)
        avg_backward = sum(r["backward_count"] for r in results) / len(results)
        
        # 保存详细结果
        with open(os.path.join(output_dir, "detailed_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 保存汇总结果
        summary = {
            "avg_verification": avg_verification,
            "avg_backtracking": avg_backtracking,
            "avg_subgoal": avg_subgoal,
            "avg_backward": avg_backward,
            "total_verification": sum(r["verification_count"] for r in results),
            "total_backtracking": sum(r["backtracking_count"] for r in results),
            "total_subgoal": sum(r["subgoal_count"] for r in results),
            "total_backward": sum(r["backward_count"] for r in results),
            "num_samples": len(results)
        }
        
        with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # 绘制图表
        self.plot_results(summary, os.path.join(output_dir, "behavior_counts.png"))
        
        return summary
    
    def evaluate_checkpoints(self, checkpoints_dir: str, dataset_path: str, output_dir: str, num_samples: int = 100) -> Dict[str, Any]:
        """
        评估多个检查点
        
        参数:
            checkpoints_dir: 检查点目录
            dataset_path: 数据集路径
            output_dir: 输出目录
            num_samples: 样本数量
            
        返回:
            评估结果
        """
        # 获取所有检查点
        checkpoints = []
        for item in os.listdir(checkpoints_dir):
            if item.startswith("checkpoint-") or item.startswith("global_step_"):
                checkpoint_path = os.path.join(checkpoints_dir, item)
                if os.path.isdir(checkpoint_path):
                    # 提取步数
                    step = int(re.search(r'(\d+)$', item).group(1))
                    checkpoints.append((step, checkpoint_path))
        
        # 按步数排序
        checkpoints.sort(key=lambda x: x[0])
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 评估结果
        all_results = {}
        
        # 评估每个检查点
        for step, checkpoint_path in checkpoints:
            logger.info(f"评估检查点: {checkpoint_path}")
            
            # 更新模型
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # 评估数据集
            checkpoint_output_dir = os.path.join(output_dir, f"step_{step}")
            summary = self.evaluate_dataset(dataset_path, checkpoint_output_dir, num_samples)
            
            all_results[step] = summary
        
        # 绘制趋势图
        self.plot_trends(all_results, os.path.join(output_dir, "behavior_trends.png"))
        
        # 保存所有结果
        with open(os.path.join(output_dir, "all_results.json"), "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        return all_results
    
    def plot_results(self, summary: Dict[str, Any], output_path: str):
        """
        绘制结果图表
        
        参数:
            summary: 汇总结果
            output_path: 输出路径
        """
        behaviors = ["verification", "backtracking", "subgoal", "backward"]
        avg_counts = [summary[f"avg_{b}"] for b in behaviors]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(behaviors, avg_counts, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        
        plt.title("平均认知行为计数")
        plt.ylabel("平均计数")
        plt.ylim(0, max(avg_counts) * 1.2)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f"{height:.2f}", ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    
    def plot_trends(self, all_results: Dict[int, Dict[str, Any]], output_path: str):
        """
        绘制趋势图
        
        参数:
            all_results: 所有结果
            output_path: 输出路径
        """
        steps = sorted(all_results.keys())
        behaviors = ["verification", "backtracking", "subgoal", "backward"]
        
        plt.figure(figsize=(12, 8))
        
        for behavior in behaviors:
            values = [all_results[step][f"avg_{behavior}"] for step in steps]
            plt.plot(steps, values, marker='o', label=behavior)
        
        plt.title("认知行为趋势")
        plt.xlabel("训练步数")
        plt.ylabel("平均计数")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='行为评估脚本')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--dataset_path', type=str, required=True, help='数据集路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--api_key', type=str, help='OpenAI API密钥')
    parser.add_argument('--num_samples', type=int, default=100, help='样本数量')
    parser.add_argument('--mode', type=str, choices=['single', 'checkpoints'], default='single', help='评估模式')
    parser.add_argument('--checkpoints_dir', type=str, help='检查点目录（仅在checkpoints模式下使用）')
    
    args = parser.parse_args()
    
    evaluator = BehaviorEvaluator(args.model_path, args.api_key)
    
    if args.mode == 'single':
        evaluator.evaluate_dataset(args.dataset_path, args.output_dir, args.num_samples)
    else:
        evaluator.evaluate_checkpoints(args.checkpoints_dir, args.dataset_path, args.output_dir, args.num_samples)

if __name__ == "__main__":
    main()