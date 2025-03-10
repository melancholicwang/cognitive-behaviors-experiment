#!/usr/bin/env python3
# 生成包含不同认知行为的思维链数据

import json
import argparse
import os
import random
from typing import List, Dict, Any
import uuid

# 模拟API调用，实际使用时替换为真实API
def mock_api_call(prompt: str, system_prompt: str) -> str:
    """
    模拟API调用，生成思维链回答
    
    参数:
        prompt: 用户提示
        system_prompt: 系统提示
        
    返回:
        生成的回答
    """
    # 从prompt中提取数字和目标
    numbers_str = prompt.split("我有以下数字：")[1].split("。")[0]
    numbers = eval(numbers_str)
    target_str = prompt.split("目标数：")[1].split("。")[0]
    target = int(target_str)
    
    # 根据system_prompt的不同，生成不同类型的思维链
    if "DO NOT backtrack" in system_prompt:
        # 负面对照组 - 直接给出答案
        return generate_negative_control(numbers, target)
    elif "backtracking" in system_prompt and "verification" in system_prompt:
        # 回溯+验证
        return generate_backtracking_verification(numbers, target)
    elif "backtracking" in system_prompt and "backward" in system_prompt:
        # 回溯+逆向推理
        return generate_backtracking_backward(numbers, target)
    elif "backtracking" in system_prompt and "subgoal" in system_prompt:
        # 回溯+子目标
        return generate_backtracking_subgoal(numbers, target)
    elif "backtracking" in system_prompt:
        # 仅回溯
        return generate_only_backtracking(numbers, target)
    else:
        # 所有策略
        return generate_all_strategies(numbers, target)

def generate_negative_control(numbers: List[int], target: int) -> str:
    """生成负面对照组的思维链"""
    # 简单计算，不包含任何认知行为
    result = f"<think> 步骤1: {numbers[0]} + {numbers[1]} = {numbers[0] + numbers[1]}. </think>\n"
    result += f"<answer> {numbers[0]} + {numbers[1]} </answer>"
    return result

def generate_only_backtracking(numbers: List[int], target: int) -> str:
    """生成仅包含回溯行为的思维链"""
    result = "<think>\n"
    result += f"步骤1: {numbers[0]} + {numbers[1]} = {numbers[0] + numbers[1]}.\n"
    result += f"步骤2: 尝试其他操作。{numbers[0]} * {numbers[1]} = {numbers[0] * numbers[1]}.\n"
    
    if len(numbers) > 2:
        result += f"步骤3: 让我尝试另一种方法。{numbers[0]} + {numbers[2]} = {numbers[0] + numbers[2]}.\n"
        result += f"步骤4: 再试一次。{numbers[0]} * {numbers[2]} = {numbers[0] * numbers[2]}.\n"
    
    result += "</think>\n"
    result += f"<answer> {numbers[0]} * {numbers[1]} </answer>"
    return result

def generate_backtracking_verification(numbers: List[int], target: int) -> str:
    """生成包含回溯和验证行为的思维链"""
    result = "<think>\n"
    result += f"步骤1: {numbers[0]} + {numbers[1]} = {numbers[0] + numbers[1]}.\n"
    result += f"步骤2: {numbers[0] + numbers[1]} 不等于 {target}，所以这个方法不行。\n"
    result += f"步骤3: 尝试 {numbers[0]} * {numbers[1]} = {numbers[0] * numbers[1]}.\n"
    
    if numbers[0] * numbers[1] == target:
        result += f"步骤4: {numbers[0] * numbers[1]} 等于 {target}，所以我们找到了答案！\n"
    else:
        result += f"步骤4: {numbers[0] * numbers[1]} 不等于 {target}，继续尝试。\n"
        if len(numbers) > 2:
            result += f"步骤5: 尝试 {numbers[0]} + {numbers[2]} = {numbers[0] + numbers[2]}.\n"
            result += f"步骤6: {numbers[0] + numbers[2]} 不等于 {target}，继续尝试。\n"
    
    result += "</think>\n"
    result += f"<answer> {numbers[0]} * {numbers[1]} </answer>"
    return result

def generate_backtracking_backward(numbers: List[int], target: int) -> str:
    """生成包含回溯和逆向推理行为的思维链"""
    result = "<think>\n"
    result += f"步骤1: {numbers[0]} + {numbers[1]} = {numbers[0] + numbers[1]}.\n"
    result += f"步骤2: 这个方法不行，让我从目标反向思考。\n"
    result += f"步骤3: 如果我想得到 {target}，可以尝试 {target} / {numbers[1]} = {target / numbers[1]}.\n"
    
    if target / numbers[1] == numbers[0]:
        result += f"步骤4: 我们需要 {numbers[0]}，刚好是我们的起始数字之一！\n"
        result += f"步骤5: 所以 {numbers[0]} * {numbers[1]} = {target}.\n"
    else:
        result += f"步骤4: 我们需要 {target / numbers[1]}，但这不是我们的起始数字。\n"
        result += f"步骤5: 让我尝试其他方法。{numbers[0]} * {numbers[1]} = {numbers[0] * numbers[1]}.\n"
    
    result += "</think>\n"
    result += f"<answer> {numbers[0]} * {numbers[1]} </answer>"
    return result

def generate_backtracking_subgoal(numbers: List[int], target: int) -> str:
    """生成包含回溯和子目标设定行为的思维链"""
    result = "<think>\n"
    result += f"步骤1: 我们的目标是 {target}。让我先设定一个子目标，尝试得到 {target // 2}，因为它是 {target} 的一半。\n"
    
    if target // 2 in numbers:
        result += f"步骤2: 我们已经有 {target // 2} 作为起始数字！\n"
        result += f"步骤3: 现在我们需要再得到 {target // 2}，然后将它们相加。\n"
    else:
        result += f"步骤2: 让我尝试得到 {target // 2}。{numbers[0]} + {numbers[1]} = {numbers[0] + numbers[1]}.\n"
        
        if numbers[0] + numbers[1] == target // 2:
            result += f"步骤3: 太好了，{numbers[0]} + {numbers[1]} = {target // 2}，这是我们的子目标。\n"
        else:
            result += f"步骤3: {numbers[0] + numbers[1]} 不等于 {target // 2}，这个子目标可能不适合。\n"
            result += f"步骤4: 让我设定另一个子目标，尝试得到 {target * 2}，然后除以2。\n"
    
    result += f"步骤5: 直接尝试 {numbers[0]} * {numbers[1]} = {numbers[0] * numbers[1]}.\n"
    
    result += "</think>\n"
    result += f"<answer> {numbers[0]} * {numbers[1]} </answer>"
    return result

def generate_all_strategies(numbers: List[int], target: int) -> str:
    """生成包含所有认知行为的思维链"""
    result = "<think>\n"
    result += f"步骤1: 我们的目标是 {target}。让我先设定一个子目标，尝试得到 {target // 2}，因为它是 {target} 的一半。\n"
    result += f"步骤2: {numbers[0]} + {numbers[1]} = {numbers[0] + numbers[1]}.\n"
    result += f"步骤3: {numbers[0] + numbers[1]} 不等于 {target}，所以这个方法不行。\n"
    result += f"步骤4: 让我从目标反向思考。如果我想得到 {target}，可以尝试 {target} / {numbers[1]} = {target / numbers[1]}.\n"
    
    if target / numbers[1] == numbers[0]:
        result += f"步骤5: 我们需要 {numbers[0]}，刚好是我们的起始数字之一！\n"
        result += f"步骤6: 所以 {numbers[0]} * {numbers[1]} = {target}.\n"
        result += f"步骤7: 验证：{numbers[0]} * {numbers[1]} = {numbers[0] * numbers[1]} = {target}，正确！\n"
    else:
        result += f"步骤5: 我们需要 {target / numbers[1]}，但这不是我们的起始数字。\n"
        result += f"步骤6: 让我尝试其他方法。{numbers[0]} * {numbers[1]} = {numbers[0] * numbers[1]}.\n"
        result += f"步骤7: 验证：{numbers[0]} * {numbers[1]} = {numbers[0] * numbers[1]}，这不等于 {target}。\n"
        result += f"步骤8: 继续尝试其他组合...\n"
    
    result += "</think>\n"
    result += f"<answer> {numbers[0]} * {numbers[1]} </answer>"
    return result

def generate_dataset(
    input_file: str,
    output_file: str,
    dataset_type: str,
    num_samples: int = 1000
):
    """
    生成包含特定认知行为的数据集
    
    参数:
        input_file: 输入文件路径
        output_file: 输出文件路径
        dataset_type: 数据集类型
        num_samples: 样本数量
    """
    # 系统提示映射
    system_prompts = {
        "negative_control": """I want to produce reasoning trajectories for the game of countdown. The goal here is to reach a target number by combining integers using basic arithmetic operations.
The answer is a series of arithmetic operations (+, -, *, /) that results in the target number.
Write your answers in <think> </think> tags. Your thinking should be of the form <think> Step X: number1 (+,-,*,/) number2 = result. </think>
Write the final answer in <answer> </answer> tags.
For the final answer, make sure that each step in the final answer is written as <answer> (number1 [+-*/] number2) [+-*/] number3 </answer>.
Answer should be a valid mathematical expression ONLY containing starting integers and NOT the target number. 
Otherwise, the grader will not be able to parse your answer.
Here are some rules:
- DO NOT backtrack, directly give me the correct answer.
- DO NOT work backwards from the goal.
- DO NOT explicitly verify your answer.
- DIRECTLY give me the solution.
- DO NOT produce any other outputs.""",
        
        "only_backtracking": """I want to produce reasoning trajectories for the game of countdown. The goal here is to reach a target number by combining integers using basic arithmetic operations.
Write your thoughts in <think> </think> tags.
The answer is a series of arithmetic operations (+, -, *, /) that results in the target number.
Write the final answer in <answer> </answer> tags.
For the final answer, make sure that each step in the final answer is written as <answer> (number1 [+-*/] number2) [+-*/] number3 </answer>.
Answer should be a valid mathematical expression ONLY containing starting integers and NOT the target number.
Otherwise, the grader will not be able to parse your answer.
Backtrack to the start or an intermediate step if you haven't reached the answer.
- DO NOT set subgoals.
- DO NOT work backwards from the goal.
- DO NOT explicitly verify your answer.""",
        
        "backtracking_verification": """I want to produce reasoning trajectories for the game of countdown. The goal here is to reach a target number by combining integers using basic arithmetic operations.
Write your thoughts in <think> </think> tags.
The answer is a series of arithmetic operations (+, -, *, /) that results in the target number.
Write the final answer in <answer> </answer> tags.
For the final answer, make sure that each step in the final answer is written as <answer> (number1 [+-*/] number2) [+-*/] number3 </answer>.
Answer should be a valid mathematical expression ONLY containing starting integers and NOT the target number.
Otherwise, the grader will not be able to parse your answer.
Verify that you have reached the answer and backtrack to the start or an intermediate step. DO NOT set subgoals or work backwards from the goal.""",
        
        "backtracking_backward": """I want to produce reasoning trajectories for the game of countdown. The goal here is to reach a target number by combining integers using basic arithmetic operations.
Write your thoughts in <think> </think> tags.
The answer is a series of arithmetic operations (+, -, *, /) that results in the target number.
Write the final answer in <answer> </answer> tags.
For the final answer, make sure that each step in the final answer is written as <answer> (number1 [+-*/] number2) [+-*/] number3 </answer>.
Answer should be a valid mathematical expression ONLY containing starting integers and NOT the target number.
Otherwise, the grader will not be able to parse your answer.
Backtrack to the start or an intermediate step if you haven't reached the answer. Work backwards from the goal if it makes things easier.
- DO NOT set subgoals.
- DO NOT explicitly verify your answer.""",
        
        "backtracking_subgoal": """I want to produce reasoning trajectories for the game of countdown. The goal here is to reach a target number by combining integers using basic arithmetic operations.
Write your thoughts in <think> </think> tags.
The answer is a series of arithmetic operations (+, -, *, /) that results in the target number.
Write the final answer in <answer> </answer> tags.
For the final answer, make sure that each step in the final answer is written as <answer> (number1 [+-*/] number2) [+-*/] number3 </answer>.
Answer should be a valid mathematical expression ONLY containing starting integers and NOT the target number.
Otherwise, the grader will not be able to parse your answer.
Decompose the answer into sub-goals and try to reach them to then reach the target, if you are unable to reach the goal or a subgoal backtrack to a previous state.
- DO NOT work backwards from the goal.
- DO NOT explicitly verify your answer.""",
        
        "all_strategies": """I want to produce reasoning trajectories for the game of countdown. The goal here is to reach a target number by combining integers using basic arithmetic operations.
Write your thoughts in <think> </think> tags.
The answer is a series of arithmetic operations (+, -, *, /) that results in the target number.
Write the final answer in <answer> </answer> tags.
For the final answer, make sure that each step in the final answer is written as <answer> (number1 [+-*/] number2) [+-*/] number3 </answer>.
Answer should be a valid mathematical expression ONLY containing starting integers and NOT the target number.
Otherwise, the grader will not be able to parse your answer.
- Verify that you have reached the answer and backtrack to the start or an intermediate step.
- Work backwards from the goal if it makes things easier.
- Decompose the answer into sub-goals and try to reach them to then reach the target, if you are unable to reach the goal or a subgoal backtrack to a previous state."""
    }
    
    # 确保数据集类型有效
    if dataset_type not in system_prompts:
        raise ValueError(f"无效的数据集类型: {dataset_type}。有效类型: {list(system_prompts.keys())}")
    
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # 如果数据量不足，则重复使用
    if len(data) < num_samples:
        data = (data * (num_samples // len(data) + 1))[:num_samples]
    # 如果数据量过多，则随机采样
    elif len(data) > num_samples:
        data = random.sample(data, num_samples)
    
    # 生成数据集
    system_prompt = system_prompts[dataset_type]
    output_data = []
    
    for item in data:
        prompt = item["query"]
        completion = mock_api_call(prompt, system_prompt)
        
        output_data.append({
            "query": prompt,
            "completion": completion
        })
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"已生成{len(output_data)}个样本到{output_file}")

def generate_empty_cot(input_file: str, output_file: str, num_samples: int = 1000):
    """
    生成空思维链数据集
    
    参数:
        input_file: 输入文件路径
        output_file: 输出文件路径
        num_samples: 样本数量
    """
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # 如果数据量不足，则重复使用
    if len(data) < num_samples:
        data = (data * (num_samples // len(data) + 1))[:num_samples]
    # 如果数据量过多，则随机采样
    elif len(data) > num_samples:
        data = random.sample(data, num_samples)
    
    # 生成数据集
    output_data = []
    
    for item in data:
        prompt = item["query"]
        numbers_str = prompt.split("我有以下数字：")[1].split("。")[0]
        numbers = eval(numbers_str)
        
        # 生成空思维链
        completion = "<think></think>\n"
        completion += f"<answer> {numbers[0]} * {numbers[1]} </answer>"
        
        output_data.append({
            "query": prompt,
            "completion": completion
        })
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"已生成{len(output_data)}个空思维链样本到{output_file}")

def generate_no_positive(input_file: str, output_file: str, num_samples: int = 1000):
    """
    生成只包含错误答案的数据集
    
    参数:
        input_file: 输入文件路径
        output_file: 输出文件路径
        num_samples: 样本数量
    """
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # 如果数据量不足，则重复使用
    if len(data) < num_samples:
        data = (data * (num_samples // len(data) + 1))[:num_samples]
    # 如果数据量过多，则随机采样
    elif len(data) > num_samples:
        data = random.sample(data, num_samples)
    
    # 生成数据集
    output_data = []
    
    for item in data:
        prompt = item["query"]
        completion = item["completion"]
        
        # 修改答案部分，使其不正确
        answer_start = completion.find("<answer>")
        answer_end = completion.find("</answer>")
        
        if answer_start != -1 and answer_end != -1:
            answer_text = completion[answer_start+8:answer_end].strip()
            # 修改答案，例如将加法改为减法
            if "+" in answer_text:
                new_answer = answer_text.replace("+", "-")
            elif "*" in answer_text:
                new_answer = answer_text.replace("*", "/")
            else:
                new_answer = answer_text.replace("-", "+")
            
            new_completion = completion[:answer_start+8] + " " + new_answer + " " + completion[answer_end:]
            
            output_data.append({
                "query": prompt,
                "completion": new_completion
            })
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"已生成{len(output_data)}个无正确答案样本到{output_file}")

def process_to_parquet(input_file: str, output_dir: str, train_ratio: float = 0.8):
    """
    将JSONL文件处理为Parquet格式，并分割为训练集和测试集
    
    参数:
        input_file: 输入文件路径
        output_dir: 输出目录
        train_ratio: 训练集比例
    """
    try:
        import pandas as pd
    except ImportError:
        print("请安装pandas: pip install pandas pyarrow")
        return
    
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    
    # 分割为训练集和测试集
    train_size = int(len(df) * train_ratio)
    train_df = df[:train_size]
    test_df = df[train_size:]
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 写入Parquet文件
    train_df.to_parquet(os.path.join(output_dir, "train.parquet"), index=False)
    test_df.to_parquet(os.path.join(output_dir, "test.parquet"), index=False)
    
    print(f"已处理{len(df)}个样本，分割为{len(train_df)}个训练样本和{len(test_df)}个测试样本")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='生成包含不同认知行为的思维链数据')
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # 生成数据集命令
    gen_parser = subparsers.add_parser('generate', help='生成数据集')
    gen_parser.add_argument('--input_file', type=str, required=True, help='输入文件路径')
    gen_parser.add_argument('--output_file', type=str, required=True, help='输出文件路径')
    gen_parser.add_argument('--dataset_type', type=str, required=True, 
                           choices=['negative_control', 'only_backtracking', 'backtracking_verification', 
                                   'backtracking_backward', 'backtracking_subgoal', 'all_strategies'],
                           help='数据集类型')
    gen_parser.add_argument('--num_samples', type=int, default=1000, help='样本数量')
    
    # 生成空思维链命令
    empty_parser = subparsers.add_parser('empty', help='生成空思维链数据集')
    empty_parser.add_argument('--input_file', type=str, required=True, help='输入文件路径')
    empty_parser.add_argument('--output_file', type=str, required=True, help='输出文件路径')
    empty_parser.add_argument('--num_samples', type=int, default=1000, help='样本数量')
    
    # 生成无正确答案命令
    no_pos_parser = subparsers.add_parser('no_positive', help='生成无正确答案数据集')
    no_pos_parser.add_argument('--input_file', type=str, required=True, help='输入文件路径')
    no_pos_parser.add_argument('--output_file', type=str, required=True, help='输出文件路径')
    no_pos_parser.add_argument('--num_samples', type=int, default=1000, help='样本数量')
    
    # 处理为Parquet命令
    parquet_parser = subparsers.add_parser('to_parquet', help='处理为Parquet格式')
    parquet_parser.add_argument('--input_file', type=str, required=True, help='输入文件路径')
    parquet_parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parquet_parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    
    args = parser.parse_args()
    
    if args.command == 'generate':
        generate_dataset(
            input_file=args.input_file,
            output_file=args.output_file,
            dataset_type=args.dataset_type,
            num_samples=args.num_samples
        )
    elif args.command == 'empty':
        generate_empty_cot(
            input_file=args.input_file,
            output_file=args.output_file,
            num_samples=args.num_samples
        )
    elif args.command == 'no_positive':
        generate_no_positive(
            input_file=args.input_file,
            output_file=args.output_file,
            num_samples=args.num_samples
        )
    elif args.command == 'to_parquet':
        process_to_parquet(
            input_file=args.input_file,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio
        )
    else:
        parser.print_help()