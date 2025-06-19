#!/usr/bin/env python3
import re
import os
from tqdm import tqdm

def generate_optimizer_configs():
    """
    生成所有optimizer配置，格式：[type, momentum1, momentum2]
    """
    configs = []
    
    # 1. default optimizer
    configs.append(["default", 0.95, None])
    
    # 2. double_momentum optimizer (双层循环)
    start0, end0, step0 = 0.85, 0.99, 0.02
    start1, end1, step1 = 0.93, 0.99, 0.02
    
    # 生成m0的值列表
    m0_values = []
    m0 = start0
    while m0 <= end0 + 1e-10:  # 加小数避免浮点精度问题
        m0_values.append(round(m0, 2))
        m0 += step0
    
    # 生成m1的值列表  
    m1_values = []
    m1 = start1
    while m1 <= end1 + 1e-10:
        m1_values.append(round(m1, 2))
        m1 += step1
    
    for m0 in m0_values:
        for m1 in m1_values:
            configs.append(["double_momentum", m0, m1])
    
    # 3. svd_momentum_v2 optimizer (单层循环)
    start, end, step = 0.85, 0.99, 0.02
    m_values = []
    m = start
    while m <= end + 1e-10:
        m_values.append(round(m, 2))
        m += step
    
    for m in m_values:
        configs.append(["svd_momentum_v2", m, None])
    
    # 4. mix optimizer (双层循环，参数范围同double_momentum)
    for m0 in m0_values:
        for m1 in m1_values:
            configs.append(["mix", m0, m1])
    
    return configs

def find_final_val_losses(filename):
    """
    查找stdout.txt文件中以step:10200/10200开头的行，提取val_loss数值
    """
    optimizer_configs = generate_optimizer_configs()
    
    print(f"生成的optimizer配置总数: {len(optimizer_configs)}")
    print("前几个配置示例:")
    for i, config in enumerate(optimizer_configs[:5]):
        print(f"  {i+1}: {config}")
    print("...")
    print()
    
    if not os.path.exists(filename):
        print(f"错误：文件 {filename} 不存在")
        return []
    
    print(f"正在搜索文件: {filename}")
    print("查找以 'step:10200/10200' 开头的行...")
    print("=" * 60)
    
    final_results = []
    
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in tqdm(enumerate(f, 1)):
            line = line.strip()
            
            # 检查是否以step:10200/10200开头
            if line.startswith("step:10200/10200"):
                # 使用正则表达式提取val_loss的值
                match = re.search(r"step:10200/10200.*?val_loss:([\d.]+)", line)
                if match:
                    val_loss = float(match.group(1))
                    final_results.append({
                        'line_number': line_num,
                        'val_loss': val_loss,
                        'full_line': line
                    })
                    print(f"行 {line_num}: val_loss = {val_loss}")
                    print(f"完整行: {line}")
                    print("-" * 40)
    
    return final_results, optimizer_configs

def main():
    filename = "stdout.txt"
    
    print("最终Val Loss提取器")
    print("=" * 60)
    
    results, optimizer_configs = find_final_val_losses(filename)
    
    for i in range(len(optimizer_configs)):
        print(f"optimizer_configs[{i}]: {optimizer_configs[i]}")
        if 2*i+1 < len(results):
            print(f"results[{2*i}]: {results[2*i]}")
            print("\n")
            #print(f"results[{2*i+1}]: {results[2*i+1]}")
        
    '''
    if results:
        print(f"\n找到 {len(results)} 个匹配的最终结果:")
        print("=" * 60)
        
        for idx, result in enumerate(results, 1):
            print(f"结果 {idx}:")
            print(f"  行号: {result['line_number']}")
            print(f"  最终val_loss: {result['val_loss']:.4f}")
            print(f"  完整行: {result['full_line']}")
            
            # 如果结果数量与配置数量匹配，显示对应的配置
            if len(results) == len(optimizer_configs):
                config = optimizer_configs[idx-1]
                config_str = f"{config[0]}"
                if config[1] is not None:
                    config_str += f", momentum_0={config[1]}"
                if config[2] is not None:
                    config_str += f", momentum_1={config[2]}"
                print(f"  对应配置: {config_str}")
            print()
        
        # 如果有多个结果，显示汇总
        if len(results) > 1:
            print("汇总:")
            print("-" * 30)
            val_losses = [r['val_loss'] for r in results]
            print(f"所有最终val_loss值: {val_losses}")
            print(f"最佳val_loss: {min(val_losses):.4f}")
            print(f"最差val_loss: {max(val_losses):.4f}")
            print(f"平均val_loss: {sum(val_losses)/len(val_losses):.4f}")
            
            # 找到最佳配置
            if len(results) == len(optimizer_configs):
                best_idx = val_losses.index(min(val_losses))
                best_config = optimizer_configs[best_idx]
                config_str = f"{best_config[0]}"
                if best_config[1] is not None:
                    config_str += f", momentum_0={best_config[1]}"
                if best_config[2] is not None:
                    config_str += f", momentum_1={best_config[2]}"
                print(f"最佳配置: {config_str} (val_loss: {min(val_losses):.4f})")
        
    else:
        print("未找到任何以 'step:10200/10200' 开头的行")
    '''
if __name__ == "__main__":
    main() 