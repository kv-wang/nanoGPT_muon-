#!/usr/bin/env python3
import re
import os

def parse_stdout_file(filename):
    """
    解析stdout.txt文件，提取每个optimizer的信息和最终loss
    """
    if not os.path.exists(filename):
        print(f"错误：文件 {filename} 不存在")
        return
    
    print(f"正在解析文件: {filename}")
    print("=" * 60)
    
    optimizers = []
    current_optimizer = {}
    
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    print(f"文件总行数: {len(lines)}")
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # 匹配optimizer类型
        if "Muon Type:" in line:
            # 如果已经有当前optimizer，保存它
            if current_optimizer:
                optimizers.append(current_optimizer.copy())
            
            # 开始新的optimizer
            current_optimizer = {}
            match = re.search(r"Muon Type:\s*(\w+)", line)
            if match:
                current_optimizer['type'] = match.group(1)
                current_optimizer['params'] = {}
                current_optimizer['val_losses'] = []
                
                print(f"\n发现Optimizer类型: {current_optimizer['type']}")
        
        # 匹配momentum参数
        elif "Muon Momentum" in line and current_optimizer:
            match = re.search(r"Muon Momentum (\d+):\s*([\d.]+)", line)
            if match:
                param_idx = match.group(1)
                param_value = float(match.group(2))
                current_optimizer['params'][f'momentum_{param_idx}'] = param_value
                print(f"  参数 momentum_{param_idx}: {param_value}")
        
        # 匹配validation loss
        elif "val_loss:" in line and current_optimizer:
            match = re.search(r"step:(\d+)/(\d+)\s+val_loss:([\d.]+)", line)
            if match:
                step = int(match.group(1))
                total_steps = int(match.group(2))
                val_loss = float(match.group(3))
                
                current_optimizer['val_losses'].append({
                    'step': step,
                    'total_steps': total_steps,
                    'val_loss': val_loss
                })
        
        i += 1
    
    # 添加最后一个optimizer
    if current_optimizer:
        optimizers.append(current_optimizer)
    
    return optimizers

def print_results(optimizers):
    """
    打印解析结果
    """
    print("\n" + "=" * 60)
    print("解析结果汇总:")
    print("=" * 60)
    
    for idx, opt in enumerate(optimizers, 1):
        print(f"\n第 {idx} 个Optimizer:")
        print(f"  类型: {opt.get('type', 'Unknown')}")
        
        # 打印参数
        if opt.get('params'):
            print("  参数:")
            for param_name, param_value in opt['params'].items():
                print(f"    {param_name}: {param_value}")
        else:
            print("  参数: 无")
        
        # 打印validation losses统计
        if opt.get('val_losses'):
            val_losses = opt['val_losses']
            print(f"  训练步数: {len(val_losses)} 个记录点")
            
            if val_losses:
                first_loss = val_losses[0]['val_loss']
                final_loss = val_losses[-1]['val_loss']
                final_step = val_losses[-1]['step']
                total_steps = val_losses[-1]['total_steps']
                
                print(f"  初始loss: {first_loss:.4f}")
                print(f"  最终loss: {final_loss:.4f} (步数: {final_step}/{total_steps})")
                print(f"  loss改善: {first_loss - final_loss:.4f}")
        else:
            print("  validation loss: 无记录")
        
        print("-" * 40)

def main():
    filename = "stdout.txt"
    
    print("Optimizer结果解析器")
    print("=" * 60)
    
    optimizers = parse_stdout_file(filename)
    
    if optimizers:
        print_results(optimizers)
        
        # 创建简要汇总
        print("\n" + "=" * 60)
        print("简要汇总:")
        print("=" * 60)
        
        for idx, opt in enumerate(optimizers, 1):
            opt_type = opt.get('type', 'Unknown')
            
            if opt.get('val_losses'):
                final_loss = opt['val_losses'][-1]['val_loss']
                print(f"Optimizer {idx}: {opt_type} -> 最终loss: {final_loss:.4f}")
            else:
                print(f"Optimizer {idx}: {opt_type} -> 无loss记录")
    else:
        print("未找到任何optimizer信息")

if __name__ == "__main__":
    main() 