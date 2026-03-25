import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
from matplotlib.backends.backend_pdf import PdfPages

def parse_log_file(log_path):
    """解析日志文件，提取关键指标"""
    metrics = {
        'rounds': [],
        'personal_acc': [],
        'global_acc': [],
        'ece': [],
        'auroc': [],
        'fpr95': []
    }
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        # 提取轮次和准确率
        if "Round" in line and "Personalized Test Accuracy" in line:
            match = re.search(r'Round (\d+).*Personalized Test Accuracy: ([\d\.]+)%.*Global Test Accuracy: ([\d\.]+)%', line)
            if match:
                round_num = int(match.group(1))
                personal_acc = float(match.group(2))
                global_acc = float(match.group(3))
                metrics['rounds'].append(round_num)
                metrics['personal_acc'].append(personal_acc)
                metrics['global_acc'].append(global_acc)
        
        # 提取ECE
        if "Expected Calibration Error" in line:
            match = re.search(r'Expected Calibration Error: ([\d\.]+)', line)
            if match:
                ece = float(match.group(1))
                metrics['ece'].append(ece)
        
        # 提取AUROC
        if "AUROC for OOD detection" in line:
            match = re.search(r'AUROC for OOD detection: ([\d\.]+)', line)
            if match:
                auroc = float(match.group(1))
                metrics['auroc'].append(auroc)
        
        # 提取FPR@95%TPR
        if "FPR at 95% TPR" in line:
            match = re.search(r'FPR at 95% TPR: ([\d\.]+)', line)
            if match:
                fpr95 = float(match.group(1))
                metrics['fpr95'].append(fpr95)
    
    return metrics

def extract_param_value(dir_name, param_prefix):
    """从目录名提取参数值"""
    match = re.search(f'{param_prefix}_(.+)', dir_name)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return match.group(1)
    return None

def analyze_sensitivity(results_dir, output_file):
    """分析敏感性并生成报告"""
    # 收集所有参数组
    param_groups = {
        'gcw': [],
        'temp': [],
        'edl': [],
        'maxev': []
    }
    
    # 查找所有日志文件
    for param_type in param_groups.keys():
        log_files = glob.glob(f"sensitivity_logs/fedug_v2_{param_type}_*.log")
        for log_file in log_files:
            param_value = extract_param_value(os.path.basename(log_file), param_type)
            if param_value is not None:
                metrics = parse_log_file(log_file)
                if metrics['rounds']:  # 确保有数据
                    # 取最后一轮的结果
                    final_idx = -1
                    param_groups[param_type].append({
                        'param_value': param_value,
                        'personal_acc': metrics['personal_acc'][final_idx],
                        'global_acc': metrics['global_acc'][final_idx],
                        'ece': metrics['ece'][final_idx] if metrics['ece'] else None,
                        'auroc': metrics['auroc'][final_idx] if metrics['auroc'] else None,
                        'fpr95': metrics['fpr95'][final_idx] if metrics['fpr95'] else None
                    })
    
    # 创建PDF报告
    with PdfPages(output_file) as pdf:
        # 标题页
        plt.figure(figsize=(12, 9))
        plt.text(0.5, 0.5, 'FedUG V2 超参数敏感性分析报告', 
                 horizontalalignment='center', verticalalignment='center', 
                 fontsize=24, transform=plt.gca().transAxes)
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        # 为每个参数组创建敏感性分析图
        param_names = {
            'gcw': '全局一致性权重 (Global Consistency Weight)',
            'temp': '温度参数 (Temperature)',
            'edl': 'EDL损失权重 (EDL Weight)',
            'maxev': '最大Evidence值 (Max Evidence)'
        }
        
        for param_type, results in param_groups.items():
            if not results:
                continue
                
            # 排序结果
            results = sorted(results, key=lambda x: x['param_value'])
            param_values = [r['param_value'] for r in results]
            personal_accs = [r['personal_acc'] for r in results]
            global_accs = [r['global_acc'] for r in results]
            eces = [r['ece'] for r in results if r['ece'] is not None]
            aurocs = [r['auroc'] for r in results if r['auroc'] is not None]
            fpr95s = [r['fpr95'] for r in results if r['fpr95'] is not None]
            
            # 创建图表
            fig, axs = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'{param_names[param_type]}敏感性分析', fontsize=16)
            
            # 准确率图
            ax = axs[0, 0]
            ax.plot(param_values, personal_accs, 'o-', label='个性化测试准确率')
            ax.plot(param_values, global_accs, 's--', label='全局测试准确率')
            ax.set_xlabel(param_names[param_type])
            ax.set_ylabel('准确率 (%)')
            ax.set_title('模型准确率')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            # ECE图
            ax = axs[0, 1]
            if eces:
                ax.plot(param_values[:len(eces)], eces, 'o-')
                ax.set_xlabel(param_names[param_type])
                ax.set_ylabel('ECE')
                ax.set_title('期望校准误差 (ECE)')
                ax.grid(True, linestyle='--', alpha=0.7)
            else:
                ax.text(0.5, 0.5, '无ECE数据', ha='center', va='center')
                ax.set_title('期望校准误差 (ECE)')
                ax.axis('off')
            
            # AUROC图
            ax = axs[1, 0]
            if aurocs:
                ax.plot(param_values[:len(aurocs)], aurocs, 'o-')
                ax.set_xlabel(param_names[param_type])
                ax.set_ylabel('AUROC')
                ax.set_title('OOD检测AUROC')
                ax.grid(True, linestyle='--', alpha=0.7)
            else:
                ax.text(0.5, 0.5, '无AUROC数据', ha='center', va='center')
                ax.set_title('OOD检测AUROC')
                ax.axis('off')
            
            # FPR@95%TPR图
            ax = axs[1, 1]
            if fpr95s:
                ax.plot(param_values[:len(fpr95s)], fpr95s, 'o-')
                ax.set_xlabel(param_names[param_type])
                ax.set_ylabel('FPR@95%TPR')
                ax.set_title('FPR at 95% TPR')
                ax.grid(True, linestyle='--', alpha=0.7)
            else:
                ax.text(0.5, 0.5, '无FPR@95%TPR数据', ha='center', va='center')
                ax.set_title('FPR at 95% TPR')
                ax.axis('off')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig)
            plt.close()
            
            # 创建表格
            plt.figure(figsize=(12, 6))
            plt.axis('off')
            
            table_data = []
            headers = ['参数值', '个性化准确率', '全局准确率']
            if eces:
                headers.append('ECE')
            if aurocs:
                headers.append('AUROC')
            if fpr95s:
                headers.append('FPR@95%TPR')
                
            for i, r in enumerate(results):
                row = [f"{r['param_value']}", f"{r['personal_acc']:.2f}%", f"{r['global_acc']:.2f}%"]
                if eces and i < len(eces):
                    row.append(f"{r['ece']:.4f}")
                if aurocs and i < len(aurocs):
                    row.append(f"{r['auroc']:.4f}")
                if fpr95s and i < len(fpr95s):
                    row.append(f"{r['fpr95']:.4f}")
                table_data.append(row)
            
            table = plt.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)
            plt.title(f'{param_names[param_type]}敏感性分析数据表', fontsize=14, pad=20)
            
            pdf.savefig()
            plt.close()
        
        # 总结页
        plt.figure(figsize=(12, 9))
        plt.text(0.5, 0.95, '敏感性分析总结', 
                 horizontalalignment='center', verticalalignment='top', 
                 fontsize=20, transform=plt.gca().transAxes)
        
        summary_text = """
        基于敏感性分析结果，我们得出以下结论：
        
        1. 全局一致性权重(GCW)：
           - 最佳值范围：[最佳值范围]
           - 影响：[描述影响]
        
        2. 温度参数(Temperature)：
           - 最佳值范围：[最佳值范围]
           - 影响：[描述影响]
        
        3. EDL损失权重：
           - 最佳值范围：[最佳值范围]
           - 影响：[描述影响]
        
        4. 最大Evidence值：
           - 最佳值范围：[最佳值范围]
           - 影响：[描述影响]
        
        综合建议：
        [综合建议]
        """
        
        plt.text(0.5, 0.5, summary_text, 
                 horizontalalignment='center', verticalalignment='center', 
                 fontsize=12, transform=plt.gca().transAxes)
        plt.axis('off')
        pdf.savefig()
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='分析FedUG V2敏感性实验结果')
    parser.add_argument('--results_dir', type=str, required=True, help='结果目录')
    parser.add_argument('--output_file', type=str, required=True, help='输出PDF文件')
    args = parser.parse_args()
    
    analyze_sensitivity(args.results_dir, args.output_file)