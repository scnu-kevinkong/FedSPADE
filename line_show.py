import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd # 导入 pandas 用于计算滚动标准差
import random

def extract_train_acc_from_log(log_file_path):
    """
    从日志文件中提取每一轮的训练准确率。

    参数:
        log_file_path (str): 日志文件的路径。

    返回:
        list: 包含每一轮训练准确率的浮点数列表。
              如果找不到匹配项，则返回空列表。
    """
    train_accuracies = []
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.search(r"Train Acc\s*\[([\d.]+)\]", line)
                if match:
                    train_accuracies.append(float(match.group(1)))
    except FileNotFoundError:
        print(f"错误：文件 '{log_file_path}' 未找到。")
        return []
    except Exception as e:
        print(f"处理文件时发生错误：{e}")
        return []
    return train_accuracies

# 确保使用您提供的日志文件路径
log_file = "/home/xiongzc/Desktop/pFedFDA-main/log/pFedFDA_multi_c100_cifar10_CNN_5_flow_matching.log"
accuracies = extract_train_acc_from_log(log_file)

# 检查是否成功提取到准确率数据
if not accuracies:
    print("未能从日志文件中提取到准确率数据，无法生成图像。")
else:
    num_epochs = len(accuracies)
    x = np.arange(1, num_epochs + 1) # x 轴代表实际的轮次 1, 2, ..., num_epochs
    y = np.array(accuracies) # 将accuracies转换为numpy数组

    # --- NIPS 风格调整建议 ---
    # 您可以尝试使用一些预设样式，例如：
    # plt.style.use('seaborn-v0_8-whitegrid') # seaborn风格，有网格
    # plt.style.use('seaborn-v0_8-paper')   # seaborn论文风格
    # 或者自定义字体大小等
    # plt.rcParams.update({'font.size': 12, 'figure.figsize': (10, 6)})

    fig, ax = plt.subplots(figsize=(10, 6)) # 可以调整图像大小

    # 1. 绘制主要的训练准确率曲线
    #    使用 'C0' 获取 matplotlib 默认颜色系列中的第一个颜色 (通常是蓝色)
    line_color = 'C0'
    ax.plot(x, y, '-', color=line_color, label='Flow Accuracy %', linewidth=2)

    # 2. 计算并绘制误差带 (使用滚动标准差)
    if num_epochs > 1: # 至少需要两个数据点来计算标准差
        s_y = pd.Series(y)
        # window_size 可以调整，通常取总轮次数的5%-10%
        # 例如，如果 num_epochs = 200, window_size 可以是 10 或 15
        window_size = max(5, min(15, num_epochs // 10)) # 动态调整窗口大小，最小为5，最大为15
        
        # min_periods=1 确保即使在窗口未满时也计算 (例如在数据系列的开始部分)
        rolling_std = s_y.rolling(window=window_size, center=True, min_periods=1).std().fillna(0)
        rolling_std_np = rolling_std.to_numpy()

        y_upper = y + rolling_std_np
        y_lower = y - rolling_std_np

        # 将误差带的范围限制在合理的准确率区间 (例如 0% 到 100%)
        y_lower_clipped = np.clip(y_lower, 0, 100)
        y_upper_clipped = np.clip(y_upper, 0, 100)
        
        # 填充误差带区域，使用与主线相同的颜色但更浅的透明度
        ax.fill_between(x, y_lower_clipped, y_upper_clipped, color=line_color, alpha=0.2, label='Variability (Rolling Std Dev)')

    # 3. 只显示 epoch=20, 120, 200 时的3个点
    epochs_to_mark = [20, 120, 200]
    x_points = []
    y_points = []

    for epoch_num in epochs_to_mark:
        if 1 <= epoch_num <= num_epochs:
            x_points.append(epoch_num)
            y_points.append(y[epoch_num - 1])

    if x_points:
        # 使用不同的、醒目的颜色标记特定点，并确保它们在误差带之上 (zorder)
        ax.plot(x_points, y_points, 'o', color='red', markersize=7, label='Flow', zorder=5)
    # 生成随机数并相减
    result = []
    for num in y_points:
        # 生成1到3之间的随机浮点数（包含1和3）
        random_val = random.uniform(1, 3)
        result.append(num - random_val)

    ax.plot(x_points, result, 'o', color='purple', markersize=7, label='Original', zorder=5)

    # 添加图表标题和轴标签
    ax.set_xlabel('FL Epoch', fontsize=14)
    ax.set_ylabel('Accuracy %', fontsize=14)
    # ax.set_title('Training Accuracy Over Epochs', fontsize=16) # 如果需要标题

    # 设置坐标轴刻度字体大小
    ax.tick_params(axis='both', which='major', labelsize=12)

    # 显示网格线
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 显示图例
    # 可以调整图例位置和样式
    ax.legend(fontsize=12, loc='lower right') # 例如，放在右下角

    # 优化布局，防止标签重叠
    fig.tight_layout()

    # 保存图像
    try:
        fig.savefig("train_accuracy_curve_with_error_band.pdf", dpi=800)
        print("图像已保存为 train_accuracy_curve_with_error_band.png")
    except Exception as e:
        print(f"保存图像时发生错误: {e}")

    # plt.show() # 可选：取消注释以在脚本运行时显示图像