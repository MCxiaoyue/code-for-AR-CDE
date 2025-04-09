import matplotlib.pyplot as plt
import numpy as np

# 模拟 EEG 数据
np.random.seed(42)  # 固定随机种子以便结果可复现
num_channels = 10   # 假设有10个通道
timestamps = 50    # 每个通道有200个时间点

# 使用正弦波作为基础信号，并添加更多的高频成分使其更加锯齿
time_points = np.linspace(0, 4 * np.pi, timestamps)
base_signal = np.sin(time_points) + 0.5 * np.sin(2 * time_points) + 0.3 * np.sin(5 * time_points)

# 添加更多随机噪声，使信号更加锯齿和不规则
eeg_data = base_signal[np.newaxis, :] + np.random.normal(0, 0.3, (num_channels, timestamps))

# 创建一个新的图形
plt.figure(figsize=(12, 6))

# 绘制每个通道的信号
for channel in range(num_channels):
    # 为每个通道添加一个垂直偏移，防止重叠
    offset = channel * 2  # 每个通道之间增加2个单位的偏移
    plt.plot(eeg_data[channel] + offset, color='blue', linewidth=2)

# 设置图形样式
plt.gca().spines["top"].set_color("black")     # 外框顶部颜色为黑色
plt.gca().spines["bottom"].set_color("black")  # 外框底部颜色为黑色
plt.gca().spines["left"].set_color("black")    # 外框左侧颜色为黑色
plt.gca().spines["right"].set_color("black")   # 外框右侧颜色为黑色

plt.xticks([])  # 不显示 x 轴刻度
plt.yticks([])  # 不显示 y 轴刻度

# 保存图像到文件
output_file = "eeg_plot.png"  # 输出文件名
plt.savefig(output_file, dpi=300, bbox_inches='tight')  # 保存为高分辨率PNG文件

# 显示图像
plt.show()