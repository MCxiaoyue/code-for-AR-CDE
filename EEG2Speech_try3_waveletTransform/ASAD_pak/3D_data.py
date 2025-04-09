import numpy as np
import pandas as pd

# 假设输入数据
batch_size = 64  # 批量大小
channels = 24    # 通道数量
timestamps = 192  # 时间戳长度

# 示例 EEG 数据 (batch_size, channels, timestamps)
EEG_1D = np.random.rand(batch_size, channels, timestamps)

# 加载 .xlsx 文件中的通道位置信息
map_df = pd.read_excel('EEG_2D1.xlsx', header=None)  # 读取 Excel 文件
map_array = map_df.values  # 转换为 NumPy 数组

# 构建通道坐标映射
axis = np.zeros((channels, 2), dtype=int)  # 初始化通道坐标数组
# channel_names = [f'Channel_{i+1}' for i in range(channels)]  # 假设通道名称为 Channel_1, Channel_2, ...
channel_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8',
                 'T3', 'T4', 'T5', 'T6', 'Sp1', 'Sp2', 'Fz', 'Cz', 'Pz', 'Oz', 'A1', 'A2']

flag = 0
# 遍历所有通道，找到其对应的二维坐标
for cha_idx, channel_name in enumerate(channel_names):
    for w in range(map_array.shape[0]):  # 遍历行
        for h in range(map_array.shape[1]):  # 遍历列
            if str(map_array[w, h]) == channel_name:  # 匹配通道名称
                print(channel_name)
                flag += 1
                print(flag)
                axis[cha_idx, 0] = w  # 行坐标
                axis[cha_idx, 1] = h  # 列坐标

# 初始化二维 EEG 数据结构
height, width = map_array.shape  # 获取二维平面的高度和宽度
EEG_2D = np.zeros((batch_size, height, width, timestamps))  # 形状为 (batch_size, height, width, timestamps)

# 将一维 EEG 数据映射到二维
for cha_idx in range(channels):
    w, h = axis[cha_idx]  # 获取当前通道的二维坐标
    EEG_2D[:, w, h, :] = EEG_1D[:, cha_idx, :]  # 将一维数据填充到对应位置

# 输出结果
print("原始 EEG 数据形状:", EEG_1D.shape)
print("转换后的 EEG 数据形状:", EEG_2D.shape)

# 可选：保存处理后的数据
np.save('EEG_2D.npy', EEG_2D)  # 保存为 .npy 文件