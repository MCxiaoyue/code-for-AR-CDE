import numpy as np
import pandas as pd

# 假设输入数据
batch_size = 64  # 批量大小
channels = 124    # 通道数量
timestamps = 2049  # 时间戳长度

# 示例 EEG 数据 (batch_size, channels, timestamps)
EEG_1D = np.random.rand(batch_size, channels, timestamps)

# 加载 .xlsx 文件中的通道位置信息
map_df = pd.read_excel('EEG_2D1.xlsx', header=None)  # 读取 Excel 文件
map_array = map_df.values  # 转换为 NumPy 数组
print(map_array)

# 构建通道坐标映射
axis = np.zeros((channels, 2), dtype=int)  # 初始化通道坐标数组
# channel_names = [f'Channel_{i+1}' for i in range(channels)]  # 假设通道名称为 Channel_1, Channel_2, ...
channel_names = ["Fp1", "Fpz", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "M1", "T7", "C3", "Cz", "C4",
                 "T8", "M2", "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "POz", "O1", "O2", "AF7", "AF3", "AF4",
                 "AF8", "F5", "F1", "F2", "F6", "FC3", "FCz", "FC4", "C5", "C1", "C2", "C6", "CP3", "CP4", "P5", "P1", "P2", "P6",
                 "PO3", "PO4", "FT7", "FT8", "TP7", "TP8", "PO7", "PO8", "FT9", "FT10", "TPP9h", "TPP10h", "PO9", "PO10", "P9", "P10",
                 "AFF1", "AFz", "AFF2", "FFC5h", "FFC3h", "FFC4h", "FFC6h", "FCC5h", "FCC3h", "FCC4h", "FCC6h", "CCP5h", "CCP3h", "CCP4h",
                 "CCP6h", "CPP5h", "CPP3h", "CPP4h", "CPP6h", "PPO1", "PPO2", "I1", "Iz", "I2", "AFp3h", "AFp4h", "AFF5h", "AFF6h", "FFT7h",
                 "FFC1h", "FFC2h", "FFT8h", "FTT9h", "FTT7h", "FCC1h", "FCC2h", "FTT8h", "FTT10h", "TTP7h", "CCP1h", "CCP2h", "TTP8h", "TPP7h",
                 "CPP1h", "CPP2h", "TPP8h", "PPO9h", "PPO5h", "PPO6h", "PPO10h", "POO9h", "POO3h", "POO4h", "POO10h", "OI1h", "OI2h"]

flag = 0
# 遍历所有通道，找到其对应的二维坐标
for cha_idx, channel_name in enumerate(channel_names):
    for w in range(map_array.shape[0]):  # 遍历行
        for h in range(map_array.shape[1]):  # 遍历列
            if str(map_array[w, h]) == channel_name:  # 匹配通道名称
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

# # 可选：保存处理后的数据
# np.save('EEG_2D.npy', EEG_2D)  # 保存为 .npy 文件