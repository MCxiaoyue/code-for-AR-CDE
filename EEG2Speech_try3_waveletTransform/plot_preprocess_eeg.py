import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from tqdm import tqdm
import pywt
import matplotlib.pyplot as plt
from kymatio import Scattering1D
from mamba.mamba import Mamba
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 手动规定的单词到标签映射
word_to_label = {
    "my": 8,
    "dad": 0,
    "is": 1,
    "a": 2,
    "policeman": 3,
    "he": 4,
    "will": 5,
    "always": 6,
    "become": 7,
    "hero": 9
}

# 处理单个文件的函数
def process_file(file_path, image_data, labels):
    with open("./orign_eeg_data/"+file_path, mode='r') as f:
        lines = f.readlines()
        ls = []
        # 采集2560行
        for line in lines[1282:52844]:  # 前三行没用，所以从3开始，每秒128帧
            line_list = line.strip().split('\t')  # 再分割成列表
            # 采集24列中的特定几列
            columns = []
            for linetime in range(1, 25):  # [5, 6, 13, 14]
                columns.append(float(line_list[linetime]))
            ls.append(columns)  # 提取指定列并转换为浮点数

        eeg = np.array(ls)

        # 定义参数
        word_duration = 1.5  # 单词时长（秒）
        rest_within_word = 5  # 同一单词间的静音间隔（秒）
        rest_between_words = 10  # 不同单词间的静音间隔（秒）
        sampling_rate = 128  # 采样率（Hz）

        # 单词序列
        words = ["my", "dad", "is", "a", "policeman", "he", "will", "always", "become", "my", "hero"]
        repetitions = 5  # 每个单词重复次数

        # 计算每个单词和静音周期的采样点数量
        samples_per_word = int(word_duration * sampling_rate)
        samples_per_rest_within_word = int(rest_within_word * sampling_rate)
        samples_per_rest_between_words = int(rest_between_words * sampling_rate)

        # 分割数据
        word_data_segments = []
        current_index = 0
        for word in words:
            for _ in range(repetitions):
                # 添加单词部分
                word_start = current_index
                word_end = word_start + samples_per_word
                word_segment = eeg[word_start:word_end, :]
                word_data_segments.append(word_segment)

                # 分配标签
                label = word_to_label[word]  # 使用单词到标签的映射来分配标签
                labels.append(label)

                # 更新索引到下一个单词或静音部分
                if _ < repetitions - 1:
                    current_index += samples_per_word + samples_per_rest_within_word
                else:
                    current_index += samples_per_word + samples_per_rest_between_words

        for eeg in word_data_segments:
            # print(eeg.shape)

            # 对每个通道进行归一化
            scaler = StandardScaler()  # MinMaxScaler() RobustScaler() StandardScaler()
            segment_normalized = scaler.fit_transform(eeg.T).T  # 注意这里转置两次是为了保持原来的shape

            arr = segment_normalized
            image_data.append(arr.T)

            # # 不进行归一化
            # image_data.append(eeg.T)


import numpy as np
import random
import torch
from torch.utils.data import DataLoader, TensorDataset

# 固定所有随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 设置GPU的随机种子
    np.random.seed(seed)  # NumPy的随机种子
    random.seed(seed)  # Python内置的随机模块种子
    torch.backends.cudnn.deterministic = True  # 确保每次返回相同的结果

seed = 42
setup_seed(seed)


if __name__ == '__main__':
    # 在循环外部初始化列表
    train_losses = []
    test_losses = []

    # 文件路径列表
    file_paths = ['vDLY-001_1.txt']

    # 创建一个空列表来存储转换后的图像数据和对应的标签
    image_data = []
    labels = []
    # 处理所有文件
    for file_path in file_paths:
        process_file(file_path, image_data, labels)

    # 转换为numpy数组
    image_data_array = np.array(image_data)
    labels_array = np.array(labels)

    print(image_data_array[0].shape)

    # 获取第一个样本的数据
    eeg_data = image_data_array[0][:10, ]  # 形状为 (24, 192)

    # 创建一个新的图形
    plt.figure(figsize=(12, 8))

    # 绘制每个通道的信号
    num_channels, timestamps = eeg_data.shape  # 获取通道数和时间点数
    for channel in range(num_channels):
        # 为每个通道添加一个垂直偏移，防止重叠
        offset = channel * 2  # 每个通道之间增加2个单位的偏移
        plt.plot(eeg_data[channel] + offset, color='blue', linewidth=2)

    # 设置图形样式
    plt.gca().spines["top"].set_color("black")  # 外框顶部颜色为黑色
    plt.gca().spines["bottom"].set_color("black")  # 外框底部颜色为黑色
    plt.gca().spines["left"].set_color("black")  # 外框左侧颜色为黑色
    plt.gca().spines["right"].set_color("black")  # 外框右侧颜色为黑色

    plt.xticks([])  # 不显示 x 轴刻度
    plt.yticks([])  # 不显示 y 轴刻度

    # 保存图像到文件
    output_file = "preprocess_eeg_plot.png"  # 输出文件名
    plt.savefig(output_file, dpi=300, bbox_inches='tight')  # 保存为高分辨率PNG文件

    # 显示图像
    plt.show()

