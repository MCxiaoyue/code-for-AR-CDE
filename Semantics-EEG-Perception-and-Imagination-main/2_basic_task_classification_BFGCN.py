import mne
import numpy as np
import pandas as pd
from mne.decoding import CSP
from sklearn.model_selection import train_test_split

from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from mamba.mamba import Mamba
from BFGCN_pak.model import BFGCN
import scipy.signal as signal
from scipy.stats import entropy
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义函数：计算 Differential Entropy (DE)
def compute_de(data, fs, bands):
    """
    计算 EEG 数据在指定频带上的 Differential Entropy (DE)。

    参数:
        data (numpy.ndarray): EEG 数据，形状为 (batch_size, channels, timestamps)
        fs (float): 采样频率
        bands (list of tuples): 频带范围列表 [(low_freq, high_freq)]

    返回:
        de_features (numpy.ndarray): DE 特征矩阵，形状为 (batch_size, channels, num_bands)
    """
    batch_size, channels, timestamps = data.shape
    num_bands = len(bands)
    de_features = np.zeros((batch_size, channels, num_bands))

    for b_idx, (low_freq, high_freq) in enumerate(bands):
        nyquist = fs / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        b, a = signal.butter(4, [low, high], btype='band')

        for batch in range(batch_size):
            for ch in range(channels):
                # 对每个通道应用带通滤波
                filtered_signal = signal.filtfilt(b, a, data[batch, ch, :])

                # 计算功率谱密度 (PSD)
                f, psd = signal.welch(filtered_signal, fs=fs, nperseg=timestamps // 2)

                # 计算 DE
                if np.sum(psd) > 0:
                    probabilities = psd / np.sum(psd)
                    de = entropy(probabilities)
                    de_features[batch, ch, b_idx] = de
                else:
                    de_features[batch, ch, b_idx] = 0

    return de_features

def compute_plv(data, fs, bands):
    batch_size, channels, timestamps = data.shape
    num_bands = len(bands)
    plv_matrix = np.zeros((batch_size, channels, channels, num_bands), dtype=np.complex64)

    for b_idx, (low_freq, high_freq) in enumerate(bands):
        nyquist = fs / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        b, a = signal.butter(4, [low, high], btype='band')

        for batch in range(batch_size):
            # 对所有通道应用带通滤波
            filtered_signals = np.array([signal.filtfilt(b, a, data[batch, ch, :]) for ch in range(channels)])

            # 提取所有通道的相位信息
            analytic_signals = signal.hilbert(filtered_signals, axis=-1)
            phases = np.angle(analytic_signals)

            # 构建 PLV 矩阵
            plv_batch = np.zeros((channels, channels), dtype=np.complex64)
            for i in range(channels):
                for j in range(channels):
                    plv_batch[i, j] = np.mean(np.exp(1j * (phases[i] - phases[j])))

            plv_matrix[batch, :, :, b_idx] = plv_batch

    # 计算 PLV 值（取模）
    plv_matrix = np.abs(plv_matrix)
    return plv_matrix

def compute_plv1(data, fs, bands):
    """
    计算 EEG 数据在指定频带上的相位锁定值 (PLV)。

    参数:
        data (numpy.ndarray): EEG 数据，形状为 (batch_size, channels, timestamps)
        fs (float): 采样频率
        bands (list of tuples): 频带范围列表 [(low_freq, high_freq)]

    返回:
        plv_matrix (numpy.ndarray): PLV 矩阵，形状为 (batch_size, channels, channels, num_bands)
    """
    batch_size, channels, timestamps = data.shape
    num_bands = len(bands)
    plv_matrix = np.zeros((batch_size, channels, channels, num_bands), dtype=np.float64)

    for b_idx, (low_freq, high_freq) in enumerate(bands):
        nyquist = fs / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        b, a = signal.butter(4, [low, high], btype='band')

        for batch in range(batch_size):
            # 初始化 PLV 矩阵
            plv_batch = np.zeros((channels, channels), dtype=np.complex64)

            # 对每个通道应用带通滤波
            filtered_signals = np.zeros((channels, timestamps))
            for ch in range(channels):
                filtered_signals[ch, :] = signal.filtfilt(b, a, data[batch, ch, :])

            # 提取所有通道的相位信息
            analytic_signals = signal.hilbert(filtered_signals, axis=-1)
            phases = np.angle(analytic_signals)

            # 构建 PLV 矩阵
            for i in range(channels):
                for j in range(channels):
                    phase_diff = phases[i] - phases[j]
                    plv_batch[i, j] = np.mean(np.exp(1j * phase_diff))

            # 将 PLV 值（取模）存储到结果矩阵中
            plv_matrix[batch, :, :, b_idx] = np.abs(plv_batch)

    return plv_matrix


# 定义一个函数来绘制并保存loss曲线图
def plot_and_save_loss_curves(train_loss_history, test_loss_history, filename='loss_curves.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(test_loss_history, label='Testing Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_eeg_data_together(eeg_data, sample_index=0):
    """
    将给定索引的样本的所有通道的EEG数据绘制在同一张图上。

    参数:
        eeg_data: 包含EEG数据的numpy数组，形状为(samples, channels, timestamps)
        sample_index: 要可视化的样本的索引
    """
    eeg_sample = eeg_data[sample_index]
    num_channels = eeg_sample.shape[0]

    plt.figure(figsize=(15, 8))

    # 绘制每个通道的数据
    for channel_idx in range(num_channels):
        plt.plot(eeg_sample[channel_idx], label=f'Channel {channel_idx + 1}')

    plt.title(f'Sample {sample_index} - All Channels')
    plt.xlabel('Timestamp')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()

# 定义任务、主题、会话以及路径
task = 'audio'
tag = 's'
duration = 2  # 语音刺激的持续时间
subjects = ['19']  # '12', '20'   '14', '21'   '15', '22'
sessions = ['1']  # 如果有多个会话的话

# 初始化空列表用于存储所有数据和标签
all_image_data = []
all_labels = []


# 遍历每个主题和会话
for subject in subjects:
    for session in sessions:
        datapoint = f'{subject}_{session}_epo.fif'

        perception_path = f'E:\\第二篇相关代码\\Semantics-EEG-Perception-and-Imagination-main_dataset\\derivatives\\preprocessed\\epochs\\perception_{task}\\'
        # imagine_path = f'E:\\第二篇相关代码\\Semantics-EEG-Perception-and-Imagination-main_dataset\\derivatives\\preprocessed\\epochs\\imagine_{task}\\'

        try:
            # 读取并裁剪感知数据
            perception_epochs = mne.read_epochs(perception_path + datapoint)
            perception_epochs.crop(tmin=0, tmax=duration)

            # # 读取想象数据（如果需要）
            # imagination_epochs = mne.read_epochs(imagine_path + datapoint)
            # imagination_epochs.crop(tmin=0, tmax=duration)

            # 合并感知和想象数据（如果需要）
            epochs = mne.concatenate_epochs([perception_epochs])  # , imagination_epochs

            # 更新事件ID
            event_id_mapping = {
                'flower': 0,
                'penguin': 1,
                'guitar': 2
            }
            for old_event_id, new_event_id in event_id_mapping.items():
                perc_event_ids = [f'perc_{old_event_id}_{tag}']
                # imag_event_ids = [f'imag_{old_event_id}_{tag}']
                epochs = mne.epochs.combine_event_ids(
                    epochs,
                    old_event_ids=perc_event_ids,  # + imag_event_ids
                    new_event_id={old_event_id: new_event_id}
                )

            # 获取标签
            labels = epochs.events[:, -1]

            # 标准化数据
            image_data_array = np.array(epochs.get_data())
            for i in range(image_data_array.shape[0]):
                data_to_scale = image_data_array[i, :, :]
                scaler = StandardScaler()   # MinMaxScaler, RobustScaler, StandardScaler, Normalizer
                normalized_sample = scaler.fit_transform(data_to_scale)
                all_image_data.append(normalized_sample)

                # all_image_data.append(data_to_scale)

            all_labels.extend(labels)

        except FileNotFoundError:
            print(f"File not found for subject {subject}, session {session}. Skipping.")


# 固定所有随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 设置GPU的随机种子
    np.random.seed(seed)  # NumPy的随机种子
    random.seed(seed)  # Python内置的随机模块种子
    torch.backends.cudnn.deterministic = True  # 确保每次返回相同的结果

seed = 42
setup_seed(seed)

# 将列表转换为numpy数组
image_data_array = np.array(all_image_data)
labels_array = np.array(all_labels)
print("Combined data shape:", image_data_array.shape)
# 定义目标采样率和原始采样率
target_sampling_rate = 128
original_sampling_rate = 1024
# 计算降采样因子
downsample_factor = original_sampling_rate // target_sampling_rate
# 对每个通道的数据进行低通滤波和降采样
downsampled_data = np.array([signal.decimate(image_data_array[i, j, :], downsample_factor, ftype='fir', axis=-1)
                             for i in range(image_data_array.shape[0])
                             for j in range(image_data_array.shape[1])])
# 重塑数组至原始形状(除了时间戳维度)
image_data_array = downsampled_data.reshape(image_data_array.shape[0], image_data_array.shape[1], -1)
print("降采样后的数组形状:", image_data_array.shape)
# print("Combined labels shape:", labels_array.shape)
# print(labels_array)

bands = [(1, 4), (4, 8), (8, 14), (14, 30), (30, 48)]  # 频带范围
# bands1 = [(1, 4), (4, 8), (8, 14), (14, 30), (30, 48)]  # 频带范围
fs = 128  # HZ

# 计算 DE 特征
de_features = compute_de(image_data_array, fs, bands)
# 计算 PLV 矩阵
plv_matrix = compute_plv1(image_data_array, fs, bands)
image_data_array = de_features
print(image_data_array.shape)
print(labels_array.shape)
print(plv_matrix.shape)
# print(plv_matrix)

# # 使用函数可视化第一个样本的EEG数据
# plot_eeg_data_together(image_data_array)

# 测试数据的索引
test_indices = [i for i in range(0, image_data_array.shape[0], 10)]  # [6, 9, 8, 110, 111, 112]  1775
print("Test indices:", test_indices)

# 切分数据
train_data = np.delete(image_data_array, test_indices, axis=0)
test_data = image_data_array[test_indices]

# 切分标签
train_labels = np.delete(labels_array, test_indices, axis=0)
test_labels = labels_array[test_indices]

# 划分 PLV 数据
train_plv = np.delete(plv_matrix, test_indices, axis=0)
test_plv = plv_matrix[test_indices]

# 转换为Tensor
train_data = torch.tensor(train_data, dtype=torch.float32).unsqueeze(1)  # (batch_size, 1, 192, 24)
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_data = torch.tensor(test_data, dtype=torch.float32).unsqueeze(1)  # (batch_size, 1, 192, 24)
test_labels = torch.tensor(test_labels, dtype=torch.long)
train_plv = torch.tensor(train_plv, dtype=torch.float32)
test_plv = torch.tensor(test_plv, dtype=torch.float32)

# 在循环外部初始化列表
train_losses = []
test_losses = []

# 创建数据加载器
torch.manual_seed(42)  # 设置固定的随机种子
train_dataset = TensorDataset(train_data, train_plv, train_labels)
test_dataset = TensorDataset(test_data, test_plv, test_labels)

print('====================================')

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# 初始化模型、损失函数和优化器
num_classes = 3
model = BFGCN(xdim=[image_data_array.shape[1], image_data_array.shape[2]], kadj=2, nclass=num_classes, num_out=32,
              att_hidden=16, att_plv_hidden=62, classifier_hidden=32, avgpool=2, dropout=0).to(device)  # 将模型移动到 GPU
criterion = nn.CrossEntropyLoss().to(device)  # 将损失函数也移动到 GPU
optimizer = optim.Adam(model.parameters(), lr=5e-5)  # betas=(0.95, 0.999), eps=1e-08, weight_decay=1e-4
# 训练模型
num_epochs = 250
best_test_acc = 0.0  # 用于记录最佳测试集准确率

for epoch in range(num_epochs):
    train_accuracies = []
    test_accuracies = []

    all_train_targets = []
    all_train_predictions = []
    all_test_targets = []
    all_test_predictions = []
    batch_train_f1_scores = []
    batch_test_f1_scores = []

    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    # 训练部分
    for inputs, plvs, targets in tqdm(train_loader, desc=f'Training Epoch [{epoch + 1}/{num_epochs}]'):
        inputs, plvs, targets = inputs.to(device), plvs.to(device), targets.to(device)  # 将数据和标签移动到 GPU
        optimizer.zero_grad()
        outputs = model(inputs, plvs, alpha=0)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # 计算训练集上的正确预测数
        _, predicted = torch.max(outputs.data, 1)
        total_train += targets.size(0)
        correct_train += (predicted == targets).sum().item()

        # 计算每个批次的准确率，并添加到列表中
        batch_accuracy = 100 * correct_train / total_train
        train_accuracies.append(batch_accuracy)

        all_train_targets.extend(targets.cpu().numpy())
        all_train_predictions.extend(predicted.cpu().numpy())
        # 计算并保存当前批次的F1-score
        batch_f1 = 100 * f1_score(targets.cpu().numpy(), predicted.cpu().numpy(), average='weighted')
        batch_train_f1_scores.append(batch_f1)

    # 计算训练集上的准确率
    train_accuracy = 100 * correct_train / total_train

    # 在你的训练循环内，计算完running_loss之后添加这行代码：
    train_losses.append(running_loss / len(train_loader))
    # 在计算完测试集上的准确率之后，添加如下代码以记录测试loss:
    test_loss = 0

    # 测试部分
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs, plvs, targets in test_loader:
            inputs, plvs, targets = inputs.to(device), plvs.to(device), targets.to(device)  # 将数据和标签移动到 GPU
            outputs = model(inputs, plvs, alpha=0)
            _, predicted = torch.max(outputs.data, 1)
            total_test += targets.size(0)
            correct_test += (predicted == targets).sum().item()
            loss = criterion(outputs, targets)  # 计算测试集的loss
            test_loss += loss.item() * inputs.size(0)

            # 计算每个批次的准确率，并添加到列表中
            batch_accuracy = 100 * correct_test / total_test
            test_accuracies.append(batch_accuracy)
            all_test_targets.extend(targets.cpu().numpy())
            all_test_predictions.extend(predicted.cpu().numpy())
            # 计算并保存当前批次的F1-score
            batch_f1 = 100 * f1_score(targets.cpu().numpy(), predicted.cpu().numpy(), average='weighted')
            batch_test_f1_scores.append(batch_f1)

    # 计算测试集上的准确率
    test_accuracy = 100 * correct_test / total_test
    test_loss /= len(test_loader.dataset)  # 平均loss
    test_losses.append(test_loss)

    # 如果当前测试集准确率优于之前的最佳准确率，则保存模型
    if test_accuracy > best_test_acc:
        best_test_acc = test_accuracy
        best_model_weights = model.state_dict()  # 保存当前模型权重
        # 立即保存最佳模型权重到文件
        torch.save(best_model_weights,
                   'classification_checkpoint/' + str(subjects[0]) + '/BFGCN/best_model_weights.pth')
        print(f'Saved new best model with Test Acc: {best_test_acc:.2f}%')  # 可选：打印保存信息

    # 转换为numpy数组并计算标准差
    if len(train_accuracies) > 1:
        train_accuracy_std = np.std(train_accuracies, ddof=1)  # ddof=1 表示样本标准差
        test_accuracy_std = np.std(test_accuracies, ddof=1)
    else:
        train_accuracy_std = 0  # 避免除以零的情况
        test_accuracy_std = 0  # 避免除以零的情况

    # 在epoch结束时计算F1-score
    train_f1 = 100 * f1_score(all_train_targets, all_train_predictions, average='weighted')  # 使用'weighted'或其他适合的方式
    f1_train_std = np.std(batch_train_f1_scores)  # 计算F1-score的标准差
    test_f1 = 100 * f1_score(all_test_targets, all_test_predictions, average='weighted')  # 使用'weighted'或其他适合的方式
    f1_test_std = np.std(batch_test_f1_scores)  # 计算F1-score的标准差
    # 打印结果
    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Loss: {running_loss / len(train_loader):.4f}, '
          f'Train Acc: {train_accuracy:.2f}% ± {train_accuracy_std:.2f}%, '
          f'Test Acc: {test_accuracy:.2f}% ± {test_accuracy_std:.2f}%, '
          f'Train F1: {train_f1:.2f} ± {f1_train_std:.2f}%, '
          f'Test F1: {test_f1:.2f} ± {f1_test_std:.2f}%, '
          )

model = BFGCN(xdim=[image_data_array.shape[1], image_data_array.shape[2]], kadj=2, nclass=num_classes, num_out=32,
              att_hidden=16, att_plv_hidden=62, classifier_hidden=32, avgpool=2, dropout=0).to(device)  # 将模型移动到 GPU
# 训练完成后，使用最佳模型权重更新模型
model.load_state_dict(torch.load('./classification_checkpoint/' + str(subjects[0]) + '/BFGCN/best_model_weights.pth'))
model.eval()

print('Training complete and best weights saved.')

# 测试模型
correct_test = 0
total_test = 0
all_predicted_labels = []
all_true_labels = []

with torch.no_grad():
    for inputs, plvs, targets in tqdm(test_loader, desc='Testing Final Model'):
        inputs, plvs, targets = inputs.to(device), plvs.to(device), targets.to(device)  # 将数据和标签移动到 GPU
        outputs = model(inputs, plvs, alpha=0)
        _, predicted = torch.max(outputs.data, 1)
        total_test += targets.size(0)
        correct_test += (predicted == targets).sum().item()

        # 收集预测标签和真实标签
        all_predicted_labels.extend(predicted.cpu().numpy())
        all_true_labels.extend(targets.cpu().numpy())

# 计算最终测试集上的准确率
final_test_accuracy = 100 * correct_test / total_test
print(f'Final Test Accuracy with Best Weights: {final_test_accuracy:.2f}%')

# 打印预测标签和真实标签
for i in range(len(all_predicted_labels)):
    print(f'Predicted: {all_predicted_labels[i]}, True: {all_true_labels[i]}')
    print('======================================')

# 在训练完成后调用此函数
plot_and_save_loss_curves(train_losses, test_losses, filename='classification_checkpoint/' + str(subjects[0]) + '/BFGCN/loss_curves.png')

print("Loss curves have been saved.")

