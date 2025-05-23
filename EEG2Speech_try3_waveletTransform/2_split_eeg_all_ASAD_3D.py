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
from ASAD_pak.model_3D import DenseNet_3D
import pandas as pd
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


if __name__ == '__main__':
    # 在循环外部初始化列表
    train_losses = []
    test_losses = []

    # 文件路径列表
    file_paths = ['vDLY-001_1.txt', 'vLCL-001_1.txt', 'vLHJ-001_1.txt', 'vLHJ-002_1.txt', 'vLCL-004_1.txt',
                  'vYHC02_1.txt', 'vLCL-01_1.txt', 'vLCL-02_1.txt', 'vCLB-001_1.txt', 'vYHC-001_1.txt', 'vLCL-003_1.txt']

    # 创建一个空列表来存储转换后的图像数据和对应的标签
    image_data = []
    labels = []
    # 处理所有文件
    for file_path in file_paths:
        process_file(file_path, image_data, labels)

    # 转换为numpy数组
    image_data_array = np.array(image_data)
    # 加载 .xlsx 文件中的通道位置信息
    map_df = pd.read_excel('./ASAD_pak/EEG_2D1.xlsx', header=None)  # 读取 Excel 文件
    map_array = map_df.values  # 转换为 NumPy 数组
    # 构建通道坐标映射
    axis = np.zeros((image_data_array.shape[1], 2), dtype=int)  # 初始化通道坐标数组
    channel_names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8',
                     'T3', 'T4', 'T5', 'T6', 'Sp1', 'Sp2', 'Fz', 'Cz', 'Pz', 'Oz', 'A1', 'A2']
    # 遍历所有通道，找到其对应的二维坐标
    for cha_idx, channel_name in enumerate(channel_names):
        for w in range(map_array.shape[0]):  # 遍历行
            for h in range(map_array.shape[1]):  # 遍历列
                if str(map_array[w, h]) == channel_name:  # 匹配通道名称
                    axis[cha_idx, 0] = w  # 行坐标
                    axis[cha_idx, 1] = h  # 列坐标
    # 初始化二维 EEG 数据结构
    height, width = map_array.shape  # 获取二维平面的高度和宽度
    EEG_2D = np.zeros((image_data_array.shape[0], height, width, image_data_array.shape[2]))  # 形状为 (batch_size, height, width, timestamps)
    # 将一维 EEG 数据映射到二维
    for cha_idx in range(image_data_array.shape[1]):
        w, h = axis[cha_idx]  # 获取当前通道的二维坐标
        EEG_2D[:, w, h, :] = image_data_array[:, cha_idx, :]  # 将一维数据填充到对应位置

    # 输出结果
    print("原始 EEG 数据形状:", image_data_array.shape)
    image_data_array = EEG_2D.transpose(0, 3, 1, 2)
    print("转换后的 EEG 数据形状:", image_data_array.shape)

    labels_array = np.array(labels)

    print(image_data_array.shape)
    print(labels_array.shape)

    # # 使用函数可视化第一个样本的EEG数据
    # plot_eeg_data_together(image_data_array)

    # 测试数据的索引
    test_indices = [i for i in range(0, image_data_array.shape[0], 10)]  # 第5, 55, 105, 155个数据（基于0索引） # 114, 165, 5
    print(test_indices)

    # 切分数据
    train_data = np.delete(image_data_array, test_indices, axis=0)
    test_data = image_data_array[test_indices]

    # 切分标签
    train_labels = np.delete(labels_array, test_indices, axis=0)
    test_labels = labels_array[test_indices]

    # 转换为Tensor
    train_data = torch.tensor(train_data, dtype=torch.float32).unsqueeze(1)  # (batch_size, 1, 192, 24)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_data = torch.tensor(test_data, dtype=torch.float32).unsqueeze(1)  # (batch_size, 1, 192, 24)
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    print(test_labels)

    # 创建数据加载器
    torch.manual_seed(42)  # 设置固定的随机种子
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False)

    # 初始化模型、损失函数和优化器
    num_classes = len(word_to_label.keys())
    model = DenseNet_3D(num_classes=num_classes).to(device)  # 将模型移动到 GPU
    criterion = nn.CrossEntropyLoss().to(device)  # 将损失函数也移动到 GPU
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)  # betas=(0.95, 0.999), eps=1e-08

    # 训练模型
    num_epochs = 500
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
        for inputs, targets in tqdm(train_loader, desc=f'Training Epoch [{epoch + 1}/{num_epochs}]'):
            inputs, targets = inputs.to(device), targets.to(device)  # 将数据和标签移动到 GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)  # 获取预测概率
            # print(outputs.shape)
            # print('22222222222222')
            # print(targets.shape)
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
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)  # 将数据和标签移动到 GPU
                outputs = model(inputs)
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
            torch.save(best_model_weights, './classification_checkpoint/ASAD_3D/best_model_weights.pth')
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


    model = DenseNet_3D(num_classes=num_classes).to(device)  # 将模型移动到 GPU
    # 训练完成后，使用最佳模型权重更新模型
    model.load_state_dict(torch.load('./classification_checkpoint/ASAD_3D/best_model_weights.pth'))
    model.eval()

    print('Training complete and best weights saved.')

    # 测试模型
    correct_test = 0
    total_test = 0
    all_predicted_labels = []
    all_true_labels = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Testing Final Model'):
            inputs, targets = inputs.to(device), targets.to(device)  # 将数据和标签移动到 GPU
            outputs = model(inputs)
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
    plot_and_save_loss_curves(train_losses, test_losses, filename='classification_checkpoint/ASAD_3D/loss_curves.png')

    print("Loss curves have been saved.")