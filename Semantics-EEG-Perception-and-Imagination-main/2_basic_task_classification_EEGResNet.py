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
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from tqdm import tqdm
from braindecode.models import EEGNetv4, ATCNet, EEGConformer, EEGITNet, ShallowFBCSPNet, EEGResNet
import random
import matplotlib.pyplot as plt

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomEEGResNet(nn.Module):
    def __init__(self, num_classes, in_chans=124, input_window_samples=2049):
        super(CustomEEGResNet, self).__init__()

        self.atcnet = EEGResNet(
            n_chans=in_chans,
            n_outputs=num_classes,
            input_window_seconds=2.0009765625,
            sfreq=1024,
            n_times=input_window_samples,
            n_first_filters=32,
            final_pool_length='auto'
        )


    def forward(self, x):
        # print(x.shape) x:[batch_size, 1, channels, time_points]
        # 确保输入张量的维度符合 EEGNetv4 的期望 [batch_size, channels, time_points]
        if x.dim() == 4:  # 如果输入是4D张量，移除不必要的维度
            x = x.squeeze(1)  # 假设多余的维度在第1个位置

        # 通过 EEGNetv4 模型进行前向传播
        x = self.atcnet(x)

        return x

# subject = '17'
# session = '1'
# task = 'audio'
#
# if task == 'pictorial':
#     tag = 'p'
#     duration = 3
# elif task == 'orthographic':
#     print("orthographic decoding task")
#     tag='t'
#     duration = 3
# elif task == 'audio':
#     tag='s'
#     duration = 2

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


# 固定所有随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 设置GPU的随机种子
    np.random.seed(seed)  # NumPy的随机种子
    random.seed(seed)  # Python内置的随机模块种子
    torch.backends.cudnn.deterministic = True  # 确保每次返回相同的结果

seed = 42
setup_seed(seed)

# 定义任务、主题、会话以及路径
task = 'audio'
tag = 's'
duration = 2
subjects = ['11']  # 添加所有需要处理的主题ID  , '12', '11', '17'
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

            # 读取想象数据（如果需要）
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
                epochs = mne.epochs.combine_event_ids(
                    epochs,
                    old_event_ids=perc_event_ids,
                    new_event_id={old_event_id: new_event_id}
                )

            # 获取标签
            labels = epochs.events[:, -1]

            # 标准化数据
            image_data_array = np.array(epochs.get_data())
            for i in range(image_data_array.shape[0]):
                data_to_scale = image_data_array[i, :, :]
                # print(data_to_scale.shape)
                scaler = StandardScaler()   # MinMaxScaler, RobustScaler, StandardScaler
                normalized_sample = scaler.fit_transform(data_to_scale)
                all_image_data.append(normalized_sample)

            all_labels.extend(labels)

        except FileNotFoundError:
            print(f"File not found for subject {subject}, session {session}. Skipping.")

# 将列表转换为numpy数组
image_data_array = np.array(all_image_data)
labels_array = np.array(all_labels)

print("Combined data shape:", image_data_array.shape)
print("Combined labels shape:", labels_array.shape)
print(labels_array)

# 测试数据的索引
test_indices = [i for i in range(0, image_data_array.shape[0], 10)]
print("Test indices:", test_indices)

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

# 创建数据加载器
torch.manual_seed(42)  # 设置固定的随机种子
train_dataset = TensorDataset(train_data, train_labels)
test_dataset = TensorDataset(test_data, test_labels)

for img, lab in test_dataset:
    # print(img)
    print(lab)

print('====================================')

train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=24, shuffle=False)

# 初始化模型、损失函数和优化器
num_classes = 3
model = CustomEEGResNet(num_classes=num_classes, in_chans=124, input_window_samples=2049).to(device)  # 将模型移动到 GPU
criterion = nn.CrossEntropyLoss().to(device)  # 将损失函数也移动到 GPU
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)  # betas=(0.95, 0.999), eps=1e-08

# 训练模型
num_epochs = 300
best_test_acc = 0.0  # 用于记录最佳测试集准确率

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    # 训练部分
    for inputs, targets in tqdm(train_loader, desc=f'Training Epoch [{epoch + 1}/{num_epochs}]'):
        inputs, targets = inputs.to(device), targets.to(device)  # 将数据和标签移动到 GPU
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # 计算训练集上的正确预测数
        _, predicted = torch.max(outputs.data, 1)
        total_train += targets.size(0)
        correct_train += (predicted == targets).sum().item()

    # 计算训练集上的准确率
    train_accuracy = 100 * correct_train / total_train

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

    # 计算测试集上的准确率
    test_accuracy = 100 * correct_test / total_test

    # 如果当前测试集准确率优于之前的最佳准确率，则保存模型
    if test_accuracy > best_test_acc:
        best_test_acc = test_accuracy
        best_model_weights = model.state_dict()  # 保存当前模型权重
        # 立即保存最佳模型权重到文件
        torch.save(best_model_weights, './best_model_weights.pth')
        print(f'Saved new best model with Test Acc: {best_test_acc:.2f}%')  # 可选：打印保存信息

    # 打印结果
    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Loss: {running_loss / len(train_loader):.4f}, '
          f'Train Acc: {train_accuracy:.2f}%, '
          f'Test Acc: {test_accuracy:.2f}%')

model = CustomEEGResNet(num_classes=num_classes, in_chans=124, input_window_samples=2049).to(device) # 将模型移动到 GPU
# 训练完成后，使用最佳模型权重更新模型
model.load_state_dict(torch.load('./best_model_weights.pth'))
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

