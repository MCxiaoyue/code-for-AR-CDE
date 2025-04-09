import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
import torch
from torch import nn
from torchvision.utils import save_image
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision.utils import make_grid
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import mne


# 启用同步CUDA调用
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # 获取所有文件名
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.png')]
        # 排序以保证顺序
        self.image_files.sort(key=lambda x: int(x.split('_')[1]))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')  # 转换为灰度图
        label = int(img_name.split('_')[2].split('.')[0])  # 从文件名中提取标签

        if self.transform:
            image = self.transform(image)

        return image, label

# 固定随机种子
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

# 定义数据集路径
root_dir = './3words2/'

# 定义变换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# 实例化数据集
custom_dataset = CustomDataset(root_dir=root_dir, transform=transform)

# test_indices 是你想要用于测试集的数据索引
test_indices = [i for i in range(0, 150, 10)]

print(test_indices)

# 获取所有数据集的索引
all_indices = list(range(len(custom_dataset)))

# 从所有索引中移除测试集索引，剩下的就是训练集索引
train_indices = [index for index in all_indices if index not in test_indices]

# 使用Subset创建训练集和测试集
train_dataset = Subset(custom_dataset, train_indices)
test_dataset = Subset(custom_dataset, test_indices)

# train_dataset = custom_dataset
# test_dataset = custom_dataset

for img, lab in test_dataset:
    # print(img)
    print(lab)
    print('====================================')

# 创建数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class CustomUNet2DModel(UNet2DModel):
    def __init__(self, *args, dropout_prob=0.1, **kwargs):
        super().__init__(*args, **kwargs)

        # 收集所有需要修改的模块
        modules_to_modify = []
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                modules_to_modify.append((name, module))

        # 修改模块
        for name, module in modules_to_modify:
            # 创建一个新的Sequential模块，包含原来的卷积层、BatchNorm和Dropout层
            new_module = nn.Sequential(
                module,
                nn.BatchNorm2d(module.out_channels),
                # nn.Dropout2d(p=dropout_prob)
            )
            # 替换原来的模块
            parts = name.split('.')
            parent = self
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_module)


class ClassConditionedUnet(nn.Module):
    def __init__(self, num_classes=3, class_emb_size=4):
        super().__init__()
        # 这个网络层会把数字所属的类别映射到一个长度为class_emb_size的特征向量上
        self.class_emb = nn.Embedding(num_classes, class_emb_size)

        # 创建自定义UNet2DModel实例
        self.model = CustomUNet2DModel(
            sample_size=128,
            in_channels=3 + class_emb_size,
            out_channels=3,
            layers_per_block=1,
            block_out_channels=(128, 128, 256, 256, 512, 512),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            # dropout_prob=0.1  # Dropout概率
        )

    # 此时扩散模型的前向计算就会含有额外的类别标签作为输入了
    def forward(self, x, t, class_labels):
        # print(class_labels)
        bs, ch, w, h = x.shape
        # 类别条件将会以额外通道的形式输入
        class_cond = self.class_emb(class_labels)  # 将类别映射为向量形式，
        # 并扩展成类似于（bs, 4, 128, 128）的张量形式

        # 归一化类别嵌入到 [0, 1] 区间
        # class_cond = (class_cond - class_cond.min()) / (class_cond.max() - class_cond.min())
        class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(bs, class_cond.shape[1], w, h)
        # print(class_cond.shape)
        # print(x.shape)
        # 将原始输入和类别条件信息拼接到一起
        # print(class_cond)
        # print('--------------------------------')
        net_input = torch.cat((x, class_cond), 1)  # (bs, 5, 128, 128)
        # print(net_input.shape)
        # 使用模型进行预测
        return self.model(net_input, t).sample  # (bs, 1, 128, 128)


# # 设置保存验证结果图片的文件夹
# output_dir1 = './checkpoint/'
# if not os.path.exists(output_dir1):
#     os.makedirs(output_dir1)


# 创建一个调度器
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')

# 定义训练参数
n_epochs = 250
loss_fn = nn.MSELoss()
net = ClassConditionedUnet().to(device)
opt = torch.optim.Adam(net.parameters(), lr=1e-4, betas=(0.95, 0.999), eps=1e-08)  # weight_decay=5e-4
# scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10, verbose=True)
best_val_loss = float('inf')

subject = '19'

def evaluate(net, dataloader, save_noise=False):
    net.eval()
    losses = []
    noises = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            noise = torch.randn_like(x)
            timesteps = torch.randint(0, 1000, (x.shape[0],)).long().to(device)
            noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
            pred = net(noisy_x, timesteps, y)
            loss = loss_fn(pred, noise)
            losses.append(loss.item())
            if save_noise:
                noises.append(noise)

    if save_noise:
        torch.save(torch.cat(noises, dim=0), './diffusion_checkpoint/'+subject+'/best_noise.pt')

    net.train()
    return sum(losses) / len(losses)

# 在训练循环外部定义保存路径
best_model_path = './diffusion_checkpoint/'+subject+'/best_model_weights.pth'
last_model_path = './diffusion_checkpoint/'+subject+'/last_model_weights.pth'

# 初始化损失列表
train_losses = []
val_losses = []

# for epoch in range(n_epochs):
#     net.train()
#     losses = []
#     for x, y in tqdm(train_dataloader):
#         x = x.to(device)
#         y = y.to(device)
#         noise = torch.randn_like(x)
#         timesteps = torch.randint(0, 1000, (x.shape[0],)).long().to(device)
#         noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
#         pred = net(noisy_x, timesteps, y)
#         loss = loss_fn(pred, noise)
#         opt.zero_grad()
#         loss.backward()
#         opt.step()
#         losses.append(loss.item())
#
#     # 计算训练集平均损失
#     train_avg_loss = sum(losses) / len(losses)
#     train_losses.append(train_avg_loss)
#
#     # 评估验证集损失
#     val_avg_loss = evaluate(net, val_dataloader)  # 首先计算验证集损失
#     val_losses.append(val_avg_loss)
#
#     # 决定是否保存噪声
#     should_save_noise = val_avg_loss < best_val_loss
#
#     # 如果验证集损失达到新低点，则保存最佳模型和噪声
#     if should_save_noise:
#         best_val_loss = val_avg_loss
#         torch.save(net.state_dict(), best_model_path)
#         print(f'Saving model at epoch {epoch} with validation loss: {val_avg_loss:.5f}')
#
#         # 重新评估一次，这次保存噪声
#         _ = evaluate(net, val_dataloader, save_noise=True)
#
#     print(f'Finished epoch {epoch}. Train loss: {train_avg_loss:.5f}, Validation loss: {val_avg_loss:.5f}')
#
#     # 在每个epoch结束后保存当前模型权重
#     torch.save(net.state_dict(), last_model_path)
#     print(f'Saved the current model weights at epoch {epoch}.')
#
# print('Training completed. Both the best and the last model weights have been saved.')
#
#
# import matplotlib.pyplot as plt
#
# # 绘制训练损失和验证损失的变化曲线
# plt.figure(figsize=(10, 6))
# plt.plot(train_losses, label='Train Loss')
# plt.plot(val_losses, label='Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss Over Epochs')
# plt.legend()
# plt.grid(True)
# plt.savefig('./diffusion_checkpoint/'+subject+'/loss_curve.png')
# plt.show()


# 加载模型

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = ClassConditionedUnet().to(device)
net.load_state_dict(torch.load('./diffusion_checkpoint/'+subject+'/best_model_weights.pth'))
net.eval()

# 手动规定的单词到标签映射
word_positions = {
    'flower': 0,
    'penguin': 1,
    'guitar': 2
}

from EEG_models import EVRNet

model = EVRNet(num_classes=3).to(device)  # 将模型移动到 GPU
# 训练完成后，使用最佳模型权重更新模型
model.load_state_dict(torch.load('./classification_checkpoint/'+subject+'/EVRNet_mamba/best_model_weights.pth'))
model.eval()

# 测试模型
correct_test = 0
total_test = 0
all_predicted_labels = []
all_true_labels = []

session = '1'
task = 'audio'

if task == 'pictorial':
    tag = 'p'
    duration = 3
elif task == 'orthographic':
    print("orthographic decoding task")
    tag='t'
    duration = 3
elif task == 'audio':
    tag='s'
    duration = 2

# load up files for one subject for one task of img vs. perc
# imagine_pictorial vs. perception_pictorial


subjects = ['19']  # 添加所有需要处理的主题ID  '12', '20'   '14', '21'   '15', '22'
sessions = ['1']  # 如果有多个会话的话

# 初始化空列表用于存储所有数据和标签
all_image_data = []
all_labels = []

# perception_path = 'E:\\第二篇相关代码\\Semantics-EEG-Perception-and-Imagination-main_dataset\\derivatives\\preprocessed\\epochs\\perception_'+task+'\\'
# # imagine_path = 'E:\\第二篇相关代码\\Semantics-EEG-Perception-and-Imagination-main_dataset\\derivatives\\preprocessed\\epochs\\imagine_'+task+'\\'
# datapoint = subject+'_'+session+ '_epo.fif'
#
# perception_epochs = mne.read_epochs(perception_path + datapoint)
# perception_events = mne.read_events(perception_path + datapoint)
# perception_epochs = perception_epochs.crop(tmin=0, tmax=duration)
#
# # imagination_epochs = mne.read_epochs(imagine_path + datapoint)
# # imagination_events = mne.read_events(imagine_path+ datapoint)
# # imagination_epochs = imagination_epochs.crop(tmin=0, tmax=duration)
# epochs = mne.concatenate_epochs([perception_epochs])  #  , imagination_epochs
#
# # 打印原始事件ID以确认
# print("Original event IDs:", epochs.event_id)
#
# # 创建新的事件ID映射
# event_id_mapping = {
#     'flower': 0,
#     'penguin': 1,
#     'guitar': 2
# }
#
# # 更新事件ID
# for old_event_id, new_event_id in event_id_mapping.items():
#     perc_event_ids = [f'perc_{old_event_id}_{tag}']
#     # imag_event_ids = [f'imag_{old_event_id}_{tag}']
#
#     # 结合感知和想象的事件ID，并为每个刺激类型分配一个新的唯一的ID
#     epochs = mne.epochs.combine_event_ids(
#         epochs,
#         old_event_ids = perc_event_ids,  # + imag_event_ids
#         new_event_id={old_event_id: new_event_id}
#     )
#
# # 打印更新后的事件ID以确认
# print("Updated event IDs:", epochs.event_id)
#
# # 更新标签
# labels = epochs.events[:, -1]
#
# # 打印标签和数据形状以确认
# print("Labels", labels)
# print("Shape of epoch data ", epochs.get_data().shape)
#
# # 转换为numpy数组
# image_data_array = np.array(epochs.get_data())
# image_data_array1 = []
# # 对每个样本的所有通道和时间点进行标准化
# for i in range(image_data_array.shape[0]):  # 遍历每个样本
#     # 将当前样本的数据重塑为二维数组 (特征数, 样本数), 这里特征数是 124 * 2049
#     data_to_scale = image_data_array[i, :, :]
#     # 实例化 StandardScaler
#     scaler = StandardScaler()  # MinMaxScaler, RobustScaler, StandardScaler
#     # 标准化该样本的数据
#     normalized_sample = scaler.fit_transform(data_to_scale)
#     # 将标准化后的数据恢复为原始形状
#     image_data_array1.append(normalized_sample)

# 遍历每个主题和会话


for subject1 in subjects:
    for session in sessions:
        datapoint = f'{subject1}_{session}_epo.fif'

        perception_path = f'E:\\第二篇相关代码\\Semantics-EEG-Perception-and-Imagination-main_dataset\\derivatives\\preprocessed\\epochs\\perception_{task}\\'
        # imagine_path = f'E:\\第二篇相关代码\\Semantics-EEG-Perception-and-Imagination-main_dataset\\derivatives\\preprocessed\\epochs\\imagine_{task}\\'
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
            print(f"File not found for subject {subject1}, session {session}. Skipping.")


# image_data_array = np.array(image_data_array1) # .transpose((0, 2, 1))
# labels_array = np.array(labels)

# 将列表转换为numpy数组
image_data_array = np.array(all_image_data)
labels_array = np.array(all_labels)

print(image_data_array.shape)
print(labels_array.shape)

# 测试数据的索引
test_indices = [i for i in range(0, image_data_array.shape[0], 10)]

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
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


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


# 生成固定的噪声和标签
saved_noise = torch.randn(len(test_indices), 3, 128, 128).to(device)
fixed_y = torch.tensor([[i] for i in all_predicted_labels]).flatten().to(device)


with torch.no_grad():
    for i, t in tqdm(enumerate(noise_scheduler.timesteps)):
        residual = net(saved_noise, t, fixed_y)
        saved_noise = noise_scheduler.step(residual, t, saved_noise).prev_sample

# 确保输出目录存在
output_dir = './generated_images/'+subject +'/'
os.makedirs(output_dir, exist_ok=True)

# 确保生成的图像张量在合适的范围内
def denormalize(image_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """Denormalize a tensor using the given mean and std."""
    mean = torch.tensor(mean).view(-1, 1, 1).to(image_tensor.device)
    std = torch.tensor(std).view(-1, 1, 1).to(image_tensor.device)
    return image_tensor * std + mean

# 使用 torchvision 的 save_image 函数保存生成的图像
for i in range(saved_noise.size(0)):
    print(i)
    img = saved_noise[i].cpu()  # 将图像数据移到CPU上
    img = img.squeeze(0)  # 单通道图像需要去掉通道维度
    img = denormalize(img)  # 反归一化
    print(img)
    print('===================')
    save_image(img, os.path.join(output_dir, f'generated_{i}_torchvision.png'), normalize=True)
