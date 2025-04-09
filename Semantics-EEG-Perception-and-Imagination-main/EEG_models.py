import torch
import torch.nn as nn
from mamba.mamba import Mamba

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义各个模块
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(TemporalBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        return self.relu(x)


class SpatialBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(SpatialBlock, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size), stride=(1, stride))
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        return self.relu(x)


class MKRB(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(MKRB, self).__init__( )

        # First part (3x3 conv)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Second part (5x5 conv)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 第一个卷积层提取特征向量 y0
        y0 = self.conv1(x)

        # 将 y0 与输入 x 融合
        fused = y0 + x

        fused = self.relu(fused)

        # 第二个卷积层进一步提取特征
        y1 = self.conv2(fused)

        # 输出最终的特征向量
        output = y1 + fused  # 残差连接

        return self.relu(output)


class EVRNet(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.1):
        super(EVRNet, self).__init__()
        self.mamba = Mamba(
            num_layers=1,  # Number of layers of the full model
            d_input=64,   # Dimension of each vector in the input sequence (i.e. token size)
            d_model=24,  # Dimension of the visible state space
            d_state=8,  # Dimension of the latent hidden states
            d_discr=16,  # Rank of the discretization matrix Δ
            ker_size=4,  # Kernel size of the convolution in the MambaBlock
            parallel=False,  # Whether to use the sequenial or the parallel implementation
        )

        # self.mamba = Mamba(
        #     num_layers=1,  # Number of layers of the full model
        #     d_input=64,   # Dimension of each vector in the input sequence (i.e. token size)
        #     d_model=32,  # Dimension of the visible state space
        #     d_state=32,  # Dimension of the latent hidden states
        #     d_discr=32,  # Rank of the discretization matrix Δ
        #     ker_size=4,  # Kernel size of the convolution in the MambaBlock
        #     parallel=False,  # Whether to use the sequenial or the parallel implementation
        # )
        self.temporal_block1 = TemporalBlock(1, 32, kernel_size=4, stride=3)
        # self.dropout1 = nn.Dropout2d(dropout_prob)  # 在第一个TemporalBlock后添加Dropout

        self.mkrb1 = MKRB(32, 32)
        # self.dropout2 = nn.Dropout2d(dropout_prob)  # 在MKRB1后添加Dropout

        self.spatial_block1 = SpatialBlock(32, 32, kernel_size=4, stride=3)
        # self.dropout3 = nn.Dropout2d(dropout_prob)  # 在第一个SpatialBlock后添加Dropout

        self.mkrb2 = MKRB(32, 32)
        # self.dropout4 = nn.Dropout2d(dropout_prob)  # 在MKRB2后添加Dropout

        self.spatial_block2 = SpatialBlock(32, 64, kernel_size=4, stride=3)
        # self.dropout5 = nn.Dropout2d(dropout_prob)  # 在第二个SpatialBlock后添加Dropout

        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化层
        self.fc_layer = nn.Linear(64, num_classes)  # 调整线性层的输入大小
        self.dropout6 = nn.Dropout(dropout_prob)  # 在全连接层前添加Dropout


    def forward(self, x):
        x = self.temporal_block1(x)
        # x = self.dropout1(x)  # 应用Dropout

        x = self.mkrb1(x)
        # x = self.dropout2(x)  # 应用Dropout

        x = self.spatial_block1(x)
        # x = self.dropout3(x)  # 应用Dropout

        x = self.mkrb2(x)
        # x = self.dropout4(x)  # 应用Dropout

        x = self.spatial_block2(x)
        # x = self.dropout5(x)  # 应用Dropout

        x = self.avg_pooling(x)

        x = x.view(x.size(0), -1)  # Flatten the tensor

        x = self.dropout6(x)  # 应用Dropout

        # print(x.shape)

        x = x.unsqueeze(-2)

        # print(x.shape)

        x = self.mamba(x)[0]

        x = x.view(x.size(0), -1)  # Flatten the tensor

        x = self.fc_layer(x)
        return x
