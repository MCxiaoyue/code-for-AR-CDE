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
from torch.nn import Parameter
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, Normalizer
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from mamba.mamba import Mamba
from mne.decoding import CSP
from scipy.io import loadmat
from DBPNet_pak.utils import makePath, cart2sph, pol2cart
from sklearn.preprocessing import scale
from scipy.interpolate import griddata
import torch.nn.functional as F
from mne.decoding import CSP
import math
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import seaborn as sns
# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_alpha(data, window_length, point0_low, point0_high):
    alpha_data = []
    for window in data:
        window_data0 = np.fft.fft(window, n=window_length, axis=0)
        window_data0 = np.abs(window_data0)
        window_data0 = np.sum(np.power(window_data0[point0_low:point0_high, :], 2), axis=0)
        window_data0 = np.log2(window_data0 / window_length)
        alpha_data.append(window_data0)
    alpha_data = np.stack(alpha_data, axis=0)
    return alpha_data


def sliding_window(eeg_datas, out_channels = 124, window_length = math.ceil(1024 * 2.0), overlap = 0.5):
    window_size = window_length
    stride = int(window_size * (1 - overlap))

    window_eeg = []

    for m in range(len(eeg_datas)):
        eeg = eeg_datas[m]
        windows = []
        for i in range(0, eeg.shape[0] - window_size + 1, stride):
            window = eeg[i:i+window_size, :]
            windows.append(window)
        window_eeg.append(np.array(windows))

    window_eeg = np.stack(window_eeg, axis=0).reshape(-1, window_size, out_channels)

    return window_eeg

def azim_proj(pos):
    """
    Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
    Imagine a plane being placed against (tangent to) a globe. If
    a light source inside the globe projects the graticule onto
    the plane the result would be a planar, or azimuthal, map
    projection.

    :param pos: position in 3D Cartesian coordinates
    :return: projected coordinates using Azimuthal Equidistant Projection
    """
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
    return pol2cart(az, math.pi / 2 - elev)


def gen_images(data, image_size=32):
    locs = loadmat('./DBPNet_pak/locs_orig1.mat')
    locs_3d = locs['data']
    locs_2d = []
    for e in locs_3d:
        locs_2d.append(azim_proj(e))

    locs_2d_final = np.array(locs_2d)
    grid_x, grid_y = np.mgrid[
                     min(np.array(locs_2d)[:, 0]):max(np.array(locs_2d)[:, 0]):image_size * 1j,
                     min(np.array(locs_2d)[:, 1]):max(np.array(locs_2d)[:, 1]):image_size * 1j]

    images = []
    for i in range(data.shape[0]):
        images.append(griddata(locs_2d_final, data[i, :], (grid_x, grid_y), method='cubic', fill_value=np.nan))
    images = np.stack(images, axis=0)

    images[~np.isnan(images)] = scale(images[~np.isnan(images)])
    images = np.nan_to_num(images)
    return images

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, (1, kernel_size, kernel_size), (1, stride, stride), (0, padding, padding), bias=bias)
        self.bn = nn.BatchNorm3d(out_channels)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, num_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = ConvLayer(in_channels, num_channels, 1, stride=1)
        self.conv2 = ConvLayer(num_channels, num_channels, 3, stride=stride, padding=1)
        self.conv3 = ConvLayer(num_channels, out_channels, 1, stride = 1)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = ConvLayer(in_channels, out_channels, 1, stride=stride)
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        x = x + identity
        x = self.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)
        return x

class FRBNet(nn.Module):
    def __init__(self):
        super(FRBNet, self).__init__()
        self.conv1 = ConvLayer(1, 32, 7, stride=2, padding=3)
        self.pool1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            ResBlock(32, 32, 64),
        )
        self.conv2 = nn.Conv3d(64, 64, 3, stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm3d(64)
        self.layer2 = nn.Sequential(
            ResBlock(64, 64, 128, stride=2),
        )
        self.conv3 = nn.Conv3d(128, 128, 3, stride = 1, padding = 1)
        self.bn3 = nn.BatchNorm3d(128)
        self.layer3 = nn.Sequential(
            ResBlock(128, 128, 256, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.conv5 = nn.Conv3d(256, 4, 1)
        self.bn5 = nn.BatchNorm3d(4)
        # self.linear = nn.Linear(8, 2)
    def forward(self, x):
        x = self.conv1(x)
        x = F.tanh(x)
        x = x.reshape(x.shape[0], -1, x.shape[3],  x.shape[4])
        x = self.pool1(x)
        x = x.reshape(x.shape[0], 32, -1, x.shape[2], x.shape[3])
        x = self.layer1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.avgpool(x)
        x = F.relu(x)
        x = torch.flatten(x, start_dim=1)
        return x


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout=0.1,
                 bias=True, add_bias_kv=False, add_zero_attn=False):
        """
        Multi-headed attention. This module can use the MULTIHEADATTENTION module built in Pytorch1.9.
        @param embed_dim: input embedding
        @param num_heads: number of heads
        @param attn_dropout: dropout applied on the attention weights
        @param bias: whether to add bias to q
        @param add_bias_kv: whether to add bias to kv
        @param add_zero_attn: add a new batch of zeros to the key and value sequences at dim=1.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, attn_mask=None):
        """
        @param query: (Time, Batch, Channel)
        @param key: (Time, Batch, Channel)
        @param value: (Time, Batch, Channel)
        @param attn_mask: mask that prevents attention to certain positions.
        @return: a tuple (output, weight), output shape (Time, Batch, Channel)
        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)

            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = attn_weights / k.shape[1] ** 0.5
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            try:
                attn_weights = attn_weights + attn_mask.unsqueeze(0)
            except:
                print(attn_weights.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # average attention weights over heads
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads
        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, layers, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                 embed_dropout=0.0, attn_mask=False):
        """
        Transformer encoder consisting of N layers. Each layer is a TransformerEncoderLayer.
        @param embed_dim: input embedding
        @param num_heads: number of heads
        @param layers: number of layers
        @param attn_dropout: dropout applied on the attention weights
        @param relu_dropout: dropout applied on the first layer of the residual block
        @param res_dropout: dropout applied on the residual block
        @param embed_dropout: dropout applied on the residual block
        @param attn_mask: whether to apply mask on the attention weights
        """
        super().__init__()
        self.dropout = embed_dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.attn_mask = attn_mask

        self.positionencoding = PositionalEncoding(embed_dim, embed_dropout)

        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = TransformerEncoderLayer(embed_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                relu_dropout=relu_dropout,
                                                res_dropout=res_dropout,
                                                attn_mask=attn_mask)
            self.layers.append(new_layer)

        self.register_buffer('version', torch.Tensor([2]))
        self.normalize = True
        if self.normalize:
            self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x_in):
        """
        @param x_in: embedded input of shape (src_len, batch, embed_dim)
        @param return_: whether to return the weight list
        @return: the last encoder layer's output of shape (src_len, batch, embed_dim).
            if return_=True, return tuple (output, weights)
        """
        # embed tokens
        x = self.positionencoding(self.embed_scale * x_in)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # encoder layers
        intermediates = [x]
        for layer in self.layers:
            x = layer(x)
            intermediates.append(x)

        if self.normalize:
            x = self.layer_norm(x)

        return x

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.max_source_positions


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                 attn_mask=False):
        """
        Encoder layer block
        @param embed_dim: input embedding
        @param num_heads: number of heads
        @param attn_dropout: dropout applied on the attention weights
        @param relu_dropout: dropout applied on the first layer of the residual block
        @param res_dropout: dropout applied on the residual block
        @param attn_mask: whether to apply mask on the attention weights
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )
        self.attn_mask = attn_mask
        self.attn_weights = None

        self.relu_dropout = relu_dropout
        self.res_dropout = res_dropout
        self.normalize_before = True

        self.fc1 = nn.Linear(self.embed_dim, self.embed_dim)  # The "Add & Norm" part in the paper
        self.fc2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None):
        """
        @param x: (seq_len, batch, embed_dim)
        @param x_k: (seq_len, batch, embed_dim)
        @param x_v: (seq_len, batch, embed_dim)
        @param return_: whether to return the weight list
        @return: encoded output of shape (batch, src_len, embed_dim).
            if return_=True, return tuple (output, weight)
        """
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        mask = buffered_future_mask(x, x_k) if self.attn_mask else None
        if x_k is None and x_v is None:
            x, _ = self.self_attn(query=x, key=x, value=x, attn_mask=mask)
        else:
            x_k = self.maybe_layer_norm(0, x_k, before=True)
            x_v = self.maybe_layer_norm(0, x_v, before=True)
            x, _ = self.self_attn(query=x, key=x_k, value=x_v, attn_mask=mask)
        self.attn_weights = _
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        # Position-wise feed forward
        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.res_dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(1, x, after=True)

        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x

def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)

def buffered_future_mask(tensor, tensor2=None):
    dim1 = dim2 = tensor.size(0)
    if tensor2 is not None:
        dim2 = tensor2.size(0)
    future_mask = torch.triu(fill_with_neg_inf(torch.ones(dim1, dim2)), 1+abs(dim2-dim1))
    if tensor.is_cuda:
        future_mask = future_mask.to(torch.device('cuda:0'))
    return future_mask[:dim1, :dim2]

#@save
class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

class TABNet(nn.Module):
    def __init__(self, in_channels):
        super(TABNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=7
            ),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.flat = nn.Flatten()
        self.linear1 = nn.Linear(64, 16)
        self.linear2 = nn.Linear(16, 4)
        self.cm_attn = TransformerEncoder(
            embed_dim=in_channels,
            num_heads=1,
            layers=1,
            attn_dropout=0,
            relu_dropout=0,
            res_dropout=0,
            embed_dropout=0,
            attn_mask=False
        )

    def forward(self, eeg):
        x = torch.squeeze(eeg, dim=1)  # x形状为(batch_size, channels, timestamps)
        x = x.permute(2, 0, 1)
        x = self.cm_attn(x)
        x = x.permute(1, 2, 0)
        x = self.conv(x)
        x = self.flat(x)
        x = self.linear1(x)
        x = F.sigmoid(x)
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.linear2(x)
        x = F.relu(x)
        return x


class DBPNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DBPNet, self).__init__()
        self.fre = FRBNet()
        self.tem = TABNet(in_channels)
        self.linear = nn.Linear(8, num_classes)
    def forward(self, x1, x2):
        seq = self.tem(x1)
        fre = self.fre(x2)
        x = torch.cat((seq, fre), dim=1)
        x = self.linear(x)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


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
subjects = ['19']  # '12', '20'   '14', '21'   '15', '2A2'
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
#
delta_low = 1
delta_high = 3
theta_low = 4
theta_high = 7
alpha_low = 8
alpha_high = 13
beta_low = 14
beta_high = 30
gamma_low = 31
gamma_high = 50
window_length = math.ceil(1024 * 2.0)
frequency_resolution = 1024 / window_length
point0_low = math.ceil(delta_low / frequency_resolution)
point0_high = math.ceil(delta_high / frequency_resolution) + 1
point1_low = math.ceil(theta_low / frequency_resolution)
point1_high = math.ceil(theta_high / frequency_resolution) + 1
point2_low = math.ceil(alpha_low / frequency_resolution)
point2_high = math.ceil(alpha_high / frequency_resolution) + 1
point3_low = math.ceil(beta_low / frequency_resolution)
point3_high = math.ceil(beta_high / frequency_resolution) + 1
point4_low = math.ceil(gamma_low / frequency_resolution)
point4_high = math.ceil(gamma_high / frequency_resolution) + 1

image_data_array = image_data_array.transpose(0, 2, 1)
window_eeg = sliding_window(image_data_array)
fre1_eeg = to_alpha(window_eeg, window_length, point0_low, point0_high)
fre1_data = gen_images(fre1_eeg)
fre2_eeg = to_alpha(window_eeg, window_length, point1_low, point1_high)
fre2_data = gen_images(fre2_eeg)
fre3_eeg = to_alpha(window_eeg, window_length, point2_low, point2_high)
fre3_data = gen_images(fre3_eeg)
fre4_eeg = to_alpha(window_eeg, window_length, point3_low, point3_high)
fre4_data = gen_images(fre4_eeg)
fre5_eeg = to_alpha(window_eeg, window_length, point4_low, point4_high)
fre5_data = gen_images(fre5_eeg)

fre_data = np.stack([fre1_data, fre2_data, fre3_data, fre4_data, fre5_data], axis=1)

image_data_array = image_data_array.transpose(0, 2, 1)
labels_array = np.array(all_labels)

# 初始化 CSP
csp = CSP(n_components=122, reg='ledoit_wolf', log=None, cov_est='concat', transform_into='csp_space',
          norm_trace=True)

# 使用 fit_transform 方法
image_data_array = csp.fit_transform(image_data_array, labels_array)
image_data_array = image_data_array.transpose(0, 2, 1)
image_data_array = sliding_window(image_data_array, out_channels=122).transpose(0, 2, 1)
print(image_data_array.shape)
print(fre_data.shape)
print(labels_array.shape)

# 使用函数可视化第一个样本的EEG数据
plot_eeg_data_together(image_data_array)

# 测试数据的索引
test_indices = [i for i in range(0, image_data_array.shape[0], 10)]  # [6, 9, 8, 110, 111, 112]  1775
print("Test indices:", test_indices)

# 切分数据
train_data = np.delete(image_data_array, test_indices, axis=0)
test_data = image_data_array[test_indices]

# 切分标签
train_labels = np.delete(labels_array, test_indices, axis=0)
test_labels = labels_array[test_indices]

#
train_fres = np.delete(fre_data, test_indices, axis=0)
test_fres = fre_data[test_indices]

# 转换为Tensor
train_data = torch.tensor(train_data, dtype=torch.float32).unsqueeze(1)  # (batch_size, 1, 22, 192)
train_fres = torch.tensor(train_fres, dtype=torch.float32).unsqueeze(1)
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_data = torch.tensor(test_data, dtype=torch.float32).unsqueeze(1)  # (batch_size, 1, 5, 32, 32)
test_fres = torch.tensor(test_fres, dtype=torch.float32).unsqueeze(1)
test_labels = torch.tensor(test_labels, dtype=torch.long)

# 在循环外部初始化列表
train_losses = []
test_losses = []

# 创建数据加载器
torch.manual_seed(42)  # 设置固定的随机种子
train_dataset = TensorDataset(train_data, train_fres, train_labels)
test_dataset = TensorDataset(test_data, test_fres, test_labels)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


# 初始化模型、损失函数和优化器
num_classes = 3
model = DBPNet(in_channels=122, num_classes=num_classes).to(device)  # 将模型移动到 GPU
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
    for inputs, fres, targets in tqdm(train_loader, desc=f'Training Epoch [{epoch + 1}/{num_epochs}]'):
        inputs, fres, targets = inputs.to(device), fres.to(device), targets.to(device)  # 将数据和标签移动到 GPU
        optimizer.zero_grad()
        outputs = model(inputs, fres)
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
        for inputs, fres, targets in test_loader:
            inputs, fres, targets = inputs.to(device), fres.to(device), targets.to(device)  # 将数据和标签移动到 GPU
            outputs = model(inputs, fres)
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
                   'classification_checkpoint/' + str(subjects[0]) + '/DBPNet/best_model_weights.pth')
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

model = DBPNet(in_channels=122, num_classes=num_classes).to(device)  # 将模型移动到 GPU
# 训练完成后，使用最佳模型权重更新模型
model.load_state_dict(torch.load('./classification_checkpoint/' + str(subjects[0]) + '/DBPNet/best_model_weights.pth'))
model.eval()

print('Training complete and best weights saved.')

# 测试模型
correct_test = 0
total_test = 0
all_predicted_labels = []
all_true_labels = []

with torch.no_grad():
    for inputs, fres, targets in tqdm(test_loader, desc='Testing Final Model'):
        inputs, fres, targets = inputs.to(device), fres.to(device), targets.to(device)  # 将数据和标签移动到 GPU
        outputs = model(inputs, fres)
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
plot_and_save_loss_curves(train_losses, test_losses, filename='classification_checkpoint/' + str(subjects[0]) + '/DBPNet/loss_curves.png')

print("Loss curves have been saved.")

