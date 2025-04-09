import librosa
import os
import numpy as np
import cv2
from PIL import Image

sr = 11025  # Sample rate. 设置采样率为 11025 Hz。采样率表示每秒对信号的采样次数，这里是音频信号的采样率。
n_fft = 2048  # fft points (samples) 设置 FFT（快速傅里叶变换）的点数为 2048，用于频谱分析和变换。
frame_shift = 0.0125  # seconds 设置帧移（frame shift）为 0.0125 秒。在音频处理中，通常将音频信号分成短时间窗口进行处理，帧移表示相邻两个窗口的时间间隔。
frame_length = 0.15  # seconds : 设置帧长（frame length）为 0.05 秒。这是短时傅里叶变换（STFT）所用的每个窗口的长度。
hop_length = int(sr*frame_shift)   # samples. 计算帧移的样本数。采样率乘以帧移得到每个窗口之间的样本数。
win_length = int(sr*frame_length)  # samples.  计算帧长的样本数。采样率乘以帧长得到每个窗口的样本数。
n_mels = 227  # 设置生成 Mel 滤波器组的数量为 80。Mel 频率倒谱系数（MFCC）是一种常用的音频特征提取方法，这里设置了用于提取 MFCC 的 Mel 滤波器组数量。
power = 1.2  # Exponent for amplifying the predicted magnitude 设置用于放大预测幅度的指数。在信号还原时，预测的幅度可能需要进行一定程度的放大。
n_iter = 100  # Number of inversion iterations 设置反变换的迭代次数。在进行信号还原时，可能需要进行多次迭代以获得更准确的结果。
preemphasis = .97  # or None设置预加重滤波器的系数。预加重滤波器用于突出高频部分，帮助改善信噪比。
max_db = 100  # 设置能量值的上限。在进行能量值计算时，可能会对能量进行限制，确保其不超过该阈值。
ref_db = 20  # 设置参考能量值的阈值。用于计算相对能量的参考值。
top_db = 15  # 设置能量的上限范围。用于对信号的能量范围进行限制或压缩，确保信号的动态范围在可接受的范围内。


def get_spectrograms(fpath):
    '''
    Returns normalized log(melspectrogram) and log(magnitude) from `sound_file`.
    用于从音频文件中提取规范化的对数梅尔频谱图（log(melspectrogram)）和对数幅度谱图（log(magnitude)）
    Args:
      sound_file: A string. The full path of a sound file.

    Returns:
      mel: A 2d array of shape (T, n_mels) <- Transposed
      mag: A 2d array of shape (T, 1+n_fft/2) <- Transposed
    '''
    # Loading sound file
    y, sr = librosa.load(fpath, sr=11025)  # 加载指定路径的音频文件 (fpath)，并指定采样率为 11025 Hz。librosa.load 函数返回音频信号 (y) 和采样率 (sr)。

    # # # Trimming 对加载的音频信号进行修剪，去除静音部分。
    y, _ = librosa.effects.trim(y, top_db=top_db)  # 通过设定的信噪比门限(top_db)进行修剪，返回修剪后的音频信号(y)和修剪的区间信息（用 _ 表示，因为在代码中未使用该信息）。

    # Preemphasis 对修剪后的音频信号进行预加重处理。该处理用于突出高频部分，通过在信号的前一个样本上减去预设的系数 (preemphasis) 乘以前一个样本的值，然后将得到的数组与原始信号的第一个样本连接起来。
    y = np.append(y[0], y[1:] - preemphasis * y[:-1])

    # stft 对预加重后的音频信号进行短时傅里叶变换（STFT），得到线性频谱表示。
    linear = librosa.stft(y=y,
                          n_fft=n_fft,  # n_fft 表示傅里叶变换的窗口大小
                          hop_length=hop_length,  # hop_length 表示相邻两个窗口的样本数
                          win_length=win_length)  # win_length 表示每个窗口的样本数。

    # magnitude spectrogram
    mag = np.abs(linear)  # (1+n_fft//2, T) 计算线性频谱的幅度谱图。取频谱的绝对值，得到每个频点的振幅。

    # mel spectrogram 通过 librosa.filters.mel 函数生成梅尔滤波器组，用于将线性频谱转换为梅尔频谱。n_mels 表示生成的梅尔滤波器的数量。
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_basis, mag)  # (n_mels, t) 线性频谱的振幅谱图通过梅尔滤波器组转换为对数梅尔频谱图。

    # to decibel 将对数梅尔频谱图和幅度谱图转换为对数刻度。避免对数值过小导致无穷大的情况，通过 np.maximum 确保值不小于一个设定的阈值（1e-5）。
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))

    # normalize 对对数梅尔频谱图和对数幅度谱图进行归一化，将数值限制在 [1e-8, 1] 的范围内。
    mel = np.clip((mel - ref_db + max_db) / max_db, 1e-8, 1)
    mag = np.clip((mag - ref_db + max_db) / max_db, 1e-8, 1)

    # Transpose 将梅尔频谱图和幅度谱图进行转置，得到最终的形状为 (T, n_mels) 和 (T, 1+n_fft/2) 的二维数组。同时将数据类型转换为 np.float32。
    mel = mel.T.astype(np.float32)  # (T, n_mels)
    mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)
    # print('mel是：',mel)
    # print('mag是：',mag)

    return mel, mag

# 定义单词序列
word_sequence = ["my", "dad", "is", "a", "policeman", "he", "will", "always", "become", "my", "hero"]


if __name__ == '__main__':
    input_directory = './orign_wav'  # 替换为包含.wav文件的目录
    file_name = './10words_repe11/'

    os.makedirs(file_name, exist_ok=True)

    # 定义单词序列
    word_sequence = ["my", "dad", "is", "a", "policeman", "he", "will", "always", "become", "my", "hero"]
    # word_sequence = ["my", "dad", "is", "a", "policeman", "he", "will", "always", "become", "hero"]
    # # 创建一个字典来存储每个单词第一次出现的位置
    # word_positions = {word: idx for idx, word in enumerate(word_sequence)}
    # for idx, word in enumerate(word_sequence):
    #     print(idx, word)

    word_positions = {
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

    index = 1

    for k in range(11):  # 4个被试
        for word in word_sequence:
            for _ in range(5):  # 每个单词重复5次
                input_file = os.path.join(input_directory, f'{word}.wav')
                if not os.path.exists(input_file):
                    print(f"Warning: File {input_file} does not exist.")
                    continue

                mel, _ = get_spectrograms(input_file)

                # print(mel.shape)

                # 图像处理
                target_len = 128 * 128
                if len(mel.reshape(-1)) < target_len:
                    original_arr = mel.flatten()
                    repeated_data = []
                    while len(repeated_data) < target_len:
                        repeated_data = np.concatenate((repeated_data, original_arr))
                    repeated_data = repeated_data[:target_len]
                    mel = repeated_data.reshape(128, 128)
                elif len(mel.reshape(-1)) >= target_len:
                    if len(mel.reshape(-1)) > target_len:
                        continue
                    else:
                        mel = cv2.resize(mel, (128, 128))

                # 将数值范围调整到0-255
                gray_image = mel * 255
                gray_image = gray_image.astype(np.uint8)

                # # 将灰度图像转换为24位颜色图像（RGB）
                # # 正确的方式是沿着最后一个轴重复三次以创建RGB图像
                rgb_image = np.stack([gray_image] * 3, axis=-1)  # 创建一个具有三个通道的图像
                # rgb_image = gray_image

                # 获取单词第一次出现的位置作为标签
                label = str(word_positions[word]) #  - 1

                # 保存为24位PNG图像，添加单词的位置作为标签
                cv2.imwrite(os.path.join(file_name, f'B_{index}_{label}.png'), rgb_image)

                # # 打开图像并确认它是24位图像
                # img = Image.open(os.path.join(file_name, f'B_{index}_{label}.png'))
                # # 如果需要，可以再次确认它是24位图像
                # img = img.convert("RGB")  # 确保图像为24位RGB

                # img.save(os.path.join(file_name, f'B_{index}_{label}.png'))
                # index += 1
                # # 打开图像并确认它是24位图像
                # img = Image.open('./' + file_name + '/B_' + str(index) + ".png")
                # # 如果需要，可以再次确认它是24位图像
                # img = img.convert("RGB")  # 确保图像为24位RGB
                #
                # img.save('./' + file_name + '/B_' + str(index) + ".png")
                index += 1

