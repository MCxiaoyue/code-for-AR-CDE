from pydub import AudioSegment

def get_wav_duration_with_pydub(file_path):
    audio = AudioSegment.from_wav(file_path)
    # 音频时长（毫秒）转换为秒
    duration = len(audio) / 1000.0
    return duration

# 示例用法
file_path = 'guitar.wav'  # 替换为你的 .wav 文件路径
duration = get_wav_duration_with_pydub(file_path)
print(f"音频时长: {duration:.2f} 秒")