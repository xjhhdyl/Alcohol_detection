import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as spi
import numpy as np
import torch
import torchaudio

def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")
    plt.show(block=False)

# 当前版本缺少批量处理
if __name__== '__main__':
    filename = "SingleWav1.wav"
    waveform, sample_rate = torchaudio.load(filename) # 读取音频文件
    plot_waveform(waveform, sample_rate)

    esd_df = pd.read_csv('../CollectedData/OutBINS/out_TEST_BINS6.csv') # 读取ESD数据文件
    esd_time = esd_df['time']
    esd_data = esd_df['max']

    # 音频下采样
    # print(audio_data_minmax.squeeze().shape)
    # audio_data_minmax_downsampled = librosa.resample(audio_data.astype(np.float32), orig_sr = fs, target_sr = 16000) # 下采样至16Khz  
