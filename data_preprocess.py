import torch
import torchaudio
import pandas as pd
import io

from utils.speechlib import *
import scipy.interpolate as spi
import torchaudio.transforms as T
from sklearn.preprocessing import MinMaxScaler
import soundfile
import auditok


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
            axes[c].set_ylabel(f"Channel {c + 1}")
    figure.suptitle("waveform")
    plt.show(block=False)


# 当前版本缺少批量处理
if __name__ == '__main__':

    wavfilename = "../CollectedData/WAV/sober/SingleWav1.wav"
    waveform, sample_rate = torchaudio.load(wavfilename)  # 读取音频文件
    print(waveform)
    # plot_waveform(waveform, sample_rate)

    esd_df = pd.read_csv('../CollectedData/OutBINS/out_TEST_BINS1.csv')  # 读取ESD数据文件
    esd_time = esd_df['time']
    esd_data = esd_df['max']

    # 音频下采样至16Khz
    resample_rate = 16000
    resampler = T.Resample(sample_rate, resample_rate, dtype=waveform.dtype)
    resampled_waveform = resampler(waveform)
    # plot_waveform(resampled_waveform, resample_rate)

    # 超声波上采样至16Khz
    num_channels, num_frames = resampled_waveform.shape
    time = torch.arange(0, num_frames) / resample_rate  # 音频的时间点
    ipo1 = spi.splrep(esd_time.values, esd_data.values, k=1)  # 样本点导入，生成参数
    upsample_esd = spi.splev(time, ipo1)  # 根据观测点和样条参数，生成插值，观测点设置为音频的时间坐标

    # 音频   min-max scaling
    min_max_scaler1 = MinMaxScaler(feature_range=(-1, 1), copy=True)  # 定义归一化的范围为[-1,1]
    audio_data_minmax = min_max_scaler1.fit_transform(resampled_waveform.numpy().reshape(-1, 1))

    # 超声波 min-max scaling
    min_max_scaler2 = MinMaxScaler(feature_range=(0, 1), copy=True)  # 定义归一化的范围为[0,1]
    esd_data_minmax = min_max_scaler2.fit_transform(esd_data.values.reshape(-1, 1))

    # 超声波和音频信号相乘
    multilsignal = np.multiply(upsample_esd, audio_data_minmax.reshape(-1, 1).squeeze())
    wav_path = r"../CollectedData/result.wav"
    soundfile.write(wav_path, multilsignal, 16000)

    # 语音分割并保存
    audio_regions = auditok.split(
        wav_path,
        min_dur=0.2,  # minimum duration of a valid audio event in seconds
        max_dur=4,  # maximum duration of an event
        max_silence=0.3,  # maximum duration of tolerated continuous silence within an event
        energy_threshold=55  # threshold of detection
    )

    gapless_region = sum(audio_regions)

    gapless_region.save("result.wav")
