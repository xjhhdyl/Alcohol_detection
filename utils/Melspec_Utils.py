import torchaudio
import numpy as np
import soundfile as sf
import librosa
import random
import torch
import torch.utils.data
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn

'''
该文件内含 hifigan的melspec提取代码 
主功能在 melspec_fn_hifigan
'''


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    torch.clamp(y, min=-1.0, max=1.0)

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[str(fmax) + '_' + str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
                                mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

    spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

    spec = torch.matmul(mel_basis[str(fmax) + '_' + str(y.device)], spec)
    spec = spectral_normalize_torch(spec)
    return spec


def melspec_fn_hifigan(wave_path):
    '''
    该函数的melspec特征提取过程和 hifigan 保持 一致

    :param wave_path: 语音路径
    :return: mel ##
    '''
    ## 读取时域波形
    audio, _ = librosa.load(wave_path, sr=22050)
    audio = torch.FloatTensor(audio)
    audio = audio.unsqueeze(0)

    mel = mel_spectrogram(audio, 1024, 80, 22050, 256, 1024, 0, 8000, center=False)

    return mel.squeeze()  # [80, frames]


def melspec_fn_hifigan_with_silencekill(wave_path):
    """
    该函数的melspec特征提取过程和 hifigan 保持 一致

    :param wave_path: 语音路径
    :return: mel ##
    """
    ## 读取时域波形
    audio, _ = librosa.load(wave_path, sr=22050)
    audio, _ = librosa.effects.trim(audio, top_db=20)
    audio = torch.FloatTensor(audio)
    audio = audio.unsqueeze(0)

    mel = mel_spectrogram(audio, 1024, 80,
                          22050, 256, 1024, 0, 8000,
                          center=False)

    return mel.squeeze()  # [80, frames]


if __name__ == '__main__':
    wav_path = '/bigdata/VCTK-Corpus/wav48/p225/p225_001.wav'
    mel = melspec_fn_hifigan(wav_path)
    print(mel.shape)  # [80, frames]
