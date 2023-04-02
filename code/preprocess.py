from pathlib import Path
from utils.Melspec_Utils import melspec_fn_hifigan
import numpy as np
import os


def extract_mel(wav_datadir, output_dir):
    """
    提取语音.wav数据集的特征
    :param wav_datadir:
    :param output_dir:

    :return:
    """

    src_wavp = Path(wav_datadir)
    for x in src_wavp.rglob('*'):
        if x.is_dir():
            Path(str(x.resolve()).replace(wav_datadir, output_dir)).mkdir(parents=True, exist_ok=True)
    print("创建相同目录,开始提取特征")

    # 提取特征
    wavpaths = [x for x in src_wavp.rglob('*.wav') if x.is_file()]

    ttsum = len(wavpaths)  # 总语音数量
    k = 0
    for wp in wavpaths:
        k += 1
        the_wavpath = str(wp)
        the_melpath = str(wp).replace(wav_datadir, output_dir).replace('.wav', '')
        print(the_melpath)
        mel = melspec_fn_hifigan(wave_path=the_wavpath)

        print(f"mel s:{mel.shape[-1]},{k}|{ttsum}")

        np.save(the_melpath, mel)


def generate_scp_dataset(dataset_dir):
    with open('Train_Scp.txt', 'a', encoding='utf-8') as txtf:
        for dirname, subdirs, files in os.walk(dataset_dir):
            for f in files:
                if f.split('.')[-1] == 'npy':
                    txtf.write(os.path.join(dirname, f) + "\n")
    print("写入表单")


if __name__ == '__main__':
    sober_wav_dir = r"D:\Research_projects\Alcohol_detection\CollectedData\WAV\sober"
    sober_mel_dir = r"D:\Research_projects\Alcohol_detection\CollectedData\meldata"
    extract_mel(sober_wav_dir, sober_mel_dir)
