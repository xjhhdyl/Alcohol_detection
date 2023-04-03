import os
import pandas as pd

"""
    生成数据集-csv文件
    目前还差融合超声波
"""

WAV_SOBER_DATA_DIR = "WAV/sober"  # 醉酒语音数据的文件路径
WAV_INTOXICATE_DATA_DIR = "WAV/intoxicate"  # 清醒语音数据的文件路径
DATASET = "../CollectedData/data.csv"  # 数据集存放的路径
#  BINS_DATA_DIR = "../CollectedData/BINS"  # 超声波数据集的文件路径

# 0-清醒  1-醉酒
dataset_df = pd.DataFrame(columns=['wav_file_name', 'classID', 'class'])  # 创建数据集

wav_sober = os.listdir(WAV_SOBER_DATA_DIR)
wav_intoxicate = os.listdir(WAV_INTOXICATE_DATA_DIR)
#  bins_folds = os.listdir(BINS_DATA_DIR)

for i in wav_sober:
    dataset_df.loc[len(dataset_df.index)] = [i, 0, "sober"]  # 向数据集插入一条清醒数据记录

for i in wav_intoxicate:
    dataset_df.loc[len(dataset_df.index)] = [i, 1, "intoxicate"]  # 向数据集插入一条醉酒数据记录

dataset_df.to_csv(DATASET, index=False)







