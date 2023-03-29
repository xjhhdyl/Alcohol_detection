import os

import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import GridSearchCV
import Extracting_features as Exfeatures
from sklearn.svm import SVC

"""
    机器学习SVM进行醉酒二分类
"""

WAV_SOBER_DATA_DIR = "../CollectedData/WAV/sober"  # 醉酒语音数据的文件路径
WAV_INTOXICATE_DATA_DIR = "../CollectedData/WAV/intoxicate"  # 清醒语音数据的文件路径

class_labels = {0: "sober",
                1: "intoxicate"}

features_list = {"ae_mean": [], "ae_var": [], "rms_mean": [], "rms_var": [], "zcr_mean": [], "zcr_var": [],
                 "chroma_stft_mean": [],
                 "chroma_stft_var": [], "spec_centroid_mean": [], "spec_centroid_var": [], "spec_cont_mean": [],
                 "spec_cont_var": [],
                 "spec_bw_mean": [], "spec_bw_var": [], "percep_mean": [], "percep_var": [], "tempo_mean": [],
                 "tempo_var": [],
                 "roll_off_mean": [], "roll_off_var": [], "roll_off50_mean": [], "roll_off50_var": [],
                 "roll_off25_mean": [], "roll_off25_var": [],
                 "log_mel_mean": [], "log_mel_var": [], "mfcc_mean": [], "mfcc_var": [], "spec_mean": [],
                 "spec_var": [],
                 "mag_spec_mean": [], "mag_spec_var": [], "mel_mean": [], "mel_var": []}

data_paths = []  # 所有语音数据的路径

wav_sober = os.listdir(WAV_SOBER_DATA_DIR)
wav_intoxicate = os.listdir(WAV_INTOXICATE_DATA_DIR)

for i in wav_sober:
    filename = WAV_SOBER_DATA_DIR + "/" + i
    data_paths.append([filename, 0])

for i in wav_intoxicate:
    filename = WAV_INTOXICATE_DATA_DIR + "/" + i
    data_paths.append([filename, 1])

for i in range(len(data_paths)):
    sample, sr = librosa.load(path=data_paths[i][0])
    ae = Exfeatures.amplitude_envelope(sample, frame_size=2048, hop_length=512)
    ae_m, ae_v = ae.mean(), ae.var()
    features_list["ae_mean"].append(ae_m)
    features_list["ae_var"].append(ae_v)

    rms = Exfeatures.Rms(sample)
    rms_m, rms_v = rms.mean(), rms.var()
    features_list["rms_mean"].append(rms_m)
    features_list["rms_var"].append(rms_v)

    zcr = Exfeatures.Zcr(sample)
    zcr_m, zcr_v = zcr.mean(), zcr.var()
    features_list["zcr_mean"].append(zcr_m)
    features_list['zcr_var'].append(zcr_v)

    mag_spec = Exfeatures.Mag_spec(sample)
    features_list["mag_spec_mean"].append(mag_spec.mean())
    features_list["mag_spec_var"].append(mag_spec.var())

    spec = Exfeatures.spectrogram(sample)
    features_list["spec_mean"].append(spec.mean())
    features_list['spec_var'].append(spec.var())

    mel_spec = Exfeatures.log_mel(sample, sr)
    features_list["mel_mean"].append(mel_spec.mean())
    features_list["mel_var"].append(mel_spec.var())

    mfcc = Exfeatures.Mfcc(sample, sr)
    features_list["mfcc_mean"].append(mfcc.mean())
    features_list["mfcc_var"].append(mfcc.var())

    chroma_stft = Exfeatures.Chroma_stft(sample, sr)
    features_list["chroma_stft_mean"].append(chroma_stft.mean())
    features_list['chroma_stft_var'].append(chroma_stft.var())

    spec_centriod = Exfeatures.Spec_centriod(sample, sr)
    features_list['spec_centroid_mean'].append(spec_centriod.mean())
    features_list['spec_centroid_var'].append(spec_centriod.var())

    spec_roll = Exfeatures.spec_roll_off(sample, sr)
    features_list["roll_off_mean"].append(spec_roll.mean())
    features_list['roll_off_var'].append(spec_roll.var())

    spec_roll50 = Exfeatures.spec_roll_off50(sample, sr)
    features_list["roll_off50_mean"].append(spec_roll50.mean())
    features_list['roll_off50_var'].append(spec_roll50.var())

    spec_roll25 = Exfeatures.spec_roll_off25(sample, sr)
    features_list["roll_off25_mean"].append(spec_roll25.mean())
    features_list['roll_off25_var'].append(spec_roll25.var())

    spec_contr = Exfeatures.spec_contrast(sample, sr)
    features_list["spec_cont_mean"].append(spec_contr.mean())
    features_list['spec_cont_var'].append(spec_contr.var())

    tempo = Exfeatures.tempogram(sample, sr)
    features_list["tempo_mean"].append(tempo.mean())
    features_list["tempo_var"].append(tempo.var())

    spec_bw = Exfeatures.spec_bandwidth(sample, sr)
    features_list["spec_bw_mean"].append(spec_bw.mean())
    features_list['spec_bw_var'].append(spec_bw.var())

    log_me = Exfeatures.log_mel(sample, sr)
    features_list["log_mel_mean"].append(log_me.mean())
    features_list['log_mel_var'].append(log_me.var())

keys = features_list.keys()

for key in keys:
    print(len(features_list[key]), key)

del features_list["percep_mean"]
del features_list["percep_var"]

feature_df = pd.DataFrame(features_list)
df1 = feature_df.copy()

file_names = [data_paths[i][0].split("/")[4] for i in range(len(data_paths))]
class_index = [data_paths[i][1] for i in range(len(data_paths))]

df1["file_name"] = file_names
df1["class_id"] = class_index

df1.to_csv('features_sounds.csv')

y = df1["class_id"]
y = np.array(y)

del df1["file_name"]
del df1["class_id"]

X = np.array(df1)

# 缩放数据
mmscale = MinMaxScaler()

X_sc = mmscale.fit_transform(X)

# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(X_sc, y, random_state=6, test_size=0.2)

clf = SVC(C=10, gamma=0.001, kernel='linear')

# clf = SVC(C=10, gamma=0.001, kernel='linear')
# grid = {"C": [10, 1e2, 1e3], "gamma": [1e-3, 1e-4, 5e-3], "kernel": ['linear', 'rbf', ]}
# grid_s = GridSearchCV(clf, grid)
# grid_s.fit(x_train, y_train)
# print(grid_s.best_params_)

clf.fit(x_train, y_train)
print("Training accuracy is :-", clf.score(x_train, y_train))

y_train_pred_svm = clf.predict(x_train)

print("Training precision score is :-", precision_score(y_train, y_train_pred_svm, average="macro"))
print("Training recall is :-", recall_score(y_train, y_train_pred_svm, average="macro"))

y_pred_svm = clf.predict(x_test)
print()
print("Testing precision score is :-", precision_score(y_test, y_pred_svm, average="macro"))
print("Testing recall is :-", recall_score(y_test, y_pred_svm, average="macro"))
print()
print("Testing f1 score is :-", f1_score(y_test, y_pred_svm, average="macro"))

print(classification_report(y_test, y_pred_svm))