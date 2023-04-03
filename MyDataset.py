import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from Create_Hparams import Create_Train_Hparams


## 下面两个函数 ，实现，将一个二维矩阵 补零到 指定长度。 （补一列一列的零）. 如果超过 指定的seglen，则切掉多余的。
def pad(x, seglen, mode='wrap'):
    pad_len = seglen - x.shape[1]
    y = np.pad(x, ((0, 0), (0, pad_len)), mode=mode)
    return y


def segment(x, seglen=128):
    '''
    :param x: npy形式的mel [80,L]
    :param seglen: padding长度
    :return: padding mel
    '''
    ## 该函数将melspec [80,len] ，padding到固定长度 seglen
    if x.shape[1] < seglen:
        y = pad(x, seglen)
    elif x.shape[1] == seglen:
        y = x
    else:
        r = np.random.randint(x.shape[1] - seglen)  ## r : [0-  (L-128 )]
        y = x[:, r:r + seglen]
    return y


class MeldataSet(Dataset):
    def __init__(self, scp_dir, seglen):
        self.scripts = []
        self.seglen = seglen
        with open(scp_dir, encoding='utf-8') as f:
            for l in f.readlines():
                self.scripts.append(l.strip('\n'))
        self.L = len((self.scripts))
        # self.speaker_names_set = list(set([p.split("\\")[1] for p in self.scripts]))
        # print(self.speaker_names_set.__len__())
        pass

    def __getitem__(self, index):
        src_path = self.scripts[index]
        # src_mel = np.load(src_path)  ## 从硬盘将数据 读入内存的一个io过程。.npy
        src_mel = segment(np.load(src_path), seglen=self.seglen)
        if (src_path.split("\\")[1] == "sober"):
            label = 0
        else:
            label = 1
        return torch.FloatTensor(src_mel), label  ## 【80,2000】

    def __len__(self):
        return self.L
        pass


def generate_pairs_scripts(meldatadir_name, save_log_dir, hp: Create_Train_Hparams):
    data_dir_p = Path(meldatadir_name)  # 转为Path（）对象
    if save_log_dir.exists() == False:
        print("表单的保存目录不存在！")
        exit()

    src_data_paths_list = []  # 保存 梅尔谱的 存储路径 的Path（）对象

    ## 获取全部梅尔谱数据集路径
    for meldatap in data_dir_p.rglob('*.npy'):
        mel_i_Len = np.load(str(meldatap)).shape[-1]  ## 加载第 i 条 melspec的长度
        if mel_i_Len >= hp.min_train_mellen:
            src_data_paths_list.append(meldatap)  # 注意添加Path()对象
            ## src data paths list 保存了每条梅尔谱的 相对路径 的 Path（）对象

    ######################   训练集和验证集的 划分，生成两个 。txt文件 ##############################
    ###  注意，我们自己随机 划分 训练集、测试集。（不划分验证集）。比例自己选择。
    ### 由于这是一个 说话人 分类任务，则，从每个说话人分类中，提取 90%比例的数据进行 训练。
    ### 剩下的10% 作为 测试。

    ##  取出该 梅尔谱数据集中的说话人编号。
    speakers_dirs = [x.parts[-1] for x in data_dir_p.glob("*") if
                     x.is_dir()]  ##['0001', '0003', '0007', '0012', '0015']

    ## 创建2个 字典 保存训练集和测试集路径。 其结构为 {  "0001" : [路径1，路径2.。。。]  }。
    train_scp_dict = {}
    test_scp_dict = {}
    for spk_name in speakers_dirs:
        train_scp_dict[spk_name] = []
        test_scp_dict[spk_name] = []
    ## 字典创建完毕

    ##  循环路径， 将路径添加到 字典中。
    for spk_name in speakers_dirs:
        this_spk_mels = []
        ## 添加当前说话人的梅尔谱路径
        for wav_p in src_data_paths_list:
            if wav_p.parts[1] == spk_name:  ## 路径的第二个字符串即为 说话人文件夹编号
                this_spk_mels.append(wav_p)
        ##  按比例对该说话人的路径，进行切分。
        Pathnums = len(this_spk_mels)
        r_r = int(Pathnums * hp.train_ratio)
        random.shuffle(this_spk_mels)  ###  随机打乱
        this_spk_train_paths = this_spk_mels[:r_r]  ## 取前90%
        this_spk_test_paths = this_spk_mels[r_r:]  ## 取 后 10%

        ## 将上面2个数组存储到 字典里
        train_scp_dict[spk_name] += this_spk_train_paths  # 这里 列表加列表 必须使用 +=
        test_scp_dict[spk_name] += this_spk_test_paths

    ## 将字典里的东西写入到文件夹。
    ###  写入训练集
    with open((save_log_dir / "train.txt"), 'a', encoding='utf-8') as f:
        ## './Experiments/vX/train.txt'
        for k, v in train_scp_dict.items():
            for p in v:
                f.write(str(p) + "\n")

    ###  写入测试集
    with open((save_log_dir / "test.txt"), 'a', encoding='utf-8') as f:
        ## './Experiments/vX/test.txt'
        for k, v in test_scp_dict.items():
            for p in v:
                f.write(str(p) + "\n")

    print("*" * 30 + "写入 数据集 的 训练测试 表单完毕" + "*" * 30)


if __name__ == '__main__':
    Mdata = MeldataSet('Train_Scp.txt', seglen=2000)
    print(Mdata[32])
