import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader


class MeldataSet_1(Dataset):
    def __init__(self, scp_dir, seglen):
        self.scripts = []
        self.seglen = seglen
        with open(scp_dir, encoding='utf-8') as f:
            for l in f.readlines():
                self.scripts.append(l.strip('\n'))
        self.L = len((self.scripts))
        pass

    def __getitem__(self, index):
        src_path = self.scripts[index]
        src_mel = np.load(src_path)  ## 从硬盘将数据 读入内存的一个io过程。.npy
        # src_mel = segment(np.load(src_path), seglen=self.seglen)
        return torch.FloatTensor(src_mel)  ## 【80,256】

    def __len__(self):
        return self.L
        pass


if __name__ == '__main__':
    Mdata = MeldataSet_1('Train_Scp.txt', seglen=256)
    print(Mdata[0])
    print(Mdata[2].shape)