import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import pandas as pd
import os


def source_target_train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    b_data = zip(*batch)
    # print('b_data is {}'.format(b_data))
    # if len(b_data) == 8:
    s_imgs, t_imgs, s_pids, t_pids, s_idx, t_idx = b_data
    # print('make dataloader collate_fn {}'.format(pids))
    # print(pids)
    s_pid = torch.tensor(s_pids, dtype=torch.int64)
    t_pid = torch.tensor(t_pids, dtype=torch.int64)
    pids = (s_pid, t_pid)

    s_idx = torch.tensor(s_idx, dtype=torch.int64)
    t_idx = torch.tensor(t_idx, dtype=torch.int64)
    idx = (s_idx, t_idx)
    img1 = torch.stack(s_imgs, dim=0)
    img2 = torch.stack(t_imgs, dim=0)
    return (img1.cpu(), img2.cpu()), pids, idx

class Transform_Dadaset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class CodeDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.id = range(len(self.data))
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.id[idx],  self.label[idx]
