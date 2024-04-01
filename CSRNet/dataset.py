import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from image import *
from data_loader.load_data import load_data


class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None, train=False, seen=0, batch_size=1,
                 num_workers=4):
        if train:
            root = root * 4
        random.shuffle(root)

        self.nSamples = len(root)  # 样本数量
        self.lines = root  # 样本列表
        self.transform = transform  # transform形参
        self.train = train  # train/test
        self.shape = shape  # 图片形状如[3,256,256]:三通道256*256
        self.seen = seen  # seen形参
        self.batch_size = batch_size  # 训练批大小形参
        self.num_workers = num_workers  # 线程默认为4

    def __len__(self):
        return self.nSamples  # 返回样本数

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.lines[index]

        img, target = load_data(img_path, self.train)  # 自定义函数加载图片

        if self.transform is not None:
            img = self.transform(img)
        return img, target
