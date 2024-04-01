import random
import os
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2


def load_data(img_path, train=True):
    # 构建输入图对应密度图路径，存储于/ground_truth/.../h5
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    # 从gr_file读取密度图数据并转化为numpy数组
    target = np.asarray(gt_file['density'])
    # 借助CV2库对密度图重缩放
    target = cv2.resize(target, (target.shape[1] // 8, target.shape[0] // 8), interpolation=cv2.INTER_CUBIC) * 64

    return img, target
