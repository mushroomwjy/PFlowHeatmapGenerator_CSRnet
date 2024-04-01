import glob
from image import *
from CSRNet.model import CSRNet
import os
import torch
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torchvision import transforms
from pprint import pprint

from PyQt5.QtWidgets import QApplication, QMainWindow
from QT_application import PFlowHeatmap

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib
from PIL import Image
import numpy as np
from fig_processer import logger as log


# matplotlib.rcParams['font.sans-serif'] = ['Fangsong']
# matplotlib.rcParams['axes.unicode_minus'] = False


def generator(filepath: list[str, str, str], outputpath, mode: str):
    transform = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]),
    ])  # RGB转换使用的转换器，其中的mean和std参数数值是从概率统计中得到的

    model = CSRNet()  # 导入网络模型
    # 单GPU或者CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    checkpoint = torch.load(r'ShanghaiTech/partA_model_best.pth.tar', map_location="cpu")  # 载入训练好的权重
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()  # 准备预测评估

    folder = filepath[0]
    filename = filepath[1]
    fileend = filepath[2]
    img = transform(Image.open('%s%s%s' % (folder, filename, fileend)).convert('RGB')).to(device)  # .cuda()
    img = img.unsqueeze(0)
    h, w = img.shape[2:4]
    h_d = h // 2
    w_d = w // 2
    # 输入图片切割成四份.to(device)更改存储设备至CPU
    img_1 = Variable(img[:, :, :h_d, :w_d].to(device))
    img_2 = Variable(img[:, :, :h_d, w_d:].to(device))
    img_3 = Variable(img[:, :, h_d:, :w_d].to(device))
    img_4 = Variable(img[:, :, h_d:, w_d:].to(device))
    density_1 = model(img_1).data.cpu().numpy()
    density_2 = model(img_2).data.cpu().numpy()
    density_3 = model(img_3).data.cpu().numpy()
    density_4 = model(img_4).data.cpu().numpy()

    # 将上部两张图片进行拼接
    up_map = np.concatenate((density_1[0, 0, ...], density_2[0, 0, ...]), axis=1)
    down_map = np.concatenate((density_3[0, 0, ...], density_4[0, 0, ...]), axis=1)
    # 将上下部合成一张完成图片
    final_map = np.concatenate((up_map, down_map), axis=0)
    plt.imshow(final_map, cmap='gist_heat')  # 展示图片cm.jet gist_heat
    plt.title("Visitor Flow Heatmap (num = %d)" % final_map.sum())
    if mode == 's':
        plt.savefig('%s%s_heat%s' % (folder, filename, fileend))
        opath_return = '%s%s_heat%s' % (folder, filename, fileend)
        log.logger.info(f"{'%s%s_heat%s' % (outputpath, filename, fileend)} Generated.")
        string_return = "%d" % final_map.sum()
        plt.show()
        log.logger.info(
            f"The estimated number of people in {filename + fileend} is %d" % final_map.sum())  # 直接输出图像预测的人数
        return string_return, opath_return
    elif mode == 'b':
        plt.savefig('%s%s_heat%s' % (outputpath, filename, fileend))
        log.logger.info(f"{'%s%s_heat%s' % (outputpath, filename, fileend)} Generated.")
        plt.show()
        log.logger.info(
            f"The estimated number of people in {filename + fileend} is %d" % final_map.sum())  # 直接输出图像预测的人数
