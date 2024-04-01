import sys
import warnings
import cv2
import os
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow

from QT_application import PFlowHeatmap
from data_loader import root_path
from fig_processer import logger as log
from data_loader import tkinter_path as tkpath
from fig_processer.figure import generator


# 点击选择批处理输入路径后获取路径并显示
def click_choose_batch_input_path():
    path = tkpath.select_folder_path()
    ui.batch_input_path.setPlainText(path)
    log.logger.info(f"Choose input path:{path}")


# 点击选择批处理输出路径后获取路径并显示
def click_choose_batch_output_path():
    path = tkpath.select_folder_path()
    ui.batch_output_path.setPlainText(path)
    log.logger.info(f"Choose output path:{path}")


# 批处理生成入口
def click_batch_generate():
    # 获取输入输出文件夹绝对路径
    input_path = ui.batch_input_path.toPlainText()
    output_path = ui.batch_output_path.toPlainText()

    if input_path != "" and input_path != "":
        log.logger.info("Heatmap generating...")
        for item in root_path.traversal(input_path):
            generator(root_path.root_spliter('b', item), output_path, mode='b')
        log.logger.info("All Heatmaps have been generated!")


def cv_imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return cv_img


# 选择单个图片路径
def click_choose_single_path():
    path = tkpath.select_file_path()
    ui.single_path.setPlainText(path)
    log.logger.info(f"Choose single path:{path}")

    # 展示图片
    if path is not None:
        img = cv_imread(path)  # 读取图像
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换图像通道
        x = img.shape[1]  # 获取图像大小
        y = img.shape[0]

        pix = PFlowHeatmap.QtGui.QPixmap(path)
        ui.input_image.setPixmap(pix)
        ui.input_image.setScaledContents(True)  # 自适应QLabel大小
        ui.input_image.show()

        log.logger.info(f"Show input image:{path}")


# 单个图片生成
def click_single_generate():
    ipath = ui.single_path.toPlainText()
    if ipath is not None:
        log.logger.info("Heatmap generating...")
        sum_people, opath = generator(root_path.root_spliter('s', ipath), ipath, mode='s')

        # 展示图片
        img = cv_imread(opath)  # 读取图像
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换图像通道
        x = img.shape[1]  # 获取图像大小
        y = img.shape[0]

        pix = PFlowHeatmap.QtGui.QPixmap(opath)
        ui.output_image.setPixmap(pix)
        ui.output_image.setScaledContents(True)  # 自适应QLabel大小
        ui.output_image.show()

        ui.people_num.setPlainText(sum_people)

        log.logger.info(f"Single Heatmap Shown!")


def click_del_fig():
    del_path = tkpath.select_file_path()
    if del_path is not None:
        os.remove(del_path)
    log.logger.info(f"Delete image:{del_path}")


def click_exit():
    sys.exit(app.exec_())


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    # 界面展示
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = PFlowHeatmap.Ui_PFlowHeatMap_Generator()
    ui.setupUi(MainWindow)
    MainWindow.show()

    ui.choose_batch_input_path.clicked.connect(click_choose_batch_input_path)
    ui.choose_batch_output_path.clicked.connect(click_choose_batch_output_path)
    ui.batch_generate.clicked.connect(click_batch_generate)

    ui.choose_single_path.clicked.connect(click_choose_single_path)
    ui.single_generate.clicked.connect(click_single_generate)
    ui.del_fig.clicked.connect(click_del_fig)

    ui.exit.clicked.connect(click_exit)
    sys.exit(app.exec_())
