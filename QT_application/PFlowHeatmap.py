# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'PFlowHeatmap.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_PFlowHeatMap_Generator(object):
    def setupUi(self, PFlowHeatMap_Generator):
        PFlowHeatMap_Generator.setObjectName("PFlowHeatMap_Generator")
        PFlowHeatMap_Generator.resize(888, 666)
        PFlowHeatMap_Generator.setMinimumSize(QtCore.QSize(888, 666))
        PFlowHeatMap_Generator.setMaximumSize(QtCore.QSize(888, 666))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(240, 240, 240))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(240, 240, 240))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(240, 240, 240))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(240, 240, 240))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        PFlowHeatMap_Generator.setPalette(palette)
        self.exit = QtWidgets.QPushButton(PFlowHeatMap_Generator)
        self.exit.setGeometry(QtCore.QRect(710, 600, 151, 31))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(11)
        font.setBold(True)
        font.setWeight(75)
        self.exit.setFont(font)
        self.exit.setObjectName("exit")
        self.groupBox = QtWidgets.QGroupBox(PFlowHeatMap_Generator)
        self.groupBox.setGeometry(QtCore.QRect(20, 20, 841, 161))
        font = QtGui.QFont()
        font.setFamily("Cambria")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox.setFont(font)
        self.groupBox.setObjectName("groupBox")
        self.batch_generate = QtWidgets.QPushButton(self.groupBox)
        self.batch_generate.setGeometry(QtCore.QRect(610, 120, 221, 31))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(10)
        font.setBold(False)
        font.setWeight(50)
        self.batch_generate.setFont(font)
        self.batch_generate.setObjectName("batch_generate")
        self.choose_batch_input_path = QtWidgets.QPushButton(self.groupBox)
        self.choose_batch_input_path.setGeometry(QtCore.QRect(700, 40, 131, 31))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.choose_batch_input_path.setFont(font)
        self.choose_batch_input_path.setObjectName("choose_batch_input_path")
        self.textBrowser = QtWidgets.QTextBrowser(self.groupBox)
        self.textBrowser.setGeometry(QtCore.QRect(20, 40, 81, 31))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.NoBrush)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.NoBrush)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.NoBrush)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        self.textBrowser.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.textBrowser.setFont(font)
        self.textBrowser.setStyleSheet("background-color: rgb(255, 255, 255, 0);")
        self.textBrowser.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textBrowser.setFrameShadow(QtWidgets.QFrame.Plain)
        self.textBrowser.setObjectName("textBrowser")
        self.textBrowser_2 = QtWidgets.QTextBrowser(self.groupBox)
        self.textBrowser_2.setGeometry(QtCore.QRect(20, 80, 81, 31))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.textBrowser_2.setFont(font)
        self.textBrowser_2.setStyleSheet("background-color: rgb(255, 255, 255, 0);")
        self.textBrowser_2.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textBrowser_2.setFrameShadow(QtWidgets.QFrame.Plain)
        self.textBrowser_2.setObjectName("textBrowser_2")
        self.batch_input_path = QtWidgets.QTextBrowser(self.groupBox)
        self.batch_input_path.setGeometry(QtCore.QRect(110, 40, 721, 31))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.batch_input_path.setFont(font)
        self.batch_input_path.setObjectName("batch_input_path")
        self.batch_output_path = QtWidgets.QTextBrowser(self.groupBox)
        self.batch_output_path.setGeometry(QtCore.QRect(110, 80, 721, 31))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.batch_output_path.setFont(font)
        self.batch_output_path.setObjectName("batch_output_path")
        self.choose_batch_output_path = QtWidgets.QPushButton(self.groupBox)
        self.choose_batch_output_path.setGeometry(QtCore.QRect(700, 80, 131, 31))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.choose_batch_output_path.setFont(font)
        self.choose_batch_output_path.setObjectName("choose_batch_output_path")
        self.textBrowser_5 = QtWidgets.QTextBrowser(self.groupBox)
        self.textBrowser_5.setGeometry(QtCore.QRect(90, 40, 16, 31))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.textBrowser_5.setFont(font)
        self.textBrowser_5.setStyleSheet("background-color: rgb(255, 255, 255, 0);")
        self.textBrowser_5.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textBrowser_5.setFrameShadow(QtWidgets.QFrame.Plain)
        self.textBrowser_5.setObjectName("textBrowser_5")
        self.textBrowser_6 = QtWidgets.QTextBrowser(self.groupBox)
        self.textBrowser_6.setGeometry(QtCore.QRect(90, 80, 16, 31))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.textBrowser_6.setFont(font)
        self.textBrowser_6.setStyleSheet("background-color: rgb(255, 255, 255, 0);")
        self.textBrowser_6.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textBrowser_6.setFrameShadow(QtWidgets.QFrame.Plain)
        self.textBrowser_6.setObjectName("textBrowser_6")
        self.batch_generate.raise_()
        self.textBrowser.raise_()
        self.textBrowser_2.raise_()
        self.batch_input_path.raise_()
        self.choose_batch_input_path.raise_()
        self.batch_output_path.raise_()
        self.choose_batch_output_path.raise_()
        self.textBrowser_5.raise_()
        self.textBrowser_6.raise_()
        self.groupBox_2 = QtWidgets.QGroupBox(PFlowHeatMap_Generator)
        self.groupBox_2.setGeometry(QtCore.QRect(20, 190, 841, 401))
        font = QtGui.QFont()
        font.setFamily("Cambria")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName("groupBox_2")
        self.groupBox_3 = QtWidgets.QGroupBox(self.groupBox_2)
        self.groupBox_3.setGeometry(QtCore.QRect(10, 70, 341, 321))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.groupBox_3.setFont(font)
        self.groupBox_3.setObjectName("groupBox_3")
        self.input_image = QtWidgets.QLabel(self.groupBox_3)
        self.input_image.setGeometry(QtCore.QRect(10, 30, 321, 281))
        self.input_image.setText("")
        self.input_image.setObjectName("input_image")
        self.groupBox_5 = QtWidgets.QGroupBox(self.groupBox_2)
        self.groupBox_5.setGeometry(QtCore.QRect(350, 70, 341, 321))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.groupBox_5.setFont(font)
        self.groupBox_5.setObjectName("groupBox_5")
        self.output_image = QtWidgets.QLabel(self.groupBox_5)
        self.output_image.setGeometry(QtCore.QRect(11, 30, 321, 281))
        self.output_image.setText("")
        self.output_image.setObjectName("output_image")
        self.choose_single_path = QtWidgets.QPushButton(self.groupBox_2)
        self.choose_single_path.setGeometry(QtCore.QRect(700, 140, 131, 31))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.choose_single_path.setFont(font)
        self.choose_single_path.setObjectName("choose_single_path")
        self.single_generate = QtWidgets.QPushButton(self.groupBox_2)
        self.single_generate.setGeometry(QtCore.QRect(700, 180, 131, 31))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.single_generate.setFont(font)
        self.single_generate.setObjectName("single_generate")
        self.del_fig = QtWidgets.QPushButton(self.groupBox_2)
        self.del_fig.setGeometry(QtCore.QRect(700, 220, 131, 31))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.del_fig.setFont(font)
        self.del_fig.setObjectName("del_fig")
        self.groupBox_6 = QtWidgets.QGroupBox(self.groupBox_2)
        self.groupBox_6.setGeometry(QtCore.QRect(700, 30, 131, 101))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.groupBox_6.setFont(font)
        self.groupBox_6.setObjectName("groupBox_6")
        self.textBrowser_8 = QtWidgets.QTextBrowser(self.groupBox_6)
        self.textBrowser_8.setGeometry(QtCore.QRect(10, 30, 81, 31))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.textBrowser_8.setFont(font)
        self.textBrowser_8.setStyleSheet("background-color: rgb(255, 255, 255, 0);")
        self.textBrowser_8.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textBrowser_8.setFrameShadow(QtWidgets.QFrame.Plain)
        self.textBrowser_8.setObjectName("textBrowser_8")
        self.people_num = QtWidgets.QTextBrowser(self.groupBox_6)
        self.people_num.setGeometry(QtCore.QRect(10, 60, 111, 31))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.people_num.setFont(font)
        self.people_num.setObjectName("people_num")
        self.textBrowser_7 = QtWidgets.QTextBrowser(self.groupBox_2)
        self.textBrowser_7.setGeometry(QtCore.QRect(700, 330, 101, 31))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.textBrowser_7.setFont(font)
        self.textBrowser_7.setStyleSheet("background-color: rgb(255, 255, 255, 0);")
        self.textBrowser_7.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textBrowser_7.setFrameShadow(QtWidgets.QFrame.Plain)
        self.textBrowser_7.setObjectName("textBrowser_7")
        self.textBrowser_10 = QtWidgets.QTextBrowser(self.groupBox_2)
        self.textBrowser_10.setGeometry(QtCore.QRect(700, 360, 131, 31))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.textBrowser_10.setFont(font)
        self.textBrowser_10.setStyleSheet("background-color: rgb(255, 255, 255, 0);")
        self.textBrowser_10.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textBrowser_10.setFrameShadow(QtWidgets.QFrame.Plain)
        self.textBrowser_10.setObjectName("textBrowser_10")
        self.textBrowser_12 = QtWidgets.QTextBrowser(self.groupBox_2)
        self.textBrowser_12.setGeometry(QtCore.QRect(700, 270, 121, 31))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.textBrowser_12.setFont(font)
        self.textBrowser_12.setStyleSheet("background-color: rgb(255, 255, 255, 0);")
        self.textBrowser_12.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textBrowser_12.setFrameShadow(QtWidgets.QFrame.Plain)
        self.textBrowser_12.setObjectName("textBrowser_12")
        self.textBrowser_13 = QtWidgets.QTextBrowser(self.groupBox_2)
        self.textBrowser_13.setGeometry(QtCore.QRect(700, 300, 121, 31))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.textBrowser_13.setFont(font)
        self.textBrowser_13.setStyleSheet("background-color: rgb(255, 255, 255, 0);")
        self.textBrowser_13.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textBrowser_13.setFrameShadow(QtWidgets.QFrame.Plain)
        self.textBrowser_13.setObjectName("textBrowser_13")
        self.single_path = QtWidgets.QTextBrowser(self.groupBox_2)
        self.single_path.setGeometry(QtCore.QRect(60, 30, 621, 31))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.single_path.setFont(font)
        self.single_path.setObjectName("single_path")
        self.textBrowser_3 = QtWidgets.QTextBrowser(self.groupBox_2)
        self.textBrowser_3.setGeometry(QtCore.QRect(10, 30, 81, 31))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.NoBrush)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.NoBrush)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.NoBrush)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        self.textBrowser_3.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(9)
        font.setBold(False)
        font.setWeight(50)
        self.textBrowser_3.setFont(font)
        self.textBrowser_3.setStyleSheet("background-color: rgb(255, 255, 255, 0);")
        self.textBrowser_3.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textBrowser_3.setFrameShadow(QtWidgets.QFrame.Plain)
        self.textBrowser_3.setObjectName("textBrowser_3")
        self.groupBox_3.raise_()
        self.groupBox_5.raise_()
        self.choose_single_path.raise_()
        self.single_generate.raise_()
        self.del_fig.raise_()
        self.groupBox_6.raise_()
        self.textBrowser_10.raise_()
        self.textBrowser_12.raise_()
        self.textBrowser_13.raise_()
        self.textBrowser_7.raise_()
        self.single_path.raise_()
        self.textBrowser_3.raise_()
        self.textBrowser_11 = QtWidgets.QTextBrowser(PFlowHeatMap_Generator)
        self.textBrowser_11.setGeometry(QtCore.QRect(730, 630, 141, 31))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(8)
        font.setBold(True)
        font.setWeight(75)
        self.textBrowser_11.setFont(font)
        self.textBrowser_11.setStyleSheet("background-color: rgb(255, 255, 255, 0);")
        self.textBrowser_11.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.textBrowser_11.setFrameShadow(QtWidgets.QFrame.Plain)
        self.textBrowser_11.setObjectName("textBrowser_11")

        self.retranslateUi(PFlowHeatMap_Generator)
        QtCore.QMetaObject.connectSlotsByName(PFlowHeatMap_Generator)

    def retranslateUi(self, PFlowHeatMap_Generator):
        _translate = QtCore.QCoreApplication.translate
        PFlowHeatMap_Generator.setWindowTitle(_translate("PFlowHeatMap_Generator", "PFlowHeatMap_Generator"))
        self.exit.setText(_translate("PFlowHeatMap_Generator", "EXIT o(´^´)o"))
        self.groupBox.setTitle(_translate("PFlowHeatMap_Generator", "Batch Mode"))
        self.batch_generate.setText(_translate("PFlowHeatMap_Generator", "Generate Heatmaps"))
        self.choose_batch_input_path.setText(_translate("PFlowHeatMap_Generator", "Choose Path"))
        self.textBrowser.setHtml(_translate("PFlowHeatMap_Generator", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'微软雅黑\',\'Cambria\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'微软雅黑\';\">#Input</span></p></body></html>"))
        self.textBrowser_2.setHtml(_translate("PFlowHeatMap_Generator", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'微软雅黑\',\'Cambria\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'微软雅黑\';\">#Output</span></p></body></html>"))
        self.choose_batch_output_path.setText(_translate("PFlowHeatMap_Generator", "Choose Path"))
        self.textBrowser_5.setHtml(_translate("PFlowHeatMap_Generator", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'微软雅黑\',\'Cambria\'; font-size:14pt; font-weight:600; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'微软雅黑\'; font-size:9pt; font-weight:400;\">:</span></p></body></html>"))
        self.textBrowser_6.setHtml(_translate("PFlowHeatMap_Generator", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'微软雅黑\',\'Cambria\'; font-size:14pt; font-weight:600; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'微软雅黑\'; font-size:9pt; font-weight:400;\">:</span></p></body></html>"))
        self.groupBox_2.setTitle(_translate("PFlowHeatMap_Generator", "Single Mode"))
        self.groupBox_3.setTitle(_translate("PFlowHeatMap_Generator", "#Input Image"))
        self.groupBox_5.setTitle(_translate("PFlowHeatMap_Generator", "# People Flow Heatmap"))
        self.choose_single_path.setText(_translate("PFlowHeatMap_Generator", "Choose Image"))
        self.single_generate.setText(_translate("PFlowHeatMap_Generator", "Generate"))
        self.del_fig.setText(_translate("PFlowHeatMap_Generator", "Delete Figure"))
        self.groupBox_6.setTitle(_translate("PFlowHeatMap_Generator", "# Messages:"))
        self.textBrowser_8.setHtml(_translate("PFlowHeatMap_Generator", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'微软雅黑\',\'微软雅黑\',\'Cambria\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">People:</p></body></html>"))
        self.people_num.setHtml(_translate("PFlowHeatMap_Generator", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'微软雅黑\',\'微软雅黑\',\'Cambria\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.textBrowser_7.setHtml(_translate("PFlowHeatMap_Generator", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'微软雅黑\',\'Cambria\'; font-size:8pt; font-weight:600; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'微软雅黑\'; font-weight:400;\"># mushroom菌</span></p></body></html>"))
        self.textBrowser_10.setHtml(_translate("PFlowHeatMap_Generator", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'微软雅黑\',\'Cambria\'; font-size:8pt; font-weight:600; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'微软雅黑\'; font-weight:400;\">   novawjy@qq.com</span></p></body></html>"))
        self.textBrowser_12.setHtml(_translate("PFlowHeatMap_Generator", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'微软雅黑\',\'Cambria\'; font-size:8pt; font-weight:600; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'微软雅黑\'; font-weight:400;\"># 08122209 wjy</span></p></body></html>"))
        self.textBrowser_13.setHtml(_translate("PFlowHeatMap_Generator", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'微软雅黑\',\'Cambria\'; font-size:8pt; font-weight:600; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'微软雅黑\'; font-weight:400;\"># 2024.03python</span></p></body></html>"))
        self.single_path.setHtml(_translate("PFlowHeatMap_Generator", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'微软雅黑\',\'Cambria\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.textBrowser_3.setHtml(_translate("PFlowHeatMap_Generator", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'微软雅黑\',\'Cambria\'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">Path :</p></body></html>"))
        self.textBrowser_11.setHtml(_translate("PFlowHeatMap_Generator", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'微软雅黑\'; font-size:8pt; font-weight:600; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:400;\">powered by CSRNet</span></p></body></html>"))
