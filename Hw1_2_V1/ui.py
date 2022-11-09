# -*- coding: utf-8 -*-
import cv2
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from scipy import signal


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(70, 190, 101, 31))
        self.pushButton.setObjectName("pushButton")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(70, 230, 111, 16))
        self.label.setObjectName("label")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(230, 140, 160, 221))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton_2 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.verticalLayout.addWidget(self.pushButton_2)
        self.pushButton_4 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_4.setObjectName("pushButton_4")
        self.verticalLayout.addWidget(self.pushButton_4)
        self.pushButton_5 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_5.setObjectName("pushButton_5")
        self.verticalLayout.addWidget(self.pushButton_5)
        self.pushButton_3 = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.pushButton_3.setObjectName("pushButton_3")
        self.verticalLayout.addWidget(self.pushButton_3)
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_2.setGeometry(
            QtCore.QRect(440, 140, 160, 221))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(
            self.verticalLayoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.pushButton_7 = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.pushButton_7.setObjectName("pushButton_7")
        self.verticalLayout_2.addWidget(self.pushButton_7)
        self.pushButton_6 = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.pushButton_6.setObjectName("pushButton_6")
        self.verticalLayout_2.addWidget(self.pushButton_6)
        self.pushButton_8 = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.pushButton_8.setObjectName("pushButton_8")
        self.verticalLayout_2.addWidget(self.pushButton_8)
        self.pushButton_9 = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.pushButton_9.setObjectName("pushButton_9")
        self.verticalLayout_2.addWidget(self.pushButton_9)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(230, 110, 131, 21))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(440, 110, 131, 21))
        self.label_3.setObjectName("label_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "2022OpenCvDl Hw1"))
        self.pushButton.setText(_translate("MainWindow", "Load Image"))
        self.pushButton.clicked.connect(self.onButtonClick)

        self.label.setText(_translate("MainWindow", "No Image Loaded"))
        self.pushButton_2.setText(_translate(
            "MainWindow", "3.1 Gaussian Blur"))
        self.pushButton_2.clicked.connect(self.onButtonClick2)

        self.pushButton_4.setText(_translate("MainWindow", "3.2 Sobel X"))
        self.pushButton_4.clicked.connect(self.onButtonClick3)

        self.pushButton_5.setText(_translate("MainWindow", "3.3 Sobel Y"))
        self.pushButton_5.clicked.connect(self.onButtonClick4)

        self.pushButton_3.setText(_translate("MainWindow", "3.4 Magnitude"))
        self.pushButton_3.clicked.connect(self.onButtonClick5)

        self.pushButton_7.setText(_translate("MainWindow", "4.1 Resize"))
        self.pushButton_7.clicked.connect(self.onButtonClick6)

        self.pushButton_6.setText(_translate("MainWindow", "4.2 Translation"))
        self.pushButton_6.clicked.connect(self.onButtonClick7)

        self.pushButton_8.setText(_translate(
            "MainWindow", "4.3 Rotation,Scaling"))
        self.pushButton_8.clicked.connect(self.onButtonClick8)

        self.pushButton_9.setText(_translate("MainWindow", "4.4 Shearing"))
        self.pushButton_9.clicked.connect(self.onButtonClick9)

        self.label_2.setText(_translate("MainWindow", "3.Edge Detection"))
        self.label_3.setText(_translate("MainWindow", "4.Transformation"))

    def onButtonClick(self):  # Image Load
        filename, filetype = QFileDialog.getOpenFileName(None,
                                                         "選取檔案",
                                                         "./")
        print(filename)
        self.label.setText(filename)
        self.img = cv2.imread(filename)

    def onButtonClick2(self):  # 3-1 Gaussian Blur
        image = self.img
        cv2.namedWindow('building')
        cv2.imshow('building', image)

        # convert to grayscale
        grayImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        cv2.namedWindow('Grayscale')
        cv2.imshow('Grayscale', grayImage)

        # 3*3 Gassian filter
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x**2+y**2))

        # Normalization
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

        gaussian = cv2.filter2D(grayImage, -1, gaussian_kernel)
        cv2.namedWindow('Gaussian Blur')
        cv2.imshow('Gaussian Blur', gaussian)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def onButtonClick3(self):  # 3-2 Sobel X
        image = self.img

        # convert to grayscale
        grayImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 3*3 Gassian filter
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x**2+y**2))

        # Normalization
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

        gaussian = cv2.filter2D(grayImage, -1, gaussian_kernel)

        sobelXfilter = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=float)
        sobelx = signal.convolve2d(gaussian, sobelXfilter, "same", "symm")
        sobelx = np.uint8(np.absolute(sobelx))
        cv2.namedWindow('sobelx')
        cv2.moveWindow('sobelx', 1000, 100)
        cv2.imshow('sobelx', sobelx)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def onButtonClick4(self):  # 3-3 Sobel Y
        image = self.img

        # convert to grayscale
        grayImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 3*3 Gassian filter
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x**2+y**2))

        # Normalization
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

        gaussian = cv2.filter2D(grayImage, -1, gaussian_kernel)

        sobelYfilter = np.array([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=float)
        sobely = signal.convolve2d(gaussian, sobelYfilter, "same", "symm")
        sobely = np.uint8(np.absolute(sobely))

        cv2.namedWindow('sobely')
        cv2.moveWindow('sobely', 1000, 100)
        cv2.imshow('sobely', sobely)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def onButtonClick5(self):  # 3-4 Magnitude
        image = self.img

        # convert to grayscale
        grayImage = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 3*3 Gassian filter
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x**2+y**2))

        # Normalization
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

        gaussian = cv2.filter2D(grayImage, -1, gaussian_kernel)

        sobelXfilter = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=float)
        sobelx = signal.convolve2d(gaussian, sobelXfilter, "same", "symm")

        sobelYfilter = np.array([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=float)
        sobely = signal.convolve2d(gaussian, sobelYfilter, "same", "symm")

        magnitude = ((sobelx * sobelx)+(sobely * sobely)) ** 0.5
        magnitude = np.uint8(np.absolute(magnitude))

        cv2.namedWindow('magnitude')
        cv2.moveWindow('magnitude', 1000, 100)
        cv2.imshow('magnitude', magnitude)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def onButtonClick6(self):  # 4-1 Resize
        img = self.img
        img1 = cv2.resize(img, (215, 215))

        size = img.shape

        w = size[0]  # width
        h = size[1]  # height

        M = np.float32([[1, 0, 0], [0, 1, 0]])  # 2x3 矩陣，x 軸平移 200，y 軸平移 100
        output = cv2.warpAffine(img1, M, (w, h))
        cv2.imshow('Resize', output)
        cv2.imwrite('Resize.jpg', output)
        cv2.waitKey(0)

    def onButtonClick7(self):  # 4-2 Translation
        img = self.img
        img1 = cv2.resize(img, (215, 215))

        size = img.shape

        w = size[0]  # width
        h = size[1]  # height

        # 2x3 矩陣，x 軸平移 215，y 軸平移 215
        M = np.float32([[1, 0, 215], [0, 1, 215]])
        output = cv2.warpAffine(img1, M, (w, h))
        #cv2.imshow('Translation', output)

        M2 = np.float32([[1, 0, 0], [0, 1, 0]])  # 2x3 矩陣，x 軸平移 0，y 軸平移 0
        output2 = cv2.warpAffine(img1, M2, (w, h))
        #cv2.imshow('Resize', output2)

        output3 = cv2.addWeighted(output, 1, output2, 1, 0)
        cv2.imshow("Translation", output3)
        cv2.imwrite("Translation.jpg", output3)
        cv2.waitKey(0)

    def onButtonClick8(self):  # 4-3 Rotation,Scaling
        #cv2.getRotationMatrix2D((x, y), angle, scale)
        # (x, y) 旋轉的中心點，angle 旋轉角度 ( - 順時針，+ 逆時針 )，scale 旋轉後的尺寸
        img = self.img
        img1 = cv2.resize(img, (215, 215))
        size = img.shape

        w = size[0]  # width
        h = size[1]  # height

        # 2x3 矩陣，x 軸平移 215，y 軸平移 215
        M = np.float32([[1, 0, 215], [0, 1, 215]])
        output = cv2.warpAffine(img1, M, (w, h))

        M2 = np.float32([[1, 0, 0], [0, 1, 0]])  # 2x3 矩陣，x 軸平移 0，y 軸平移 0
        output2 = cv2.warpAffine(img1, M2, (w, h))

        output3 = cv2.addWeighted(output, 1, output2, 1, 0)

        # 中心點 (215, 215)，旋轉 45 度，尺寸 0.5
        M = cv2.getRotationMatrix2D((w/2, h/2), 45, 0.5)
        output = cv2.warpAffine(output3, M, (430, 430))

        cv2.imshow('Rotate and Scale', output)
        cv2.imwrite('Rotate and Scale.jpg', output)
        cv2.waitKey(0)

    def onButtonClick9(self):  # 4.4 Shearing
        img = self.img
        img1 = cv2.resize(img, (215, 215))
        size = img.shape

        w = size[0]  # width
        h = size[1]  # height

        # 2x3 矩陣，x 軸平移 215，y 軸平移 215
        M = np.float32([[1, 0, 215], [0, 1, 215]])
        output = cv2.warpAffine(img1, M, (w, h))

        M2 = np.float32([[1, 0, 0], [0, 1, 0]])  # 2x3 矩陣，x 軸平移 0，y 軸平移 0
        output2 = cv2.warpAffine(img1, M2, (w, h))

        output3 = cv2.addWeighted(output, 1, output2, 1, 0)

        # 中心點 (215, 215)，旋轉 45 度，尺寸 0.5
        M = cv2.getRotationMatrix2D((w/2, h/2), 45, 0.5)
        output = cv2.warpAffine(output3, M, (430, 430))

        p1 = np.float32([[50, 50], [200, 50], [50, 200]])
        p2 = np.float32([[10, 100], [100, 50], [100, 250]])

        M1 = cv2.getAffineTransform(p1, p2)
        output4 = cv2.warpAffine(output, M1, (430, 430))
        cv2.imshow('Shearing', output4)
        cv2.imwrite('Shearing.jpg', output4)
        cv2.waitKey(0)


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
