# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(718, 502)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(70, 190, 121, 28))

        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(70, 320, 121, 28))

        self.pushButton_2.setObjectName("pushButton_2")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(80, 230, 121, 16))

        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(80, 360, 121, 16))
        self.label_2.setObjectName("label_2")

        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(220, 150, 180, 31))  # (x,y,w,h)
        self.label_3.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_3.setObjectName("label_3")
        font = QtGui.QFont()                       # 加入文字設定
        font.setFamily('Verdana')                  # 設定字體
        font.setPointSize(10)                      # 文字大小
        self.label_3.setFont(font)

        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(220, 180, 160, 231))

        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)

        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton_4 = QtWidgets.QPushButton(self.verticalLayoutWidget)

        self.pushButton_4.setObjectName("pushButton_4")
        self.verticalLayout.addWidget(self.pushButton_4)
        self.pushButton_5 = QtWidgets.QPushButton(self.verticalLayoutWidget)

        self.pushButton_5.setObjectName("pushButton_5")
        self.verticalLayout.addWidget(self.pushButton_5)
        self.pushButton_6 = QtWidgets.QPushButton(self.verticalLayoutWidget)

        self.pushButton_6.setObjectName("pushButton_6")
        self.verticalLayout.addWidget(self.pushButton_6)
        self.pushButton_3 = QtWidgets.QPushButton(self.verticalLayoutWidget)

        self.pushButton_3.setObjectName("pushButton_3")
        self.verticalLayout.addWidget(self.pushButton_3)

        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(450, 150, 180, 31))
        self.label_4.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_4.setObjectName("label_4")
        font = QtGui.QFont()                       # 加入文字設定
        font.setFamily('Verdana')                  # 設定字體
        font.setPointSize(10)                      # 文字大小
        self.label_4.setFont(font)

        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_2.setGeometry(
            QtCore.QRect(450, 180, 160, 231))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(
            self.verticalLayoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.pushButton_8 = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.pushButton_8.setObjectName("pushButton_8")
        self.verticalLayout_2.addWidget(self.pushButton_8)
        self.pushButton_9 = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.pushButton_9.setObjectName("pushButton_9")
        self.verticalLayout_2.addWidget(self.pushButton_9)
        self.pushButton_7 = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.pushButton_7.setObjectName("pushButton_7")
        self.verticalLayout_2.addWidget(self.pushButton_7)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate(
            "MainWindow", "2022 Opencvdl Hw1"))
        self.pushButton.setText(_translate("MainWindow", "Load Image 1"))
        self.pushButton.clicked.connect(self.onButtonClick)

        self.pushButton_2.setText(_translate("MainWindow", "Load Image 2"))
        self.pushButton_2.clicked.connect(self.onButtonClick2)

        self.label.setText(_translate("MainWindow", "No Image Loaded"))

        self.label_2.setText(_translate("MainWindow", "No Image Loaded"))

        self.label_3.setText(_translate("MainWindow", "1.Image Processing"))

        self.pushButton_4.setText(_translate(
            "MainWindow", "1.1 Color Seperation"))
        self.pushButton_4.clicked.connect(self.onPushButtonClicked)

        self.pushButton_5.setText(_translate(
            "MainWindow", "1.2 Color Transformation"))
        self.pushButton_5.clicked.connect(self.onPushButtonClicked2)

        self.pushButton_6.setText(_translate(
            "MainWindow", "1.3 Color Detection"))
        self.pushButton_6.clicked.connect(self.onPushButtonClicked3)

        self.pushButton_3.setText(_translate("MainWindow", "1.4 Blending"))
        self.pushButton_3.clicked.connect(self.onPushButtonClicked4)

        self.label_4.setText(_translate("MainWindow", "2.Image Smoothing"))
        self.pushButton_8.setText(_translate(
            "MainWindow", "2.1 Gaussian Blur"))
        self.pushButton_8.clicked.connect(self.onPushButtonClicked5)

        self.pushButton_9.setText(_translate(
            "MainWindow", "2.2 Bilateral Filter"))
        self.pushButton_9.clicked.connect(self.onPushButtonClicked6)

        self.pushButton_7.setText(_translate(
            "MainWindow", "2.3 Median Filter"))
        self.pushButton_7.clicked.connect(self.onPushButtonClicked7)

    def onButtonClick(self):  # Image Load 1
        filename, filetype = QFileDialog.getOpenFileName(None,
                                                         "選取檔案",
                                                         "./")
        print(filename)
        self.label.setText(filename)
        self.img1 = cv2.imread(filename)

    def onButtonClick2(self):  # Image Load 2
        filename, filetype = QFileDialog.getOpenFileName(None,
                                                         "選取檔案",
                                                         "./")
        print(filename)
        self.label_2.setText(filename)
        self.img2 = cv2.imread(filename)

    def onPushButtonClicked(self):  # Color Separation

        img = self.img1
        img = cv2.resize(self.img1, (480, 270))
        # split() 出來為灰階
        b, g, r = cv2.split(img)

        # 生成一個值爲0的單通道數組
        zeros = np.zeros(img.shape[:2], dtype='uint8')

        # 分別擴展B、G、R成爲三通道。另外兩個通道用上面的值爲0的數組填充
        merged_r = cv2.merge([zeros, zeros, r])
        merged_g = cv2.merge([zeros, g, zeros])
        merged_b = cv2.merge([b, zeros, zeros])

        cv2.imshow('image', img)
        cv2.imshow('R_channel', merged_r)
        cv2.imshow('G_channel', merged_g)
        cv2.imshow('B_channel', merged_b)

        cv2.waitKey(0)

    def onPushButtonClicked2(self):  # Color Transformation
        img = self.img1
        img = cv2.resize(self.img1, (480, 270))

        b, g, r = cv2.split(img)
        #averweighted = (r+g+b)/3
        mergergb = cv2.merge([r//3+g//3+b//3, r//3+g//3+b//3, r//3+g//3+b//3])
        # cvtColor
        imgbgr2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("OpenCV function", imgbgr2gray)
        cv2.imshow("Average weighted", mergergb)

        cv2.waitKey(0)

    def onPushButtonClicked3(self):  # Color Detection
        img = self.img1
        img = cv2.resize(self.img1, (480, 270))
        # 預設排列為BGR HSV即色相、飽和度、明度（Hue, Saturation, Value）
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Green Range : (40-80,50-255,20-255)
        lower_green = np.array([40, 50, 20])
        upper_green = np.array([80, 255, 255])

        # White Range : (0-180,0-20,200-255)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 20, 255])

        # 遮罩
        mask = cv2.inRange(hsv, lower_green, upper_green)
        # 將遮罩與原圖做AND
        green = cv2.bitwise_and(img, img, mask=mask)
        cv2.imshow('green mask', mask)

        # 遮罩
        mask = cv2.inRange(hsv, lower_white, upper_white)
        # 將遮罩與原圖做AND
        white = cv2.bitwise_and(img, img, mask=mask)
        cv2.imshow('white mask', mask)

        cv2.imshow('', img)
        cv2.imshow('Green', green)
        cv2.imshow('White', white)
        cv2.waitKey(0)

    def onPushButtonClicked4(self):  # Blending

        imgone = self.img1
        imgone = cv2.resize(self.img1, (480, 270))

        imgtwo = self.img2
        imgtwo = cv2.resize(self.img2, (480, 270))

        def on_change(val):
            # print(val)
            alpha = float(val)/255.0
            beta = (1.0 - alpha)
            result = cv2.addWeighted(imgone, alpha, imgtwo, beta, 0.0)
            cv2.imshow('Blend', result)

        cv2.namedWindow('Blend')

        # Use Trackbar to change the weights and show the result in the new window
        cv2.createTrackbar('Blend', 'Blend', 0, 255, on_change)
        cv2.imshow('Blend', imgtwo)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def onPushButtonClicked5(self):  # Gaussian Blur
        img = self.img1
        img = cv2.resize(self.img1, (400, 400))

        def on_change(self):
            ksize = cv2.getTrackbarPos('magnitude', 'Gaussian_Filter')
            ksize = 2*ksize+1  # k=2m+1 奇數
            Gaussian = cv2.GaussianBlur(img, (ksize, ksize), 0)
            cv2.imshow('Gaussian_Filter', Gaussian)
            # Blures input image
            # cv2.GaussianBlur(src, ksize, sigmaX, sigmaY, borderType)

        # Creates window
        cv2.namedWindow('Gaussian_Filter')

        # Creates Trackbar with slider position and callback function
        low_k = 0
        high_k = 10

        cv2.createTrackbar('magnitude', 'Gaussian_Filter',
                           low_k, high_k, on_change)
        cv2.imshow('Gaussian_Filter', img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def onPushButtonClicked6(self):  # Bilateral Filter

        img = self.img1
        img = cv2.resize(self.img1, (400, 400))

        def on_change(self):
            ksize = cv2.getTrackbarPos('magnitude', 'Bilateral_filter')
            ksize = 2*ksize+1  # k=2m+1 奇數
            # cv2.GaussianBlur(src, ksize, sigmaX, sigmaY, borderType)
            Bilateral = cv2.bilateralFilter(img, ksize, 90, 90)
            cv2.imshow('Bilateral_filter', Bilateral)

        # Creates window
        cv2.namedWindow('Bilateral_filter')

        # Creates Trackbar with slider position and callback function
        low_k = 0
        high_k = 10

        cv2.createTrackbar('magnitude', 'Bilateral_filter',
                           low_k, high_k, on_change)
        cv2.imshow('Bilateral_filter', img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def onPushButtonClicked7(self):  # Median Filter

        img = self.img1
        img = cv2.resize(self.img1, (400, 400))

        def on_change(self):
            ksize = cv2.getTrackbarPos('magnitude', 'median_filter')
            ksize = 2*ksize+1  # k=2m+1 奇數
            # cv2.dst = cv.medianBlur(src, ksize[, dst] )
            median = cv2.medianBlur(img, ksize)
            cv2.imshow('median_filter', median)

        # Creates window
        cv2.namedWindow('median_filter')

        # Creates Trackbar with slider position and callback function
        low_k = 0
        high_k = 10

        cv2.createTrackbar('magnitude', 'median_filter',
                           low_k, high_k, on_change)
        cv2.imshow('median_filter', img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
