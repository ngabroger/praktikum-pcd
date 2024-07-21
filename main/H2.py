import cv2
import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QMessageBox
from PyQt5.uic import loadUi
import numpy as np


class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('showimage2.ui', self)
        self.image = None
        self.loadButton.clicked.connect(self.loadClicked)
        self.actionMaxFilter.triggered.connect(self.applyMaxFilter)

        # Connect thresholding actions
        self.actionbinary.triggered.connect(lambda: self.applyThreshold(cv2.THRESH_BINARY))
        self.actionBinary_Inverse.triggered.connect(lambda: self.applyThreshold(cv2.THRESH_BINARY_INV))
        self.actionTrunch.triggered.connect(lambda: self.applyThreshold(cv2.THRESH_TRUNC))
        self.actionTo_Zero.triggered.connect(lambda: self.applyThreshold(cv2.THRESH_TOZERO))
        self.actionTo_Zero_Invers.triggered.connect(lambda: self.applyThreshold(cv2.THRESH_TOZERO_INV))

        self.actionMean_Thresholding.triggered.connect(self.applyMeanThresholding)
        self.actionGaussian.triggered.connect(self.applyGaussianThresholding)
        self.actionOtsu_Thresholding.triggered.connect(self.applyOtsuThresholding)

    @pyqtSlot()
    def loadClicked(self):
        self.loadImage('noisy_image.jpg')  # Load the image with noise

    @pyqtSlot()
    def applyMaxFilter(self):
        try:
            if self.image is not None:
                gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                max_filtered_image = self.max_filter(gray_image)
                self.image = max_filtered_image
                self.displayImage(2)
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def applyThreshold(self, thresh_type):
        try:
            if self.image is not None:
                gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                _, thresh_image = cv2.threshold(gray_image, 127, 255, thresh_type)
                self.image = thresh_image
                self.displayImage(2)
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def applyMeanThresholding(self):
        try:
            if self.image is not None:
                gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                mean_thresh_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                          cv2.THRESH_BINARY, 11, 2)
                self.image = mean_thresh_image
                self.displayImage(2)
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def applyGaussianThresholding(self):
        try:
            if self.image is not None:
                gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                gaussian_thresh_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                              cv2.THRESH_BINARY, 11, 2)
                self.image = gaussian_thresh_image
                self.displayImage(2)
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    @pyqtSlot()
    def applyOtsuThresholding(self):
        try:
            if self.image is not None:
                gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                _, otsu_thresh_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                self.image = otsu_thresh_image
                self.displayImage(2)
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def loadImage(self, fname):
        try:
            self.image = cv2.imread(fname)
            self.displayImage(1)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def displayImage(self, windows=1):
        try:
            qformat = QImage.Format_Indexed8

            if len(self.image.shape) == 3:
                if self.image.shape[2] == 4:
                    qformat = QImage.Format_RGBA8888
                else:
                    qformat = QImage.Format_RGB888

            img = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)
            img = img.rgbSwapped()

            if windows == 1:
                self.imgLabel.setPixmap(QPixmap.fromImage(img))
                self.imgLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
                self.imgLabel.setScaledContents(True)

            if windows == 2:
                self.imgLabel.setPixmap(QPixmap.fromImage(img))
                self.imgLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
                self.imgLabel.setScaledContents(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def max_filter(self, image):
        h, w = image.shape
        img_out = image.copy()
        for i in range(3, h - 3):
            for j in range(3, w - 3):
                max_val = 0
                for k in range(-3, 4):
                    for l in range(-3, 4):
                        pixel_value = image[i + k, j + l]
                        if pixel_value > max_val:
                            max_val = pixel_value
                img_out[i, j] = max_val
        return img_out


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = ShowImage()
    window.setWindowTitle('Praktikum Pengolahan Citra Digital')
    window.show()
    sys.exit(app.exec_())
