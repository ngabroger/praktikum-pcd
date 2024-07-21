import cv2
import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QMessageBox
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt
import numpy as np


class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('showimage.ui', self)
        self.image = None
        self.loadButton.clicked.connect(self.loadClicked)
        self.grayButton.clicked.connect(self.grayClicked)

        # Initialize the 'Filtering' menu action
        self.actionFilter.triggered.connect(self.filteringClicked)

    def loadClicked(self):
        # Load image from a predefined path
        filename = 'gambar.jpg'
        self.image = cv2.imread(filename)
        if self.image is not None:
            self.displayImage()
        else:
            QMessageBox.information(self, "Error", "Image not found or unable to load.")

    def grayClicked(self):
        # Convert image to grayscale
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            self.image = gray_image
            self.displayImage()

    def displayImage(self):
        # Display image in the GUI
        qformat = QImage.Format_Indexed8 if len(self.image.shape) == 2 else QImage.Format_RGB888
        img = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)
        img = img.rgbSwapped()  # Convert BGR to RGB
        self.imageLabel.setPixmap(QPixmap.fromImage(img))
        self.imageLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

    def filteringClicked(self):
        if self.image is not None:
            # Convert the image to grayscale if it is not already
            if len(self.image.shape) == 3:
                gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = self.image

            # Define a sample kernel for convolution
            kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])

            # Perform convolution
            img_out = cv2.filter2D(gray_image, -1, kernel)

            # Display the filtered image using matplotlib
            plt.imshow(img_out, cmap='gray', interpolation='bicubic')
            plt.xticks([]), plt.yticks([])
            plt.show()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ShowImage()
    window.show()
    sys.exit(app.exec_())
