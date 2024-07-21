import cv2
import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QMessageBox
from PyQt5.uic import loadUi
import numpy as np
from matplotlib import pyplot as plt

class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('showimage.ui', self)
        self.image = None
        self.loadButton.clicked.connect(self.loadClicked)
        self.actionSharpening_Image.triggered.connect(self.sharpeningImage)  # Add this line
        self.contrast_value = 1.6

    @pyqtSlot()
    def loadClicked(self):
        self.loadImage('gambar.jpg')


    @pyqtSlot()
    def sharpeningImage(self):
        try:
            if self.image is not None:
                # Define sharpening kernels
                kernel1 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                kernel2 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                kernel3 = np.array([[1, -2, 1], [-2, 5, -2], [1, -2, 1]])

                # Apply sharpening kernels
                sharpened1 = cv2.filter2D(self.image, -1, kernel1)
                sharpened2 = cv2.filter2D(self.image, -1, kernel2)
                sharpened3 = cv2.filter2D(self.image, -1, kernel3)

                # Display results
                self.displayImageResult(sharpened1, 'Sharpened Image with Kernel 1')
                self.displayImageResult(sharpened2, 'Sharpened Image with Kernel 2')
                self.displayImageResult(sharpened3, 'Sharpened Image with Kernel 3')
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during sharpening: {str(e)}")

    def displayImageResult(self, img, title):
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

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
                self.hasilLabel.setPixmap(QPixmap.fromImage(img))
                self.hasilLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
                self.hasilLabel.setScaledContents(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = ShowImage()
    window.setWindowTitle('Praktikum Pengolahan Citra Digital')
    window.show()
    sys.exit(app.exec_())
