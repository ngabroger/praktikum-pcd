import cv2
import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QMessageBox
from PyQt5.uic import loadUi
import numpy as np
from matplotlib import pyplot as plt  # Import matplotlib


class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('showimage.ui', self)
        self.image = None
        self.loadButton.clicked.connect(self.loadClicked)

        # Connect Sobel edge detection action
        self.actionSobel.triggered.connect(self.sobelClicked)

        self.contrast_value = 1.6

    @pyqtSlot()
    def loadClicked(self):
        self.loadImage('images.jpg')



    @pyqtSlot()
    def sobelClicked(self):
        try:
            if self.image is not None:
                gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

                sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

                Gx = cv2.filter2D(gray_image, -1, sobel_x)
                Gy = cv2.filter2D(gray_image, -1, sobel_y)

                gradient = np.sqrt(Gx ** 2 + Gy ** 2)
                gradient = (gradient / gradient.max()) * 255

                self.image = gradient.astype(np.uint8)
                self.displayImage(2)  # Display the Sobel edge detected image

                plt.imshow(self.image, cmap='gray', interpolation='bicubic')
                plt.title('Sobel Edge Detection')
                plt.axis('off')
                plt.show()
            else:
                QMessageBox.critical(self, "Error", "No image loaded.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred during Sobel edge detection: {str(e)}")

    def loadImage(self, flname):
        try:
            self.image = cv2.imread(flname)
            if self.image is None:
                raise FileNotFoundError(f"Could not load image: {flname}")
            self.displayImage()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def displayImage(self, windows=1):
        if self.image is not None:
            qformat = QImage.Format_Indexed8

            if len(self.image.shape) == 3:
                if self.image.shape[2] == 4:
                    qformat = QImage.Format_RGBA8888
                else:
                    qformat = QImage.Format_RGB888
            img = QImage(self.image.data, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)

            img = img.rgbSwapped()
            if windows == 1:
                # Storing the loaded image in imgLabel
                self.imgLabel.setPixmap(QPixmap.fromImage(img))

                # Positioning the image at the center
                self.imgLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)

            if windows == 2:
                self.hasilLabel.setPixmap(QPixmap.fromImage(img))
                self.hasilLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
                self.hasilLabel.setScaledContents(True)
        else:
            QMessageBox.critical(self, "Error", "No image loaded.")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ShowImage()
    window.setWindowTitle('Show Image GUI')
    window.show()
    sys.exit(app.exec_())
