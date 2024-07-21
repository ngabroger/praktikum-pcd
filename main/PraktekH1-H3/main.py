import sys
import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt


class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('showgui.ui', self)

        # Connect buttons and actions
        self.loadButton.clicked.connect(self.fungsi)
        self.grayButton.clicked.connect(self.grayscale)
        self.actionOperasi_Pencerahan.triggered.connect(self.brightness)
        self.actionSimple_Contrast.triggered.connect(self.contrast)
        self.actionContrast_Streching.triggered.connect(self.contrastStretching)
        self.actionNegative_Image.triggered.connect(self.negativeImage)
        self.actionBiner.triggered.connect(self.binaryImage)

        self.actionTranslasi.triggered.connect(self.translation)
        self.actionRotate_45.triggered.connect(self.rotate45)
        self.actionRotate_46.triggered.connect(self.rotateMin45)
        self.actionRotate_90.triggered.connect(self.rotate90)
        self.actionRotate_91.triggered.connect(self.rotateMin90)
        self.actionRotate_180.triggered.connect(self.rotate180)

        self.actionZoom_in.triggered.connect(self.zoomIn)
        self.actionZoom_Out.triggered.connect(self.zoomOut)
        self.actionDimension.triggered.connect(self.dimension)

        self.actionCrop.triggered.connect(self.crop)

        self.actionAdd.triggered.connect(self.arithmathicAdd)
        self.actionSubstract.triggered.connect(self.arithmathicSubstract)

        self.actionAnd.triggered.connect(self.booleanAnd)
        self.actionOr.triggered.connect(self.booleanOr)
        self.actionNot.triggered.connect(self.booleanNot)
        self.actionXor.triggered.connect(self.booleanXor)

        self.action2D_Convolusion.triggered.connect(self.Konvolusi2d)
        self.actionMean_Filter.triggered.connect(self.meanFilter)
        self.actionGauss_Filter.triggered.connect(self.gaussFilter)
        self.actionI.triggered.connect(self.sharpeningI)
        self.actionII.triggered.connect(self.sharpeningII)
        self.actionIII.triggered.connect(self.sharpeningIII)
        self.actionIV.triggered.connect(self.sharpeningIV)
        self.actionV.triggered.connect(self.sharpeningV)
        self.actionVI.triggered.connect(self.sharpeningVI)
        self.actionMedian_Filter.triggered.connect(self.medianFilter)
        self.actionMax_Filter.triggered.connect(self.maxFilter)
        self.actionMin_Filter.triggered.connect(self.minFilter)

        self.actionSobel.triggered.connect(self.deteksiSobel)
        self.actionCanny.triggered.connect(self.deteksiCanny)

        self.actionDilasi.triggered.connect(self.dilasi)
        self.actionErosi.triggered.connect(self.erosi)
        self.actionOpening.triggered.connect(self.opening)
        self.actionClosing.triggered.connect(self.closing)

        self.actionBinary.triggered.connect(self.binary)
        self.actionBinary_Invers.triggered.connect(self.binaryInvers)
        self.actionTrunc.triggered.connect(self.trunc)
        self.actionZero.triggered.connect(self.zero)
        self.actionZero_Invers.triggered.connect(self.zeroInvers)

        self.actionMean.triggered.connect(self.meanThresholding)
        self.actionGaussian.triggered.connect(self.gaussThresholding)
        self.actionOtsu.triggered.connect(self.otsuThresholding)

        self.actionCountour_Image.triggered.connect(self.contourImage)

    def fungsi(self):
        self.Image = cv2.imread('Screenshot 2024-05-04 212501.png')
        self.displayImage(1)

    def grayscale(self):
        if self.Image is not None:
            gray = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
            self.Image = gray
            self.displayImage(2)

    def brightness(self):
        if self.Image is not None:
            try:
                self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
            except:
                pass
            brightness = 80
            bright_img = cv2.convertScaleAbs(self.Image, alpha=1, beta=brightness)
            self.Image = bright_img
            self.displayImage(1)

    def contrast(self):
        if self.Image is not None:
            try:
                self.Image = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
            except:
                pass
            contrast = 1.7
            contrast_img = cv2.convertScaleAbs(self.Image, alpha=contrast, beta=0)
            self.Image = contrast_img
            self.displayImage(1)

    def contrastStretching(self):
        if self.Image is not None:
            min_val = np.min(self.Image)
            max_val = np.max(self.Image)
            stretched_img = cv2.normalize(self.Image, None, 0, 255, cv2.NORM_MINMAX)
            self.Image = stretched_img
            self.displayImage(1)

    def negativeImage(self):
        if self.Image is not None:
            negative_img = 255 - self.Image
            self.Image = negative_img
            self.displayImage(1)

    def binaryImage(self):
        if self.Image is not None:
            if len(self.Image.shape) == 3:  # Check if the image is colored
                gray = cv2.cvtColor(self.Image, cv2.COLOR_BGR2GRAY)
            else:
                gray = self.Image
            # Apply binary threshold
            threshold_value, binary_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            self.Image = binary_img
            self.displayImage(1)

    def grayscaleHist(self):
        if self.Image is not None:
            print("Generating Grayscale Histogram")
            plt.hist(self.Image.ravel(), 255, [0, 255])
            plt.show()

    def rgbHist(self):
        if self.Image is not None:
            print("Generating RGB Histogram")
            color = ('b', 'g', 'r')
            for i, col in enumerate(color):
                histo = cv2.calcHist([self.Image], [i], None, [256], [0, 256])
                plt.plot(histo, color=col)
            plt.xlim([0, 256])
            plt.show()

    def equalHist(self):
        if self.Image is not None:
            print("Generating Equalization Histogram")
            hist, bins = np.histogram(self.Image.flatten(), 256, [0, 256])
            cdf = hist.cumsum()
            cdf_normalized = cdf * hist.max() / cdf.max()
            cdf_m = np.ma.masked_equal(cdf, 0)
            cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() -
                                                   cdf_m.min())
            cdf = np.ma.filled(cdf_m, 0).astype("uint8")
            self.displayImage(2)
            plt.plot(cdf_normalized, color="b")
            plt.hist(self.Image.flatten(), 256, [0, 256], color="r")
            plt.xlim([0, 256])
            plt.legend(("cdf", "histogram"), loc="upper left")
            plt.show()

    def translation(self):
        h, w = self.Image.shape[:2]
        quarter_h,quarter_w = h/4,w/4
        T = np.float32([[1,0,quarter_w],[0,1,quarter_h]])
        img = cv2.warpAffine(self.Image,T,(w,h))
        self.Image = img
        self.displayImage(2)

    def rotate(self, degree):
        h, w = self.Image.shape[:2]
        rotationMatrix = cv2.getRotationMatrix2D((w / 2, h / 2), degree, 1.0)  # Gunakan skala 1.0
        cos = np.abs(rotationMatrix[0, 0])
        sin = np.abs(rotationMatrix[0, 1])

        # Hitung dimensi baru dari gambar
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))  # Perbaiki menjadi (w * sin) untuk nH

        # Perbaiki translasi
        rotationMatrix[0, 2] += (nW / 2) - (w / 2)
        rotationMatrix[1, 2] += (nH / 2) - (h / 2)

        # Rotasi gambar dengan dimensi baru
        rot_image = cv2.warpAffine(self.Image, rotationMatrix, (nW, nH))
        self.Image = rot_image
        self.displayImage(2)

    def rotate45(self):
        self.rotate(45)

    def rotateMin45(self):
        self.rotate(-45)

    def rotate90(self):
        self.rotate(90)

    def rotateMin90(self):
        self.rotate(-90)

    def rotate180(self):
        self.rotate(180)

    def zoomIn(self):
        skala = 2
        resize_image = cv2.resize(self.Image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('original', self.Image)
        cv2.imshow('Zoom In', resize_image)
        cv2.waitKey()

    def zoomOut(self):
        skala = 0.75
        resize_image = cv2.resize(self.Image, None, fx=skala, fy=skala)
        cv2.imshow('original', self.Image)
        cv2.imshow('Zoom Out', resize_image)
        cv2.waitKey()

    def dimension(self):
        resize_image = cv2.resize(self.Image, (900, 400), interpolation=cv2.INTER_AREA)
        cv2.imshow('original', self.Image)
        cv2.imshow('dimensi', resize_image)
        cv2.waitKey()

    def crop(self):
        start_row = 30
        end_row = 250
        start_col = 100
        end_col = 300
        crop_image = self.Image[start_row:end_row, start_col:end_col]
        cv2.imshow('original', self.Image)
        cv2.imshow('Crop Image', crop_image)
        cv2.waitKey()

    def arithmathicAdd(self):
        img1 = cv2.imread('Screenshot 2024-05-04 212501.png', 0)
        img2 = cv2.imread('Screenshot 2024-05-04 212501.png', 0)
        add_img = img1 + img2
        cv2.imshow('Image 1 Original', img1)
        cv2.imshow('Image 2 Original', img2)
        cv2.imshow('Image add', add_img)
        cv2.waitKey()

    def arithmathicSubstract(self):
        img1 = cv2.imread('Screenshot 2024-05-04 212501.png', 0)
        img2 = cv2.imread('Screenshot 2024-05-04 212501.png', 0)
        subtract = img1 - img2
        cv2.imshow('Image 1 Original', img1)
        cv2.imshow('Image 2 Original', img2)
        cv2.imshow('Image Subtract', subtract)
        cv2.waitKey()

    def booleanAnd(self):
        img1 = cv2.imread('Screenshot 2024-05-04 212501.png', 1)
        img2 = cv2.imread('Screenshot 2024-05-04 212501.png', 1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        boolean_and = cv2.bitwise_and(img1, img2)
        cv2.imshow('Image 1 Original', img1)
        cv2.imshow('Image 2 Original', img2)
        cv2.imshow('Image Boolean Add', boolean_and)
        cv2.waitKey()

    def booleanOr(self):
        img1 = cv2.imread('Screenshot 2024-05-04 212501.png', 1)
        img2 = cv2.imread('Screenshot 2024-05-04 212501.png', 1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        boolean_or = cv2.bitwise_or(img1, img2)
        cv2.imshow('Image 1 Original', img1)
        cv2.imshow('Image 2 Original', img2)
        cv2.imshow('Image Boolean Or', boolean_or)
        cv2.waitKey()

    def booleanNot(self):
        img1 = cv2.imread('Screenshot 2024-05-04 212501.png', 1)
        img2 = cv2.imread('Screenshot 2024-05-04 212501.png', 1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        boolean_not = cv2.bitwise_not(img1, img2)
        cv2.imshow('Image 1 Original', img1)
        cv2.imshow('Image 2 Original', img2)
        cv2.imshow('Image Boolean Not', boolean_not)
        cv2.waitKey()

    def booleanXor(self):
        img1 = cv2.imread('Screenshot 2024-05-04 212501.png', 1)
        img2 = cv2.imread('Screenshot 2024-05-04 212501.png', 1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        boolean_xor = cv2.bitwise_xor(img1, img2)
        cv2.imshow('Image 1 Original', img1)
        cv2.imshow('Image 2 Original', img2)
        cv2.imshow('Image Boolean Xor', boolean_xor)
        cv2.waitKey()

    def Konvolusi2d(self):
        kernel = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ])
        img_input = cv2.imread('Screenshot 2024-05-04 212501.png')
        gray_image = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        img_out = cv2.filter2D(gray_image, -1, kernel)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        cv2.imshow('Image belum konvolusi', img_input)
        plt.show()

    def meanFilter(self):
        mean = (1 / 9) * np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ])
        img_input = cv2.imread('Screenshot 2024-05-04 212501.png')
        gray_image = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        img_out = cv2.filter2D(gray_image, -1, mean)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        cv2.imshow('Image belum konvolusi', img_input)
        plt.show()

    def gaussFilter(self):
        gauss = (1.0 / 345) * np.array([
            [1, 5, 7, 5, 1],
            [5, 20, 33, 20, 5],
            [7, 33, 55, 33, 7],
            [5, 20, 33, 20, 5],
            [1, 5, 7, 5, 1]
        ])
        img_input = cv2.imread('Screenshot 2024-05-04 212501.png')
        gray_image = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        img_out = cv2.filter2D(gray_image, -1, gauss)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        cv2.imshow('Image belum konvolusi', img_input)
        plt.show()

    def sharpeningI(self):
        sharpI = np.array([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ])
        img_input = cv2.imread('Screenshot 2024-05-04 212501.png')
        gray_image = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        img_out = cv2.filter2D(gray_image, -1, sharpI)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        cv2.imshow('Image belum konvolusi', img_input)
        plt.show()

    def sharpeningII(self):
        sharpII = np.array([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]
        ])
        img_input = cv2.imread('Screenshot 2024-05-04 212501.png')
        gray_image = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        img_out = cv2.filter2D(gray_image, -1, sharpII)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        cv2.imshow('Image belum konvolusi', img_input)
        plt.show()

    def sharpeningIII(self):
        sharpIII = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
        img_input = cv2.imread('Screenshot 2024-05-04 212501.png')
        gray_image = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        img_out = cv2.filter2D(gray_image, -1, sharpIII)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        cv2.imshow('Image belum konvolusi', img_input)
        plt.show()

    def sharpeningIV(self):
        sharpIV = np.array([
            [1, 2, 1],
            [-2, 5, -2],
            [1, -2, 1]
        ])
        img_input = cv2.imread('Screenshot 2024-05-04 212501.png')
        gray_image = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        img_out = cv2.filter2D(gray_image, -1, sharpIV)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        cv2.imshow('Image belum konvolusi', img_input)
        plt.show()

    def sharpeningV(self):
        sharpV = np.array([
            [1, -2, 1],
            [-2, 4, -2],
            [1, -2, 1]
        ])
        img_input = cv2.imread('Screenshot 2024-05-04 212501.png')
        gray_image = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        img_out = cv2.filter2D(gray_image, -1, sharpV)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        cv2.imshow('Image belum konvolusi', img_input)
        plt.show()

    def sharpeningVI(self):
        sharpVI = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ])
        img_input = cv2.imread('Screenshot 2024-05-04 212501.png')
        gray_image = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        img_out = cv2.filter2D(gray_image, -1, sharpVI)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        cv2.imshow('Image belum konvolusi', img_input)
        plt.show()

    def medianFilter(self):
        img_input = cv2.imread('Screenshot 2024-05-04 212501.png')
        gray_image = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        height, width = gray_image.shape
        img_out = np.zeros((height, width), dtype=np.uint8)
        for i in range(3, height - 3):
            for j in range(3, width - 3):
                neighbors = []
        for k in range(-3, 4):
            for l in range(-3, 4):
                neighbors.append(gray_image[i + k][j + l])
        neighbors.sort()
        img_out[i, j] = neighbors[24]
        plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(img_input,
                                                      cv2.COLOR_BGR2RGB))
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(1, 2, 2), plt.imshow(img_out, cmap='gray')
        plt.title('Median Filtered Image'), plt.xticks([]), plt.yticks([])
        plt.show()

    def maxFilter(self):
        img_input = cv2.imread('Screenshot 2024-05-04 212501.png')
        gray_image = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        height, width = gray_image.shape
        img_out = np.zeros((height, width), dtype=np.uint8)
        for i in range(3, height - 3):
            for j in range(3, width - 3):
                max_val = 0
                for k in range(-3, 4):
                    for l in range(-3, 4):
                        pixel_val = gray_image[i + k][j + l]
                        if pixel_val > max_val:
                            max_val = pixel_val
                img_out.itemset((i, j), max_val)
        plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(img_input,
                                                      cv2.COLOR_BGR2RGB))
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(1, 2, 2), plt.imshow(img_out, cmap='gray')
        plt.title('Max Filtered Image'), plt.xticks([]), plt.yticks([])
        plt.show()

    def minFilter(self):
        img_input = cv2.imread('Screenshot 2024-05-04 212501.png')
        gray_image = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        height, width = gray_image.shape
        img_out = np.zeros((height, width), dtype=np.uint8)
        for i in range(3, height - 3):
            for j in range(3, width - 3):
                min_val = 255
                for k in range(-3, 4):
                    for l in range(-3, 4):
                        pixel_val = gray_image[i + k][j + l]
                        if pixel_val < min_val:
                            min_val = pixel_val
                img_out.itemset((i, j), min_val)
        plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(img_input,
                                                      cv2.COLOR_BGR2RGB))
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(1, 2, 2), plt.imshow(img_out, cmap='gray')
        plt.title('Min Filtered Image'), plt.xticks([]), plt.yticks([])
        plt.show()

    def deteksiSobel(self):
        # input image
        img_input = cv2.imread('Screenshot 2024-06-20 184738.png')
        # image convert rgb to grayscale
        gray_image = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        # inisialisasi sobel
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        # hitung gradient
        Gx = cv2.filter2D(gray_image, cv2.CV_64F, sobel_x)
        Gy = cv2.filter2D(gray_image, cv2.CV_64F, sobel_y)
        gradient = np.sqrt(Gx ** 2 + Gy ** 2)
        gradient = (gradient / np.max(gradient)) * 255
        gradient = gradient.astype(np.uint8)
        # output
        plt.imshow(gradient, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()
        cv2.imshow('Image Before', img_input)
        cv2.waitKey()

    def deteksiCanny(self):
        # Input image
        img_input = cv2.imread('Screenshot 2024-06-20 184738.png')
        if img_input is None:
            print("Image not found")
            return

        # Image convert RGB to grayscale
        gray_image = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)

        gauss = (1.0 / 57) * np.array(
            [[0, 1, 2, 1, 0],
             [1, 3, 5, 3, 1],
             [2, 5, 9, 5, 2],
             [1, 3, 5, 3, 1],
             [0, 1, 2, 1, 0]]
        )

        # Convolution of image with Gaussian kernel
        smoothed_image = cv2.filter2D(gray_image, -1, gauss)

        # Initialize Sobel kernels
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        Gx = cv2.filter2D(smoothed_image, cv2.CV_64F, sobel_x)
        Gy = cv2.filter2D(smoothed_image, cv2.CV_64F, sobel_y)

        gradient = np.sqrt(Gx ** 2 + Gy ** 2)
        gradient = (gradient / np.max(gradient)) * 255
        gradient = gradient.astype(np.uint8)

        theta = np.arctan2(Gy, Gx)

        # Non-Maximum Suppression
        H, W = gradient.shape
        Z = np.zeros((H, W), dtype=np.uint8)
        angle = theta * 180. / np.pi
        angle[angle < 0] += 180

        for i in range(1, H - 1):
            for j in range(1, W - 1):
                try:
                    q = 255
                    r = 255
                    # Angle 0
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = gradient[i, j + 1]
                        r = gradient[i, j - 1]
                    # Angle 45
                    elif (22.5 <= angle[i, j] < 67.5):
                        q = gradient[i + 1, j - 1]
                        r = gradient[i - 1, j + 1]
                    # Angle 90
                    elif (67.5 <= angle[i, j] < 112.5):
                        q = gradient[i + 1, j]
                        r = gradient[i - 1, j]
                    # Angle 135
                    elif (112.5 <= angle[i, j] < 157.5):
                        q = gradient[i - 1, j - 1]
                        r = gradient[i + 1, j + 1]

                    if (gradient[i, j] >= q) and (gradient[i, j] >= r):
                        Z[i, j] = gradient[i, j]
                    else:
                        Z[i, j] = 0
                except IndexError as e:
                    pass

        img_N = Z.astype("uint8")

        # Double Thresholding
        weak = 100
        strong = 150

        for i in range(H):
            for j in range(W):
                a = img_N[i, j]
                if a > strong:
                    img_N[i, j] = 255
                elif a < weak:
                    img_N[i, j] = 0
                else:
                    img_N[i, j] = a

        img_H1 = img_N.astype("uint8")
        cv2.imshow("Hysteresis I", img_H1)

        strong = 255
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                if img_H1[i, j] == weak:
                    try:
                        if ((img_H1[i + 1, j - 1] == strong) or (img_H1[i + 1, j] == strong) or
                                (img_H1[i + 1, j + 1] == strong) or (img_H1[i, j - 1] == strong) or
                                (img_H1[i, j + 1] == strong) or (img_H1[i - 1, j - 1] == strong) or
                                (img_H1[i - 1, j] == strong) or (img_H1[i - 1, j + 1] == strong)):
                            img_H1[i, j] = strong
                        else:
                            img_H1[i, j] = 0
                    except IndexError as e:
                        pass

        img_H2 = img_H1.astype("uint8")
        cv2.imshow("Hysteresis II", img_H2)
        cv2.imshow('Image Before', img_input)
        cv2.waitKey()

    def dilasi(self):
        img_input = cv2.imread('Screenshot 2024-06-18 200238.png', cv2.IMREAD_GRAYSCALE)
        _, binary_image = cv2.threshold(img_input, 127, 255, cv2.THRESH_BINARY)
        strel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        dilasi = cv2.dilate(binary_image, strel, iterations=1)
        cv2.imshow('Image asli', img_input)
        cv2.imshow('Dilated Image', dilasi)
        cv2.waitKey()

    def erosi(self):
        img_input = cv2.imread('Screenshot 2024-06-18 200238.png', cv2.IMREAD_GRAYSCALE)
        _, binary_image = cv2.threshold(img_input, 127, 255, cv2.THRESH_BINARY)
        strel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        erosi = cv2.erode(binary_image, strel, iterations=1)
        cv2.imshow('Image asli', img_input)
        cv2.imshow('Eroded Image', erosi)
        cv2.waitKey()

    def opening(self):
        img_input = cv2.imread('Screenshot 2024-06-18 200238.png', cv2.IMREAD_GRAYSCALE)
        _, binary_image = cv2.threshold(img_input, 127, 255, cv2.THRESH_BINARY)
        strel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, strel)
        cv2.imshow('Image asli', img_input)
        cv2.imshow('Opened Image', opening)
        cv2.waitKey()

    def closing(self):
        img_input = cv2.imread('Screenshot 2024-06-18 200238.png', cv2.IMREAD_GRAYSCALE)
        _, binary_image = cv2.threshold(img_input, 127, 255, cv2.THRESH_BINARY)
        strel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, strel)
        cv2.imshow('Image asli', img_input)
        cv2.imshow('Closed Image', closing)
        cv2.waitKey()

    def binary(self):
        img_input = cv2.imread('Screenshot 2024-06-18 200238.png', cv2.IMREAD_GRAYSCALE)
        T = 127
        max = 255
        ret, thresh = cv2.threshold(img_input, T, max, cv2.THRESH_BINARY)
        cv2.imshow("Binary", thresh)
        print(thresh)
        cv2.waitKey()

    def binaryInvers(self):
        img_input = cv2.imread('Screenshot 2024-06-18 200238.png', cv2.IMREAD_GRAYSCALE)
        T = 127
        max = 255
        ret, thresh = cv2.threshold(img_input, T, max, cv2.THRESH_BINARY_INV)
        cv2.imshow("Binary Invers", thresh)
        print(thresh)
        cv2.waitKey()

    def trunc(self):
        img_input = cv2.imread('Screenshot 2024-06-18 200238.png', cv2.IMREAD_GRAYSCALE)
        T = 127
        max = 255
        ret, thresh = cv2.threshold(img_input, T, max, cv2.THRESH_TRUNC)
        cv2.imshow("Trunc Image", thresh)
        print(thresh)
        cv2.waitKey()

    def zero(self):
        img_input = cv2.imread('Screenshot 2024-06-18 200238.png', cv2.IMREAD_GRAYSCALE)
        T = 127
        max = 255
        ret, thresh = cv2.threshold(img_input, T, max, cv2.THRESH_TOZERO)
        cv2.imshow("To Zero", thresh)
        print(thresh)
        cv2.waitKey()

    def zeroInvers(self):
        img_input = cv2.imread('Screenshot 2024-06-18 200238.png', cv2.IMREAD_GRAYSCALE)
        T = 127
        max = 255
        ret, thresh = cv2.threshold(img_input, T, max, cv2.THRESH_TOZERO_INV)
        cv2.imshow("To Zero Invers", thresh)
        print(thresh)
        cv2.waitKey()

    def meanThresholding(self):
        img_input = cv2.imread('Screenshot 2024-06-18 200238.png', cv2.IMREAD_GRAYSCALE)
        thresh = cv2.adaptiveThreshold(img_input, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 2)
        cv2.imshow("Mean Thresholding", thresh)
        print(thresh)
        cv2.waitKey()

    def gaussThresholding(self):
        img_input = cv2.imread('Screenshot 2024-06-18 200238.png', cv2.IMREAD_GRAYSCALE)
        thresh = cv2.adaptiveThreshold(img_input, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 2)
        cv2.imshow("Gaussian Thresholding", thresh)
        print(thresh)
        cv2.waitKey()

    def otsuThresholding(self):
        img_input = cv2.imread('Screenshot 2024-06-18 200238.png', cv2.IMREAD_GRAYSCALE)
        T = 130
        ret, thresh = cv2.threshold(img_input, T, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow("Otsu Thresholding", thresh)
        print(thresh)

    def contourImage(self):
        # Membaca gambar input
        img_input = cv2.imread('bangun.jpg')
        if img_input is None:
            print("Error: Gambar tidak ditemukan.")
            return

        gray_image = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        T = 127
        ret, thresh = cv2.threshold(gray_image, T, 255, cv2.THRESH_BINARY)

        # Menampilkan hasil thresholding untuk debugging
        cv2.imshow('Thresholded Image', thresh)

        # Menemukan kontur
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # Menampilkan gambar dengan semua kontur yang ditemukan
        img_contours = img_input.copy()
        cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
        cv2.imshow('All Contours', img_contours)

        # Warna untuk masing-masing bentuk
        shape_colors = {
            "Triangle": (0, 255, 0),
            "Square": (255, 0, 0),
            "Rectangle": (0, 0, 255),
            "Star": (255, 255, 0),
            "Circle": (0, 255, 255)
        }

        # Membuat gambar berwarna untuk mewarnai bagian dalam
        colored_image = np.zeros_like(img_input)

        for contour in contours:
            # Mengaproksimasi kontur menjadi poligon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Mengidentifikasi bentuk berdasarkan jumlah sisi
            num_vertices = len(approx)
            shape = "Unknown"

            if num_vertices == 3:
                shape = "Triangle"
            elif num_vertices == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                # Debug print removed
                if 0.9 <= aspect_ratio <= 1.1:
                    shape = "Square"
                else:
                    shape = "Rectangle"
            elif num_vertices == 10:  # Typically, stars have more vertices
                shape = "Star"
            else:
                # Menggunakan kebundaran untuk membedakan lingkaran dari bintang
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter != 0:  # Menghindari pembagian dengan nol
                    circularity = 4 * np.pi * (area / (perimeter * perimeter))
                    print(f"Circularity: {circularity} for contour with {num_vertices} vertices")
                    if 0.5 <= circularity <= 1.5:  # Perluasan rentang circularity
                        shape = "Circle"
                    else:
                        shape = "Unknown"

            print(f"Detected shape: {shape} with {num_vertices} vertices")

            if shape != "Unknown":
                # Menemukan pusat kontur
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0

                # Mendapatkan warna untuk bentuk
                color = shape_colors.get(shape, (255, 255, 255))

                # Menggambar kontur dan nama bentuk
                cv2.drawContours(colored_image, [contour], -1, color, -1)  # Mengisi bagian dalam bentuk
                cv2.putText(colored_image, shape, (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Menampilkan kontur dan bentuk yang terdeteksi
            cv2.imshow('Contours and Shapes', colored_image)
            cv2.waitKey()  # Penundaan untuk melihat hasil setiap langkah

        # Menampilkan hasil akhir
        cv2.imshow('Contours and Shapes', colored_image)
        cv2.waitKey(0)

    def displayImage(self, window=1):
        if self.Image is not None:
            qformat = QImage.Format_Indexed8
            if len(self.Image.shape) == 3:
                if self.Image.shape[2] == 4:
                    qformat = QImage.Format_RGBA8888
                else:
                    qformat = QImage.Format_RGB888
            img = QImage(self.Image, self.Image.shape[1], self.Image.shape[0], self.Image.strides[0], qformat)
            img = img.rgbSwapped()
            if window == 1:
                self.imgLabel.setPixmap(QPixmap.fromImage(img))
            elif window == 2:
                self.hasilLabel.setPixmap(QPixmap.fromImage(img))


app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('Show Image GUI')
window.show()
sys.exit(app.exec_())
