import cv2
import numpy as np

img1 = cv2.imread('gambar.jpg', 0)
img2 = cv2.imread('gambar2.jpg', 0)

if img1 is None or img2 is None:
    print("One or more images could not be loaded.")
else:
    add_img = cv2.add(img1, img2)

    subtract_img = cv2.subtract(img1, img2)

    cv2.imshow("Added Image", add_img)
    cv2.imshow("Subtracted Image", subtract_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()