import cv2
import numpy as np

img1 = cv2.imread('gambar.jpg', 1)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

img2 = cv2.imread('gambar2.jpg', 1)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

result = cv2.bitwise_and(img1, img2)

cv2.imshow("Bitwise AND Result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()