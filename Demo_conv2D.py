import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('FB_IMG_1665457195977.jpg')
img = cv2.resize(img, (200, 200))
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# print(img_gray)


def conv2d( input , kernelSize):
    height, width = input.shape
    kernel = np.random.randn(kernelSize, kernelSize)
    # print(kernel)
    results = np.zeros((height - kernelSize + 1, width - kernelSize + 1))
    # print(results, results.shape)
    for row in range(0, height - kernelSize + 1):
        for col in range(0, width - kernelSize + 1):
            results[row, col] = np.sum(input[row: row + kernelSize, col: col + kernelSize] * kernel)
    return results


img_gray_conv2d = conv2d(img_gray, 3)
plt.imshow(img_gray_conv2d, cmap='gray')
plt.show()
