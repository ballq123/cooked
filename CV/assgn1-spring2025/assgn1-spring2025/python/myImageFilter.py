import os
import cv2 
import math
import numpy as np
from scipy import signal  
from matplotlib import pyplot as plt

'''
@param img0:    2D numpy array
                Grayscale image
@param h:       2D numpy array
                Matrix storing a convolution filter
'''
def myImageFilter(img0, h):
    imageH, imageW = np.shape(img0)
    fH, fW = np.shape(h)
    paddingH, paddingW = fH // 2, fW // 2
    padded = np.pad(img0, ((paddingH, paddingH), (paddingW, paddingW)), mode='edge')

    img1 = np.zeros((imageH, imageW))
    for i in range(fH):
        for j in range(fW):
            subsection = padded[i: i + imageH, j: j + imageW]
            hSubSection = h[i, j]
            img1 += (subsection * hSubSection)
    return img1


def showRes():
    sigma     = 2
    threshold = 0.03
    rhoRes    = 2
    thetaRes  = np.pi / 90
    
    img = cv2.imread('img06.jpg', cv2.IMREAD_GRAYSCALE)
    if (img.ndim == 3):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = np.float32(img) / 255

    hsize = 2 * np.ceil(3 * sigma) + 1
    kernel = signal.windows.gaussian(hsize, sigma)
    kernel = np.outer(kernel, kernel)
    kernel = (1 / np.sum(kernel)) * kernel

    img_edge = myImageFilter(img, kernel)

    resultsdir = '../results'
    os.makedirs(resultsdir, exist_ok=True)

    # Save using OpenCV
    cv2.imwrite(os.path.join(resultsdir, 'img06_filter.png'), (img_edge * 255).astype(np.uint8))
    plt.imshow(img_edge, cmap='gray')
    plt.axis('off')
    plt.show()

# showRes()
