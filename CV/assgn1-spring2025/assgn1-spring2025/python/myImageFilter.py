import numpy as np
import cv2

'''
@param img0: grayscale (2D) image, ** assuming it's a np array **
@param h:    matrix storing  a convolution filter
'''
def myImageFilter(img0, h):
    imageH, imageW = np.shape(img0)
    fH, fW = np.shape(h)
    paddingH, paddingW = fH // 2, fW // 2

    padded = np.pad(img0, ((paddingH, paddingH), (paddingW, paddingW)), mode='edge')
    img1 = np.zeros_like(img0)

    for i in range(imageH):
        for j in range(imageW):
            subsection = padded[i: i + fH, j: j + fW]
            img1[i, j] = np.sum(subsection * h)
    return img1
