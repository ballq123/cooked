import cv2
import math
import numpy as np
from scipy import signal    
from cv2 import dilate
from matplotlib import pyplot as plt
from myImageFilter import myImageFilter

'''
@param img0:    2D numpy array
                Grayscale (2D) image 
@param sigma:   scalar
                StdDev of Gaussian smoothing kernel to be used before edge detection  
'''
def myEdgeFilter(img0, sigma):
    hsize = 2 * np.ceil(3 * sigma) + 1
    # mighttt have to do some weird stuff with this kernel's shape ngl
    kernel = signal.windows.gaussian(hsize, sigma)
    kernel = np.outer(kernel, kernel)
    kernel = (1 / np.sum(kernel)) * kernel

    # print("kernel shape after: ", kernel.shape)
    smoothed = myImageFilter(img0, kernel)
    xOrientedSobel = np.array([[1., 0., -1.]])
    yOrientedSobel = np.array([[1.],
                               [0.],
                               [-1.]])
    imgX = myImageFilter(smoothed, xOrientedSobel)
    imgY = myImageFilter(smoothed, yOrientedSobel)

    # angles below are in range [-pi, pi) = [-180, 180)
    gradAngleRadians = np.arctan2(imgY, imgX)
    gradAngle = ((gradAngleRadians * 180) / np.pi) % 180
    gradMag = np.sqrt((imgX ** 2) + (imgY ** 2))

    img1 = nMaxSuppression(smoothed, gradMag, gradAngle)
    return img1, imgX, imgY
    

'''
@param gradMag:     2D numpy array
                    Gradient magnitude matrix of image
@param gradAngle:   2D numpy array
                    Gradient angle matrix of image
'''
def nMaxSuppression(smoothed, gradMag, gradAngle):
    zeroDeg = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0]
    ], dtype=np.uint8)
    fourFiveDeg = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ], dtype=np.uint8)
    ninetyDeg = np.array([
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0]
    ], dtype=np.uint8)
    oneThreeFiveDeg = np.array([
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0]
    ], dtype=np.uint8)

    # print("gradMag shape:", gradMag.shape)
    # print("Kernel shape:", zeroDeg.shape)

    dilateZero = dilate(gradMag, zeroDeg)
    dilateFour = dilate(gradMag, fourFiveDeg)
    dilateNine = dilate(gradMag, ninetyDeg)
    dilateOne = dilate(gradMag, oneThreeFiveDeg)
    dilations = [dilateZero, dilateFour, dilateNine, dilateOne]

    # mod 4 wraps 180 degrees back to 0
    # use .astype(int) because using int() on an array throws errors :(
    nearestAngle = np.round(gradAngle / float(45))
    index = (nearestAngle.astype(int)) % 4

    # Visualize dilations (for debugging)
    # for i, dilation in enumerate(dilations):
    #     plt.imshow(dilation, cmap='gray')
    #     plt.title(f'Dilation for angle {i * 45} degrees')
    #     plt.show()

    rows, cols = smoothed.shape
    shaved = np.zeros_like(smoothed)
    for r in range(rows):
        for c in range(cols):
            angle_index = index[r, c]
            current_dilation = dilations[angle_index]
            if gradMag[r, c] >= current_dilation[r, c]:
                shaved[r, c] = gradMag[r, c]
            else:
                shaved[r, c] = 0
    return shaved


def showRes():
    sigma = 2
    img = cv2.imread('img01.jpg', cv2.IMREAD_GRAYSCALE)
    assert img is not None, "file DNE? we can't read it"
    edges_custom, _, _ = myEdgeFilter(img, sigma)
    plt.imshow(edges_custom, cmap='gray')
    plt.title('My Edge Filter')
    plt.axis('off')
    plt.show()

showRes()
