import cv2
import math
import numpy as np
from scipy import signal    # For signal.gaussian function
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
    hsize = 2 * math.ceil(3 * sigma) + 1
    # mighttt have to do some weird stuff with this kernel's shape ngl
    kernel = signal.windows.gaussian(hsize, sigma)
    kernel = np.outer(kernel, kernel)
    # normalize dat bihh (TAKE THIS OUT LATER)
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
    rows, cols = smoothed.shape
    shaved = np.zeros_like(smoothed)

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

    gradMag = gradMag.astype(np.float32)  # Ensure float32 for dilation
    gradMag = np.expand_dims(gradMag, axis=-1)

    print("gradMag shape after conversion:", gradMag.shape)
    print("Kernel shape:", zeroDeg.shape)


    # mod 4 wraps 180 degrees back to 0
    # use .astype(int) because using int() on an array throws errors :(
    nearestAngle = np.round(gradAngle / float(45))
    index = (nearestAngle.astype(int)) % 4

    dilateZero = dilate(gradMag, zeroDeg)
    dilateFour = dilate(gradMag, fourFiveDeg)
    dilateNine = dilate(gradMag, ninetyDeg)
    dilateOne = dilate(gradMag, oneThreeFiveDeg)
    dilations = [dilateZero, dilateFour, dilateNine, dilateOne]

    for row in range(rows):
        for col in range(cols):
            # select corresponding dilation based on angle idx
            current_dilation = dilations[index[row, col]]
            if gradMag[row, col] == current_dilation[row, col]:
                shaved[row, col] = gradMag[row, col]
            else:
                shaved[row, col] = 0
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