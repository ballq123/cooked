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
    # have to do some weird stuff with this kernel's shape ngl
    kernel = signal.windows.gaussian(hsize, sigma)
    kernel = np.outer(kernel, kernel)
    # normalize dat bihh
    kernel = (1 / np.sum(kernel)) * kernel

    # print("kernel shape after: ", kernel.shape)
    smoothed = myImageFilter(img0, kernel)

    xOrientedSobel = np.array([[1., 0., -1.]])
    imgX = myImageFilter(smoothed, xOrientedSobel)
    yOrientedSobel = np.array([[1.],
                               [0.],
                               [-1.]])
    imgY = myImageFilter(smoothed, yOrientedSobel)

    gradMag = np.sqrt((imgX ** 2) + (imgY ** 2))
    # returns angles in range [-pi, pi) = [-180, 180)
    gradAngleRadians = np.arctan2(imgY, imgX)
    gradAngle = ((gradAngleRadians * 180) / np.pi) % 180

    img1 = nMaxSuppression(smoothed, gradMag, gradAngle)
    return img1, imgX, imgY

'''
@param gradMag:     2D numpy array
                    Gradient magnitude matrix of image
@param gradAngle:   2D numpy array
                    Gradient angle matrix of image
'''
def nMaxSuppression(smoothed, gradMag, gradAngle):
    gradH, gradW = gradMag.shape
    shaved = np.zeros_like(gradMag)

    zeroDeg = np.array([
        [[0, 0, 0],
         [1, 1, 1],
         [0, 0, 0]]
    ])
    fourFiveDeg = np.array([
        [[0, 0, 1],
         [0, 1, 0],
         [1, 0, 0]],
    ])
    ninetyDeg = np.array([
        [[0, 1, 0],
         [0, 1, 0],
         [0, 1, 0]],
    ])
    oneThreeFiveDeg = np.array([
        [[1, 0, 0],
         [0, 1, 0],
         [0, 0, 1]],
    ])

    dilateZero = dilate(smoothed, zeroDeg)
    dilateFour = dilate(smoothed, fourFiveDeg)
    dilateNine = dilate(smoothed, ninetyDeg)
    dilateOne = dilate(smoothed, oneThreeFiveDeg)




    '''

    # skip bordering pixels bc they don't have all necessary neighbors 😞  
    # DO THIS, OR SHOULD WE PAD THE IMAGE & ALLOW FOR BORDER INCLUSION?  
    for i in range(1, gradH - 1):
        for j in range(1, gradW - 1):
            angle = gradAngle[i, j]
            # horizontal (left -> right, or vice-versa)
            if ((angle >= 0 and angle < 22.5) or (angle >= 157.5 and angle <= 180)):
                left = gradMag[i, j-1]
                right = gradMag[i, j+1]
            # diagonal (bottom-left -> top-right, or vice-versa)
            elif (angle >= 22.5 and angle < 67.5):
                left = gradMag[i+1, j-1]
                right = gradMag[i-1, j+1]
            # vertical (top -> bottom, or vice-versa)
            elif (angle >= 67.5 and angle < 112.5):
                left = gradMag[i-1, j]
                right = gradMag[i+1, j]
            # diagonal (top-left -> bottom-right, or vice-versa)
            else:
                # angle >= 112.5 and angle < 157.5
                left = gradMag[i-1, j-1]
                right = gradMag[i+1, j+1]

            # "value of a center pixel is set to 0 if its gradient magnitude is 
            #   less than either one or both of its neighbors' values"
            curMag = gradMag[i, j]
            if (curMag > left) and (curMag > right):
                shaved[i, j] = curMag
            else:
                shaved[i, j] = 0
    '''
    return shaved

def showRes():
    sigma = 2
    img = cv2.imread('img01.jpg', cv2.IMREAD_GRAYSCALE)
    assert img is not None, "file DNE? we can't read it"
    edges_custom = myEdgeFilter(img, sigma)[2]
    plt.imshow(edges_custom, cmap='gray')
    plt.title('My Edge Filter')
    plt.axis('off')
    plt.show()

showRes()