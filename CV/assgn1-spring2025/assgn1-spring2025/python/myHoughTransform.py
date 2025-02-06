import cv2 
import math
import numpy as np
from scipy import signal  
from matplotlib import pyplot as plt
from myImageFilter import myImageFilter
from myEdgeFilter import myEdgeFilter


'''
@param Im:              2D numpy array
                        Edge magnitude image, thresholded to ignore pixels 
                        with a low edge filter response
@param rhoRes:          scalar
                        Distance resolution of the Hough transform accumulator
                        in pixels
@param thetaRes:        scalar
                        Angular resolution of the accumulator in radians

@return img_hough:      2D numpy array
                        Hough transform accumulator that contains the number of "votes"
                        for all the possible lines passing through the image
@return rhoScale:       2D numpy array
                        Array of rho values over which myHoughTransform() generates the 
                        Hough transform matrix img_hough
@return thetaScale:     2D numpy array
                        Array of theta values over which myHoughTransform() generates the 
                        Hough transform matrix img_hough
'''
def myHoughTransform(Im, rhoRes, thetaRes):
    rows, cols = Im.shape

    diagonal = np.sqrt((rows ** 2) + (cols ** 2))
    maxRho = np.ceil(diagonal)

    # possible rho vals = 0 -> maxRho+rhoRes, stepping by rhoRes
    rhoScale = np.arange(0, maxRho + rhoRes, rhoRes)
    # possible theta vals = 0 -> 2pi, stepping by thetaRes
    thetaScale = np.arange(0, 2 * np.pi, thetaRes)
    lenRho, lenTheta = len(rhoScale), len(thetaScale)

    img_hough = np.zeros((lenRho, lenTheta))
    for row in range(rows):
        for col in range(cols):
            # apparently this is faster than doing Im[row][col] in numpy
            # edges will have a value of 1 at Im[row, col], else 0
            if Im[row, col] == 1:
                for j, theta in enumerate(thetaScale):
                    rho = (col * np.cos(theta)) + (row * np.sin(theta))
                    # theta vals that correspond to negative rho vals are invalid
                    if rho >= 0:
                        i = int(np.round(rho / rhoRes))
                        img_hough[i, j] += 1
    res = [img_hough, rhoScale, thetaScale]
    return res


def showRes():
    sigma     = 2
    threshold = 0.03
    rhoRes    = 2
    thetaRes  = np.pi / 90
    
    img = cv2.imread('img01.jpg', cv2.IMREAD_GRAYSCALE)
    if (img.ndim == 3):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = np.float32(img) / 255

    img_edge = myEdgeFilter(img, sigma)
    img_threshold = np.float32(img_edge > threshold)
    assert img is not None, "file DNE? we can't read it"
    img_hough, rhoScale, thetaScale = myHoughTransform(img_threshold, rhoRes, thetaRes)

    plt.imshow(img_hough, cmap='gray')
    plt.title('My Edge Filter')
    plt.axis('off')
    plt.show()

# showRes()