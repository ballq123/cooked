import math
import cv2
import numpy as np
from scipy import signal  
from matplotlib import pyplot as plt
from myImageFilter import myImageFilter


'''
@param Im:   2D numpy array
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
@return rhoScale:       Array of rho values over which myHoughTransform() generates the 
                        Hough transform matrix img_hough
@return thetaScale:     Array of theta values over which myHoughTransform() generates the 
                        Hough transform matrix img_hough
'''
def myHoughTransform(Im, rhoRes, thetaRes):
    row, col = Im.shape
    diagonal = np.sqrt((row ** 2) + (col ** 2))
    maxRho = np.ceil(diagonal)

    # possible rho vals = 0 -> maxRho+rhoRes, stepping by rhoRes
    rhoScale = np.arange(0, maxRho + rhoRes, rhoRes)
    # possible theta vals = 0 -> 2pi, stepping by thetaRes
    thetaScale = np.arange(0, 2 * np.pi, thetaRes)
    lenRho, lenTheta = len(rhoScale), len(thetaScale)
    img_hough = np.zeros((lenRho, lenTheta))

    for i in range(row):
        for idx in enumerate(thetaScale):


    res = [img_hough, rhoScale, thetaScale]
    return res


def showRes():
    sigma = 2
    img = cv2.imread('img01.jpg', cv2.IMREAD_GRAYSCALE)
    assert img is not None, "file DNE? we can't read it"
    img_hough, rhoScale, thetaScale = myHoughTransform(img, sigma)
    plt.imshow(img_hough, cmap='gray')
    plt.title('My Edge Filter')
    plt.axis('off')
    plt.show()

showRes()