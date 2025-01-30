import math
import cv2
import numpy as np
from scipy import signal  
from matplotlib import pyplot as plt
from myImageFilter import myImageFilter


'''
@param img_threshold:   2D numpy array
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
@return rhoScale:       array of rho values over which myHoughTransform() generates the 
                        Hough transform matrix img_hough
@return thetaScale:     array of theta values over which myHoughTransform() generates the 
                        Hough transform matrix img_hough
'''
def myHoughTransform(Im, rhoRes, thetaRes):
    # YOUR CODE HERE
    return


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