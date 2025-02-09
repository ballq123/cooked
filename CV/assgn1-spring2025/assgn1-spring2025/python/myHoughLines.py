import cv2 
import math
import numpy as np
from cv2 import dilate
from scipy import signal 
from scipy import ndimage 
from matplotlib import pyplot as plt
from myImageFilter import myImageFilter
from myEdgeFilter import myEdgeFilter
from myHoughTransform import myHoughTransform

'''
@param H:       2D numpy array
                Hough transform accumulator
@param nLines:  scalar
                Number of lines to return

@return rhos:   2D numpy array with shape (nLines, 1)
                Contains the row coordinates of peaks in H
@return thetas: 2D numpy array with shape (nLines, 1)
                Contains the column coordinates of peaks in H
'''
def myHoughLines(H, nLines):
    suppressed = nonMaxSuppression(H)
    filtered = np.where(H == suppressed, H, 0)

    sortedH = np.argsort(filtered, axis=None)
    # convert indices of sortedH from 1D -> 2D
    converted = np.dstack(np.unravel_index(sortedH, H.shape))[0]
    
    rhos, thetas = converted[-nLines:].T
    res = [rhos, thetas]
    return res

'''
@param H:   2D numpy array
            Image to be suppressed
'''
def nonMaxSuppression(H):
    # need to consider all neighbors of a pixel in this case, not just those
    # lying along the gradient direction
    kernel = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]])
    res = cv2.dilate(H, kernel)
    return res


def showRes():
    sigma     = 2
    threshold = 0.03
    rhoRes    = 2
    thetaRes  = np.pi / 90
    nLines    = 15
    
    img = cv2.imread('img01.jpg', cv2.IMREAD_GRAYSCALE)
    if (img.ndim == 3):
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = np.float32(img) / 255

    img_edge = myEdgeFilter(img, sigma)
    img_threshold = np.float32(img_edge > threshold)
    assert img is not None, "file DNE? we can't read it"
    img_hough, rhoScale, thetaScale = myHoughTransform(img_threshold, rhoRes, thetaRes)

    [rhos, thetas] = myHoughLines(img_hough, nLines)
    lines = cv2.HoughLinesP(np.uint8(255 * img_threshold), rhoRes, thetaRes, \
                                50, minLineLength = 20, maxLineGap = 5)
    img_lines = np.dstack([img,img,img])
    # display line results from myHoughLines function in red
    for k in np.arange(nLines):
        a = np.cos(thetaScale[thetas[k]])
        b = np.sin(thetaScale[thetas[k]])
        
        x0 = a*rhoScale[rhos[k]]
        y0 = b*rhoScale[rhos[k]]
        
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        
        cv2.line(img_lines,(x1,y1),(x2,y2),(0,0,255),1)
    
    # display line segment results from cv2.HoughLinesP in green
    for line in lines:
        coords = line[0]
        cv2.line(img_lines, (coords[0], coords[1]), (coords[2], coords[3]), \
                    (0, 255, 0), 1)

    plt.imshow(img_lines, cmap='gray')
    plt.title('My Edge Filter')
    plt.axis('off')
    plt.show()

# showRes()
