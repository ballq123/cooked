import cv2 
import math
import numpy as np
from scipy import signal  
from matplotlib import pyplot as plt
from myImageFilter import myImageFilter
from myEdgeFilter import myEdgeFilter
from myHoughTransform import myHoughTransform

def myHoughLines(H, nLines):
    # YOUR CODE HERE
