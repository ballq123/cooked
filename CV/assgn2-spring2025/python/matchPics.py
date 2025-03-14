import numpy as np
import cv2
import skimage.color
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection
from helper import plotMatches

def matchPics(I1, I2):
	#I1, I2 : Images to match
	

	#Convert Images to GrayScale
	I1 = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY) 
	I2 = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
	
	#Detect Features in Both Images
	feat1 = corner_detection(I1)
	feat2 = corner_detection(I2)
	
	#Obtain descriptors for the computed feature locations
	d1, locs1 = computeBrief(I1, feat1)
	d2, locs2 = computeBrief(I2, feat2)	

	#Match features using the descriptors
	matches = briefMatch(d1, d2)
	
	return matches, locs1, locs2
