import cv2
import numpy as np
import scipy.ndimage
from helper import plotMatches
from matchPics import matchPics
from matplotlib import pyplot as plt


#Q3.5
#Read the image and convert to grayscale, if necessary
cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
I1 = cv2.cvtColor(cv_cover, cv2.COLOR_BGR2GRAY) 
I2 = cv2.cvtColor(cv_desk, cv2.COLOR_BGR2GRAY)
# 0 -> 360 degrees, incrementing by 10
rotations = np.arange(0, 360, 10)
histogram = np.zeros(36)

for i in range(36):
	#Rotate Image
	r1 = scipy.ndimage.rotate(cv_cover, rotations[i])
	#Compute features, descriptors and Match features
	m, l1, l2 = matchPics(cv_cover, r1)
	#Update histogram
	mH, mW = m.shape
	histogram[i] = mH

# histogram
plt.figure(figsize=(15, 4))
# plt.axis('off')
plt.bar(rotations, histogram, width=5, color='red', alpha=0.7)
plt.title('Histogram of BRIEF Matches for Different Rotations')
plt.xlabel('Rotation Angle (degrees)')
plt.ylabel('Number of Matches')
plt.xticks(rotations)  # Set x-ticks to correspond to the rotation angles
plt.grid(True)
plt.savefig("../results/briefHistogram.png", bbox_inches='tight', pad_inches=0)
plt.show()

# BRIEF result @ specific orientation
orientations = [45, 180, 277]
for deg in range(len(orientations)):
	rot_image = scipy.ndimage.rotate(cv_cover, orientations[deg])
	m, l1, l2 = matchPics(cv_cover, rot_image)
	plotMatches(cv_cover, rot_image, m, l1, l2)
