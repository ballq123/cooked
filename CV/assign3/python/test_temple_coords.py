import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

import submission as sub
from submission import *
import helper

# 1. Load the two temple images and the points from data/some_corresp.npz
im1 = io.imread("../data/im1.png")
im2 = io.imread("../data/im2.png")

with np.load("../data/some_corresp.npz") as data:
    pts1 = data['pts1']
    pts2 = data['pts2']

# 2. Run eight_point to compute F
#TODO implement
M = max(im1.shape[1], im1.shape[0])
F = eight_point(pts1, pts2, M)
print("Fundamental matrix\n", F, end="\n\n")
# for visualization
helper.displayEpipolarF(im1, im2, F)

'''
# 3. Load points in image 1 from data/temple_coords.npz
with np.load("../data/temple_coords.npz") as data:
    pts1 = data['pts1']

# 4. Run epipolar_correspondences to get points in image 2
#TODO implement
pts2 = None
# for visualization, calls epipolar_correspondences inside
helper.epipolarMatchGUI(im1, im2, F)

# 5. Load intrinsics data from data/intrinsics.npz
with np.load("../data/intrinsics.npz") as data:
    K1 = data["K1"]
    K2 = data["K2"]

# 6. Compute the essential matrix
#TODO implement
E = None
print("Essential Matrix\n", E, "\n")

# 7. Create the camera projection matrix P1
# Assume that extrinsics = (I | 0)
#TODO implement
P1 = None

# 8. Use camera2 to get 4 camera projection matrices M2
#TODO implement
M2s = None

# 9. Run triangulate using the projection matrices and find out correct M2, P2, and pts_3d
#TODO implement
pts_3d = None
M2 = None
P2 = None

# 10. Get reprojection error
#TODO implement
reprojectionError = None
print("\nReprojection error: ", reprojectionError)

# 11. Scatter plot the correct 3D points
# From https://pythonprogramming.net/matplotlib-3d-scatterplot-tutorial/
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xData = pts_3d[:, 0]
yData = pts_3d[:, 1]
zData = pts_3d[:, 2]

ax.scatter(xData, yData, zData, c='b', marker='.')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.tight_layout(pad=0.1)
plt.show()

# 12. Save the computed extrinsic parameters (R1,R2,t1,t2) to results/extrinsics.npz
R1 = np.eye(3)
t1 = np.zeros((3,1))
R2 = M2[:, :3]
t2 = M2[:, 3:]
np.savez('../results/extrinsics.npz', R1=R1, t1=t1, R2=R2, t2=t2)
'''