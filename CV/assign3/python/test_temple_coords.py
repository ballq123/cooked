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
# helper.displayEpipolarF(im1, im2, F)

# 3. Load points in image 1 from data/temple_coords.npz
with np.load("../data/temple_coords.npz") as data:
    pts1 = data['pts1']

# 4. Run epipolar_correspondences to get points in image 2
pts2 = epipolar_correspondences(im1, im2, F, pts1)
# for visualization, calls epipolar_correspondences inside
# helper.epipolarMatchGUI(im1, im2, F)

# 5. Load intrinsics data from data/intrinsics.npz
with np.load("../data/intrinsics.npz") as data:
    K1 = data["K1"]
    K2 = data["K2"]

# 6. Compute the essential matrix
#TODO implement
E = essential_matrix(F, K1, K2)
print("Essential Matrix\n", E, "\n")

# 7. Create the camera projection matrix P1
# Assume that extrinsics = (I | 0)
#TODO implement
iden = np.array([[1. ,0. ,0. ,0.],
                 [0. ,1., 0., 0.],
                 [0. ,0., 1., 0.]])
P1 = K1 @ iden

# 8. Use camera2 to get 4 camera projection matrices M2
#TODO implement
M2s = helper.camera2(E)

# 9. Run triangulate using the projection matrices and find out correct M2, P2, and pts_3d
#TODO implement
numCands = 4
p2Candidates = []
for i in range(numCands):
    p = K2 @ M2s[:, : ,i]
    p2Candidates.append(p)

threeDimPts = []
for i in range(numCands):
    threeDimPts.append(sub.triangulate(P1, pts1, p2Candidates[i], pts2))

cands = [0, 0, 0, 0]
for i in range(numCands):
    cands[i] = np.sum(threeDimPts[i][:, 2] > 0)

best = np.argmax(cands)
M2 = M2s[:, :, best]
P2 = p2Candidates[best]
pts_3d = triangulate(P1, pts1, P2, pts2)
# print("best:", best)

# 10. Get reprojection error
#TODO implement
ones = np.ones((pts_3d.shape[0], 1))
pts1Homo, pts2Homo =  P1 @ np.hstack((pts_3d, ones)).T, P2 @ np.hstack((pts_3d, ones)).T
proj1, proj2 = pts1Homo[:2] / pts1Homo[2], pts2Homo[:2] / pts2Homo[2]
p1T, p2T = pts1.T, pts2.T
err1 = np.linalg.norm(proj1 - p1T, axis=0)
err2 = np.linalg.norm(proj2 - p2T, axis=0)
e1 = np.sum(err1)
e2 = np.sum(err2)
reprojection_error = e1 + e2
denom = pts_3d.shape[0]
reprojection_error /= (2 * denom)
print("\nReprojection error: ", reprojection_error)

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
