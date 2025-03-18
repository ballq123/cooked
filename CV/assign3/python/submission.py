"""
Homework 5
Submission Functions
"""

# import packages here
from helper import *
import numpy as np
import scipy
import cv2


"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""
def eight_point(pts1, pts2, M):
    T = np.array([[1/M, 0], [0, 1/M]])
    T0 = np.array([[1/M, 0, 0], [0, 1/M, 0], [0, 0, 1]])
    
    pts1_norm = pts1 @ T
    pts2_norm = pts2 @ T
    
    A = np.zeros((9, 9))
    for i in range(9):
        x1, y1 = pts1_norm[i]
        x2, y2 = pts2_norm[i]
        A[i] = [x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1]
        
    _, _, V = np.linalg.svd(A)
    F = V[8].reshape(3, 3)
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    F_rank2 = np.diag(S)
    F_unnorm = U @ F_rank2 @ Vt
    F_unnorm = T0 @ F_unnorm @ T0
    
    F_refined = refineF(F_unnorm, pts1, pts2)
    
    return F_refined


"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""
def epipolar_correspondences(im1, im2, F, pts1):
    pts2 = np.zeros_like(pts1)

    half_w = window_size // 2
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY) if len(im1.shape) == 3 else im1
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY) if len(im2.shape) == 3 else im2

    for i, (x1, y1) in enumerate(pts1):
        # Compute epipolar line l' = F * [x1, y1, 1]
        line = F @ np.array([x1, y1, 1])
        a, b, c = line  # Line equation: ax + by + c = 0

        # Generate candidate points along the epipolar line
        y_candidates = np.arange(max(half_w, 0), min(im2.shape[0] - half_w, im2.shape[0]))
        x_candidates = (- (b * y_candidates + c) / a).astype(int)

        # Filter valid candidates
        valid_idx = (x_candidates >= half_w) & (x_candidates < im2.shape[1] - half_w)
        y_candidates, x_candidates = y_candidates[valid_idx], x_candidates[valid_idx]

        # Extract patch around (x1, y1) in image 1
        patch1 = im1[y1 - half_w:y1 + half_w + 1, x1 - half_w:x1 + half_w + 1]

        best_match = None
        min_dist = float("inf")

        for x2, y2 in zip(x_candidates, y_candidates):
            patch2 = im2[y2 - half_w:y2 + half_w + 1, x2 - half_w:x2 + half_w + 1]

            # Compute Euclidean distance between patches
            dist = np.sum((patch1 - patch2) ** 2)

            if dist < min_dist:
                min_dist = dist
                best_match = (x2, y2)

        pts2[i] = best_match

    return pts2


"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    # replace pass by your implementation
    pass


"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):
    # replace pass by your implementation
    pass


"""
Q3.2.1 Image Rectification
       [I] K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] M1 M2, rectification matrices (3x3 matrix)
           K1p K2p, rectified camera matrices (3x3 matrix)
           R1p R2p, rectified rotation matrices (3x3 matrix)
           t1p t2p, rectified translation vectors (3x1 matrix)
"""
def rectify_pair(K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation
    pass


"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp, win_size):
    # replace pass by your implementation
    pass


"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    # replace pass by your implementation
    pass


"""
Q3.3.1 Camera Matrix Estimation
       [I] x, 2D points (Nx2 matrix)
           X, 3D points (Nx3 matrix)
       [O] P, camera matrix (3x4 matrix)
"""
def estimate_pose(x, X):
    # replace pass by your implementation
    pass


"""
Q3.3.2 Camera Parameter Estimation
       [I] P, camera matrix (3x4 matrix)
       [O] K, camera intrinsics (3x3 matrix)
           R, camera extrinsics rotation (3x3 matrix)
           t, camera extrinsics translation (3x1 matrix)
"""
def estimate_params(P):
    # replace pass by your implementation
    pass
