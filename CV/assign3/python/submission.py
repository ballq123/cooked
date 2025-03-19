"""
Homework 5
Submission Functions
"""

# import packages here
from helper import *
import numpy as np
import scipy
from scipy import signal
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
    
    pts1Norm = pts1 @ T
    pts2Norm = pts2 @ T
    
    I = 9
    A = np.zeros((I, I))
    for i in range(I):
        x1, y1 = pts1Norm[i]
        x2, y2 = pts2Norm[i]
        a0, a1 = x1 * x2, x1 * y2
        a2, a3 = x1, y1 * x2
        a4, a5 = y1 * y2, y1
        a6, a7 = x2, y2
        A[i] = [a0, a1, a2, a3, a4, a5, a6, a7, 1]
        
    _, _, V = np.linalg.svd(A)
    F = V[8].reshape(3, 3)
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    fRank2 = np.diag(S)
    Funnorm = U @ fRank2 @ Vt
    Funnorm = T0 @ Funnorm @ T0
    
    F_refined = refineF(Funnorm, pts1, pts2)
    
    return F_refined


"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""
def epipolar_correspondences(im1, im2, F, pts1, windowSize=5):
    halfW = (windowSize // 2)
    minDist = windowSize ** 2
    kGauss = signal.windows.gaussian(windowSize, 1, sym=True)
    kGaussR = kGauss.reshape(1, windowSize)
    k = scipy.signal.convolve2d(kGaussR, kGaussR.T)

    pts2 = np.zeros_like(pts1)
    h1, _ = pts1.shape
    pts1 = np.hstack((pts1, np.ones((h1, 1))))
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY).astype(np.float32)

    for i, point in enumerate(pts1):
        x, y = point[0], point[1]
        top, bottom = (y - halfW), ((y - halfW) + 1)
        l, r = (x - halfW), ((x - halfW) + 1)
        patch1 = im1[int(top):int(bottom), int(l):int(r)]
        w1 = patch1 * k

        # normalize epipolar line => div by euclidean dist eq
        line = F @ point
        a = line[0]
        b = line[1]
        c = line[2]
        aSq = a ** 2
        bSq = b ** 2
        line = line / (np.sqrt(aSq + bSq)) 

        height, width = im2.shape
        curP = width - 1
        if (b == 0):
            curP = im2.shape[0] - 1

        d = []
        xVals, yVals = [], []
        for val in range(curP):
            xVal, yVal = val, -1 * (a * val + c) // b
            if (b == 0):
                xVal, yVal = -1 * ((b * val) + c) // a, val
            if ((xVal < halfW) or ((width - halfW - 1) < xVal) 
                or (yVal < halfW) or ((height - halfW - 1) < yVal)):
                continue

            top = (yVal - halfW)
            bottom = (top + windowSize)
            left = (xVal - halfW) 
            r = (left + windowSize)
            patch2 = im2[int(top):int(bottom), int(left):int(r)]
            w2 = np.multiply(patch2, k)

            a1 = point[:2]
            arr = np.array([xVal, yVal])
            dist = np.linalg.norm(a1 - arr)

            if dist < minDist:
                d.append(np.linalg.norm(w1 - w2))
                xVals.append(xVal)
                yVals.append(yVal)
        loc = np.argmin(d)
        arr = np.array([int(xVals[loc]), int(yVals[loc])])
        pts2[i] = arr

    return pts2


"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    k2T = K2.T
    E = k2T @ F @ K1
    return E


"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):
    numPoints = pts1.shape[0]
    # store homogeneous 3D points
    pts3dHomogen = np.zeros((numPoints, 4))  
    for i in range(numPoints):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]
        A = np.vstack([
            x1 * P1[2, :] - P1[0, :],
            y1 * P1[2, :] - P1[1, :],
            x2 * P2[2, :] - P2[0, :],
            y2 * P2[2, :] - P2[1, :] 
        ])

        _, _, Vh = np.linalg.svd(A)
        xHomogen = Vh[-1]  
        pts3dHomogen[i] = xHomogen / xHomogen[-1]
    return pts3dHomogen[:, :3]


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
    c1 = -np.linalg.inv(K1 @ R1) @ (K1 @ t1)
    c2 = -np.linalg.inv(K2 @ R2) @ (K2 @ t2)
    diff = c1 - c2

    r1 = diff / np.linalg.norm(diff)
    r1First = r1.T[0]
    r2a = np.cross(R1[2,:].T, r1[:,0]) 
    r2 = r2a[:,None]
    r2First = r2.T[0]
    r3a = np.cross(r1[:,0], r2[:,0])
    r3 = r3a[:,None]
    r3First = r3.T[0]

    R = np.array([r1First, -r2First, r3First])
    trans1 = -(R @ c1)
    trans2 = -(R @ c2)
    
    k2Dot = K2 @ R
    M1 = (k2Dot) @ np.linalg.inv(K1 @ R1)
    M2 = (k2Dot) @ np.linalg.inv(K2 @ R2)
    
    return M1, M2, K2, K2, R, R, trans1, trans2


"""
Q3.2.2 Disparity Map
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           max_disp, scalar maximum disparity value
           win_size, scalar window size value
       [O] dispM, disparity map (H1xW1 matrix)
"""
def get_disparity(im1, im2, max_disp, win_size):
    frame = []
    maxDisp = max_disp + 1
    for i in range(maxDisp):
        frame.append(0)
    arr = np.ones((win_size, win_size))
    for i in range(len(frame)):
        shift = np.roll(im2, -i, axis=-1)
        diff = im1 - shift
        dist = diff ** 2
        conv = scipy.signal.convolve2d(dist, arr, "same")
        frame[i] = conv
    frame = np.array(frame)
    dispM = frame.argmin(axis=0)
    dispM = np.where(dispM > 255, 255, dispM)
    return dispM


"""
Q3.2.3 Depth Map
       [I] dispM, disparity map (H1xW1 matrix)
           K1 K2, camera matrices (3x3 matrix)
           R1 R2, rotation matrices (3x3 matrix)
           t1 t2, translation vectors (3x1 matrix)
       [O] depthM, depth map (H1xW1 matrix)
"""
def get_depth(dispM, K1, K2, R1, R2, t1, t2):
    c1 = -np.linalg.inv(K1 @ R1) @ (K1 @ t1)
    c2 = -np.linalg.inv(K2 @ R2) @ (K2 @ t2)
    b = np.linalg.norm(c2 - c1)  # Baseline
    
    f = K1[0, 0]
    depthM = np.zeros_like(dispM, dtype=np.float32)
    for y in range(dispM.shape[0]):
        for x in range(dispM.shape[1]):
            disp = dispM[y, x]
            if disp > 0:
                depthM[y, x] = (b * f) / disp
            else:
                depthM[y, x] = 0

    return depthM


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
