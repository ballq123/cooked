import numpy as np
import cv2


def computeH(x1, x2):
	"""
	OUTPUT:
	H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
	equation
	"""
	#Q3.6
	#Compute the homography between two sets of points
	N, _ = x1.shape
	A = []
	for i in range(N):
		x1_i, y1_i = x1[i]
		x2_i, y2_i = x2[i]
		A.append([-x2_i, -y2_i, -1, 0, 0, 0, x1_i * x2_i, x1_i * y2_i, x1_i])
		A.append([0, 0, 0, -x2_i, -y2_i, -1, y1_i * x2_i, y1_i * y2_i, y1_i])
	A = np.array(A)

	_, _, Vh = np.linalg.svd(A)
	H2to1 = Vh[-1].reshape(3, 3)
	return H2to1


def computeH_norm(x1, x2):
	#Q3.7
	#Compute the centroid of the points
	centroid1 = np.mean(x1, axis=0)
	centroid2 = np.mean(x2, axis=0)

	#Shift the origin of the points to the centroid
	x1Shift = x1 - centroid1
	x2Shift = x2 - centroid2

	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
	dist1 = np.max(np.linalg.norm(x1Shift, axis=1))
	dist2 = np.max(np.linalg.norm(x2Shift, axis=1))
	scale1 = np.sqrt(2) / dist1
	scale2 = np.sqrt(2) / dist2
	x1Norm = x1Shift * scale1
	x2Norm = x2Shift * scale2

	#Similarity transform 1
	t1 = np.array([
        [scale1, 0,      -scale1 * centroid1[0]],
        [0,      scale1, -scale1 * centroid1[1]],
        [0,      0,       1]
    ])

	#Similarity transform 2
	t2 = np.array([
        [scale2, 0,      -scale2 * centroid2[0]],
        [0,      scale2, -scale2 * centroid2[1]],
        [0,      0,       1]
    ])

	#Compute homography
	hNorm = computeH(x1Norm, x2Norm)

	#Denormalization
	H2to1 = np.linalg.inv(t1) @ hNorm @ t2

	return H2to1




def computeH_ransac(x1, x2):
	"""
	OUTPUTS
	bestH2to1 - homography matrix with the most inliers found during RANSAC
	inliers - a vector of length N (len(matches)) with 1 at the those matches
		that are part of the consensus set, and 0 elsewhere.
	"""
	#Q3.8
	#Compute the best fitting homography given a list of matching points
	N = x1.shape[0]
	x1, x2 = np.fliplr(x1), np.fliplr(x2)
	x2H = np.hstack((x2, np.ones((N, 1))))  
	bestH2to1 = None
	max_inliers = 0
	best_inliers = None
	e = 1e-7
	numIters = 1000
	tolerance = 2
	for i in range(numIters):
		idx = np.random.choice(N, 4, replace=False)
		H = computeH_norm(x1[idx], x2[idx])
		x1_proj = (H @ x2H.T).T 

		# avoid division by zero
		projs = []
		for proj in x1_proj:
			p0, p1 = proj[0], proj[1]
			if proj[2] != 0:
				projs.append([p0/proj[2], p1/proj[2]])
			else:
				projs.append([p0/e, p1/e])

		distances = np.linalg.norm(x1 - projs, axis=1)
		inliers = distances <= tolerance
		num_inliers = np.sum(inliers)
		if num_inliers > max_inliers:
			max_inliers = num_inliers
			bestH2to1 = H
			best_inliers = inliers
		if i >= N or num_inliers >= N:
			return bestH2to1, best_inliers
		
	return bestH2to1, best_inliers



def compositeH(H2to1, template, img):
	
	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	# x_template = H2to1*x_photo
	# For warping the template to the image, we need to invert it.
	hInv = np.linalg.inv(H2to1)
	
	#Create mask of same size as template
	tShape = template.shape
	m = np.ones(tShape)

	#Warp mask by appropriate homography
	imH = img.shape[0]
	imW = img.shape[1]
	mWarp = cv2.warpPerspective(m, hInv, (imW, imH))

	#Warp template by appropriate homography
	tWarp = cv2.warpPerspective(template, hInv, (imW, imH))

	#Use mask to combine the warped template and the image
	composite_img = tWarp + img * np.logical_not(mWarp)
	
	return composite_img

