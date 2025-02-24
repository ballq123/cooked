import numpy as np
import cv2
from loadVid import loadVid
from matchPics import matchPics
from planarH import computeH_ransac, compositeH, computeH_norm, computeH

def makeVid():
    cv_cover = cv2.imread('../data/cv_cover.jpg')

    ar_source = loadVid("../data/ar_source.mov")
    np.save("../data/ar_source.npy", ar_source)
    ar_source = np.load("../data/ar_source.npy")

    book = loadVid("../data/book.mov")
    np.save("../data/book.npy", book)
    book = np.load("../data/book.npy")

    arH, arW = ar_source[0].shape[:2]
    bH, bW = book[0].shape[:2]

    c1 = (arW / 2) - (bW / 2)
    c2 = (arW / 2) + (bW / 2)
    cropI, cropJ = np.round([c1, c2]).astype(int)

    arCrop = []
    for f in ar_source:
        # Crop each frame
        cropped_frame = f[:, cropI:cropJ]  
        arCrop.append(cropped_frame)
    ar_source = np.array(arCrop)

    ar = []
    cvS = cv_cover.shape
    for i in range(len(book)):
        # print("i:", i)
        template = cv2.resize(ar_source[i], (cvS[1], cvS[0]))
        matches, locs1, locs2 = matchPics(cv_cover, book[i])
        matchCov, matchDesk = matches[:,0], matches[:,1]
        x1, x2 = locs1[matchCov], locs2[matchDesk]
        bestH2to1, _ = computeH_ransac(x1, x2)
        comp = compositeH(bestH2to1, template, book[i])
        ar.append(comp)

    ar = np.array(ar)
    arS = ar.shape
    fps = 29
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('../result/ar.avi', fourcc, fps, (arS[2], arS[1]))
    for frame in ar:
        out.write(frame)

    out.release()
    return

makeVid()