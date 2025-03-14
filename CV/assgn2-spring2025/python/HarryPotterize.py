import numpy as np
import cv2
from matchPics import matchPics
from matplotlib import pyplot as plt
from planarH import compositeH, computeH_ransac, computeH_norm, computeH

def hPotterize():
    cv_cover = cv2.imread('../data/cv_cover.jpg')
    cvS = cv_cover.shape
    cv_desk = cv2.imread('../data/cv_desk.png')
    hp = cv2.imread('../data/hp_cover.jpg')
    hp = cv2.resize(hp, (cvS[1], cvS[0]))

    matches, locs1, locs2 = matchPics(cv_cover, cv_desk)
    matchCov, matchDesk = matches[:,0], matches[:,1]
    x1, x2 = locs1[matchCov], locs2[matchDesk]
    bestH2to1, _ = computeH_ransac(x1, x2)
    comp = compositeH(bestH2to1, hp, cv_desk)

    plt.figure()
    plt.axis('off')
    plt.imshow(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB))
    # save to put in writeup
    plt.savefig("../results/harryP.png", bbox_inches='tight', pad_inches=0)
    plt.show()

hPotterize()
