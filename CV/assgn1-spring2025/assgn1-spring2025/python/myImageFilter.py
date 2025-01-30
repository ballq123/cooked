import numpy as np

'''
@param img0:    2D numpy array
                Grayscale image
@param h:       2D numpy array
                Matrix storing a convolution filter
'''
def myImageFilter(img0, h):
    imageH, imageW = np.shape(img0)
    fH, fW = np.shape(h)
    paddingH, paddingW = fH // 2, fW // 2
    padded = np.pad(img0, ((paddingH, paddingH), (paddingW, paddingW)), mode='edge')

    img1 = np.zeros((imageH, imageW))
    for i in range(fH):
        for j in range(fW):
            subsection = padded[i: i + imageH, j: j + imageW]
            hSubSection = h[i, j]
            img1 += (subsection * hSubSection)
    return img1


