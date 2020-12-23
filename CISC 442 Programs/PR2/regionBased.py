import numpy as np
import cv2
from numpy.lib.stride_tricks import as_strided

# Let user select the desired number of levels for multi-resolution
levels = int(input("Enter desired number of levels: "))

def getOriginalImages():
    # Change the images here to test
    left = cv2.imread('leftPic.png')
    right = cv2.imread('rightPic.png')

    originalLeft = left.copy()
    originalRight = right.copy()

    return originalLeft, originalRight

def resolution(image, levels):

    h, w, c = image.shape

    outputImage = image

    # Duplicating 1-pixel to the corresponding 4-pixels
    for i in range(0, h, 2**(levels-1)):
        for j in range(0, w, 2**(levels-1)):
            for k in range(0, 3, 1):
                outputImage[i:i+2**(levels-1), j:j+2**(levels-1), k] = image[i, j, k]

    return outputImage

def initImage():
    left, right = getOriginalImages()

    left = resolution(left, levels)
    right = resolution(right, levels)

    cv2.imshow('Input left image', left)
    cv2.imshow('Input right image', right)

    left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    # Set the template size here
    templateSize = int(input("Enter your desired template size: "))

    # Set the matching window here
    window = int(input("Enter your desired matching window size: "))

    return left, right, templateSize, window


# Stereo matching using SSD
def sumOfSquaredDiff(image, template, mask=None):
    if mask is None:
        mask = np.ones_like(template)
    window_size = template.shape

    updatedImage = as_strided(image, shape=(image.shape[0] - window_size[0] + 1, image.shape[1] - window_size[1] + 1,) +
                                           window_size, strides=image.strides * 2)

    # Compute the sum of squared differences
    ssd = ((updatedImage - template) ** 2 * mask).sum(axis=-1).sum(axis=-1)
    return ssd

def disparity_ssd(left, right, templateSize, window, lambdaV):
    im_rows, im_cols = left.shape
    tpl_rows = tpl_cols = templateSize
    disparity = np.zeros(left.shape, dtype=np.float32)

    # Double division signs used to make all division integer division
    for r in range(int(tpl_rows/2), int(im_rows-tpl_rows/2)):
        tr_min, tr_max = max(r-tpl_rows//2, 0), min(r+tpl_rows//2+1, im_rows)
        for c in range(int(tpl_cols/2), int(im_cols-tpl_cols/2)):
            tc_min = max(c-tpl_cols//2, 0)
            tc_max = min(c+tpl_cols//2+1, im_cols)
            tpl = left[tr_min:tr_max, tc_min:tc_max].astype('int')
            rc_min = max(c - window // 2, 0)
            rc_max = min(c + window // 2 + 1, im_cols)
            R_strip = right[tr_min:tr_max, rc_min:rc_max].astype('int')
            error = sumOfSquaredDiff(R_strip, tpl)
            c_tf = max(c-rc_min-tpl_cols//2, 0)
            dist = np.arange(error.shape[1]) - c_tf
            cost = error + np.abs(dist * lambdaV)
            _,_,min_loc,_ = cv2.minMaxLoc(cost)
            disparity[r, c] = dist[min_loc[0]]
    return disparity

def ssd():

    # Calculate disparity maps of the left and right images
    left, right, templateSize, window = initImage()
    lDisparity = np.abs(disparity_ssd(left, right, templateSize=templateSize, window=window, lambdaV=0.0))
    rDisparity = np.abs(disparity_ssd(right, left, templateSize=templateSize, window=window, lambdaV=0.0))

    # Scale disparity maps
    lDisparity = cv2.normalize(lDisparity, lDisparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    rDisparity = cv2.normalize(rDisparity, rDisparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return lDisparity, rDisparity


# Stereo matching using SAD
def sumOfAbsDiff(image, template, mask=None):
    if mask is None:
        mask = np.ones_like(template)
    window_size = template.shape

    updatedImage = as_strided(image, shape=(image.shape[0] - window_size[0] + 1, image.shape[1] - window_size[1] + 1,) +
                                           window_size, strides=image.strides * 2)

    # Compute the sum of squared differences
    ssd = ((abs(updatedImage - template)) * mask).sum(axis=-1).sum(axis=-1)
    return ssd

def disparity_sad(left, right, templateSize, window, lambdaV):
    im_rows, im_cols = left.shape
    tpl_rows = tpl_cols = templateSize
    disparity = np.zeros(left.shape, dtype=np.float32)

    for r in range(tpl_rows//2, im_rows-tpl_rows//2):
        tr_min, tr_max = max(r-tpl_rows//2, 0), min(r+tpl_rows//2+1, im_rows)
        for c in range(tpl_cols//2, im_cols-tpl_cols//2):
            tc_min = max(c-tpl_cols//2, 0)
            tc_max = min(c+tpl_cols//2+1, im_cols)
            tpl = left[tr_min:tr_max, tc_min:tc_max]
            rc_min = max(c - window // 2, 0)
            rc_max = min(c + window // 2 + 1, im_cols)
            R_strip = right[tr_min:tr_max, rc_min:rc_max]
            error = sumOfAbsDiff(R_strip, tpl)
            c_tf = max(c-rc_min-tpl_cols//2, 0)
            dist = np.arange(error.shape[1]) - c_tf
            cost = error + np.abs(dist * lambdaV)
            _,_,min_loc,_ = cv2.minMaxLoc(cost)
            disparity[r, c] = dist[min_loc[0]]
    return disparity

def sad():

    # Calculate disparity maps of the left and right images
    left, right, templateSize, window = initImage()
    lDisparity = np.abs(disparity_sad(left, right, templateSize=templateSize, window=window, lambdaV=0.0))
    rDisparity = np.abs(disparity_sad(right, left, templateSize=templateSize, window=window, lambdaV=0.0))

    # Scale disparity maps
    lDisparity = cv2.normalize(lDisparity, lDisparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    rDisparity = cv2.normalize(rDisparity, rDisparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return lDisparity, rDisparity

# Stereo matching using normalized correlation

def disparity_ncorr(left, right, templateSize, window, lambdaV):
    im_rows, im_cols = left.shape
    tpl_rows = tpl_cols = templateSize
    disparity = np.zeros(left.shape, dtype=np.float32)

    for r in range(tpl_rows//2, im_rows-tpl_rows//2):
        tr_min, tr_max = max(r-tpl_rows//2, 0), min(r+tpl_rows//2+1, im_rows)
        for c in range(tpl_cols//2, im_cols-tpl_cols//2):
            tc_min = max(c-tpl_cols//2, 0)
            tc_max = min(c+tpl_cols//2+1, im_cols)
            tpl = left[tr_min:tr_max, tc_min:tc_max]
            rc_min = max(c - window // 2, 0)
            rc_max = min(c + window // 2 + 1, im_cols)
            R_strip = right[tr_min:tr_max, rc_min:rc_max]
            error = cv2.matchTemplate(R_strip, tpl, method=cv2.TM_CCORR_NORMED)
            c_tf = max(c-rc_min-tpl_cols//2, 0)
            dist = np.arange(error.shape[1]) - c_tf
            cost = error - np.abs(dist * lambdaV)
            _,_,_,max_loc = cv2.minMaxLoc(cost)
            disparity[r, c] = dist[max_loc[0]]
    return disparity

def ncc():

    # Calculate disparity maps of the left and right images
    left, right, templateSize, window = initImage()
    lDisparity = np.abs(disparity_ncorr(left, right, templateSize=templateSize, window=window, lambdaV=0.0))
    rDisparity = np.abs(disparity_ncorr(right, left, templateSize=templateSize, window=window, lambdaV=0.0))

    # Scale disparity maps
    lDisparity = cv2.normalize(lDisparity, lDisparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                        dtype=cv2.CV_8U)
    rDisparity = cv2.normalize(rDisparity, rDisparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                        dtype=cv2.CV_8U)

    return lDisparity, rDisparity


# Validity check of the two images
def validity(left, right):
    print("Validating image")
    r1, c1 = left.shape
    r2, c2 = right.shape

    # Validate left image by calculating left - right image disparities
    for i in range(0, r1, 1):
        for j in range(0, c1, 1):
            if left[i,j] != right[i,j]:
                left[i,j] = 0

    # Validate right image by calculating right - left image disparities
    for i in range(0, r2, 1):
        for j in range(0, c2, 1):
            if right[i, j] != left[i, j]:
                right[i, j] = 0

# Averaging is performed in the neighborhood to fill the gaps (zeroes)
def averaging(left, right):
    print("Averaging image")
    kernel = np.ones((5, 5), np.float32) / 25
    left = cv2.filter2D(left, -1, kernel)
    right = cv2.filter2D(right, -1, kernel)


# Propogating disparity to the lower level of the pyramid and updating the disparities
def propogate(left, right):
    print("Propgating image")
    h, w = left.shape

    for k in range(levels-1, 0, -1):
        outputLeft = left.copy()
        for i in range(0, h, 2 ** (k)):
            for j in range(0, w, 2 ** (k)):
                outputLeft[i:i + 2 ** (k), j:j + 2 ** (k)] = left[i, j]

    for k in range(levels-1, 0, -1):
        outputRight = right.copy()
        for i in range(0, h, 2 ** (k)):
            for j in range(0, w, 2 ** (k)):
                outputRight[i:i + 2 ** (k), j:j + 2 ** (k)] = left[i, j]

    nLeft, nRight = getOriginalImages()
    nLeft = cv2.cvtColor(nLeft, cv2.COLOR_BGR2GRAY)
    nRight = cv2.cvtColor(nRight, cv2.COLOR_BGR2GRAY)
    nld = np.abs(disparity_ssd(nLeft, right, templateSize=7, window=100, lambdaV=0.0))
    nrd = np.abs(disparity_ssd(nRight, left, templateSize=7, window=100, lambdaV=0.0))

    # Scale disparity maps
    # nld is the new left disparity given while nrd is the new right dispairty
    nld = cv2.normalize(nld, nld, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
    nrd = cv2.normalize(nrd, nrd, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)

    updatedLeft = nld + outputLeft
    updatedRight = nrd + outputRight


print("Choose your desired matching score\nPress 1 for Sum of Absolute Differences\nPress 2 for Sum of Squared Differences\nPress 3 for Normalized Cross-Correlation")
score = int(input("Enter your desired matching score: "))

if score == 1:
    left, right = ssd()
    cv2.imshow('Left Disparity', left)
    cv2.imshow('Right Disparity', right)
    validity(left, right)
    averaging(left,right)
    propogate(left, right)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif score == 2:
    left, right = sad()
    cv2.imshow('Left Disparity', left)
    cv2.imshow('Right Disparity', right)
    validity(left, right)
    averaging(left, right)
    propogate(left, right)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif score == 3:
    left, right = ncc()
    cv2.imshow('Left Disparity', left)
    cv2.imshow('Right Disparity', right)
    validity(left, right)
    averaging(left, right)
    propogate(left, right)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print ("Select a valid option for matching")

