
import numpy as np
import cv2
from numpy.lib.stride_tricks import as_strided

def findCorners(image, window_size, k, thresh):
    # Find x and y derivatives
    dy, dx = np.gradient(image)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2
    height = image.shape[0]
    width = image.shape[1]

    cornerList = []
    copiedImage = image.copy()
    outputImage = cv2.cvtColor(copiedImage, cv2.COLOR_GRAY2RGB)
    offset = window_size//2

    # Detect all corners using Harris Corner Detector
    print ("Finding Corners of the Image")
    for y in range(offset, height-offset):
        for x in range(offset, width-offset):
            windowIxx = Ixx[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIxy = Ixy[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIyy = Iyy[y-offset:y+offset+1, x-offset:x+offset+1]
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()

            # Calculating corner response using determinant and trace (Harris)
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            r = det - k*(trace**2)

            # If corner response crosses threshold, point is marked
            if r > thresh:
                # print x, y, r
                cornerList.append([x, y, r])
                outputImage.itemset((y, x, 0), 0)
                outputImage.itemset((y, x, 1), 0)
                outputImage.itemset((y, x, 2), 255)
    return outputImage, cornerList


# Let user select the desired number of levels
levels = int(input("Enter desired number of levels: "))

def getOriginalImages():
    # Change the images here to test
    left = cv2.imread('leftPic.png')
    right = cv2.imread('rightPic.png')

    originalLeft = left.copy()
    originalRight = right.copy()

    return originalLeft, originalRight

def resolution(image, levels):
    if(levels < 1):
        print ("Invalid number of levels")

    h, w, c = image.shape

    outputImage = image

    for i in range(0, h, 2**(levels-1)):
        for j in range(0, w, 2**(levels-1)):
            for k in range(0, 3, 1):
                outputImage[i:i+2**(levels-1), j:j+2**(levels-1), k] = image[i, j, k]

    return outputImage

def initImage():
    left,right = getOriginalImages()
    left = resolution(left, levels)
    right = resolution(right, levels)
    cv2.imshow('Input left image', left)
    cv2.imshow('Input right image', right)

    templateSize = 7

    window = 100

    left = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
    right = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)

    window_size = 3
    k = 0.15
    thresh = 100000

    finalLeft, cornerList = findCorners(left, int(window_size), float(k), int(thresh))
    finalRight, cornerList = findCorners(right, int(window_size), float(k), int(thresh))

    return finalLeft, finalRight, templateSize, window


# Stereo matching using SSD
def sumOfSquaredDifferences(image, template, mask=None):
    if mask is None:
        mask = np.ones_like(template)
    window_size = template.shape

    updatedImage = as_strided(image, shape=(image.shape[0] - window_size[0] + 1, image.shape[1] - window_size[1] + 1,) +
                                           window_size, strides=image.strides * 2)

    # Compute the sum of squared differences
    ssd = ((updatedImage - template) ** 2 * mask).sum(axis=-1).sum(axis=-1)
    return ssd

def disparity_ssd(left, right, templateSize, window, lambdaValue):
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
            error = sumOfSquaredDifferences(R_strip, tpl)
            c_tf = max(c-rc_min-tpl_cols//2, 0)
            dist = np.arange(error.shape[1]) - c_tf
            cost = error + np.abs(dist * lambdaValue)
            _,_,min_loc,_ = cv2.minMaxLoc(cost)
            disparity[r, c] = dist[min_loc[0]]
    return disparity

def ssd():

    # Calculate disparity maps of the left and right images
    left, right, templateSize, window = initImage()
    left = cv2.cvtColor(left,cv2.COLOR_RGB2GRAY)
    right = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
    lDisparity = np.abs(disparity_ssd(left, right, templateSize=templateSize, window=window, lambdaValue=0.0))
    rDisparity = np.abs(disparity_ssd(right, left, templateSize=templateSize, window=window, lambdaValue=0.0))
    # Scale disparity maps
    lDisparity = cv2.normalize(lDisparity, lDisparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    rDisparity = cv2.normalize(rDisparity, rDisparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return lDisparity, rDisparity

# Stereo matching using SAD
def sumOfAbsDifferences(image, template, mask=None):
    if mask is None:
        mask = np.ones_like(template)
    window_size = template.shape

    updatedImage = as_strided(image, shape=(image.shape[0] - window_size[0] + 1, image.shape[1] - window_size[1] + 1,) +
                                           window_size, strides=image.strides * 2)

    # Compute the sum of squared differences
    ssd = ((abs(updatedImage - template)) * mask).sum(axis=-1).sum(axis=-1)
    return ssd

def disparity_sad(left, right, templateSize, window, lambdaValue):
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
            error = sumOfAbsDifferences(R_strip, tpl)
            c_tf = max(c-rc_min-tpl_cols//2, 0)
            dist = np.arange(error.shape[1]) - c_tf
            cost = error + np.abs(dist * lambdaValue)
            _,_,min_loc,_ = cv2.minMaxLoc(cost)
            disparity[r, c] = dist[min_loc[0]]
    return disparity

def sad():

    # Calculate disparity maps of the left and right images
    left, right, templateSize, window = initImage()
    left = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
    right = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
    lDisparity = np.abs(disparity_sad(left, right, templateSize=templateSize, window=window, lambdaValue=0.0))
    rDisparity = np.abs(disparity_sad(right, left, templateSize=templateSize, window=window, lambdaValue=0.0))
    # Scale disparity maps
    lDisparity = cv2.normalize(lDisparity, lDisparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    rDisparity = cv2.normalize(rDisparity, rDisparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return lDisparity, rDisparity

# Stereo matching using normalized correlation
def disparity_ncorr(left, right, templateSize, window, lambdaValue):
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
            cost = error - np.abs(dist * lambdaValue)
            _,_,_,max_loc = cv2.minMaxLoc(cost)
            disparity[r, c] = dist[max_loc[0]]
    return disparity

def ncc():

    # Calculate disparity maps of the left and right images
    left, right, templateSize, window = initImage()
    cv2.imshow('Corner Response Left Image', left)
    cv2.imshow('Corner Response Right Image', right)
    left = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
    right = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
    lDisparity = np.abs(disparity_ncorr(left, right, templateSize=templateSize, window=window, lambdaValue=0.0))
    rDisparity = np.abs(disparity_ncorr(right, left, templateSize=templateSize, window=window, lambdaValue=0.0))
    # Scale disparity maps
    lDisparity = cv2.normalize(lDisparity, lDisparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
    rDisparity = cv2.normalize(rDisparity, rDisparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)

    return lDisparity, rDisparity


# Valididty check that places 0's if  the left-to-right match does not correspond to right-to-left match
def validate(left, right):
    print("Validating image")
    r1, c1 = left.shape
    r2, c2 = right.shape

    # Validate left image by calculating left - right image disparities
    for i in range(0, r1, 1):
        for j in range(0, c1, 1):
            if left[i,j] != right[i,j]:
                left[i,j] = 0

    # Validate left image by calculating right - left image disparities
    for i in range(0, r2, 1):
        for j in range(0, c2, 1):
            if right[i, j] != left[i, j]:
                right[i, j] = 0


# Average fills gaps
def average(left, right):
    print("Averaging image")
    kernel = np.ones((5, 5), np.float32) / 25
    left = cv2.filter2D(left, -1, kernel)
    right = cv2.filter2D(right, -1, kernel)


# Propogating disparity to the lower level of the pyramid
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
    nld = np.abs(disparity_ssd(nLeft, right, templateSize=7, window=100, lambdaValue=0.0))
    nrd = np.abs(disparity_ssd(nRight, left, templateSize=7, window=100, lambdaValue=0.0))

    # Scale disparity maps
    nld = cv2.normalize(nld, nld, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
    nrd = cv2.normalize(nrd, nrd, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)

    updatedLeft = nld + outputLeft
    updatedRight = nrd + outputRight

print("Choose your desired matching score\nPress 1 for Sum of Absolute Differences\nPress 2 for Sum of Squared Differences\nPress 3 for Normalized Cross-Correlation")
score = int(input("Enter your desired matching score: "))

if score == 1:
    left, right = ssd()
    cv2.imshow('Left Disparity', left)
    cv2.imshow('Right Disparity', right)
    validate(left, right)
    average(left,right)
    propogate(left, right)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif score == 2:
    left, right = sad()
    cv2.imshow('Left Disparity', left)
    cv2.imshow('Right Disparity', right)
    validate(left, right)
    average(left, right)
    propogate(left, right)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif score == 3:
    left, right = ncc()
    cv2.imshow('Left Disparity', left)
    cv2.imshow('Right Disparity', right)
    validate(left, right)
    average(left, right)
    propogate(left, right)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    print ("Select a valid option for matching")