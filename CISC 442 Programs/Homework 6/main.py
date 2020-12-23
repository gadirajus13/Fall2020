# Sohan Gadiraju
# Homework 6

import cv2
import numpy as np
from matplotlib import pyplot as plt

flower = cv2.imread("flower.jpg")
flower = cv2.resize(flower,(600,600))

def getEdge(img):
    edges = cv2.Canny(img,100,200)
    return edges

def getCorner(img):
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
    grayscale = np.float32(grayscale) 
    corners = cv2.cornerHarris(grayscale, 2, 5, 0.07) 

    corners = cv2.dilate(corners, None) 
    img[corners > 0.01 * corners.max()]=[255, 0, 0] 

    return img

def rotate(image, angle):
    center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image, rot_matrix, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

# 1: Run an in-built edge detector and a corner detector, produce 2 output images ( i) image with edges, ii) image with corners)

# 1a
orgEdges = getEdge(flower)
cv2.imshow('Flower with Edges', orgEdges)

# 1b
orgCorners = getCorner(flower)
cv2.imshow('Flower with Corners', orgCorners)



# 2: Rotate the original image by 45-degrees and perform (1)
flowerRotated = rotate(flower, 45)

# 2a
rotEdges = getEdge(flowerRotated)
cv2.imshow('Rotated Flower with Edges', rotEdges)
# 2b
rotCorners = getCorner(flowerRotated)
cv2.imshow('Rotated Flower with Corners', rotCorners)



# 3: Scale the original image by 1.5 in both the x and y-directions and perform (1)
resized = cv2.resize(flower, (0,0), fx=1.5, fy=1.5)

# 3a
resizedEdges = getEdge(resized)
cv2.imshow('Resized Flower with Edges', resizedEdges)

# 3b
resizedCorners = getCorner(resized)
cv2.imshow('Resized Flower with Corners', resizedCorners)



# 4: Shear the original image in the x-direction by 1.3 and perform (1)
rows, cols, dim = flower.shape
M = np.float32([[1, 0.5, 0],
                [0, 1, 0],
                [0 , 0, 1]])
shearedX = cv2.warpPerspective(flower,M,(int(cols*1.3), int(rows*1.3)))

shearedEdgesX = getEdge(shearedX)
cv2.imshow('Sheared Flower with Edges in X Direction', shearedEdgesX)

shearedCornersX = getCorner(shearedX)
cv2.imshow('Sheared Flower with Corners in X Direction', shearedCornersX)



# 5: Shear the original image in the y-direction by 1.3 and perform (1)
M = np.float32([[1, 0, 0],
                [0.5, 1, 0],
                [0 , 0, 1]])
shearedY = cv2.warpPerspective(flower,M,(int(cols*1.3), int(rows*1.3)))

shearedEdgesY = getEdge(shearedY)
cv2.imshow('Sheared Flower with Edges in Y Direction', shearedEdgesY)

shearedCornersY = getCorner(shearedX)
cv2.imshow('Sheared Flower with Corners in Y Direction', shearedCornersY)


while(1):
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif k == ord('q'):
        break      
if flower is None:
    print("File does not exit\n")