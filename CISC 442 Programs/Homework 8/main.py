
import cv2 
import numpy as np 
from matplotlib import pyplot as plt 
  
# Q1:  How many correspondences are needed to solve the matrix shown on the slide(page)-12.
index = 0
correspondences = 1

calculated = np.zeros((4,1), dtype=np.float32)
# 4 correspondences


# Questions 2 to 4
pts1 = np.zeros((4,2), dtype=np.float32)
pts2 = np.zeros((4,2), dtype=np.float32)
boat = cv2.imread('img.jpeg') 
boat = cv2.resize(boat, (1000, 600)) 
rows, cols, ch = boat.shape 
together = np.zeros((rows, 2*cols, 3))

def select_correspondences(event, x, y, flags, param):
    global index, pts1, pts2
    center_coordinates = (x, y)
    radius = 3
    thickness = 5
    color = (0, 0, 0)
    if event == cv2.EVENT_LBUTTONDOWN:
        a = 0
    elif event == cv2.EVENT_LBUTTONUP:
        if index < (correspondences*2):
            if x > 256:
                color = (255, 255, 0)
                pts1[int(index/2)] = (center_coordinates[0] - 256, center_coordinates[1]) 
            else:
                color = (255, 0, 255)
                pts2[int(index/2)] = center_coordinates

ptsA = np.float32([[50, 50],  
                   [200, 50], 
                   [50, 200],
                   [40, 140]]) 
  
ptsB = np.float32([[45, 55], 
                   [200, 50],  
                   [100, 250],
                   [60, 160]]) 

center = (cols/2, rows/2)
focal_length = center[0] / np.tan(60/2 * np.pi / 180)
K = np.array([[focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]], dtype = "double")

S, R = cv2.findHomography(ptsA, ptsB) 

matrix = cv2.getPerspectiveTransform(ptsA, ptsB)

num, Ts,Rs,  Ns  = cv2.decomposeHomographyMat(S, K)
print(Rs[1][0][0])
tranMatrix = np.zeros((3, 3), dtype=np.float32)

tranMatrix[0][0] = Rs[0][0][0]
tranMatrix[0][1] = Rs[0][1][0]
tranMatrix[0][2] = Ts[0][0][0]
tranMatrix[1][0] = Rs[0][1][0]
tranMatrix[1][1] = Rs[0][1][0]
tranMatrix[1][2] = Ts[0][1][0]
tranMatrix[2][0] = 0
tranMatrix[2][1] = 0
tranMatrix[2][2] = 1

dst = cv2.warpPerspective(boat, matrix, (cols, rows))

cv2.imshow('Input',boat) 

cv2.imshow('Output',dst) 


# Displaying the image 
while(1):
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif k == ord('q'):
        break      
if boat is None:
    print("File does not exit\n")