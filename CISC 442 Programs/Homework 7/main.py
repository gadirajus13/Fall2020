# Sohan Gadiraju
# Homework 6

import cv2
import numpy as np
from matplotlib import pyplot as plt

flower = cv2.imread("flower.jpg")
flower = cv2.resize(flower,(400,400))

# Question 1: Generate and show four levels of multi-resolution. Use a Gaussian kernel of your choice. 
res1 = cv2.GaussianBlur(flower,(5,5),0)
res2 = cv2.GaussianBlur(res1,(5,5),0)
res3 = cv2.GaussianBlur(res2,(5,5),0)

# Original Image is Level One, Each Res is the next level
question1 = np.concatenate((flower, res1, res2, res3), axis=1)
cv2.imshow('4 Levels of Multi-Resolution', question1)


# Question 2: Generate and show four levels of multi-scale. Use the same Gaussian kernel as above.
scale1 = cv2.pyrDown(flower)
scale2 = cv2.pyrDown(scale1)
scale3 = cv2.pyrDown(scale2)

# Original Image is Level One, Each Scale is the next pyramid down

cv2.imshow("Level One",flower);
cv2.imshow("Level Two",scale1);
cv2.imshow("Level Three",scale2);
cv2.imshow("Level Four",scale3);

# Question 3: Generate Laplacian planes using a Laplacian kernel of your choice (can use Laplacian of Gaussian, or Laplacian).
plane1 = cv2.Laplacian(flower, cv2.CV_64F)
plane2 = cv2.Laplacian(scale1, cv2.CV_64F)
plane3 = cv2.Laplacian(scale2, cv2.CV_64F)
plane4 = cv2.Laplacian(scale3, cv2.CV_64F)

cv2.imshow("Plane 1",plane1);
cv2.imshow("Plane 2",plane2);
cv2.imshow("Plane 3",plane3);
cv2.imshow("Plane 4",plane4);

# Question 4: Generate approximation to Laplacian using the difference of Gaussian planes from (1). Note, you need to do 'Expand' on images before taking the difference.

# Using first guassian plane
approx1 = flower - res1
approx2 = res1 - res2
approx3 = res2 - res3


cv2.imshow("Approximation 1", approx1)
cv2.imshow("Approximation 2", approx2)
cv2.imshow("Approximation 3", approx3)


# Question 5: Generate approximation to Laplacian using the difference of Gaussian planes from (2)
pyr1 = cv2.pyrUp(scale1)
dog1 = flower - pyr1

pyr2 = cv2.pyrUp(scale2)
dog2 = scale1 - pyr2

pyr3 = cv2.pyrUp(scale3)
dog3 = scale2 - pyr3

cv2.imshow("Problem 5 DOG 1", dog1)
cv2.imshow("Problem 5 DOG 2", dog2)
cv2.imshow("Problem 5 DOG 3", dog3)

while(1):
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif k == ord('q'):
        break      
if flower is None:
    print("File does not exit\n")