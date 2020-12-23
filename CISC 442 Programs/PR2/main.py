# Sohan Gadiraju
# Programming Assignment 2

import cv2
import numpy as np
from matplotlib import pyplot as plt


print("Welcome to a Basic Stereo Analysis System")
print("\nChoose your desired method\nPress 1 for Region-Based Analysis\nPress 2 for Feature-Based Analysis")
q1 = int(input("Enter Your Selection: "))

if q1 == 1:
    print("\nRegion-Based Analysis was selected!")
    exec(open("regionBased.py").read())
elif q1 == 2:
    print("\nFeauture-Based Analysis was selected!")
    exec(open("featureBased.py").read())
else:
    print("\nInvalid Selection. Please try again!")
