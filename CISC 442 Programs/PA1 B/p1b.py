import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("img.png")
img2 = cv2.imread("img.png")
pCopy = img.copy()
aCopy = img.copy()
pComp = img.copy()
aComp = img.copy()
count1 = 4
count2 = 4
countA1 = 3
countA2 = 3
countFP = 8
countFA = 6
countOVP = 0
countOVA = 0

def create_perspective():
    global mouseX,mouseY, a, b, count1
    rows, cols, ch = img.shape 
    a = []
    b = []
    cv2.imshow('Perspective Transformation',img)
    print("Select four initial points to use to create a perspective transformation")
    cv2.setMouseCallback('Perspective Transformation',create_perspectiveClick)
    print("Select four more points to finish the perspective transformation")
    cv2.setMouseCallback('Perspective Transformation',create_perspectiveClick)

def create_affine():
    global mouseX,mouseY, c, d, countA1
    rows, cols, ch = img2.shape 
    c = []
    d = []
    cv2.imshow('Affine Transformation',img2)

    print("Select three initial points to use to create affine transformation")
    cv2.setMouseCallback('Affine Transformation',create_affineClick)
    print("Select three more points to finish the affine transformation")
    cv2.setMouseCallback('Affine Transformation',create_affineClick)    

def calc_perspective():
    global mouseX,mouseY, fp1, fp2, countfp, image1, image2, numpy_horizontal, pCopy
    rows, cols, ch = img2.shape 
    fp1 = []
    fp2 = []
    image1 = cv2.resize(img, (0, 0), None, .5, .5)
    image2 = cv2.resize(pCopy, (0, 0), None, .5, .5)

    numpy_horizontal = np.hstack((image1, image2))
    cv2.imshow('Choose 4 Points on each',numpy_horizontal)

    print("Select four points on left pic and four on right pic:\n")
    cv2.setMouseCallback('Choose 4 Points on each',perspectiveComp_Click)  

def calc_overConsP():
    global mouseX,mouseY, ovp1, ovp2, countfp, image1, image2, numpy_horizontal, pCopy
    rows, cols, ch = img2.shape 
    ovp1 = []
    ovp2 = []
    image1 = cv2.resize(img, (0, 0), None, .5, .5)
    image2 = cv2.resize(pCopy, (0, 0), None, .5, .5)

    numpy_horizontal = np.hstack((image1, image2))
    numpy_horizontal = np.hstack((image1, image2))
    cv2.imshow('Choose Unlimited Points (P to stop)',numpy_horizontal)

    print("Select an unlimted amount of points:\n")
    cv2.setMouseCallback('Choose Unlimited Points (P to stop)',ovp_Click) 

def calc_affine():
    global mouseX,mouseY, fp1, fp2, countfp, image1, image2, numpy_horizontal, pCopy
    rows, cols, ch = img2.shape 
    fp1 = []
    fp2 = []
    image1 = cv2.resize(img, (0, 0), None, .5, .5)
    image2 = cv2.resize(aCopy, (0, 0), None, .5, .5)

    numpy_horizontal = np.hstack((image1, image2))
    cv2.imshow('Choose 3 Points on each',numpy_horizontal)

    print("Select four points on left pic and four on right pic:\n")
    cv2.setMouseCallback('Choose 3 Points on each',affineComp_Click)  

def calc_overConsA():
    global mouseX,mouseY, ovp1, ovp2, countfp, image1, image2, numpy_horizontal, pCopy
    rows, cols, ch = img2.shape 
    ovp1 = []
    ovp2 = []
    image1 = cv2.resize(img, (0, 0), None, .5, .5)
    image2 = cv2.resize(pCopy, (0, 0), None, .5, .5)

    numpy_horizontal = np.hstack((image1, image2))
    numpy_horizontal = np.hstack((image1, image2))
    cv2.imshow('Choose Unlimited Points (P to stop)',numpy_horizontal)

    print("Select an unlimted amount of points:\n")
    cv2.setMouseCallback('Choose Unlimited Points (P to stop)',ova_Click) 


def create_perspectiveClick(event,x,y,flags,param):
    global mouseX,mouseY, a, b, count1, count2, pCopy
    rows, cols, ch = img.shape 
    cv2.imshow('Perspective Transformation',img)
    if (event == cv2.EVENT_LBUTTONDBLCLK) and (count1 > 0):
        cv2.circle(img,(x,y),10,(255,0,0),-1)
        mouseX,mouseY = x,y
        coords = tuple([x,y])
        a.append(coords)
        count1 -=1 
        print(str(count1) + "Clicks Remaining")                 
    elif (event == cv2.EVENT_LBUTTONDBLCLK) and (count1 == 0) and (count2 > 0):
        cv2.circle(img,(x,y),10,(255,0,0),-1)
        mouseX,mouseY = x,y
        coords = tuple([x,y])
        b.append(coords)
        count2 -= 1
        print(str(count2) + "Clicks Remaining")
        pts1 = np.float32(a)
        pts2 = np.float32(b)
    elif (count1 == 0) and (count2 == 0):
        pts1 = np.float32(a)
        pts2 = np.float32(b)
        retval = cv2.getPerspectiveTransform(pts1,pts2)
        print("Perspective Parameters:\n")
        print(str(retval)+"\n")
        dst= cv2.warpPerspective(img,retval,(cols,rows))
        pCopy = dst
        cv2.imshow('Perspective Version', pCopy)
        create_affine()   

def create_affineClick(event,x,y,flags,param):
    global mouseX,mouseY, c, d, countA1, countA2, aCopy
    rows, cols, ch = img2.shape 
    cv2.imshow('Affine Transformation',img2)
    if (event == cv2.EVENT_LBUTTONDBLCLK) and (countA1 > 0):
        cv2.circle(img2,(x,y),10,(255,0,0),-1)
        mouseX,mouseY = x,y
        coords = tuple([x,y])
        c.append(coords)
        countA1 -=1  
        print(str(countA1) + "Clicks Remaining")                
    elif (event == cv2.EVENT_LBUTTONDBLCLK) and (countA1 == 0) and (countA2 > 0):
        cv2.circle(img2,(x,y),10,(255,0,0),-1)
        mouseX,mouseY = x,y
        coords = tuple([x,y])
        d.append(coords)
        print(str(countA2) + "Clicks Remaining")
        countA2 -= 1
        pts1 = np.float32(c)
        pts2 = np.float32(d)
    elif (countA1 == 0) and (countA2 == 0):
        pts1 = np.float32(c)
        pts2 = np.float32(d)
        retval = cv2.getAffineTransform(pts1,pts2)
        print("Affine Parameters:\n")
        print(str(retval)+"\n")
        dst= cv2.warpAffine(img2,retval,(cols,rows))
        aCopy = dst
        cv2.imshow('Affine Version', aCopy)
        calc_perspective()

def perspectiveComp_Click(event,x,y,flags,param):
    global mouseX,mouseY, fp1, fp2, countFP, fval
    rows, cols, ch = img.shape 
    cv2.imshow('Choose 4 Points on each',numpy_horizontal)
    if (event == cv2.EVENT_LBUTTONDBLCLK) and (countFP > 4) :
        cv2.circle(image1,(x,y),8,(255,0,0),-1)
        mouseX,mouseY = x,y
        coords = tuple([x,y])
        fp1.append(coords)
        countFP -= 1 
        print(str(countFP - 4) + "Clicks Remaining on Image 1")                 
    elif (event == cv2.EVENT_LBUTTONDBLCLK) and (countFP <= 4) and (countFP > 0):
        cv2.circle(image2,(x,y),10,(255,255,0),-1)
        mouseX,mouseY = x,y
        coords = tuple([x,y])
        fp2.append(coords)
        countFP -= 1 
        print(str(countFP) + "Clicks Remaining on Image 2")
        ptsfp1 = np.float32(fp1)
        ptsfp2 = np.float32(fp2)
    elif (countFP == 0) :
        ptsfp1 = np.float32(fp1)
        ptsfp2 = np.float32(fp2)
        fval = cv2.getPerspectiveTransform(ptsfp1,ptsfp2)
        print("Perspective Parameters:\n")
        print(str(fval)+"\n")
        calc_overConsP()

def ovp_Click(event,x,y,flags,param):
    global mouseX,mouseY, ovp1, ovp2, countOVP
    rows, cols, ch = img.shape 
    print("Clicks so far: ", countOVP, "\nPress p to stop and calculate\n")
    
    if (event == cv2.EVENT_LBUTTONDBLCLK) :
        cv2.circle(image1,(x,y),8,(255,0,0),-1)
        mouseX,mouseY = x,y
        coords = tuple([x,y])
        ovp1.append(coords)
        countOVP += 1 
        print(str(countOVP) + "Clicks\nPress p to stop and calculate\n")                 
    elif (event == cv2.EVENT_LBUTTONDBLCLK) :
        cv2.circle(image2,(x,y),10,(255,255,0),-1)
        mouseX,mouseY = x,y
        coords = tuple([x,y])
        ovp2.append(coords)
        countOVP += 1 
        print(str(countOVP) + "Clicks\nPress p to stop and calculate\n")
        ptsOvp1 = np.float32(ovp1)
        ptsOvp2 = np.float32(ovp2)
    elif (cv2.EVENT_RBUTTONDOWN) :
        ptsOvp1 = np.float32(ovp1)
        ptsOvp2 = np.float32(ovp2)
        IA = np.linalg.pinv(ptsOvp1)
        retval = np.dot(IA, ptsOvp2)
        print("Perspective Parameters (Overcomp):\n")
        print(str(retval)+"\n")
        countFA

def affineComp_Click(event,x,y,flags,param):
    global mouseX,mouseY, ova1, ova2, countFA, rval
    rows, cols, ch = img.shape 
    cv2.imshow('Choose 3 Points on each',numpy_horizontal)
    if (event == cv2.EVENT_LBUTTONDBLCLK) and (countFA > 3) :
        cv2.circle(image1,(x,y),8,(255,0,0),-1)
        mouseX,mouseY = x,y
        coords = tuple([x,y])
        fp1.append(coords)
        countFA -= 1 
        print(str(countFP - 3) + "Clicks Remaining on Image 1")                 
    elif (event == cv2.EVENT_LBUTTONDBLCLK) and (countFA <= 3) and (countFA > 0):
        cv2.circle(image2,(x,y),10,(255,255,0),-1)
        mouseX,mouseY = x,y
        coords = tuple([x,y])
        fp2.append(coords)
        countFA -= 1 
        print(str(countFP) + "Clicks Remaining on Image 2")
        ptsfp1 = np.float32(fp1)
        ptsfp2 = np.float32(fp2)
    elif (countFA == 0) :
        ptsfp1 = np.float32(fp1)
        ptsfp2 = np.float32(fp2)
        rval = cv2.getAffineTransform(ptsfp1,ptsfp2)
        print("Affine Parameters:\n")
        print(str(fval)+"\n")
        calc_overConsA()

def ova_Click(event,x,y,flags,param):
    global mouseX,mouseY, ova1, ova2, countOVA
    rows, cols, ch = img.shape 
    print("Clicks so far: ", countOVA, "\nPress p to stop and calculate\n")
    k = cv2.waitKey(20) & 0xFF
    if (event == cv2.EVENT_LBUTTONDBLCLK) :
        cv2.circle(image1,(x,y),8,(255,0,0),-1)
        mouseX,mouseY = x,y
        coords = tuple([x,y])
        ova1.append(coords)
        countOVA += 1 
        print(str(countOVA) + "Clicks\nPress p to stop and calculate\n")                 
    elif (event == cv2.EVENT_LBUTTONDBLCLK) :
        cv2.circle(image2,(x,y),10,(255,255,0),-1)
        mouseX,mouseY = x,y
        coords = tuple([x,y])
        ova2.append(coords)
        countOVA += 1 
        print(str(countOVA) + "Clicks\nPress p to stop and calculate\n")
        ptsOvp1 = np.float32(ova1)
        ptsOvp2 = np.float32(ova2)
    elif (k == ord('p')) :
        ptsOvp1 = np.float32(ova1)
        ptsOvp2 = np.float32(ova2)
        IA = np.linalg.pinv(ptsOvp1)
        retval = np.dot(IA, ptsOvp2)
        print("Affine Parameters (Overcomp):\n")
        print(str(retval)+"\n")
        print("Least-Squares Error:\n")
        print(np.sum(np.subtract(hom,configV(V)),dtype=np.float))
    
def getHomography(source, dest):
    n = 0
    arr = []
    if len(source) == len(dest):
        for source0,dest0 in zip(source, dest):
            for source1, dest1 in zip(source0,dest0):
                arr.append([source1[0],source1[1],1,0,0,0,-source1[0]*dest1[0], -source1[1]*dest1[0], -dest1[0]])
                arr.append([0,0,0,source1[0],source1[1],1,-source1[0]*dest1[1], -source1[1]*dest1[1], -dest1[1]])
    arr = np.array(arr)
    U, S, V = np.linalg.svd(arr.T @ arr)
    return U,S,V

def configV(V):
    V2 = V[-1]
    V2 /= V2[-1]
    V2 = V2.reshape(3,3)
    return V2

def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),10,(255,0,0),-1)
        mouseX,mouseY = x,y


if (count1 == 4):
    create_perspective()

while(1):
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif k == ord('a'):
        print (mouseX,mouseY);
    elif k == ord('q'):
        break      
if img is None:
    print("File does not exit\n")

cv2.destroyAllWindows()        