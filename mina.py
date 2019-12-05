import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import skimage as io

img = cv.imread('tnr.jpg')
gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
thresh, thresh_img = cv.threshold(gray_img,127,255,cv.THRESH_BINARY_INV)
thresh_img = np.asarray(thresh_img)
thresh_img = thresh_img / 255


listOfGaps = []
listOfGapsLength = []
Words = [] # list of segmented words

isWord = 0
projection = np.sum(thresh_img,axis=0).tolist()
projectionArray = np.asarray(projection)
for i in range(len(projectionArray)):
    if projectionArray[i] != 0.0 and isWord == 0:
        listOfGaps.append(i)
        isWord = 1
    elif projectionArray[i] == 0.0 and isWord == 1:
        listOfGaps.append(i)
        isWord = 0
        
print(len(listOfGaps))
word = []
listOfGapsIndex = 0
imgArray = np.asarray(img)
print(imgArray.shape)
for i in range(imgArray.shape[1]):
    if i == listOfGaps[listOfGapsIndex]:
        imgTemp = imgArray[:,listOfGaps[listOfGapsIndex]:listOfGaps[listOfGapsIndex + 1]]
        word.append(imgTemp)
        cv.imshow('label',imgTemp)
        cv.waitKey(0)
        cv.destroyAllWindows()
        listOfGapsIndex += 2
        Words.append(word)
        if len(listOfGaps) <= listOfGapsIndex:
            break
        
print(Words)