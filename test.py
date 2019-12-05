import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.filters import gaussian

import matplotlib.patches as mpatches
from skimage.measure import label, regionprops
from skimage.util import invert
from scipy.spatial.distance import euclidean

import numpy as np
from skimage.draw import polygon_perimeter

from commonfunctions import *
import cv2
import numpy as np


from skimage.measure import compare_ssim
import imutils
from imutils import contours
from skimage.segmentation import clear_border
alphabetics_dict = {}
Alpha_numeric_list=['A','B','C','D','E','F','G','H','I','G','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
                    '0','1','2','3','4','5','6','7','8','9',
                    'a','b','c','d','e','f','g','h','i','g','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

# import image
def read_image(image):
    image = cv2.imread(image)
    return image

#read letters images and save their histograms
def readletters():
    for i in range(len(Alpha_numeric_list)):
        alphabetics_dict.update( { Alpha_numeric_list[i] : histogram(
            rgbtogray(read_image( "alphanumeric/" + Alpha_numeric_list[i] + ".png"))) } )

    return

def rgbtogray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray
# cv2.imshow('threshold', thresh)

# dilation
def dilation(thresh):
    kernel = np.ones((10, 1), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)
    return img_dilation
# cv2.imshow('dilated', img_dilation)
    
# cv2.imshow('gray', gray)

# binary
def convert_to_binary(gray):
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    return ret,thresh

# find contours and sort them
def find_contours(img_dilation):
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    return sorted_ctrs

def extract_digits_and_symbols(image, charCnts):
    # grab the internal Python iterator for the list of character
    # contours, then  initialize the character ROI and location
    # lists, respectively

    rois = []
    locs = []

    for i, ctr in enumerate(charCnts):
        x, y, w, h = cv2.boundingRect(ctr)


        roi = image[y:y + h, x:x + w]
        
        show_images([roi],['ROI'])
        
        rois.append(roi)
        locs.append((x, y, x+w, y + h))

    # return a tuple of the ROIs and locations
    return (rois, locs)


def ConstructOurDictionary():
    ref = read_image('alphabets.jpg')
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    ref = imutils.resize(ref, width=2000)
    ref = cv2.threshold(ref, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # find contours in the MICR image (i.e,. the outlines of the
    # characters) and sort them from left to right
    refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #refCnts = imutils.grab_contours(refCnts)
    #refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]

    # extract the digits and symbols from the list of contours, then
    # initialize a dictionary to map the character name to the ROI
    refROIs = extract_digits_and_symbols(ref, refCnts)

    for (name,roi) in zip(Alpha_numeric_list, refROIs):
        #print("her")
        roi=cv2.resize(roi,(36, 36))
        alphabetics_dict[name] = roi
    
#     for key in alphabetics_dict:
#         print(key)
#         show_images([alphabetics_dict[key]])
    return

#ConstructOurDictionary()


def ExtractAlphabets():
    # import image
    image = cv2.imread('alphabets.jpg')

    # grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # binary
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)


    # dilation
    kernel = np.ones((5, 5), np.uint8)
    img_dilation = cv2.dilate(thresh, kernel, iterations=1)


    # find contours

    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    
    rois = []
    locs = []
    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)

        # Getting ROI
        roi = image[y:y + h, x:x + w]
        if w > 15 and h > 15:
            cv2.imwrite('images//{}.png'.format(i), roi)
            #show_images([roi])
            rois.append(roi)
            locs.append((x, y, x+w, y + h))

ExtractAlphabets()