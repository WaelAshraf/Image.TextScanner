# USAGE
# python bank_check_ocr.py --image example_check.png --reference micr_e13b_reference.png

# import the necessary packages
from commonfunctions import *
from skimage.segmentation import clear_border
from imutils import contours
import imutils
import numpy as np
import argparse
import cv2

alphabetics_dict = {}
Alpha_numeric_list=['A','B','C','D','E','F','G','H','I','G','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
                    '0','1','2','3','4','5','6','7','8','9',
                    'a','b','c','d','e','f','g','h','i','g','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

# import image
def read_image(image):
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    return thresh

#read letters images and save their histograms
def readletters():
    for i in range(26):
        roi = read_image( 'alphanumeric/capital/' + Alpha_numeric_list[i] + '.png')
        roi=cv2.resize(roi,(36, 36))
        #show_images([roi],["roi"])
        alphabetics_dict.update( { Alpha_numeric_list[i] : roi } )
        
    for i in range(26,62):
        roi = read_image( "alphanumeric/" + Alpha_numeric_list[i] + ".png")
        roi=cv2.resize(roi,(36, 36))
        alphabetics_dict.update( { Alpha_numeric_list[i] : roi } )

    return

def extract_digits_and_symbols(image, charCnts, minW=5, minH=15):
	# grab the internal Python iterator for the list of character
	# contours, then  initialize the character ROI and location
	# lists, respectively
	charIter = charCnts.__iter__()
	rois = []
	locs = []

	# keep looping over the character contours until we reach the end
	# of the list
	while True:
		try:
			# grab the next character contour from the list, compute
			# its bounding box, and initialize the ROI
			c = next(charIter)
			(cX, cY, cW, cH) = cv2.boundingRect(c)
			roi = None

			# check to see if the width and height are sufficiently
			# large, indicating that we have found a digit
			# if cW >= minW and cH >= minH:
				# extract the ROI
			roi = image[cY:cY + cH, cX:cX + cW]
			show_images([roi])
			roi=cv2.resize(roi,(50, 50))
			rois.append(roi)
			show_images([roi])
			locs.append((cX, cY, cX + cW, cY + cH))

			# otherwise, we are examining one of the special symbols
			# else:
			# 	# MICR symbols include three separate parts, so we
			# 	# need to grab the next two parts from our iterator,
			# 	# followed by initializing the bounding box
			# 	# coordinates for the symbol
			# 	parts = [c, next(charIter), next(charIter)]
			# 	(sXA, sYA, sXB, sYB) = (np.inf, np.inf, -np.inf,
			# 		-np.inf)

			# 	# loop over the parts
			# 	for p in parts:
			# 		# compute the bounding box for the part, then
			# 		# update our bookkeeping variables
			# 		(pX, pY, pW, pH) = cv2.boundingRect(p)
			# 		sXA = min(sXA, pX)
			# 		sYA = min(sYA, pY)
			# 		sXB = max(sXB, pX + pW)
			# 		sYB = max(sYB, pY + pH)

			# 	# extract the ROI
			# 	roi = image[sYA:sYB, sXA:sXB]
			# 	rois.append(roi)
			# 	locs.append((sXA, sYA, sXB, sYB))

		# we have reached the end of the iterator; gracefully break
		# from the loop
		except StopIteration:
			break

	# return a tuple of the ROIs and locations
	return (rois, locs)
readletters()
# for key in alphabetics_dict:
#show_images([alphabetics_dict['n']])

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False,
	help="path to input image")
args = vars(ap.parse_args())

# initialize a rectangular kernel (wider than it is tall) along with
# an empty list to store the output of the check OCR
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7))
output = []

# load the input image, grab its dimensions, and apply array slicing
# to keep only the bottom 20% of the image (that's where the account
# information is)
image = cv2.imread("TNRoman.png")
delta=0
#show_images([image])
# convert the bottom image to grayscale, then apply a blackhat
# morphological operator to find dark regions against a light
# background (i.e., the routing and account numbers)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

# compute the Scharr gradient of the blackhat image, then scale
# the rest back into the range [0, 255]
gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0,
	ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype("uint8")

# apply a closing operation using the rectangular kernel to help
# cloes gaps in between rounting and account digits, then apply
# Otsu's thresholding method to binarize the image
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(gradX, 0, 255,
	cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# remove any pixels that are touching the borders of the image (this
# simply helps us in the next step when we prune contours)
thresh = clear_border(thresh)
show_images([thresh],["thresh"])

# find contours in the thresholded image, then initialize the
# list of group locations6we`2
groupCnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
groupCnts = groupCnts[0]
groupLocs = []

# loop over the group contours
for (i, c) in enumerate(groupCnts):
	# compute the bounding box of the contour
	(x, y, w, h) = cv2.boundingRect(c)

	# only accept the contour region as a grouping of characters if
	# the ROI is sufficiently large
	#if w > 50 and h > 15:
	groupLocs.append((x, y, w, h))

# sort the digit locations from left-to-right
groupLocs = sorted(groupLocs, key=lambda x:x[0])

# loop over the group locations
for (gX, gY, gW, gH) in groupLocs:
	# initialize the group output of characters
	groupOutput = []

	# extract the group ROI of characters from the grayscale
	# image, then apply thresholding to segment the digits from
	# the background of the credit card
	group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
	#show_images([group])
	group = cv2.threshold(group, 0, 255,
		cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

	#cv2.imshow("Group", group)
	#cv2.waitKey(0)

	# find character contours in the group, then sort them from
	# left to right
	charCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	charCnts = imutils.grab_contours(charCnts)
	charCnts = contours.sort_contours(charCnts,
		method="left-to-right")[0]
	#show_images([group])
	# find the characters and symbols in the group
	(rois, locs) = extract_digits_and_symbols(group, charCnts)

	# loop over the ROIs from the group
	for roi in rois:
		# initialize the list of template matching scores and
		# resize the ROI to a fixed size
		scores = []
		roi = cv2.resize(roi, (36, 36))
		#show_images([roi])
		print("ROIIII")
		# loop over the reference character name and corresponding
		# ROI
		for charName in Alpha_numeric_list:
			# apply correlation-based template matching, take the
			# score, and update the scores list
			if charName=='r':
				#print("Ay 7aga")
				pass
			result = cv2.matchTemplate(roi, alphabetics_dict[charName],
				cv2.TM_CCOEFF)
			(_, score, _, _) = cv2.minMaxLoc(result)
			scores.append(score)

		# the classification for the character ROI will be the
		# reference character name with the *largest* template
		# matching score
		groupOutput.append(Alpha_numeric_list[np.argmax(scores)])

	# add the group output to the overall check OCR output
	output.append("".join(groupOutput))

# display the output check OCR information to the screen
print("Check OCR: {}".format(" ".join(output)))
cv2.imshow("Check OCR", image)
cv2.waitKey(0)