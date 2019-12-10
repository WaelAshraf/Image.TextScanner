import cv2
import numpy as np
from commonfunctions import *
import matplotlib.pyplot as plt
from skimage import measure
#from skimage.measure import structural_similarity as ssim
# def minimum_bounding_rectangle(hull_points):

#     # calculate edge angles
#     edges = np.zeros((len(hull_points)-1, 2))
#     edges = hull_points[1:] - hull_points[:-1]

#     angles = np.zeros((len(edges)))
#     angles = np.arctan2(edges[:, 1], edges[:, 0])

#     angles = np.abs(np.mod(angles, pi2))
#     angles = np.unique(angles)

#     # find rotation matrices
#     # XXX both work
#     rotations = np.vstack([
#         np.cos(angles),
#         np.cos(angles-pi2),
#         np.cos(angles+pi2),
#         np.cos(angles)]).T
# #     rotations = np.vstack([
# #         np.cos(angles),
# #         -np.sin(angles),
# #         np.sin(angles),
# #         np.cos(angles)]).T
#     rotations = rotations.reshape((-1, 2, 2))

#     # apply rotations to the hull
#     rot_points = np.dot(rotations, hull_points.T)

#     # find the bounding points
#     min_x = np.nanmin(rot_points[:, 0], axis=1)
#     max_x = np.nanmax(rot_points[:, 0], axis=1)
#     min_y = np.nanmin(rot_points[:, 1], axis=1)
#     max_y = np.nanmax(rot_points[:, 1], axis=1)

#     # find the box with the best area
#     areas = (max_x - min_x) * (max_y - min_y)
#     best_idx = np.argmin(areas)

#     # return the best box
#     x1 = max_x[best_idx]
#     x2 = min_x[best_idx]
#     y1 = max_y[best_idx]
#     y2 = min_y[best_idx]
#     r = rotations[best_idx]

#     rval = np.zeros((4, 2))
#     rval[0] = np.dot([x1, y2], r)
#     rval[1] = np.dot([x2, y2], r)
#     rval[2] = np.dot([x2, y1], r)
#     rval[3] = np.dot([x1, y1], r)

#     return rval
# img = cv2.imread('TNRoman.png')
# img2 = img.copy()
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, threshed_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
# contours, hier = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

# #--- Black image to be used to draw individual convex hull ---
# black = np.zeros_like(img)
# #cv2.imshow("black.jpg", black)

# contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0]) #added by OP : this sorts contours left to right, so images come in order

# for cnt in contours:
#     hull = cv2.convexHull(cnt)
#     x,y,w,h = cv2.boundingRect(cnt)
#     cv2.rectangle(threshed_img,(x,y),(x+w,y+h),(0,255,0),2)
#     cv2.imshow("img.jpg", threshed_img)
#     img3 = img.copy()
#     black2 = black.copy()

#     #--- Here is where I am filling the contour after finding the convex hull ---
#     cv2.drawContours(black2, [hull], -1, (255, 255, 255), -1)
#     g2 = cv2.cvtColor(black2, cv2.COLOR_BGR2GRAY)
#     r, t2 = cv2.threshold(g2, 127, 255, cv2.THRESH_BINARY)
#     cv2.imshow("t2.jpg", t2)

#     masked = cv2.bitwise_and(img2, img2, mask = t2)    
#     cv2.imshow("masked.jpg", masked)

#     print(len(hull))
#     cv2.waitKey(0)

# cv2.destroyAllWindows()

# mser = cv2.MSER_create()

# #Resize the image so that MSER can work better
# img = cv2.resize(img, (img.shape[1]*2, img.shape[0]*2))

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# vis = img.copy()

# regions = mser.detectRegions(gray)

# hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
# bbox=minimum_bounding_rectangle(hulls)
# plt.fill(bbox[:,0], bbox[:,1], alpha=0.2)
# plt.axis('equal')
# plt.show()
# cv2.polylines(vis, hulls, 1, (0,255,0)) 

# cv2.namedWindow('img', 0)
# cv2.imshow('img', vis)
# while(cv2.waitKey()!=ord('q')):
#     continue
# cv2.destroyAllWindows()

def SegmentImg2Lines(img):
    roi_list=[]
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, threshed_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((15,60), np.uint8)
    img_dilation = cv2.dilate(threshed_img, kernel, iterations=1)
    #show_images([img_dilation],["img_dilation ROI"])
    # find contours
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        # Getting ROI
        roi = image[y:y + h, x:x + w]
        #show_images([roi],["Line ROI"])
        roi_list.append(roi)
    return roi_list


def Segmentline2word(line):
    roi_list=[]
    locs=[]
    gray = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)
    ret, threshed_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5,20), np.uint8)
    img_dilation = cv2.dilate(threshed_img, kernel, iterations=1)
    #show_images([img_dilation],["img_dilation ROI"])
    # find contours
    ctrs, hier = cv2.findContours(img_dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # sort contours
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    for i, ctr in enumerate(sorted_ctrs):
        # Get bounding box
        x, y, w, h = cv2.boundingRect(ctr)
        # Getting ROI
        roi = line[y:y + h, x:x + w]
        locs.append((x, y, x + w,y + h))
        #show_images([roi],["Line ROI"])
        roi_list.append(roi)
    return roi_list,locs    

def SegmentLine2Char(word):
    roi_list=[]
    locs=[]
    #img = cv2.imread(line)
    gray = cv2.cvtColor(word, cv2.COLOR_BGR2GRAY)
    ret, threshed_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, hier = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0]) 
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        roi = threshed_img[y:y + h, x:x + w]
        roi=cv2.resize(roi,(160, 160))
        #show_images([roi])
        roi_list.append(roi)
        locs.append((x, y, x + w,y + h))

    return roi_list,locs

def ExtractText(roi_list,roi_locs,loc_list_bold,alphabetics_dict,Alpha_numeric_list):
    output=[]
    for roi,loc in zip(roi_list,roi_locs):
        # initialize the list of template matching scores and
        # resize the ROI to a fixed size
        scores = []
        #roi = cv2.resize(roi, (36, 36))
        # loop over the reference character name and corresponding
        # ROI
        #show_images([roi],["roi"])
        for charName in Alpha_numeric_list:
            # apply correlation-based template matching, take the
            # score, and update the scores list
            result = cv2.matchTemplate(roi, alphabetics_dict[charName], cv2.TM_CCOEFF_NORMED)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)
    # the classification for the character ROI will be the
    # reference character name with the *largest* template
    # matching score
        if loc in loc_list_bold:
            print("Bold")
        output.append(Alpha_numeric_list[np.argmax(scores)])
    return output



# import image
def read_image(image):
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, threshed_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    return threshed_img

#read letters images and save their histograms
def construct_alphabetics_dict(path,path_cap):
    alphabetics_dict = {}
    Alpha_numeric_list=['A','B','C','D','E','F','G','H','I','G','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
                    '0','1','2','3','4','5','6','7','8','9',
                    'a','b','c','d','e','f','g','h','i','g','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    
    Alphabets_list=['A','B','C','D','E','F','G','H','I','G','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
                    'a','b','c','d','e','f','g','h','i','g','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    for i in range(26):
        input_char = read_image( path_cap + Alphabets_list[i] + '.png')
        ctrs, hier = cv2.findContours(input_char, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # sort contours
        sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
        for j, ctr in enumerate(sorted_ctrs):

            # Get bounding box
            x, y, w, h = cv2.boundingRect(ctr)
            # Getting ROI
            roi = input_char[y:y + h, x:x + w]
            roi=cv2.resize(roi,(160, 160))
        #show_images([roi],["roi"])
        alphabetics_dict.update( { Alphabets_list[i] : roi } )
        
    for i in range(26,52):
        input_char = read_image( path+ Alphabets_list[i] + ".png")
        ctrs, hier = cv2.findContours(input_char, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # sort contours
        sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
        for j, ctr in enumerate(sorted_ctrs):
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(ctr)
            # Getting ROI
            roi = input_char[y:y + h, x:x + w]
            roi=cv2.resize(roi,(160, 160))
        #show_images([roi],["roi"])
        alphabetics_dict.update( { Alphabets_list[i] : roi } )

    return alphabetics_dict,Alphabets_list





def SegmentItalic(image):
    roi_list=[]
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, threshed_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, hier = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0]) 
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        rect = cv2.minAreaRect(hull)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img,[box],0,(0,0,255),2)
    show_images([img])
#SegmentItalic('TNRoman.png')


def DetectBold(image):
    locs=[]
    roi_list_bold=[]
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, threshed_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    kernel=np.ones((11,11),np.uint8)
    opening = cv2.morphologyEx(threshed_img, cv2.MORPH_OPEN, kernel)
    neg=255-opening
    #show_images([neg])
    contours, hier = cv2.findContours(opening, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0]) 
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        roi = opening[y:y + h, x:x + w]
        #show_images([roi])
        roi_list_bold.append(roi)
        locs.append((x, y, x + w,y + h))
    return roi_list_bold,locs
            
def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err

def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	#s = ssim(imageA, imageB)
    
	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("MSE: %.2f" % (m))

	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA, cmap = plt.cm.gray)
	plt.axis("off")

	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB, cmap = plt.cm.gray)
	plt.axis("off")

	# show the images
	plt.show()    
#DetectBold('google-font-noto-sans.png')    

#alphabetics_dict_Arial,Alphabets_list_Arial=construct_alphabetics_dict('Arial/','Arial/Cap/')
#alphabetics_dict_calibri,Alphabets_list_calibri=construct_alphabetics_dict('calibri/','calibri/Cap/')
alphabetics_dict_times,Alphabets_list_times=construct_alphabetics_dict('times/','times/Cap/')
lines=SegmentImg2Lines('google-font-noto-sans.png')
#show_images([lines[2]])
worddes,_=Segmentline2word(lines[2])
roi_list,roi_locs=SegmentLine2Char(words[0])
show_images([words[0]])
roi_list_bold,locs_bold=DetectBold('google-font-noto-sans.png')
output=ExtractText(roi_list,roi_locs,locs_bold,alphabetics_dict_times,Alphabets_list_times)
print(output)
#output=[]




#show_images([alphabetics_dict_times['C']])

for charName in Alphabets_list_times:

            # apply correlation-based template matching, take the
            # score, and update the scores list
    m=mse(roi_list[0],alphabetics_dict_times[charName])
    # the classification for the character ROI will be the
    # reference character name with the *largest* template
    # matching score
    output.append(m)
#sortederr=min(output)
#print(sortederr)
#m=mse(roi_list[0],alphabetics_dict_times['E'])
#compare_images(roi_list[0],alphabetics_dict_times['E'],'EF')
#compare_images(roi_list[0],alphabetics_dict_times['F'],'EF')
# result = cv2.matchTemplate(roi_list[0], alphabetics_dict_times['F'], cv2.TM_CCOEFF_NORMED)
# (_, score, _, _) = cv2.minMaxLoc(result)
# print(result)

# result = cv2.matchTemplate(roi_list[0], alphabetics_dict_times['E'], cv2.TM_CCOEFF_NORMED)
# (_, score, _, _) = cv2.minMaxLoc(result)
# print(result)
#compare_images(alphabetics_dict_times['E'],alphabetics_dict_times['F'],'EF')



# hsv_roi = cv2.cvtColor(roi_list[0], cv2.COLOR_BGR2HSV)
# hist_roi = cv2.calcHist([hsv_roi], channels, None, histSize, ranges, accumulate=False)
# cv2.normalize(hist_roi, hist_roi, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

# hsv_F = cv2.cvtColor(alphabetics_dict_times['F'], cv2.COLOR_BGR2HSV)
# hist_F = cv2.calcHist([hsv_F], channels, None, histSize, ranges, accumulate=False)
# cv2.normalize(hist_F, hist_F, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)


# hsv_E = cv2.cvtColor(alphabetics_dict_times['E'], cv2.COLOR_BGR2HSV)
# hist_E = cv2.calcHist([hsv_E], channels, None, histSize, ranges, accumulate=False)
# cv2.normalize(hist_E, hist_E, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

# roi_F = cv2.compareHist(hist_roi, hist_F, cv2.COMP_CORREL)
# print(roi_f)
# roi_F = cv2.compareHist(hist_roi, hist_F, cv2.COMP_CHISQR)
# print(roi_f)
# roi_F = cv2.compareHist(hist_roi, hist_F, cv2.COMP_INTERSECT)
# print(roi_f)
# roi_F = cv2.compareHist(hist_roi, hist_F, cv2.COMP_BHATTACHARYYA)
# print(roi_f)
# print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
# roi_E = cv2.compareHist(hist_roi, hist_E, cv2.COMP_CORREL)
# print(roi_E)
# roi_E = cv2.compareHist(hist_roi, hist_E, cv2.COMP_CHISQR)
# print(roi_E)
# roi_E = cv2.compareHist(hist_roi, hist_E, cv2.COMP_INTERSECT)
# print(roi_E)
# roi_E = cv2.compareHist(hist_roi, hist_E, cv2.COMP_BHATTACHARYYA)
# print(roi_E)