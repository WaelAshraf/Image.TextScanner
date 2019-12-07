import cv2
import numpy as np
from commonfunctions import *
import matplotlib.pyplot as plt
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

def SegmentLine2Char(line):
    roi_list=[]
    #img = cv2.imread(image)
    gray = cv2.cvtColor(line, cv2.COLOR_BGR2GRAY)
    ret, threshed_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, hier = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0]) 
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        roi = threshed_img[y:y + h, x:x + w]
        roi_list.append(roi)
    return roi_list

def ExtractText(roi_list,alphabetics_dict,Alpha_numeric_list):
    output=[]
    for roi in roi_list:
        # initialize the list of template matching scores and
        # resize the ROI to a fixed size
        scores = []
        roi = cv2.resize(roi, (36, 36))
        # loop over the reference character name and corresponding
        # ROI
        show_images([roi],["roi"])
        show_images([alphabetics_dict['u']],["u"])
        for charName in Alpha_numeric_list:
            # apply correlation-based template matching, take the
            # score, and update the scores list

            result = cv2.matchTemplate(roi, alphabetics_dict[charName], cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)
    # the classification for the character ROI will be the
    # reference character name with the *largest* template
    # matching score
        output.append(Alpha_numeric_list[np.argmax(scores)])
    return output



# import image
def read_image(image):
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    ret, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    return thresh

#read letters images and save their histograms
def construct_alphabetics_dict():
    alphabetics_dict = {}
    Alpha_numeric_list=['A','B','C','D','E','F','G','H','I','G','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
                    '0','1','2','3','4','5','6','7','8','9',
                    'a','b','c','d','e','f','g','h','i','g','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    for i in range(26):
        roi = read_image( 'alphanumeric/capital/' + Alpha_numeric_list[i] + '.png')
        roi=cv2.resize(roi,(36, 36))
        #show_images([roi],["roi"])
        alphabetics_dict.update( { Alpha_numeric_list[i] : roi } )
        
    for i in range(26,62):
        roi = read_image( "alphanumeric/" + Alpha_numeric_list[i] + ".png")
        roi=cv2.resize(roi,(36, 36))
        alphabetics_dict.update( { Alpha_numeric_list[i] : roi } )

    return alphabetics_dict,Alpha_numeric_list


alphabetics_dict,Alpha_numeric_list=construct_alphabetics_dict()
lines=SegmentImg2Lines('TNRoman.png')
roi_list=SegmentLine2Char(lines[0])
output=ExtractText(roi_list,alphabetics_dict,Alpha_numeric_list)
print (output)