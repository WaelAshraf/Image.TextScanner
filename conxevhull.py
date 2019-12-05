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
img = cv2.imread('TNRoman.png')
img2 = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, threshed_img = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
contours, hier = cv2.findContours(threshed_img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

#--- Black image to be used to draw individual convex hull ---
black = np.zeros_like(img)
#cv2.imshow("black.jpg", black)

contours = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0]) #added by OP : this sorts contours left to right, so images come in order

for cnt in contours:
    hull = cv2.convexHull(cnt)
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("img.jpg", img)
    img3 = img.copy()
    black2 = black.copy()

    #--- Here is where I am filling the contour after finding the convex hull ---
    cv2.drawContours(black2, [hull], -1, (255, 255, 255), -1)
    g2 = cv2.cvtColor(black2, cv2.COLOR_BGR2GRAY)
    r, t2 = cv2.threshold(g2, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("t2.jpg", t2)

    masked = cv2.bitwise_and(img2, img2, mask = t2)    
    cv2.imshow("masked.jpg", masked)

    print(len(hull))
    cv2.waitKey(0)

cv2.destroyAllWindows()

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


