import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
#from PIL import Image


#Implement the number of vertical and horizontal corners
nb_vertical = 9
nb_horizontal = 6


# termination criteria
#criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((nb_horizontal*nb_vertical,3), np.float32)
objp[:,:2] = np.mgrid[0:nb_vertical,0:nb_horizontal].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints_left = [] # 3d point in real world space
imgpoints_left = [] # 2d points in image plane.

images = glob.glob('rs/left-*.png')
assert images
d=0

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Implement findChessboardCorners here
    ret, corners = cv2.findChessboardCorners(gray,(nb_vertical,nb_horizontal), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints_left.append(objp)

        imgpoints_left.append(corners)
        print("undistorting {} left image".format(d))
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints_left, imgpoints_left, gray.shape[::-1], None, None)
        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))   
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imwrite('rs_undistorted/left-undis-{}.png'.format(d), dst)
        d+=1


# Arrays to store object points and image points from all the images.
objpoints_right = [] # 3d point in real world space
imgpoints_right = [] # 2d points in image plane.

images = glob.glob('rs/right-*.png')
assert images
d=0
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Implement findChessboardCorners here
    ret, corners = cv2.findChessboardCorners(gray,(nb_vertical,nb_horizontal), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints_right.append(objp)

        imgpoints_right.append(corners)

        print("undistorting {} right image".format(d))
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints_right, imgpoints_right, gray.shape[::-1], None, None)
        h,  w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))   
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imwrite('rs_undistorted/right-undis-{}.png'.format(d), dst)
        d+=1

