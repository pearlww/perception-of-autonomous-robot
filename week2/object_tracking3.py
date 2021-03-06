import cv2
import numpy as np
import imutils

#cap = cv2.VideoCapture('Robots.mp4')
cap = cv2.VideoCapture('Challenge.mp4')


ret, frame = cap.read()
if ret == True:
    b,g,r = cv2.split(frame)
    img = cv2.merge([r,g,b])
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
else:
    print("error to read the frame")

# fail, the firstFrame need a pure background
firstFrame = gray

# loop over the frames of the video
while True:

    ret, frame = cap.read()
    if ret:
        b,g,r = cv2.split(frame)
        img = cv2.merge([r,g,b])
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        print("error to read the frame")
        break

    gray=cv2.GaussianBlur(gray,(21,21),0)
    
    # compute the absolute difference between the current frame and
    # first frame
    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 100, 255, cv2.THRESH_BINARY)[1]
    # dilate the thresholded image to fill in holes, then find contours
    # on thresholded image
    thresh = cv2.erode(thresh, None, iterations=5)
    thresh = cv2.dilate(thresh, None, iterations=40)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # loop over the contours
    for c in cnts:
        # if the contour is too small, ignore it
        if cv2.contourArea(c) > 20000:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # compute the bounding box for the contour, draw it on the frame,
        # and update the text
        

    cv2.imshow('thresh', thresh)
    cv2.imshow('frameDelta', frameDelta)
    cv2.imshow('image', img)
 
    if cv2.waitKey(20) == ord('q'): 
        cap.release()
        cv2.destroyAllWindows()
        break