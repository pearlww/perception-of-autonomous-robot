import cv2
import numpy as np

# import os

# os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# img=cv2.imread('Lego.jpg')
# cv2.imshow('image', img )
# cv2.waitKey(200)

cap = cv2.VideoCapture('Robots.mp4')

ret, frame1 = cap.read()
if ret==True:
    b,g,r = cv2.split(frame1)
    img1 = cv2.merge([r,g,b])
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
else:
    print("error to read the frame")


feat1_hist=[]
feat2_hist=[]
while True:

    ret, frame2 = cap.read()
    if ret == True:
        b,g,r = cv2.split(frame2)
        img2 = cv2.merge([r,g,b])
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        print("error to tead the frame")


    #find features
    feat1 = cv2.goodFeaturesToTrack(gray1, maxCorners=100, qualityLevel=0.3, minDistance=7)
    feat2, status, error = cv2.calcOpticalFlowPyrLK(gray1, gray2, feat1, None)

    rowindex= np.array([ i>2 for i in error ])
    feat1_ext = feat1[rowindex]
    feat2_ext = feat2[rowindex]
    feat1_hist.append(feat1_ext)
    feat2_hist.append(feat2_ext)

    stop=len(feat1_hist)-30 if len(feat1_hist)>30 else 0
    start=len(feat1_hist)-1
    for j in range(start,stop,-1):
        for i in range(len(feat1_hist[j])):
            cv2.line(img2, (feat1_hist[j][i][0], feat1_hist[j][i][1]), (feat2_hist[j][i][0], feat2_hist[j][i][1]), (0, 255, 0), 2)
            
    (x, y, w, h) = cv2.boundingRect(feat1_ext)
    cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 2)    

    cv2.imshow('image', img2)
    #cv2.waitKey(20) 
 
    if cv2.waitKey(20) == ord('q'): 
        cap.release()
        cv2.destroyAllWindows()
        break 

    gray1 = gray2       






