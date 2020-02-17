import cv2
import numpy as np


def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

cap = cv2.VideoCapture('Robots.mp4')

while True:

    #read two continous frames
    ret, frame1 = cap.read()
    if ret==True:
        b,g,r = cv2.split(frame1)
        img1 = cv2.merge([r,g,b])
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    else:
        print("error to read the frame")

    ret, frame2 = cap.read()
    if ret == True:
        b,g,r = cv2.split(frame2)
        img2 = cv2.merge([r,g,b])
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    else:
        print("error to tead the frame")


    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.5, 0)
    img2=draw_flow(gray2, flow)

    cv2.imshow('image', img2)
    #cv2.waitKey(20)  
    key = cv2.waitKey(20)
    if key == ord('q'): 
        cap.release()
        cv2.destroyAllWindows()
        break
   






