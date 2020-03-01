import cv2
import numpy as np
from matplotlib import pyplot as plt

def compare(template,img):
    sum=0
    delta=cv2.absdiff(template,img)
    h,w=delta.shape
    for i in range(h):
        for j in range(w):
            sum+=delta[i][j]
    return sum

def scan(template,img_span,kenalsize,step):
    h,w=img_span.shape
    min=255*h*h

    for i in range(0,w-kenalsize,step):
        s=compare(template,img_span[:,i:i+h])
        if s<min: min=s
    return min

def full_match(img1,img2,kenalsize,step):
    H,W=img1.shape
    matrix=np.zeros(img1.shape)

    for i in range(0,H-kenalsize,step):
        img_span=img2[i:i+kenalsize,:]
        print("scaning row:{}".format(i))

        for j in range(0,W-kenalsize,step):
            template=img1[i:i+kenalsize,j:j+kenalsize]
            value=scan(template,img_span,kenalsize,step)
            matrix[i:i+kenalsize,j:j+kenalsize]=value
    
    # res_matrix=np.zeros(matrix.shape)
    # maxi=np.max(matrix)
    # mini=np.min(matrix) 
    # for r in range(H):
    #     for c in range(W):
    #         res_matrix[r,c] = matrix[r,c]/(maxi-mini)*255
 
    return matrix



img_left = cv2.imread("tsukuba_left.png")
gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)

img_right = cv2.imread("tsukuba_right.png")
gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

res=full_match(gray_left,gray_right,7,4)
print(res)


plt.imshow(res,cmap='Greys')
plt.show()
#what this picture? how to change this to disparity picture?
