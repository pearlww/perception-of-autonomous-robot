import cv2
from matplotlib import pyplot as plt

def compare(template,img):
    sum=0
    delta=cv2.absdiff(template,img)
    h,w=delta.shape
    for i in range(h):
        for j in range(w):
            sum+=delta[i][j]
    return sum

def scan(template,img_span,step):
    h,w=img_span.shape
    min=255*h*h

    for i in range(0,w-step,step):
        s=compare(template,img_span[:,i:i+h])
        if s<min: min=s
    return min

plt.gray()
img = cv2.imread("nose_left.png")

img_left = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray_left = cv2.cvtColor(img_left, cv2.COLOR_RGB2GRAY)


span = cv2.imread("nose_span.png")
gray_span = cv2.cvtColor(span, cv2.COLOR_BGR2GRAY)

f, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10,10))
ax_left.imshow(gray_left)
ax_right.imshow(gray_span)
plt.show()


value=scan(gray_left,gray_span,7)
print(value)
