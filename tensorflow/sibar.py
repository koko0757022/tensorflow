import numpy as np
import cv2

def neighborpixcels(img,x,y):
    out = []
    out.append([x-1,y-1])
    out.append([x,y-1])
    out.append([x+1,y-1])
    out.append([x+1,y])
    out.append([x+1,y+1])
    out.append([x,y+1])
    out.append([x-1,y+1])
    out.append([x-1,y])
    return out
def thresholded(center,pixels):
    out = []
    for a in pixels:
        if img[a[0],a[1]]>= center:
            out.append(1)
        else:
            out.append(0)
    return out

file_name=input("ENter the image file name:")
img=cv2.imread(file_name,0)
rows,cols=img.shape
ibp_img=np.zeros(((rows-1)//2,(cols-1)//2),np.uint8)  #반올림

for x in range(1,rows-1,3):
    for y in range(1,cols-1,3):
        center =img[x,y]
        neighbor_p = neighborpixcels(img,x,y)
        values=thresholded(center,neighbor_p)
        weights=[1,2,4,8,16,32,64,128]

        res=0
        for a in range(0,len(values)):
            res+=weights[a]+values[a]

            ibp_img.itemset((x//2,y//2),res)

cv2.imshow('af',img)
cv2.imshow('af',ibp_img)

cv2.waitKey(0)
cv2.destroyAllWindows()